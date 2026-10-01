"""Sandbox-rule emulator. Compile only, no GPU.

The target environment runs our Python file through a RestrictedPython-style rewrite. Observed rules:
functions decorated with triton.jit / triton_dist.jit are left as written (else no Triton kernel
could compile); everything else is rewritten (attribute access -> _getattr_(obj, "name"),
subscripts -> _getitem_, ...) and attribute names that start with "_" plus annotated
assignments (x: T = v) are rejected. Kernels with any other decorator
are NOT exempt: their bodies get rewritten and no longer compile. Also: the target harness catches
an exception on one rank while the other ranks hang, so never raise on one rank only.
Use this to check that a new kernel file still passes before it goes into the production file.

This script rewrites a whole module the same way (mode skip = target-like; all = also rewrite the
Triton kernels; none = control), loads it, and compiles the module's kernels listed in TARGETS
for sm_90a with a stub driver (no GPU needed). (The TMA kernels take tensor descriptors and are
not compiled here; the rewrite check covers them: decorated @triton.jit, so left as written.)
usage: python rp_check.py <file.py> <skip|all|none> <outdir>
Output per target: "COMPILED ..." or "FAILED ..."; first line tells whether the rewrite itself
was accepted. Needs: pip install RestrictedPython.
"""
import ast
import importlib.util
import os
import sys

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

SKIP = {"triton.jit", "triton_dist.jit"}

# kernel name -> (pointer signature, constexprs, num_warps); scalars default to i32
P_REF = dict(xq_ptr="*i32", sx_ptr="*fp32", tok_ptr="*i64", rw_ptr="*fp32", wg_ptr="*i32",
             wu_ptr="*i32", sg_ptr="*fp32", su_ptr="*fp32", counts_ptr="*i32",
             segstart_ptr="*i32", a_ptr="*bf16", amax_ptr="*fp32", map_ptr="*i32")
P_DN = dict(aq_ptr="*i8", sa_ptr="*fp32", wd_ptr="*i8", sd_ptr="*fp32", counts_ptr="*i32",
            segstart_ptr="*i32", c_ptr="*bf16", map_ptr="*i32")
TARGETS = {
    "_epk_gate_up_kernel": (P_REF, dict(BM=128, BN=128, BK=128, GROUP_M=8, IN_I8=False, GUW=True,
                                        R32=True), 8),
    "_epk_dn2_kernel": (P_DN, dict(BM=128, BMT=128, BN=256, BK=128, GROUP_M=8, TL=True), 8),
}


def transform(src, mode):
    if mode == "none":
        return src
    from RestrictedPython.transformer import RestrictingNodeTransformer

    class Policy(RestrictingNodeTransformer):
        def check_name(self, node, name, allow_magic_methods=False):
            return  # (module-level "_" names are tolerated here; attributes are still checked)

        def visit_AugAssign(self, node):
            if isinstance(node.target, (ast.Subscript, ast.Attribute)):
                return self.node_contents_visit(node)
            return super().visit_AugAssign(node)

        def visit_FunctionDef(self, node):
            if mode == "skip":
                for d in node.decorator_list:
                    f = d.func if isinstance(d, ast.Call) else d
                    if ast.unparse(f) in SKIP:
                        return node
            return super().visit_FunctionDef(node)

    tree = ast.parse(src)
    errors = []
    tree = Policy(errors, [], {}).visit(tree)
    if errors:
        raise SyntaxError("; ".join(errors[:5]))
    ast.fix_missing_locations(tree)

    def conv(e):
        if isinstance(e, ast.Slice):
            return ast.Call(ast.Name("slice", ast.Load()), [e.lower or ast.Constant(None),
                            e.upper or ast.Constant(None), e.step or ast.Constant(None)], [])
        if isinstance(e, ast.Tuple):
            e.elts = [conv(x) for x in e.elts]
        return e

    class Sl(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "_getitem_":
                node.args = [conv(x) for x in node.args]
            return node
    tree = Sl().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def main():
    src_path, mode, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(src_path)))
    try:
        txt = transform(open(src_path).read(), mode)
    except SyntaxError as e:
        print(f"[{mode}] REWRITE REJECTED: {e}")
        return 1
    tp = os.path.join(out, "tx_" + mode + "_" + os.path.basename(src_path))
    open(tp, "w").write(txt)
    name = os.path.basename(tp)[:-3]
    spec = importlib.util.spec_from_file_location(name, tp)
    m = importlib.util.module_from_spec(spec)
    from RestrictedPython import Guards
    m._getattr_ = Guards.safer_getattr
    m._getitem_ = lambda o, k: o[k]
    m._write_ = lambda o: o
    m._getiter_ = iter
    m._inplacevar_ = lambda op, x, y: {"+=": lambda: x + y, "-=": lambda: x - y,
                                       "*=": lambda: x * y}[op]()
    sys.modules[name] = m
    spec.loader.exec_module(m)
    print(f"[{mode}] rewrite accepted, module loaded", flush=True)
    tgt = GPUTarget("cuda", 90, 32)
    from triton.runtime import driver as drv

    class Stub:
        def get_current_target(self):
            return tgt
    drv.set_active(Stub())
    for kname, (ptrs, cx, nw) in TARGETS.items():
        fn = getattr(m, kname, None)
        if fn is None:
            continue
        sig, consts, attrs = {}, {}, {}
        for i, n in enumerate(fn.arg_names):
            if n in cx:
                sig[n] = "constexpr"
                consts[(i,)] = cx[n]
            else:
                sig[n] = ptrs.get(n, "i32")
                if n in ptrs or n in ("H", "I"):
                    attrs[(i,)] = [["tt.divisibility", 16]]
        try:
            k = triton.compile(ASTSource(fn, sig, consts, attrs), target=tgt,
                               options=dict(num_warps=nw))
            print(f"[{mode}] {kname}: COMPILED (shared {k.metadata.shared} B, "
                  f"wgmma in PTX: {k.asm['ptx'].count('wgmma.mma_async')})", flush=True)
        except Exception as e:
            print(f"[{mode}] {kname}: FAILED {type(e).__name__}: {str(e)[-400:]!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

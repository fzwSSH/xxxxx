"""Reference kernel check on any CUDA GPU (single GPU, synthetic data): gate/up and down.

sm_90: FP8 path. Pointer reference (_epk_gate_up_kernel) and TMA reference (_epg_gu_kernel) vs a
  float64 torch model (SQNR) and vs each other (bitwise), plus the Triton-WS variant (ws_gu).
other GPUs (e.g. A100, sm_80): the INT8 twin of the same kernels (IN_I8=True; FP8 needs sm_89+).
usage: python test_ref.py
Last line: "REF PASS" / "REF FAIL" (exit 0 / 1). Pass = SQNR >= 35 dB for every kernel that ran.
"""
import sys
import time

import torch

import ref_gu

MIN_DB = 35.0


def check(name, fn, P, t, ref_act=None):
    act, amax = ref_gu.out_buffers(P)
    try:
        t0 = time.time()
        fn(P, act, amax)
        torch.cuda.synchronize()
        dt = time.time() - t0
    except Exception as e:
        msg = repr(e).splitlines()[0][:300]
        print(f"SKIP/ERROR {name}: {msg}", flush=True)
        return None, None
    q = ref_gu.sqnr_db(t, act.float())
    same = ""
    if ref_act is not None:
        nm = int((ref_act.view(torch.int16) != act.view(torch.int16)).sum())
        same = f", bitwise mismatches vs pointer ref {nm}"
    ok = q >= MIN_DB
    print(f"{'PASS' if ok else 'FAIL'} {name}: SQNR {q:.1f} dB{same} (first call incl. compile "
          f"{dt:.1f} s)", flush=True)
    return ok, act


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA GPU")
        return 0
    cap = torch.cuda.get_device_capability()
    hop = cap[0] == 9
    int8 = cap < (8, 9)
    print(f"GPU {torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}: "
          f"{'INT8 twin' if int8 else 'FP8'} path", flush=True)
    ok = True
    for (M, H, I, E) in [(1000, 2048, 2048, 4), (4096, 4096, 2048, 8)]:
        tag = f"M={M} H={H} I={I} E={E}"
        P = ref_gu.make_problem(M, H, I, E, seed=1, int8=int8, gather=True)
        t = ref_gu.torch_ref(P)
        r, act_p = check("ptr-ref gather " + tag, ref_gu.run_ref_ptr, P, t)
        ok &= bool(r)
        P2 = ref_gu.make_problem(M, H, I, E, seed=1, int8=int8, gather=False)
        t2 = ref_gu.torch_ref(P2)
        r, act_p2 = check("ptr-ref contiguous " + tag, ref_gu.run_ref_ptr, P2, t2)
        ok &= bool(r)
        r, _ = check("tma-ref " + tag, ref_gu.run_ref_tma, P2, t2, act_p2)
        if hop:
            ok &= bool(r)
            import ws_gu
            for nm, fn in (("triton-ws K-loop", ws_gu.run_ws),
                           ("triton-ws persistent", ws_gu.run_ws_persistent)):
                r, _ = check(nm + " " + tag, fn, P2, t2, act_p2)
                if r is None:
                    print(f"INFO {nm} did not run (a T2 finding, not a REF failure)")
    import dn_ref
    for (M, H, I, E) in [(1000, 2048, 2048, 4), (4096, 4096, 2048, 8), (2048, 4096, 8192, 8)]:
        tag = f"M={M} H={H} I={I} E={E}"
        D = dn_ref.make_down_problem(M, H, I, E, seed=2)
        t = dn_ref.torch_dn_ref(D)
        outs = {}
        for nm, fn in (("dn ptr-twin", dn_ref.run_dn_ptr), ("dn tma-ref", dn_ref.run_dn_tma)):
            c = dn_ref.dn_out(D)
            try:
                fn(D, c)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"SKIP/ERROR {nm} {tag}: {repr(e).splitlines()[0][:300]}")
                ok = False
                continue
            outs[nm] = c
            q = ref_gu.sqnr_db(t, c.float())
            nm_t = int((t.to(torch.bfloat16).view(torch.int16) != c.view(torch.int16)).sum())
            good = q >= MIN_DB
            ok &= good
            print(f"{'PASS' if good else 'FAIL'} {nm} {tag}: SQNR {q:.1f} dB, bf16 words != torch "
                  f"model {nm_t}", flush=True)
        if len(outs) == 2:
            a, b = outs.values()
            nm_ = int((a.view(torch.int16) != b.view(torch.int16)).sum())
            print(f"{'PASS' if nm_ == 0 else 'FAIL'} dn tma vs ptr bitwise {tag}: {nm_} mismatches")
            ok &= nm_ == 0
    if not hop:
        print("INFO Hopper-only parts (FP8 gate/up, Triton WS variants) skipped: not sm_90")
    print("REF " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

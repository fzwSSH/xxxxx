"""Reference FP8 MoE gate/up GEMM (Triton 3.4) + cache packing + synthetic problem builder.

Kernels (copied unchanged from the production build, "od15"; only @triton_dist.jit -> @triton.jit):
  _epk_gate_up_kernel  pointer loads, rows gathered through tok[] (works on sm_80 with IN_I8=True)
  _epg_gu_kernel       TMA loads through host TensorDescriptors (contiguous A rows)
Both compute, per routed row p (expert e) and activation channel c:
  acc_g = x[tok[p]] . Wg[e, c],  acc_u = x[tok[p]] . Wu[e, c]          (FP8 x FP8 -> FP32)
  g = bf16(acc_g * sx * sg[e, c]),  u = bf16(acc_u * sx * su[e, c])
  act = bf16(silu(g) * u * rw[p])
  out = bf16(act @ blockdiag(Had32))  (R32=True)  and  amax[p, c // 128] = max |out| (FP32)
Weight cache layout (GUW=True): ONE interleaved [E * 2I, H] 8-bit tensor, row 2c = gate channel
c, row 2c + 1 = up channel c of expert e (rows e * 2I ...); per-channel scales sg / su [E * I].
Tile map: one int32 per M tile of BM=128 rows: expert << 16 | tile index inside the expert
(-1 = no tile).
"""
import math

import torch
import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except Exception:  # pragma: no cover
    TensorDescriptor = None

BM = 128
FP8_MAX = 448.0


def is_hopper():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@triton.jit
def _had128():
    # 128x128 Sylvester Hadamard, +-1 entries: Had[i, j] = (-1)^popcount(i & j). BF16.
    i = tl.arange(0, 128)
    p = i[:, None] & i[None, :]
    p = p ^ (p >> 4)
    p = p ^ (p >> 2)
    p = p ^ (p >> 1)
    return (1 - 2 * (p & 1)).to(tl.bfloat16)


@triton.jit
def _map_tile(pid_m, map_ptr):
    # (expert, tile index inside the expert) of M-tile pid_m; expert = -1 past the end.
    v = tl.load(map_ptr + pid_m)
    return tl.where(v >= 0, v >> 16, -1), v & 65535


@triton.jit
def _had32():
    # 32x32 Sylvester Hadamard, +-1 entries: Had[i, j] = (-1)^popcount(i & j). BF16.
    i = tl.arange(0, 32)
    p = i[:, None] & i[None, :]
    p = p ^ (p >> 4)
    p = p ^ (p >> 2)
    p = p ^ (p >> 1)
    return (1 - 2 * (p & 1)).to(tl.bfloat16)


@triton.jit
def _rot32(v, ROWS: tl.constexpr, COLS: tl.constexpr):
    # H32: v [ROWS, COLS] (BF16) @ blockdiag(Had32) -> FP32. Exact products, FP32 sums.
    return tl.reshape(tl.dot(tl.reshape(v, (ROWS * COLS // 32, 32)), _had32()), (ROWS, COLS))


@triton.jit
def _epk_map2(pid_m, map_ptr):
    # (expert, first row of the tile inside the expert, tail flag) of M-tile pid_m;
    # expert = -1 past the end. Entry = tail << 30 | expert << 16 | tile index (tiles of BMR rows).
    v = tl.load(map_ptr + pid_m)
    return tl.where(v >= 0, (v >> 16) & 16383, -1), v & 65535, (v >> 30) & 1


@triton.jit(do_not_specialize=["E", "NUM_N"])
def _epk_gate_up_kernel(xq_ptr, sx_ptr, tok_ptr, rw_ptr, wg_ptr, wu_ptr, sg_ptr, su_ptr,
                        counts_ptr, segstart_ptr, a_ptr, amax_ptr, E, H, I, NUM_N, map_ptr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        GROUP_M: tl.constexpr, IN_I8: tl.constexpr, GUW: tl.constexpr,
                        R32: tl.constexpr):
    # _gate_up_i8_kernel (ROT = True) with the A rows gathered from the per-token 8-bit x:
    # row p reads x row tok[p] and its scale sx[tok[p]]. FP8 operands with FP32
    # accumulation (IN_I8 False) or INT8 with INT32 (A100 twin). Epilogue:
    # g = bf16(sx * sg * acc_g), u likewise, a = bf16(silu(g) * u * rw), stored rotated
    # (a @ Had128) as BF16 plus its per-row amax amax[p, pid_n].
    if IN_I8:
        xq = xq_ptr.to(tl.pointer_type(tl.int8), bitcast=True)
        wg = wg_ptr.to(tl.pointer_type(tl.int8), bitcast=True)
        wu = wu_ptr.to(tl.pointer_type(tl.int8), bitcast=True)
    else:
        xq = xq_ptr.to(tl.pointer_type(tl.float8e4nv), bitcast=True)
        wg = wg_ptr.to(tl.pointer_type(tl.float8e4nv), bitcast=True)
        wu = wu_ptr.to(tl.pointer_type(tl.float8e4nv), bitcast=True)
    pid = tl.program_id(0)
    num_pid_m = tl.num_programs(0) // NUM_N
    num_in_group = GROUP_M * NUM_N
    group_id = pid // num_in_group
    first_m = group_id * GROUP_M
    gsize = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % num_in_group) % gsize
    pid_n = (pid % num_in_group) // gsize
    e_sel, tile_in = _map_tile(pid_m, map_ptr)
    if e_sel >= 0:
        cnt = tl.load(counts_ptr + e_sel)
        seg = tl.load(segstart_ptr + e_sel)
        local_rows = tile_in * BM + tl.arange(0, BM)
        rmask = local_rows < cnt
        rows = (seg + local_rows).to(tl.int64)
        xrow = tl.load(tok_ptr + rows, mask=rmask, other=0)
        offs_n = pid_n * BN + tl.arange(0, BN)
        nmask = offs_n < I
        offs_k = tl.arange(0, BK)
        wbase = e_sel.to(tl.int64) * I * H
        if GUW:
            # interleaved caches: rows 2c / 2c + 1 = gate / up of channel c; one (BK x 2BN) tile
            offs_w = 2 * pid_n * BN + tl.arange(0, 2 * BN)
            wmk = offs_w < 2 * I
            if IN_I8:
                acc = tl.zeros((BM, 2 * BN), dtype=tl.int32)
            else:
                acc = tl.zeros((BM, 2 * BN), dtype=tl.float32)
            for k0 in range(0, H, BK):
                kk = k0 + offs_k
                kmask = kk < H
                woff = 2 * wbase + offs_w[None, :].to(tl.int64) * H + kk[:, None]
                wmask = wmk[None, :] & kmask[:, None]
                if IN_I8:
                    a = tl.load(xq + xrow[:, None] * H + kk[None, :],
                                mask=rmask[:, None] & kmask[None, :], other=0)
                    bw = tl.load(wg + woff, mask=wmask, other=0)
                    acc = tl.dot(a, bw, acc, out_dtype=tl.int32)
                else:
                    a = tl.load(xq + xrow[:, None] * H + kk[None, :],
                                mask=rmask[:, None] & kmask[None, :], other=0.0)
                    bw = tl.load(wg + woff, mask=wmask, other=0.0)
                    acc = tl.dot(a, bw, acc)
            acc_g, acc_u = tl.split(tl.reshape(acc, (BM, BN, 2)))
        else:
            if IN_I8:
                acc_g = tl.zeros((BM, BN), dtype=tl.int32)
                acc_u = tl.zeros((BM, BN), dtype=tl.int32)
            else:
                acc_g = tl.zeros((BM, BN), dtype=tl.float32)
                acc_u = tl.zeros((BM, BN), dtype=tl.float32)
            for k0 in range(0, H, BK):
                kk = k0 + offs_k
                kmask = kk < H
                woff = wbase + offs_n[None, :].to(tl.int64) * H + kk[:, None]
                wmask = nmask[None, :] & kmask[:, None]
                if IN_I8:
                    a = tl.load(xq + xrow[:, None] * H + kk[None, :],
                                mask=rmask[:, None] & kmask[None, :], other=0)
                    bg = tl.load(wg + woff, mask=wmask, other=0)
                    bu = tl.load(wu + woff, mask=wmask, other=0)
                    acc_g = tl.dot(a, bg, acc_g, out_dtype=tl.int32)
                    acc_u = tl.dot(a, bu, acc_u, out_dtype=tl.int32)
                else:
                    a = tl.load(xq + xrow[:, None] * H + kk[None, :],
                                mask=rmask[:, None] & kmask[None, :], other=0.0)
                    bg = tl.load(wg + woff, mask=wmask, other=0.0)
                    bu = tl.load(wu + woff, mask=wmask, other=0.0)
                    acc_g = tl.dot(a, bg, acc_g)
                    acc_u = tl.dot(a, bu, acc_u)
        sx = tl.load(sx_ptr + xrow, mask=rmask, other=0.0)
        rw = tl.load(rw_ptr + rows, mask=rmask, other=0.0)
        cidx = e_sel.to(tl.int64) * I + offs_n
        sg = tl.load(sg_ptr + cidx, mask=nmask, other=0.0)
        su = tl.load(su_ptr + cidx, mask=nmask, other=0.0)
        g = (acc_g.to(tl.float32) * sx[:, None] * sg[None, :]).to(tl.bfloat16).to(tl.float32)
        u = (acc_u.to(tl.float32) * sx[:, None] * su[None, :]).to(tl.bfloat16).to(tl.float32)
        silu = g / (1.0 + tl.exp(-g))
        act = (silu * u * rw[:, None]).to(tl.bfloat16)
        if R32:
            ar = _rot32(act, BM, BN)
        else:
            ar = tl.dot(act, _had128())
        tl.store(a_ptr + rows[:, None] * I + offs_n[None, :], ar.to(tl.bfloat16),
                 mask=rmask[:, None] & nmask[None, :])
        tl.store(amax_ptr + rows * NUM_N + pid_n, tl.max(tl.abs(ar), axis=1), mask=rmask)


# ---------------------------------------------------------------------------------------
# ep_g GEMMs: TMA loads (host tensor descriptors) of the A rows and the 8-bit weight tiles.
# A: the routed rows in expert order (row p of xs / aq is GEMM row p), descriptor [r_max, K]
# with box [BM, 128] (and [BMT, 128] for tail tiles); rows past the expert's count belong to
# the next expert (computed, never stored: every output row depends only on its own A row) and
# rows past r_max / columns past K are zero-filled by the TMA unit. B: the weight caches of
# this rank's experts, K-major: plain [rows, K] with box [BN, 128], or the blocked layout
# (TL: (128 x 128) gate/up, (256 x 128) down blocks) viewed as rows of 128 bytes, box [BN, 128].
# Same tiles, k order (BK = 128) and epilogue as _epk_gu_tile / _epk_dn_tile.
@triton.jit
def _epg_gu_tile(a_desc, g_desc, u_desc, sx_ptr, tok_ptr, rw_ptr, sg_ptr, su_ptr, a_ptr,
                 amax_ptr, e_sel, r0, cnt, seg, pid_n, H, I, NUM_N, KP,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, IN_I8: tl.constexpr,
                 TL: tl.constexpr, GUW: tl.constexpr, R32: tl.constexpr):
    # GUW: g_desc is the interleaved gate/up cache (box [2BN, 128]); one dot per k step.
    local_rows = r0 + tl.arange(0, BM)
    rmask = local_rows < cnt
    rows = (seg + local_rows).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    nmask = offs_n < I
    arow = seg + r0
    if GUW:
        n0 = e_sel * I + pid_n * BN
        if TL:
            # (256 x 128) blocks of interleaved rows: interleaved row 2 * n0 of the tile, k
            # block kb at 128-byte row (n0 // 128) * 2 * KP + kb * 256 + 2 * (n0 % 128)
            brow = (n0 // 128) * 2 * KP + 2 * (n0 % 128)
        else:
            brow = 2 * n0
        if IN_I8:
            acc = tl.zeros((BM, 2 * BN), dtype=tl.int32)
        else:
            acc = tl.zeros((BM, 2 * BN), dtype=tl.float32)
        for k0 in range(0, H, BK):
            a = a_desc.load([arow, k0])
            if TL:
                bw = g_desc.load([brow + 2 * k0, 0])
            else:
                bw = g_desc.load([brow, k0])
            if IN_I8:
                acc = tl.dot(a, bw.T, acc, out_dtype=tl.int32)
            else:
                acc = tl.dot(a, bw.T, acc)
        acc_g, acc_u = tl.split(tl.reshape(acc, (BM, BN, 2)))
    else:
        if TL:
            # (128 x 128) blocks: row n of the tile, k block kb at 128-byte row
            # (n0 // 128) * KP + kb * 128 + n0 % 128 (n0 = first output channel of the tile)
            n0 = e_sel * I + pid_n * BN
            brow = (n0 // 128) * KP + n0 % 128
        else:
            brow = e_sel * I + pid_n * BN
        if IN_I8:
            acc_g = tl.zeros((BM, BN), dtype=tl.int32)
            acc_u = tl.zeros((BM, BN), dtype=tl.int32)
        else:
            acc_g = tl.zeros((BM, BN), dtype=tl.float32)
            acc_u = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, H, BK):
            a = a_desc.load([arow, k0])
            if TL:
                bg = g_desc.load([brow + k0, 0])
                bu = u_desc.load([brow + k0, 0])
            else:
                bg = g_desc.load([brow, k0])
                bu = u_desc.load([brow, k0])
            if IN_I8:
                acc_g = tl.dot(a, bg.T, acc_g, out_dtype=tl.int32)
                acc_u = tl.dot(a, bu.T, acc_u, out_dtype=tl.int32)
            else:
                acc_g = tl.dot(a, bg.T, acc_g)
                acc_u = tl.dot(a, bu.T, acc_u)
    xrow = tl.load(tok_ptr + rows, mask=rmask, other=0)
    sx = tl.load(sx_ptr + xrow, mask=rmask, other=0.0)
    rw = tl.load(rw_ptr + rows, mask=rmask, other=0.0)
    cidx = e_sel.to(tl.int64) * I + offs_n
    sg = tl.load(sg_ptr + cidx, mask=nmask, other=0.0)
    su = tl.load(su_ptr + cidx, mask=nmask, other=0.0)
    g = (acc_g.to(tl.float32) * sx[:, None] * sg[None, :]).to(tl.bfloat16).to(tl.float32)
    u = (acc_u.to(tl.float32) * sx[:, None] * su[None, :]).to(tl.bfloat16).to(tl.float32)
    silu = g / (1.0 + tl.exp(-g))
    act = (silu * u * rw[:, None]).to(tl.bfloat16)
    if R32:
        ar = _rot32(act, BM, BN)
    else:
        ar = tl.dot(act, _had128())
    tl.store(a_ptr + rows[:, None] * I + offs_n[None, :], ar.to(tl.bfloat16),
             mask=rmask[:, None] & nmask[None, :])
    tl.store(amax_ptr + rows * NUM_N + pid_n, tl.max(tl.abs(ar), axis=1), mask=rmask)


@triton.jit(do_not_specialize=["E", "NUM_N"])
def _epg_gu_kernel(a_desc, at_desc, g_desc, u_desc, sx_ptr, tok_ptr, rw_ptr, sg_ptr, su_ptr,
                   counts_ptr, segstart_ptr, a_ptr, amax_ptr, E, H, I, NUM_N, map_ptr, KP,
                   BM: tl.constexpr, BMT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   GROUP_M: tl.constexpr, IN_I8: tl.constexpr, TL: tl.constexpr,
                   GUW: tl.constexpr, R32: tl.constexpr):
    # _epk_gu2_kernel with TMA loads: at_desc = the A descriptor with a BMT-row box (tail
    # tiles, BMT < BM), TL = blocked weight caches. BMT == BM, TL False: the v1 kernel.
    pid = tl.program_id(0)
    num_pid_m = tl.num_programs(0) // NUM_N
    num_in_group = GROUP_M * NUM_N
    group_id = pid // num_in_group
    first_m = group_id * GROUP_M
    gsize = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % num_in_group) % gsize
    pid_n = (pid % num_in_group) // gsize
    e_sel, tile_in, tail = _epk_map2(pid_m, map_ptr)
    if e_sel >= 0:
        cnt = tl.load(counts_ptr + e_sel)
        seg = tl.load(segstart_ptr + e_sel)
        if BMT < BM and tail != 0:
            _epg_gu_tile(at_desc, g_desc, u_desc, sx_ptr, tok_ptr, rw_ptr, sg_ptr, su_ptr, a_ptr,
                         amax_ptr, e_sel, tile_in * BM, cnt, seg, pid_n, H, I, NUM_N, KP, BMT,
                         BN, BK, IN_I8, TL, GUW, R32)
        else:
            _epg_gu_tile(a_desc, g_desc, u_desc, sx_ptr, tok_ptr, rw_ptr, sg_ptr, su_ptr, a_ptr,
                         amax_ptr, e_sel, tile_in * BM, cnt, seg, pid_n, H, I, NUM_N, KP, BM,
                         BN, BK, IN_I8, TL, GUW, R32)


@triton.jit
def _epk_boff(rowp, cols, KP, RB: tl.constexpr, TL: tl.constexpr):
    # Offset of (row, col) in an 8-bit weight cache: TL: (RB x 128) blocks (row block, then
    # K block, row-major inside; rows KP long, KP % 128 == 0), else plain row-major (KP = K).
    if TL:
        return ((rowp // RB) * (RB * KP) + (rowp % RB) * 128)[:, None] + \
            ((cols // 128) * (RB * 128) + cols % 128)[None, :]
    else:
        return rowp[:, None] * KP + cols[None, :]


# --------------------------------------------------------------------------------------------
# Cache packing (gate/up part of the production warm-up quantizer _epk_wquant2_kernel, in torch;
# same formulas: per-channel scale s = max(amax, 1e-12) / 448, q = clamp(w / s, +-448) -> e4m3,
# round-to-nearest-even). INT8 variant (A100 twin): s = amax / 127, q = round(w / s).
# --------------------------------------------------------------------------------------------
def pack_gate_up(wg, wu, int8=False):
    """wg, wu: [E, I, H] (bf16/fp32). Returns (q [E * 2I, H] fp8/int8 interleaved, sg [E*I],
    su [E*I] fp32)."""
    E, I, H = wg.shape

    def q1(w):
        w = w.float()
        amax = w.abs().amax(dim=2)
        if int8:
            s = torch.clamp(amax, min=1e-12) / 127.0
            q = torch.round(w / s[:, :, None]).clamp(-127, 127).to(torch.int8)
        else:
            s = torch.clamp(amax, min=1e-12) / FP8_MAX
            q = torch.clamp(w / s[:, :, None], -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return q, s.reshape(-1).contiguous()

    qg, sg = q1(wg)
    qu, su = q1(wu)
    q = torch.stack([qg, qu], dim=2).reshape(E * 2 * I, H).contiguous()
    return q, sg, su


def quant_x(x, int8=False):
    """Per-row 8-bit activation quant: x [T, H] -> (xq [T, H] fp8/int8, sx [T] fp32)."""
    x = x.float()
    amax = torch.clamp(x.abs().amax(dim=1), min=1e-12)
    if int8:
        s = amax / 127.0
        return torch.round(x / s[:, None]).clamp(-127, 127).to(torch.int8), s.contiguous()
    s = amax / FP8_MAX
    return torch.clamp(x / s[:, None], -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), s.contiguous()


def tile_map(counts, bm=BM):
    """counts: list of rows per expert -> int32 map (expert << 16 | tile), MT entries."""
    m = []
    for e, c in enumerate(counts):
        for t in range((c + bm - 1) // bm):
            m.append((e << 16) | t)
    if not m:
        m = [-1]
    return m


def make_problem(M, H, I, E, seed=0, skew=0.0, int8=False, gather=True, dev="cuda"):
    """Synthetic routed problem with M routed rows over E local experts.
    gather=True: rows read x through a permuted tok[] (as after dispatch, xrow gather);
    gather=False: A rows are already contiguous in routed order (tok = iota), as the TMA kernel needs.
    skew > 0 puts more rows on low expert ids (Zipf-like weights i^-skew)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    wts = torch.tensor([(i + 1) ** (-skew) for i in range(E)], dtype=torch.float64)
    counts = torch.floor(wts / wts.sum() * M).to(torch.int64)
    counts[0] += M - int(counts.sum())
    counts = [int(c) for c in counts]
    seg = [0]
    for c in counts[:-1]:
        seg.append(seg[-1] + c)
    T = M
    x = torch.randn(T, H, generator=g).to(torch.bfloat16)
    wg = (torch.randn(E, I, H, generator=g) * 0.02).to(torch.bfloat16)
    wu = (torch.randn(E, I, H, generator=g) * 0.02).to(torch.bfloat16)
    tok = torch.randperm(T, generator=g) if gather else torch.arange(T)
    rw = torch.rand(M, generator=g) * 0.5 + 0.25
    xq, sx = quant_x(x, int8)
    q, sg, su = pack_gate_up(wg, wu, int8)
    tm = tile_map(counts)
    P = dict(M=M, H=H, I=I, E=E, int8=int8, counts=counts, seg=seg,
             xq=xq.to(dev), sx=sx.to(dev), q=q.to(dev), sg=sg.to(dev), su=su.to(dev),
             tok=tok.to(dev), rw=rw.float().to(dev),
             cnt_t=torch.tensor(counts, dtype=torch.int32, device=dev),
             seg_t=torch.tensor(seg, dtype=torch.int32, device=dev),
             map_t=torch.tensor(tm, dtype=torch.int32, device=dev), mt=len(tm))
    return P


def out_buffers(P):
    dev = P["xq"].device
    act = torch.empty((P["M"], P["I"]), dtype=torch.bfloat16, device=dev)
    amax = torch.empty((P["M"] * (P["I"] // 64),), dtype=torch.float32, device=dev)
    return act, amax


def i32(t):
    return t.view(torch.int32)


def run_ref_ptr(P, act, amax, num_stages=4):
    """The pointer reference (_epk_gate_up_kernel, od15 v1 tiles: 128 x 128 x 128, GROUP_M 8)."""
    nn = P["I"] // 128
    _epk_gate_up_kernel[(P["mt"] * nn,)](
        i32(P["xq"]), P["sx"], P["tok"], P["rw"], i32(P["q"]), i32(P["q"]), P["sg"], P["su"],
        P["cnt_t"], P["seg_t"], act, amax, P["E"], P["H"], P["I"], nn, P["map_t"], BM=BM,
        BN=128, BK=128, GROUP_M=8, IN_I8=P["int8"], GUW=True, R32=True, num_warps=8,
        num_stages=num_stages)


def map_for(P, bm):
    """Tile map of problem P for M tiles of bm rows (cached in P)."""
    key = "map_%d" % bm
    if key not in P:
        tm = tile_map(P["counts"], bm)
        P[key] = (torch.tensor(tm, dtype=torch.int32, device=P["xq"].device), len(tm))
    return P[key]


# the od15 TMA gate/up config: EPG_GU = (BM 128, BN 128, num_warps 8, num_stages 4), BK 128,
# GROUP_M 8
GU_CUR = dict(BM=128, BN=128, BK=128, num_warps=8, num_stages=4, GROUP_M=8)


def tma_descs(P, BM=128, BN=128, BK=128):
    """Host descriptors of the TMA gate/up: A [M, H] box [BM, BK], interleaved weights
    [E * 2I, H] box [2 BN, BK]. Needs tok = iota (make_problem(gather=False))."""
    xq, q = P["xq"], P["q"]
    a = TensorDescriptor(xq, [P["M"], P["H"]], [P["H"], 1], [BM, BK])
    w = TensorDescriptor(q, [q.shape[0], P["H"]], [P["H"], 1], [2 * BN, BK])
    return a, w


def run_ref_tma(P, act, amax, descs=None, cfg=None):
    """The TMA gate/up (_epg_gu_kernel; cfg defaults to the od15 config GU_CUR). amax has
    I // BN entries per row."""
    c = dict(GU_CUR)
    if cfg:
        c.update(cfg)
    a, w = descs if descs is not None else tma_descs(P, c["BM"], c["BN"], c["BK"])
    mp, mt = map_for(P, c["BM"])
    nn = P["I"] // c["BN"]
    _epg_gu_kernel[(mt * nn,)](
        a, a, w, w, P["sx"], P["tok"], P["rw"], P["sg"], P["su"], P["cnt_t"], P["seg_t"], act,
        amax, P["E"], P["H"], P["I"], nn, mp, P["H"], BM=c["BM"], BMT=c["BM"], BN=c["BN"],
        BK=c["BK"], GROUP_M=c["GROUP_M"], IN_I8=P["int8"], TL=False, GUW=True, R32=True,
        num_warps=c["num_warps"], num_stages=c["num_stages"])


def had32_t(dev):
    i = torch.arange(32)
    p = i[:, None] & i[None, :]
    par = torch.zeros_like(p)
    for b in range(5):
        par ^= (p >> b) & 1
    return (1 - 2 * par).to(torch.float64).to(dev)


def torch_ref(P):
    """float64 torch model of the same math (for tolerance checks, not bitwise)."""
    H, I = P["H"], P["I"]
    xq = P["xq"].to(torch.float64)
    q = P["q"].to(torch.float64)
    out = torch.zeros((P["M"], I), dtype=torch.float64, device=xq.device)
    for e, (c, s) in enumerate(zip(P["counts"], P["seg"])):
        if c == 0:
            continue
        rows = torch.arange(s, s + c, device=xq.device)
        xr = P["tok"][rows]
        acc = xq[xr] @ q[e * 2 * I:(e + 1) * 2 * I].T
        ag, au = acc[:, 0::2].float(), acc[:, 1::2].float()
        sx = P["sx"][xr][:, None]
        g = (ag * sx * P["sg"][e * I:(e + 1) * I][None, :]).to(torch.bfloat16).float()
        u = (au * sx * P["su"][e * I:(e + 1) * I][None, :]).to(torch.bfloat16).float()
        a = (g / (1.0 + torch.exp(-g)) * u * P["rw"][rows][:, None]).to(torch.bfloat16)
        out[rows] = (a.to(torch.float64).reshape(-1, 32) @ had32_t(xq.device)).reshape(c, I)
    return out


def sqnr_db(ref, got):
    ref = ref.to(torch.float64)
    err = (got.to(torch.float64) - ref).pow(2).sum()
    sig = ref.pow(2).sum()
    if err == 0:
        return math.inf
    return 10.0 * math.log10(float(sig / err))

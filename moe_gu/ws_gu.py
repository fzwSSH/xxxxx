"""T2 candidates: the TMA gate/up (ref_gu._epg_gu_kernel, GUW single-dot path, same tiles, k
order and epilogue) with Triton's built-in automatic warp specialization:
  _ws_gu_kernel       one program per tile, warp_specialize on the K loop
  _ws_gu_pers_kernel  persistent grid (one program per SM), warp_specialize on the tile loop
                      (the pattern of Triton 3.4 tutorial 09 matmul_kernel_descriptor_persistent)
WS=False gives the same kernel without warp specialization (control). Not used in the production
build; untested on Hopper. If Triton 3.4 rejects or ignores warp_specialize here on sm_90a, that
is a valid T2 finding (report the error, or grep the TTGIR for "ttg.warp_specialize").
"""
import torch
import triton
import triton.language as tl

from ref_gu import _epk_map2, _rot32, _had32, BM, tma_descs  # noqa: F401


@triton.jit(do_not_specialize=["E", "NUM_N"])
def _ws_gu_kernel(a_desc, g_desc, sx_ptr, tok_ptr, rw_ptr, sg_ptr, su_ptr, counts_ptr,
                  segstart_ptr, a_ptr, amax_ptr, E, H, I, NUM_N, map_ptr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
                  WS: tl.constexpr, STAGES: tl.constexpr):
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
        r0 = tile_in * BM
        local_rows = r0 + tl.arange(0, BM)
        rmask = local_rows < cnt
        rows = (seg + local_rows).to(tl.int64)
        offs_n = pid_n * BN + tl.arange(0, BN)
        nmask = offs_n < I
        arow = seg + r0
        brow = 2 * (e_sel * I + pid_n * BN)
        acc = tl.zeros((BM, 2 * BN), dtype=tl.float32)
        for k0 in tl.range(0, H, BK, num_stages=STAGES, warp_specialize=WS):
            a = a_desc.load([arow, k0])
            bw = g_desc.load([brow, k0])
            acc = tl.dot(a, bw.T, acc)
        acc_g, acc_u = tl.split(tl.reshape(acc, (BM, BN, 2)))
        xrow = tl.load(tok_ptr + rows, mask=rmask, other=0)
        sx = tl.load(sx_ptr + xrow, mask=rmask, other=0.0)
        rw = tl.load(rw_ptr + rows, mask=rmask, other=0.0)
        cidx = e_sel.to(tl.int64) * I + offs_n
        sg = tl.load(sg_ptr + cidx, mask=nmask, other=0.0)
        su = tl.load(su_ptr + cidx, mask=nmask, other=0.0)
        g = (acc_g * sx[:, None] * sg[None, :]).to(tl.bfloat16).to(tl.float32)
        u = (acc_u * sx[:, None] * su[None, :]).to(tl.bfloat16).to(tl.float32)
        silu = g / (1.0 + tl.exp(-g))
        act = (silu * u * rw[:, None]).to(tl.bfloat16)
        ar = _rot32(act, BM, BN)
        tl.store(a_ptr + rows[:, None] * I + offs_n[None, :], ar.to(tl.bfloat16),
                 mask=rmask[:, None] & nmask[None, :])
        tl.store(amax_ptr + rows * NUM_N + pid_n, tl.max(tl.abs(ar), axis=1), mask=rmask)


def run_ws(P, act, amax, descs=None, ws=True, num_warps=4, stages=4):
    a, w = descs if descs is not None else tma_descs(P)
    nn = P["I"] // 128
    _ws_gu_kernel[(P["mt"] * nn,)](
        a, w, P["sx"], P["tok"], P["rw"], P["sg"], P["su"], P["cnt_t"], P["seg_t"], act, amax,
        P["E"], P["H"], P["I"], nn, P["map_t"], BM=BM, BN=128, BK=128, GROUP_M=8, WS=ws,
        STAGES=stages, num_warps=num_warps)


@triton.jit(do_not_specialize=["E", "NUM_N", "MT"])
def _ws_gu_pers_kernel(a_desc, g_desc, sx_ptr, tok_ptr, rw_ptr, sg_ptr, su_ptr, counts_ptr,
                       segstart_ptr, a_ptr, amax_ptr, E, H, I, NUM_N, map_ptr, MT,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                       GROUP_M: tl.constexpr, WS: tl.constexpr, FLAT: tl.constexpr,
                       STAGES: tl.constexpr):
    # every map entry is a valid tile here (map built with exactly MT entries)
    nprog = tl.num_programs(0)
    num_in_group = GROUP_M * NUM_N
    for t in tl.range(tl.program_id(0), MT * NUM_N, nprog, flatten=FLAT, warp_specialize=WS):
        group_id = t // num_in_group
        first_m = group_id * GROUP_M
        gsize = tl.minimum(MT - first_m, GROUP_M)
        pid_m = first_m + (t % num_in_group) % gsize
        pid_n = (t % num_in_group) // gsize
        v = tl.load(map_ptr + pid_m)
        e_sel = (v >> 16) & 16383
        r0 = (v & 65535) * BM
        cnt = tl.load(counts_ptr + e_sel)
        seg = tl.load(segstart_ptr + e_sel)
        local_rows = r0 + tl.arange(0, BM)
        rmask = local_rows < cnt
        rows = (seg + local_rows).to(tl.int64)
        offs_n = pid_n * BN + tl.arange(0, BN)
        nmask = offs_n < I
        arow = seg + r0
        brow = 2 * (e_sel * I + pid_n * BN)
        acc = tl.zeros((BM, 2 * BN), dtype=tl.float32)
        for k0 in tl.range(0, H, BK, num_stages=STAGES):
            a = a_desc.load([arow, k0])
            bw = g_desc.load([brow, k0])
            acc = tl.dot(a, bw.T, acc)
        acc_g, acc_u = tl.split(tl.reshape(acc, (BM, BN, 2)))
        xrow = tl.load(tok_ptr + rows, mask=rmask, other=0)
        sx = tl.load(sx_ptr + xrow, mask=rmask, other=0.0)
        rw = tl.load(rw_ptr + rows, mask=rmask, other=0.0)
        cidx = e_sel.to(tl.int64) * I + offs_n
        sg = tl.load(sg_ptr + cidx, mask=nmask, other=0.0)
        su = tl.load(su_ptr + cidx, mask=nmask, other=0.0)
        g = (acc_g * sx[:, None] * sg[None, :]).to(tl.bfloat16).to(tl.float32)
        u = (acc_u * sx[:, None] * su[None, :]).to(tl.bfloat16).to(tl.float32)
        silu = g / (1.0 + tl.exp(-g))
        act = (silu * u * rw[:, None]).to(tl.bfloat16)
        ar = _rot32(act, BM, BN)
        tl.store(a_ptr + rows[:, None] * I + offs_n[None, :], ar.to(tl.bfloat16),
                 mask=rmask[:, None] & nmask[None, :])
        tl.store(amax_ptr + rows * NUM_N + pid_n, tl.max(tl.abs(ar), axis=1), mask=rmask)


def run_ws_persistent(P, act, amax, descs=None, ws=True, flat=False, num_warps=4, stages=4,
                      nprog=0):
    a, w = descs if descs is not None else tma_descs(P)
    nn = P["I"] // 128
    mt = P["mt"]
    if nprog == 0:
        nprog = torch.cuda.get_device_properties(0).multi_processor_count
    nprog = max(1, min(nprog, mt * nn))
    _ws_gu_pers_kernel[(nprog,)](
        a, w, P["sx"], P["tok"], P["rw"], P["sg"], P["su"], P["cnt_t"], P["seg_t"], act, amax,
        P["E"], P["H"], P["I"], nn, P["map_t"], mt, BM=BM, BN=128, BK=128, GROUP_M=8, WS=ws,
        FLAT=flat, STAGES=stages, num_warps=num_warps)

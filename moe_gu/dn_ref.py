"""Reference INT8 MoE down GEMM (Triton 3.4): the od15 TMA kernel + a pointer twin, the blocked
down-weight cache packing, and a synthetic problem builder.

Kernels (copied unchanged from the production build; only @triton_dist.jit -> @triton.jit):
  _epg_dn_kernel   TMA loads through INT8-typed host TensorDescriptors (current H800 path)
  _epk_dn2_kernel  pointer loads, same tiles / k order / epilogue (correctness twin)
Math, per routed row p (expert e) and output column h:
  acc = sum_k aq[p, k] * qd[e, h, k]        (INT8 x INT8 -> INT32, exact)
  out[p, h] = bf16(acc * sa[p] * sd[e, h])
aq: per-row INT8 activation [M, I] (the rotated gate/up output after per-row quant), sa [M].
qd: rotated per-output-channel INT8 down weights; sd [E * H].
Blocked cache (TL=True, the od15 layout): rows padded to HP = 256 * ceil(H / 256) per expert;
(256 x 128) blocks, block (row block, k block) is one contiguous 32 KB chunk; viewed as rows of
128 bytes: row (n0 // 256) * 2I + kb * 256 + n0 % 256. TMA box [BN, 128] (BN divides 256, BK 128).
Plain cache (TL=False): [E * H, I] row-major, box [BN, BK].
DIRECT=False here (the DIRECT epilogue needs the multi-GPU combine buffers).
"""
import torch
import triton
import triton.language as tl

from ref_gu import _epk_map2, tile_map, had32_t  # noqa: F401
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _epk_dn_tile(aq, sa_ptr, wd, sd_ptr, c_ptr, e_sel, r0, cnt, seg, pid_n, H, I, HP,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, TL: tl.constexpr):
    local_rows = r0 + tl.arange(0, BM)
    rmask = local_rows < cnt
    rows = (seg + local_rows).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    nl = tl.arange(0, BN)
    nmask = offs_n < H
    offs_k = tl.arange(0, BK)
    if TL:
        # (BN x BK) blocks, BK = 128: block (row block, k step) is contiguous
        wb = (e_sel.to(tl.int64) * HP + pid_n * BN) * I
        wrow = nl.to(tl.int64) * BK
        wstep = BN
    else:
        wb = e_sel.to(tl.int64) * H * I
        wrow = offs_n.to(tl.int64) * I
        wstep = 1
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(0, I, BK):
        kk = k0 + offs_k
        a = tl.load(aq + rows[:, None] * I + kk[None, :], mask=rmask[:, None], other=0)
        b = tl.load(wd + wb + wrow[None, :] + (k0 * wstep + offs_k)[:, None], mask=nmask[None, :],
                    other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    sa = tl.load(sa_ptr + rows, mask=rmask, other=0.0)
    sd = tl.load(sd_ptr + e_sel.to(tl.int64) * H + offs_n, mask=nmask, other=0.0)
    c = acc.to(tl.float32) * sa[:, None] * sd[None, :]
    tl.store(c_ptr + rows[:, None] * H + offs_n[None, :], c.to(tl.bfloat16),
             mask=rmask[:, None] & nmask[None, :])


@triton.jit(do_not_specialize=["E", "NUM_N"])
def _epk_dn2_kernel(aq_ptr, sa_ptr, wd_ptr, sd_ptr, counts_ptr, segstart_ptr, c_ptr,
                    E, H, I, NUM_N, map_ptr, HP,
                    BM: tl.constexpr, BMT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    GROUP_M: tl.constexpr, TL: tl.constexpr):
    # _down_i8_kernel with the tail tiles and the blocked cache layout of _epk_gu2_kernel
    # (TL: HP = H rounded up to BN rows per expert).
    aq = aq_ptr.to(tl.pointer_type(tl.int8), bitcast=True)
    wd = wd_ptr.to(tl.pointer_type(tl.int8), bitcast=True)
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
        cnt = tl.load(counts_ptr + e_sel).to(tl.int32)
        seg = tl.load(segstart_ptr + e_sel)
        if BMT < BM and tail != 0:
            _epk_dn_tile(aq, sa_ptr, wd, sd_ptr, c_ptr, e_sel, tile_in * BM, cnt, seg, pid_n,
                         H, I, HP, BMT, BN, BK, TL)
        else:
            _epk_dn_tile(aq, sa_ptr, wd, sd_ptr, c_ptr, e_sel, tile_in * BM, cnt, seg, pid_n,
                         H, I, HP, BM, BN, BK, TL)


# ---------------------------------------------------------------------------------------
# DN_TMA8: TMA loads of the INT8 down GEMM (A rows aq, weight cache qd) through INT8-typed
# descriptors (the tiles need no bitcast). A: descriptor [r_max, I], box [BM, 128] (and
# [BMT, 128] for tail tiles); rows past the expert's count are computed and never stored (as
# in the gate/up TMA kernel). B: plain [Ep * H, I] with box [256, 128], or the blocked cache
# ((256 x 128) blocks, rows padded to HP per expert) viewed as rows of 128 bytes: row
# (n0 // 256) * 2 * I + kb * 256 + n0 % 256, box [256, 128]. Columns past H (the next
# expert's rows or zero fill) are computed and never stored / never enter the row scale.
# od12: the blocked layout (TL) is used whenever S["tld"] (v2, or DN_BLK on this class).
# Same tiles, k order (BK = 128) and epilogue as _epk_dn_tile / _epk_dnd_tile.
@triton.jit
def _epg_dn_tile(a_desc, d_desc, sa_ptr, sd_ptr, c_ptr, e_sel, r0, cnt, seg, pid_n, H, I, HP,
                 sgl_ptr, peer_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 TL: tl.constexpr, DIRECT: tl.constexpr):
    local_rows = r0 + tl.arange(0, BM)
    rmask = local_rows < cnt
    rows = (seg + local_rows).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    nmask = offs_n < H
    arow = seg + r0
    if TL:
        n0 = e_sel * HP + pid_n * BN
        brow = (n0 // 256) * (2 * I) + n0 % 256
    else:
        brow = e_sel * H + pid_n * BN
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(0, I, BK):
        a = a_desc.load([arow, k0])
        if TL:
            b = d_desc.load([brow + 2 * k0, 0])
        else:
            b = d_desc.load([brow, k0])
        acc = tl.dot(a, b.T, acc, out_dtype=tl.int32)
    sa = tl.load(sa_ptr + rows, mask=rmask, other=0.0)
    sd = tl.load(sd_ptr + e_sel.to(tl.int64) * H + offs_n, mask=nmask, other=0.0)
    c = acc.to(tl.float32) * sa[:, None] * sd[None, :]
    if DIRECT:
        # the DN_DIRECT epilogue of _epk_dnd_tile (BN = 256 = one scale block)
        cb = c.to(tl.bfloat16)
        code = tl.load(sgl_ptr + rows, mask=rmask, other=-1)
        one = code >= 0
        tl.store(c_ptr + rows[:, None] * H + offs_n[None, :], cb,
                 mask=(rmask & (code < 0))[:, None] & nmask[None, :])
        if tl.max(one.to(tl.int32), axis=0) > 0:
            v = tl.where(nmask[None, :], cb.to(tl.float32), 0.0)
            s = tl.maximum(tl.max(tl.abs(v), axis=1), 1e-30) / 127.0
            y = v * (1.0 / s)[:, None]
            r = (y + tl.where(y >= 0, 0.5, -0.5)).to(tl.int32)
            q = tl.minimum(tl.maximum(r, -127), 127).to(tl.int8)
            src = code >> 20
            slot = (code & 1048575).to(tl.int64)
            qa = tl.where(src == 0, tl.load(peer_ptr + 0),
                          tl.where(src == 1, tl.load(peer_ptr + 1),
                                   tl.where(src == 2, tl.load(peer_ptr + 2),
                                            tl.load(peer_ptr + 3))))
            pa = tl.where(src == 0, tl.load(peer_ptr + 4),
                          tl.where(src == 1, tl.load(peer_ptr + 5),
                                   tl.where(src == 2, tl.load(peer_ptr + 6),
                                            tl.load(peer_ptr + 7))))
            # pq base, slot * H and the columns: multiples of 16 (see _epk_dnd_tile)
            qp = tl.multiple_of(qa.to(tl.pointer_type(tl.int8)), 16)
            tl.store(qp[:, None] + (slot * H)[:, None] + offs_n[None, :], q,
                     mask=one[:, None] & nmask[None, :])
            tl.store(pa.to(tl.pointer_type(tl.float32)) + slot * ((H + 255) // 256) + pid_n, s,
                     mask=one)
    else:
        tl.store(c_ptr + rows[:, None] * H + offs_n[None, :], c.to(tl.bfloat16),
                 mask=rmask[:, None] & nmask[None, :])


@triton.jit(do_not_specialize=["E", "NUM_N"])
def _epg_dn_kernel(a_desc, at_desc, d_desc, sa_ptr, sd_ptr, counts_ptr, segstart_ptr, c_ptr,
                   sgl_ptr, peer_ptr, E, H, I, NUM_N, map_ptr, HP,
                   BM: tl.constexpr, BMT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   GROUP_M: tl.constexpr, TL: tl.constexpr, DIRECT: tl.constexpr):
    # _epk_dn2_kernel / _epk_down_kernel (DIRECT False) or _epk_dnd_kernel (DIRECT) with TMA
    # loads; at_desc: the A descriptor with a BMT-row box (tail tiles, BMT < BM).
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
        cnt = tl.load(counts_ptr + e_sel).to(tl.int32)
        seg = tl.load(segstart_ptr + e_sel)
        if BMT < BM and tail != 0:
            _epg_dn_tile(at_desc, d_desc, sa_ptr, sd_ptr, c_ptr, e_sel, tile_in * BM, cnt, seg,
                         pid_n, H, I, HP, sgl_ptr, peer_ptr, BMT, BN, BK, TL, DIRECT)
        else:
            _epg_dn_tile(a_desc, d_desc, sa_ptr, sd_ptr, c_ptr, e_sel, tile_in * BM, cnt, seg,
                         pid_n, H, I, HP, sgl_ptr, peer_ptr, BM, BN, BK, TL, DIRECT)


# the od15 TMA down config (v1 shapes): BM 128, BN 256, BK 128, EPG_DN = (num_warps 8,
# num_stages 4), GROUP_M 8, blocked cache
DN_CUR = dict(BM=128, BN=256, BK=128, num_warps=8, num_stages=4, GROUP_M=8, TL=True)


def round_i8(v):
    """round half away from zero, clamp to [-127, 127] (as the production _round_i8)."""
    r = torch.trunc(v + torch.where(v >= 0, 0.5, -0.5))
    return r.clamp(-127, 127).to(torch.int8)


def pack_down(wd):
    """wd [E, H, I] -> (plain [E * H, I] int8, blocked [E * HP * I // 128, 128] int8, sd [E * H]).
    Per output channel: r = w @ blockdiag(Had32) along I, q = round(r * 127 / amax(r)),
    sd = amax / (127 * 32)."""
    E, H, I = wd.shape
    r = (wd.float().reshape(E, H, I // 32, 32) @ had32_t(wd.device).float()).reshape(E, H, I)
    amax = torch.clamp(r.abs().amax(dim=2), min=1e-30)
    q = round_i8(r * (127.0 / amax)[:, :, None])
    sd = (amax / (127.0 * 32.0)).reshape(-1).contiguous()
    HP = 256 * ((H + 255) // 256)
    qp = torch.zeros((E, HP, I), dtype=torch.int8, device=wd.device)
    qp[:, :H] = q
    blk = qp.reshape(E * HP // 256, 256, I // 128, 128).permute(0, 2, 1, 3).contiguous()
    return q.reshape(E * H, I).contiguous(), blk.reshape(-1, 128), sd, HP


def make_down_problem(M, H, I, E, seed=0, skew=0.0, dev="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    wts = torch.tensor([(i + 1) ** (-skew) for i in range(E)], dtype=torch.float64)
    counts = torch.floor(wts / wts.sum() * M).to(torch.int64)
    counts[0] += M - int(counts.sum())
    counts = [int(c) for c in counts]
    seg = [0]
    for c in counts[:-1]:
        seg.append(seg[-1] + c)
    aq = torch.randint(-127, 128, (M, I), generator=g, dtype=torch.int8)
    sa = torch.rand(M, generator=g) * 1e-3 + 1e-4
    wd = (torch.randn(E, H, I, generator=g) * 0.02).to(dev)
    qd, qb, sd, HP = pack_down(wd)
    return dict(M=M, H=H, I=I, E=E, HP=HP, counts=counts, seg=seg, aq=aq.to(dev),
                sa=sa.float().to(dev), qd=qd, qb=qb, sd=sd,
                cnt_t=torch.tensor(counts, dtype=torch.int32, device=dev),
                seg_t=torch.tensor(seg, dtype=torch.int32, device=dev))


def dn_map(P, bm):
    key = "map_%d" % bm
    if key not in P:
        tm = tile_map(P["counts"], bm)
        P[key] = (torch.tensor(tm, dtype=torch.int32, device=P["aq"].device), len(tm))
    return P[key]


def dn_out(P):
    return torch.empty((P["M"], P["H"]), dtype=torch.bfloat16, device=P["aq"].device)


def dn_descs(P, BM, BN, BK, TL):
    a = TensorDescriptor(P["aq"], [P["M"], P["I"]], [P["I"], 1], [BM, BK])
    if TL:
        assert BK == 128 and 256 % BN == 0
        d = TensorDescriptor(P["qb"], [P["qb"].shape[0], 128], [128, 1], [BN, 128])
    else:
        d = TensorDescriptor(P["qd"], [P["qd"].shape[0], P["I"]], [P["I"], 1], [BN, BK])
    return a, d


def run_dn_tma(P, c_out, cfg=None, descs=None):
    c = dict(DN_CUR)
    if cfg:
        c.update(cfg)
    a, d = descs if descs is not None else dn_descs(P, c["BM"], c["BN"], c["BK"], c["TL"])
    mp, mt = dn_map(P, c["BM"])
    nn = triton.cdiv(P["H"], c["BN"])
    _epg_dn_kernel[(mt * nn,)](
        a, a, d, P["sa"], P["sd"], P["cnt_t"], P["seg_t"], c_out, P["sa"], P["sa"], P["E"],
        P["H"], P["I"], nn, mp, P["HP"], BM=c["BM"], BMT=c["BM"], BN=c["BN"], BK=c["BK"],
        GROUP_M=c["GROUP_M"], TL=c["TL"], DIRECT=False, num_warps=c["num_warps"],
        num_stages=c["num_stages"])


def run_dn_ptr(P, c_out):
    """Pointer twin (_epk_dn2_kernel, TL blocked cache, v1 tiles 128 x 256 x 128)."""
    mp, mt = dn_map(P, 128)
    nn = triton.cdiv(P["H"], 256)
    _epk_dn2_kernel[(mt * nn,)](
        P["aq"], P["sa"], P["qb"], P["sd"], P["cnt_t"], P["seg_t"], c_out, P["E"], P["H"],
        P["I"], nn, mp, P["HP"], BM=128, BMT=128, BN=256, BK=128, GROUP_M=8, TL=True,
        num_warps=8, num_stages=4)


def torch_dn_ref(P):
    I, H = P["I"], P["H"]
    out = torch.zeros((P["M"], H), dtype=torch.float64, device=P["aq"].device)
    for e, (c, s) in enumerate(zip(P["counts"], P["seg"])):
        if c == 0:
            continue
        acc = P["aq"][s:s + c].double() @ P["qd"][e * H:(e + 1) * H].double().T
        out[s:s + c] = (acc.float() * P["sa"][s:s + c, None] * P["sd"][e * H:(e + 1) * H][None, :]
                        ).to(torch.bfloat16).double()
    return out

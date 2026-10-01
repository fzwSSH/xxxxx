# Instructions for the H20 agent

## Your role

You are a GPU performance engineer working alone on one machine with one Hopper GPU (an H20).
You have no other context, and you need none: everything is in this repo. You tune the tile
configs of two existing Triton kernels (T1) and test one Triton feature (T2). You report only
through files in this repo, committed and pushed as COOPERATION.md says. Another agent (MAIN)
reads your pushes; it cannot talk to you except through INBOX.md.

First do the README quick start (Steps 0-6). Do not start T1 before Step 6 shows `REF PASS`.

## Hard rules (never break these)

1. Synthetic data only. The scripts make random data; never load any other data.
2. Kernels stay pure Triton: `@triton.jit`, `tl.*`, TMA tensor descriptors. No CUDA libraries
   (cuBLAS, CUTLASS, cuDNN), no custom CUDA C++, no Gluon, no inline PTX / asm.
3. Never change the numerics. A config counts only if its output bytes are EQUAL to the current
   config's output (`bytes_equal: true`, `mism` 0). Do not change the math in the epilogue, the
   data types, or the weight quantization.
4. One GPU (`CUDA_VISIBLE_DEVICES=0`). Nothing else may run on it while you time kernels
   (check `nvidia-smi` shows no other process).
5. Report a config as a win only if it is **>= 5 % faster on the kernel AND byte-identical**, and
   the gain holds in 3 repeated runs. H20 speed ratios differ from the H800 where these kernels
   run in the end (the H20 has much less FP8 compute but similar memory bandwidth), so small
   gains are noise for us.
6. Keep every kernel file passing `python moe_gu/rp_check.py <file> skip /tmp/rp`
   (`rewrite accepted`): no annotated assignments `x: T = v`, no attribute names that start
   with `_`, kernels decorated `@triton.jit`.
7. Commit and push small steps (see COOPERATION.md). Push partial results at least every
   30 minutes and before 2026-10-01 20:00 UTC+8.

## The kernels (1 paragraph)

An MoE layer sends each token to a few "experts"; the rows of all tokens routed to one expert
are contiguous (`segstart[e]`, `counts[e]`). **Gate/up** (`moe_gu/ref_gu.py::_epg_gu_kernel`):
an FP8 x FP8 GEMM with FP32 accumulation. The gate and up weight matrices of an expert are
interleaved in one `[2I, H]` matrix, so one 128 x 256 `tl.dot` per k step computes both; the
epilogue applies scales, SwiGLU (`silu(gate) * up`), a routing weight, a Hadamard-32 rotation
(a fixed +-1 32x32 matrix applied per 32-column block), stores BF16 and a per-row max. **Down**
(`moe_gu/dn_ref.py::_epg_dn_kernel`): an INT8 x INT8 GEMM with INT32 accumulation of the
quantized activation `[M, I]` with the expert's down weights `[H, I]`, scaled and stored as BF16.
Its weights use a "blocked" cache: (256 rows x 128 columns) blocks, each one contiguous 32 KB
chunk, so one TMA box `[256, 128]` reads one block. Both kernels walk a tile map (one entry per
BM-row tile: expert << 16 | tile index) with a grouped raster (GROUP_M).

## Glossary

- **TMA** (Tensor Memory Accelerator): Hopper hardware unit that copies a 2-D tile ("box")
  from global to shared memory; in Triton you use a host `TensorDescriptor` and `desc.load([r, c])`.
  The box shape is fixed by the descriptor, so it must match the kernel's tile (each side <= 256).
- **BM / BN / BK**: tile sizes: rows of the output tile, columns of the output tile, and the
  depth of one k step. For gate/up the weight box is `[2 * BN, BK]` (gate + up rows).
- **num_warps**: warps per program (4 = one warpgroup, 8 = two).
- **num_stages**: software pipeline depth of the k loop (how many k tiles are in flight in
  shared memory). Shared memory per program is about num_stages x (BM x BK + B-tile bytes);
  Hopper allows 227 KB.
- **GROUP_M**: grouped tile order: GROUP_M row tiles are walked together over all column tiles,
  for better L2 reuse of the B tiles.
- **Warp specialization**: some warps only load (producer), others only compute (consumers),
  overlapping copies and math. In Triton 3.4: `tl.range(..., warp_specialize=True)`.
- **Persistent kernel**: grid = number of SMs; each program loops over many tiles.
- **mism / bytes_equal**: number of BF16 output words that differ from the current config /
  whether that number is 0.

## T1: tile-config sweep (main task, budget about 3 h)

Current configs (tuned on H800; `ref_gu.GU_CUR`, `dn_ref.DN_CUR`):

| kernel | BM | BN | BK | num_warps | num_stages | GROUP_M | weight layout |
|---|---|---|---|---|---|---|---|
| gu (gate/up) | 128 | 128 | 128 | 8 | 4 | 8 | interleaved plain `[E * 2I, H]` |
| dn (down) | 128 | 256 | 128 | 8 | 4 | 8 | blocked (256 x 128) |

Shapes (M = routed rows on the GPU, split evenly over E = 8 experts):
gu: M in {2048, 8192, 16384} x H = 4096 x I in {2048, 8192} (6 shapes);
dn: M in {2048, 8192, 16384} x H in {4096, 2048} x I in {2048, 8192} (12 shapes).

Grids in `moe_gu/sweep.py` (configs over the 227 KB shared-memory limit are dropped):

| grid | gu | dn |
|---|---|---|
| quick | BM {64,128} x BN {64,128} x BK 128 x warps {4,8} x stages {3,4} x GROUP_M 8 | blocked, BM {64,128} x BN {128,256} x BK 128 x warps {4,8} x stages {3,4} x GROUP_M 8 |
| full | BM {64,128} x BN {64,128} x BK {64,128} x warps {4,8} x stages {2..5} x GROUP_M {1,8,16} | blocked: BM {64,128} x BN {64,128,256} x BK 128 x warps {4,8} x stages {2..5} x GROUP_M {1,8,16}; plus plain layout: BM {64,128} x BN {128,256} x BK {64,128} x warps {4,8} x stages {3,4} |

### Step T1.1: quick sweep (20-40 min)

```bash
cd moe_gu
python sweep.py --grid quick --json ../results/T1_sweep.json --md ../results/T1_sweep.md 2>&1 | tee ../logs/T1_quick.log
```

Expected output: for each shape a block like

```
### gu M=8192 H=4096 I=2048 E=8: current [BM128 BN128 BK128 w8 s4 g8] 0.9123 ms (301 TFLOPS); 12 configs in 40 s
| rank | config | ms | vs current | mism |
|---|---|---|---|---|
| 1 | BM128 BN64 BK128 w8 s4 g8 | 0.8512 | +7.2% | 0 |
...
```

and at the end one `BEST <kernel> <shape>: [config] ... gain +x.x%` line per shape. The files
results/T1_sweep.json (rows: kernel, shape, config, time_us, ref_time_us, gain_pct,
bytes_equal, ...) and results/T1_sweep.md (best byte-equal config per shape) are written.
Then: update STATUS.md, commit `[H20] T1: quick sweep`, push.

If X fails, do Y:
- a config prints `skip ... OutOfResources`: normal (too much shared memory); ignore.
- every config prints `skip` or the current config itself fails: copy the first error into
  OUTBOX.md, mark T1 blocked, go to T2.
- `mism` is not 0 for configs that only change BM / num_warps / num_stages / GROUP_M: write it
  into OUTBOX.md (unexpected; FP8 accumulation order should not change) and still only report
  byte-equal configs.
- one shape takes more than 10 min: run shapes one by one with `--kind`, `--M`, `--I`, `--H` /
  `--Hdn` (results merge into the same JSON).

### Step T1.2: confirm the wins (20-40 min)

For every shape whose `BEST` gain is >= 5 %: rerun that shape 3 times with the same grid and
note the gain each time, e.g.

```bash
for r in 1 2 3; do python sweep.py --grid quick --kind gu --M 8192 --I 2048 --H 4096 2>&1 | grep BEST; done
```

A config is confirmed if it is the best (or within 1 % of the best) and >= 5 % faster than the
current config in all 3 runs. Write the confirmations into results/T1_sweep.md under a heading
`## Confirmed` (one line per shape: config, 3 gains). Commit `[H20] T1: confirmed wins`, push.

### Step T1.3: full grid on the promising classes (up to 2 h, stop at 18:30 UTC+8)

A "class" is one kernel with one (H, I) pair, all M. Run the full grid class by class, the
class with the largest quick-sweep gain first:

```bash
python sweep.py --grid full --kind dn --Hdn 4096 --I 2048 --json ../results/T1_sweep.json --md ../results/T1_sweep.md 2>&1 | tee ../logs/T1_full_dn_4096_2048.log
```

Commit and push after each class (`[H20] T1: full grid dn H=4096 I=2048`). Confirm new wins as
in T1.2.

### Step T1.4: one config per class (15 min)

For each class choose ONE config that is >= 5 % faster than the current config on every M of
the class and byte-equal (or write "no clear win"). Add a table to results/T1_sweep.md:

```
## Recommendation per class
| kernel | H | I | config | gain at M=2048 / 8192 / 16384 | bytes_equal |
```

Commit `[H20] T1: recommendation`, push, set T1 = done in STATUS.md.

## T2: Triton built-in warp specialization (budget about 1 h)

`moe_gu/ws_gu.py` has the same gate/up math as `_epg_gu_kernel` in two forms:
- `ws-k`: one program per tile, `warp_specialize=True` on the K loop;
- `ws-p`: persistent (grid = SM count), `warp_specialize=True` on the tile loop.
Both also run with WS off as a control ("ws-k off", "ws-p off").

Known before you start (compile only, fork, target sm_90a, on a non-Hopper host): both
variants compile with `warp_specialize=True`, but the generated TTGIR has no
`ttg.warp_specialize` op and the PTX has no `setmaxnreg`, so this Triton version may silently
ignore warp specialization on Hopper. T2 checks this on the real GPU.

### Step T2.1: does WS change the code? (15 min)

```bash
cd moe_gu
python test_ref.py 2>&1 | grep -i "ws"        # correctness lines for both variants (PASS + 0 mismatches expected)
export TRITON_CACHE_DIR=$PWD/../logs/tcache_ws && rm -rf $TRITON_CACHE_DIR
python bench.py --M 8192 --I 2048 --warps 4 > /dev/null
grep -l "warp_specialize" $TRITON_CACHE_DIR/*/*.ttgir; grep -c "setmaxnreg" $TRITON_CACHE_DIR/*/*.ptx
unset TRITON_CACHE_DIR
```

Write into results/T2_ws.md: does any `_ws_gu*` TTGIR contain `warp_specialize`; does any PTX
contain `setmaxnreg`. If neither: "WS ignored on sm_90 in this Triton".

### Step T2.2: timing (15-30 min)

```bash
python bench.py --json ../results/T2_ws.json --md ../results/T2_ws.md 2>&1 | tee ../logs/T2_bench.log
```

Expected: a markdown table `| M | I | kernel | warps | ms | vs tma-cur | mism |` with rows
tma-cur, ws-k, ws-k off, ws-p, ws-p off for warps 4 and 8, for 6 shapes. Decision: a variant
is a win only if mism = 0 and it is >= 5 % faster than tma-cur in 3 runs; also compare with
T1's best config for that shape, if T1 found one. Commit `[H20] T2: timing`, push.

If X fails, do Y:
- a WS variant raises at compile time: put the first error line in results/T2_ws.md; that is a
  valid result.
- `mism` > 0 for a WS variant: report it; do not count it as a win.
- the persistent variant is slower only because of the grid size: try
  `run_ws_persistent(..., nprog=2 * SMs)` once and note it.

## Final report

When T1 and T2 are done (or at 19:30 UTC+8 at the latest): make sure results/T1_sweep.md,
results/T1_sweep.json, results/T2_ws.md, results/T2_ws.json and STATUS.md are up to date;
write a 5-line summary at the top of OUTBOX.md (environment: fork or fallback; best T1 win per
class; T2 verdict); commit `[H20] final report`; push.

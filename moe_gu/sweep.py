"""T1: tile-config sweep of the two TMA kernels on one Hopper GPU (synthetic data).

  gu: TMA FP8 gate/up (ref_gu._epg_gu_kernel; GUW interleaved cache, H32 epilogue)
  dn: TMA INT8 down   (dn_ref._epg_dn_kernel; od15 blocked cache TL=True, or plain TL=False)
For every shape: run the current config (od15) first, then every candidate config. Each
candidate's output (BF16 act / out) is compared word by word with the current config's output
("mism" column; must be 0 to be a drop-in replacement). Time = triton.testing.do_bench median.
Configs that do not fit (shared memory / registers) or fail to compile print as "skip".

usage: python sweep.py [--kind gu,dn] [--grid quick|full] [--E 8] [--M 2048,8192,16384]
                       [--I 2048,8192] [--H 4096] [--Hdn 4096,2048] [--csv out.csv]
                       [--json ../results/T1_sweep.json] [--md ../results/T1_sweep.md]
--json / --md MERGE into existing files (rows with the same kernel + shape + config are replaced),
so several partial runs build one result file.
M = routed rows on this GPU, split evenly over E local experts.
Output: per shape the top 5 configs and the current one, then a summary table
"BEST <kind> <shape> <config> <time> vs current <time> gain <pct>%".
"""
import argparse
import csv
import itertools
import json
import os
import time

import torch
import triton

import dn_ref
import ref_gu

SMEM = 227 * 1024


def gu_grid(full):
    if full:
        g = itertools.product([64, 128], [64, 128], [64, 128], [4, 8], [2, 3, 4, 5], [1, 8, 16])
    else:
        g = itertools.product([64, 128], [64, 128], [128], [4, 8], [3, 4], [8])
    for BM, BN, BK, nw, ns, gm in g:
        if ns * (BM * BK + 2 * BN * BK) > SMEM - 8192:
            continue
        yield dict(BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns, GROUP_M=gm)


def dn_grid(full):
    if full:
        g = list(itertools.product([True], [64, 128], [64, 128, 256], [128], [4, 8], [2, 3, 4, 5],
                                   [1, 8, 16]))
        g += list(itertools.product([False], [64, 128], [128, 256], [64, 128], [4, 8], [3, 4],
                                    [8]))
    else:
        g = itertools.product([True], [64, 128], [128, 256], [128], [4, 8], [3, 4], [8])
    for TL, BM, BN, BK, nw, ns, gm in g:
        if ns * (BM * BK + BN * BK) > SMEM - 8192:
            continue
        yield dict(TL=TL, BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns, GROUP_M=gm)


def cfg_str(c):
    s = f"BM{c['BM']} BN{c['BN']} BK{c['BK']} w{c['num_warps']} s{c['num_stages']} g{c['GROUP_M']}"
    if "TL" in c:
        s += " blk" if c["TL"] else " plain"
    return s


def bench_one(fn):
    fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=10, rep=50)


def sweep_shape(kind, M, H, I, E, full, rows, int8=False):
    if kind == "gu":
        P = ref_gu.make_problem(M, H, I, E, seed=11, gather=False, int8=int8)
        out_c, am_c = ref_gu.out_buffers(P)
        cur = dict(ref_gu.GU_CUR)
        t_cur = bench_one(lambda: ref_gu.run_ref_tma(P, out_c, am_c, cfg=cur))
        grid = list(gu_grid(full))
        flop = 2.0 * M * H * 2 * I
    else:
        P = dn_ref.make_down_problem(M, H, I, E, seed=12)
        out_c = dn_ref.dn_out(P)
        cur = dict(dn_ref.DN_CUR)
        t_cur = bench_one(lambda: dn_ref.run_dn_tma(P, out_c, cfg=cur))
        grid = list(dn_grid(full))
        flop = 2.0 * M * H * I
    ref_bytes = out_c.view(torch.int16).clone()
    shape = f"M={M} H={H} I={I} E={E}"
    res = []
    t0 = time.time()
    for c in grid:
        if kind == "gu":
            out, am = ref_gu.out_buffers(P)
            fn = (lambda c=c, out=out, am=am: ref_gu.run_ref_tma(P, out, am, cfg=c))
        else:
            out = dn_ref.dn_out(P)
            fn = (lambda c=c, out=out: dn_ref.run_dn_tma(P, out, cfg=c))
        try:
            t = bench_one(fn)
            mism = int((out.view(torch.int16) != ref_bytes).sum())
        except Exception as e:
            print(f"  skip {cfg_str(c)}: {type(e).__name__} {str(e).splitlines()[0][:120] if str(e) else ''}")
            continue
        res.append((t, c, mism))
        rows.append(dict(kernel=kind, shape=shape, config=cfg_str(c),
                         time_us=round(t * 1000, 2), ref_time_us=round(t_cur * 1000, 2),
                         gain_pct=round(100 * (t_cur / t - 1), 1), bytes_equal=(mism == 0),
                         mismatch_words=mism, is_current=(cfg_str(c) == cfg_str(cur))))
    res.sort(key=lambda r: r[0])
    print(f"\n### {kind} {shape}: current [{cfg_str(cur)}] {t_cur:.4f} ms "
          f"({flop / t_cur / 1e9:.0f} TFLOPS); {len(res)} configs in {time.time() - t0:.0f} s")
    print("| rank | config | ms | vs current | mism |")
    print("|---|---|---|---|---|")
    for i, (t, c, mism) in enumerate(res[:5]):
        print(f"| {i + 1} | {cfg_str(c)} | {t:.4f} | {100 * (t_cur / t - 1):+.1f}% | {mism} |")
    exact = [r for r in res if r[2] == 0]
    best = exact[0] if exact else None
    return shape, cur, t_cur, best


def write_json_md(jpath, mdpath, rows, gpu, triton_path):
    """Merge rows into jpath (list of dicts) and rewrite mdpath with the best byte-equal config
    per (kernel, shape)."""
    old = []
    if os.path.exists(jpath):
        try:
            old = json.load(open(jpath))["rows"]
        except Exception:
            old = []
    key = lambda r: (r["kernel"], r["shape"], r["config"])
    merged = {key(r): r for r in old}
    for r in rows:
        merged[key(r)] = r
    allrows = sorted(merged.values(), key=lambda r: (r["kernel"], r["shape"], r["time_us"]))
    os.makedirs(os.path.dirname(os.path.abspath(jpath)), exist_ok=True)
    json.dump(dict(gpu=gpu, triton=triton.__version__, triton_path=triton_path,
                   updated=time.strftime("%Y-%m-%d %H:%M:%S %z"), rows=allrows),
              open(jpath, "w"), indent=1)
    print("json:", jpath, len(allrows), "rows")
    if not mdpath:
        return
    best = {}
    for r in allrows:
        k = (r["kernel"], r["shape"])
        if r["bytes_equal"] and (k not in best or r["time_us"] < best[k]["time_us"]):
            best[k] = r
    with open(mdpath, "w") as f:
        f.write(f"# T1 sweep (auto-generated by sweep.py)\n\nGPU {gpu}, triton {triton.__version__} "
                f"({triton_path}), updated {time.strftime('%Y-%m-%d %H:%M:%S %z')}\n\n")
        f.write("Best byte-equal config per shape. WIN = gain >= 5 %.\n\n")
        f.write("| kernel | shape | best config | time_us | current time_us | gain % | verdict |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for (kn, sh), r in sorted(best.items()):
            v = "WIN" if r["gain_pct"] >= 5.0 else "no clear win"
            f.write(f"| {kn} | {sh} | {r['config']} | {r['time_us']} | {r['ref_time_us']} | "
                    f"{r['gain_pct']} | {v} |\n")
    print("md:", mdpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="gu,dn")
    ap.add_argument("--grid", default="quick", choices=["quick", "full"])
    ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--M", default="2048,8192,16384")
    ap.add_argument("--I", default="2048,8192")
    ap.add_argument("--H", default="4096")
    ap.add_argument("--Hdn", default="4096,2048")
    ap.add_argument("--csv", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    ap.add_argument("--smoke", action="store_true",
                    help="non-Hopper code-path test: INT8 gate/up twin, tiny grid (no real timing)")
    a = ap.parse_args()
    if not ref_gu.is_hopper() and not a.smoke:
        print("SKIP sweep: needs sm_90 (FP8 TMA gate/up); this GPU is",
              torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
        return
    p = torch.cuda.get_device_properties(0)
    print(f"GPU {p.name}, {p.multi_processor_count} SMs, triton {triton.__version__}, grid {a.grid}")
    rows, summ = [], []
    for kind in a.kind.split(","):
        Hs = a.H if kind == "gu" else a.Hdn
        for H in [int(v) for v in Hs.split(",")]:
            for I in [int(v) for v in a.I.split(",")]:
                for M in [int(v) for v in a.M.split(",")]:
                    summ.append((kind,) + sweep_shape(kind, M, H, I, a.E, a.grid == "full", rows,
                                                     int8=not ref_gu.is_hopper()))
    print("\n## Summary (best config with output bytes equal to the current config)")
    for kind, shape, cur, t_cur, best in summ:
        if best is None:
            print(f"BEST {kind} {shape}: no byte-equal config ran")
            continue
        t, c, _ = best
        print(f"BEST {kind} {shape}: [{cfg_str(c)}] {t:.4f} ms vs current {t_cur:.4f} ms "
              f"gain {100 * (t_cur / t - 1):+.1f}%")
    if a.json:
        write_json_md(a.json, a.md, rows, gpu=f"{p.name} ({p.multi_processor_count} SMs)",
                      triton_path=os.path.dirname(triton.__file__))
    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("csv:", a.csv)


if __name__ == "__main__":
    main()

# MoE expert-GEMM tuning bench for one Hopper GPU (H20)

This repo tunes two GPU kernels of a Mixture-of-Experts (MoE) layer on ONE Hopper GPU, using
synthetic data only: the FP8 **gate/up** GEMM and the INT8 **down** GEMM, both written in
Triton 3.4 with TMA loads. They currently use tile configs tuned on an H800. Your job (details
in PROMPT.md) is to find tile configs that are clearly faster (>= 5 %) while producing exactly
the same output bytes (T1), and to test whether Triton's built-in warp specialization helps the
gate/up kernel (T2). You report through files in this repo (COOPERATION.md). Results are useful
only if pushed before **2026-10-01 20:00 UTC+8**; push partial results early.

## Quick start (do these steps in order)

All commands run from the repo root unless a step says otherwise. Times are rough.

**Step 0. Prerequisites (2 min).** Linux x86_64, one Hopper GPU, NVIDIA driver for CUDA 12.8
(driver >= 570), about 25 GB free disk, internet, `git`, `curl`, `gcc`/`g++`. Install `uv` if
`uv --version` fails:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
```

**Step 1. Clone (1 min).** The repo is `git@github.com:fzwSSH/xxxxx.git` (HTTPS: `https://github.com/fzwSSH/xxxxx.git`); you need collaborator access.

```bash
git clone git@github.com:fzwSSH/xxxxx.git h20_bundle && cd h20_bundle
```

**Step 2. Check the GPU (1 min).**

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
```

Expected: one line like `NVIDIA H20, 9.0, 5xx.xx, 97871 MiB` (the name may differ; compute_cap
MUST be 9.0). If it is not 9.0: stop, write the output into OUTBOX.md, commit, push.

**Step 3. Sources, venv, LLVM, wheels (10-30 min, downloads about 4 GB).**

```bash
export ROOT=$HOME/tdist34
bash setup/setup34.sh 2>&1 | tee logs/setup34.log | tail -3
```

Success: the last line is `SETUP_DONE`. If the LLVM download (1.2 GB) is very slow, stop it,
run `bash setup/pdl.sh` (parallel download), then rerun `setup34.sh`.

**Step 4. Build the Triton fork (10-40 min, depends on CPU cores).**

```bash
bash setup/build34.sh 2>&1 | tee logs/build34.log | tail -3
```

Success: a line `triton 3.4.0 /.../Triton-distributed/3rdparty/triton/python/triton/__init__.py`
and the last line `BUILD_DONE`.

If it fails:
- error with undefined `std::__glibcxx_assert_fail`, `std::__throw_bad_array_new_length` or
  `exception_ptr::_M_release`: run
  `rm -f $ROOT/Triton-distributed/3rdparty/triton/build/cmake*/CMakeCache.txt; SHIM=1 bash setup/build34.sh`.
- any other error, or Steps 3 + 4 together take more than 90 minutes: use the **fallback**
  below, and write "FALLBACK: upstream triton 3.4.0" in STATUS.md and in every results file
  note. (The real target uses the fork; upstream 3.4.0 is close but not identical.)

Fallback (5 min):

```bash
uv venv -p 3.10 $HOME/t34up && source $HOME/t34up/bin/activate
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
uv pip install triton==3.4.0 numpy RestrictedPython
```

**Step 5. Check the environment (1 min).**

```bash
source $ROOT/.venv/bin/activate        # or: source $HOME/t34up/bin/activate (fallback)
python -c "import sys, torch, triton; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('triton', triton.__version__, triton.__file__); print('cuda', torch.cuda.is_available(), torch.cuda.get_device_capability(), torch.cuda.get_device_properties(0).multi_processor_count, 'SMs')"
```

Expected (fork):

```
python 3.10.x
torch 2.8.0+cu128
triton 3.4.0 /home/<you>/tdist34/Triton-distributed/3rdparty/triton/python/triton/__init__.py
cuda True (9, 0) <N> SMs
```

With the fallback the triton path is inside `site-packages`. Any other torch / triton version:
fix the install first. A message `torch ... with triton 3.4.0. inductor patch is not applied`
is harmless.

**Step 6. Correctness checks (1-5 min).**

```bash
bash run_all.sh
```

Success: the output contains `REF PASS` and three lines `[skip] rewrite accepted, module
loaded`. Failure: any `FAIL` line or `REF FAIL`; then do not tune: copy the failing lines into
OUTBOX.md, set STATUS.md to blocked, commit, push.

**Step 7. Tasks.** Read PROMPT.md (the work) and COOPERATION.md (how to report), then do T1,
then T2. Shortcuts: `T1=1 bash run_all.sh` (quick sweep, 15-30 min), `T2=1 bash run_all.sh`
(warp-specialization timing, 5-15 min).

## Files

| path | what |
|---|---|
| PROMPT.md | the task instructions (read fully) |
| COOPERATION.md | how to report through git: STATUS.md, INBOX.md, OUTBOX.md, results/ |
| moe_gu/ref_gu.py | gate/up kernels (TMA `_epg_gu_kernel` = the one to tune; pointer twin `_epk_gate_up_kernel`), weight packing, synthetic problems, float64 torch model |
| moe_gu/dn_ref.py | down kernels (TMA `_epg_dn_kernel` = the one to tune; pointer twin `_epk_dn2_kernel`), blocked weight packing, problems, torch model |
| moe_gu/ws_gu.py | gate/up with Triton's automatic warp specialization (T2) |
| moe_gu/test_ref.py | correctness of all kernels |
| moe_gu/sweep.py | T1 tile-config sweep, writes results/T1_sweep.json + .md |
| moe_gu/bench.py | T2 timing, writes results/T2_ws.json + .md |
| moe_gu/rp_check.py | checks that a kernel file passes the target sandbox's source rewrite |
| setup/ | environment scripts (Triton-distributed 8260bc3 + ByteDance Triton fork f53694a) |
| run_all.sh | runs the checks; `T1=1` / `T2=1` add the tasks; logs to logs/ |

## Environment details

The target runs Triton-distributed commit `8260bc3` with its submodule `3rdparty/triton` =
ByteDance-Seed/triton `f53694a` (`triton.__version__ == "3.4.0"`, LLVM `8957e64a`, ptxas
12.8.93), Python 3.10, torch 2.8.0+cu128. `setup34.sh` fetches exactly these commits.
The fork is installed through the `triton_dist` editable package (its import finder exposes
`3rdparty/triton/python/triton`); there must be no separate `triton` wheel in that venv
(`setup34.sh` removes the one torch pulls in). The build forces gcc
(`CC=gcc CXX=g++ LDSHARED=...`) because the uv Python's sysconfig asks for clang. With a driver
older than CUDA 12.8, add `/usr/local/cuda/compat/lib` (or your CUDA compat dir) to
`LD_LIBRARY_PATH`.

What was checked before handing over (A100 host, fork env): `run_all.sh` passes (gate/up
INT8 twin SQNR 55.6 dB, TMA = pointer bitwise; down GEMM bitwise equal to the torch model);
`sweep.py --smoke` runs end to end. The FP8 path has not run on a Hopper GPU in this exact
form, so Step 6 on the H20 is the first real FP8 check.

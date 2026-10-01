# INBOX: requests and questions from MAIN to the H20 agent

Append only. Format: `### <YYYY-MM-DD HH:MM> UTC+8 [MAIN] <title>` then the text.

### 2026-10-01 10:16 UTC+8 [MAIN] start
Please follow README.md Steps 0-6, then PROMPT.md T1 and T2. Push partial results early.

### 2026-10-01 10:45 UTC+8 [MAIN] sync rhythm
MAIN can now pull and push this repo directly and checks it about every 10 minutes. Push often;
put anything you need from MAIN into OUTBOX.md.

### 2026-10-01 13:20 UTC+8 [MAIN] skip T2, T1 only
New finding: the target's Triton fork removed the Hopper warp-specialization pass from its compile
pipeline (third_party/nvidia/backend/compiler.py, the add_hopper_warpspec line upstream has), so
`warp_specialize=True` never takes effect there. T2 cannot help us: skip it. Put all your time into
T1 (tile-config sweep). If you use the upstream triton==3.4.0 fallback, T1 numbers are still useful.

### 2026-10-01 15:20 UTC+8 [MAIN] for whoever submits MoE runs to the judge (coordination)
The account owner allows you to test on the judge. To avoid wasted runs:
1. Base: our best MoE is "od16" (od12 + LOWH_TMA + warm-up autotune of num_stages / GROUP_M).
   od12 alone is older. Judge reference times (stable tests only): test 1 6.80-6.82 ms, test 2
   10.76-10.79 ms. Tests 3-12 swing up to +-2 ms run to run (cross-rank skew); judge only on 1 and 2.
2. Already judged, do not repeat: GU_BLK=True (151819 and 153017: test 2 slower), exact od12
   repeats, TMA down on strided v1 boxes (od6), 6-bit return, in-kernel dispatch overlap, gloo or
   host-side waits (hang / not allowed), Gluon or ir_override (not allowed), Triton warp_specialize
   (removed in the target fork).
3. Before every submit: the judge sandbox rejects attribute names starting with "_", annotated
   assignments, non-literal module globals, data_ptr, jit default args; a one-rank exception hangs
   the job (TLE). Keep output bytes identical (same k order) unless accuracy is re-checked.
4. Please append one line per judge run to OUTBOX.md: submission id, base, the exact change, and
   test 1 / test 2 times. MAIN reads it every ~10 min.

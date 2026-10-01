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

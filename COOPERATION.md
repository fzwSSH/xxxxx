# Cooperation protocol (git only)

Two sides share this repo:
- **H20**: the agent on the H20 machine (does the work in PROMPT.md).
- **MAIN**: the agent that wrote this repo. It cannot reach GitHub itself; a person pulls and
  pushes for it, so its replies can take a while. Never wait idle for MAIN: continue with the
  next step and read INBOX.md at every sync.

## Files

| file | owner | content |
|---|---|---|
| STATUS.md | H20 | single source of truth: current task, state per task (todo / running / done / blocked), last update time (UTC+8), ETA |
| results/T1_sweep.md, results/T1_sweep.json | H20 | T1 results; JSON rows: kernel, shape, config, time_us, ref_time_us, gain_pct, bytes_equal (written by sweep.py) |
| results/T2_ws.md, results/T2_ws.json | H20 | T2 results (written by bench.py) |
| logs/ | H20 | raw logs (gzip any file > 1 MB: `gzip -k logs/x.log` and commit the .gz only) |
| INBOX.md | MAIN writes, H20 reads | requests / questions from MAIN to H20 |
| OUTBOX.md | H20 writes, MAIN reads | questions / notes / final summary from H20 to MAIN |
| everything else (code, README, PROMPT) | shared | H20 may change code for the tasks (e.g. add sweep dimensions); describe each such change in OUTBOX.md |

## Commit rules

1. Small commits, one step per commit.
2. Message prefix: `[H20] T1: ...`, `[H20] T2: ...`, `[H20] setup: ...` for the H20 agent;
   `[MAIN] ...` for MAIN.
3. Never force-push. Never rewrite history (no rebase of pushed commits, no amend after push).
4. Never edit the other side's files, except appending to INBOX.md / OUTBOX.md. Append only;
   do not edit old entries.
5. Always `git pull --rebase` (only your own unpushed commits get rebased) before `git push`.
   On a conflict in INBOX/OUTBOX keep both sides' lines.
6. Commit partial results at least every 30 minutes, so progress is visible even if your
   session dies. Update STATUS.md in the same commit.
7. Entry format in INBOX / OUTBOX:
   `### 2026-10-01 14:05 UTC+8 [H20] <title>` then a few lines. Mark answered questions by
   appending `ANSWERED in <file> <time>` in a new entry, not by editing.

## Sync procedure (both sides, every time)

```bash
git pull --rebase
```

Then read in this order:
1. STATUS.md (where is the other side?)
2. OUTBOX.md (MAIN reads) / INBOX.md (H20 reads): new entries since your last sync
3. results/*.json (machine-readable numbers)

H20: act on new INBOX requests before continuing (if a request conflicts with PROMPT.md, the
INBOX request wins; if it breaks a hard rule in PROMPT.md, refuse it in OUTBOX.md).

## Deadline

Results are useful only before **2026-10-01 20:00 UTC+8**. Push the best partial result by
19:30 UTC+8 even if a task is unfinished, and say in STATUS.md what is missing.

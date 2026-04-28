# Optimizing the o11y skill with GEPA

This skill is shaped to be the target of [GEPA](https://gepa-ai.github.io/gepa)
prompt optimization, mirroring the RLM♥GEPA pattern Trampoline used for
SpreadsheetBench: a cheap-executor RLM produces traces, a reflection LM reads
those traces and proposes patches to `instructions.md`, and GEPA's loop picks
the best version on a held-out val split.

## What gets optimized

Just one artifact: `agents/o11y_skill/instructions.md`. The candidate dict GEPA
manipulates is `{"instructions": "<markdown>"}`. The seed is preserved on disk;
the winner is written to a sibling `instructions.evolved.md`.

## Splits

`agents/o11y_skill/splits.toml` pins 11 / 11 / 11 stratified train / val / test
task IDs. **Do not run optimization against `test`** — it's reserved for a
single final evaluation after a skill version is locked in.

## Running

Smoke (proves the loop closes; ~$2–4, ~15 min):

```bash
set -a && . .env && set +a
uv run python scripts/gepa_optimize.py --budget smoke
```

Small (real iteration; ~$15–30, ~1–2 h):

```bash
uv run python scripts/gepa_optimize.py --budget small
```

Medium (full splits; ~$50–150, ~hours):

```bash
uv run python scripts/gepa_optimize.py --budget medium
```

Knobs you might tweak:

- `--executor-model` — the cheap RLM that *produces* traces. Default
  `openrouter/anthropic/claude-haiku-4-5`.
- `--reflection-model` — the LM GEPA uses to *propose* new instructions.
  Default `openrouter/anthropic/claude-sonnet-4-6`.
- `--max-steps`, `--timeout-s` — passed to the executor RLM via env, identical
  to what `scripts/predict_rlm_eval.py` uses.

## Mechanism (in one paragraph)

The adapter (`agents/o11y_skill/adapter.py`) wraps every GEPA `evaluate()` call
in a swap of `instructions.md`: it writes the candidate to disk, runs the
executor on each task in the batch via the same `mise run bench:job:quiet`
code path the eval harness uses, parses per-trial `result.json` and
`trajectory.json`, and returns scores plus a distilled `O11yTrajectory` per
task. `OptimizationContext` brackets the entire `gepa.optimize` call so the
seed is restored on exit (and on Ctrl-C, the backup file
`instructions.seed.md.bak` survives for manual restore). Per-candidate runs
land in `jobs/<eval-id>/cand###/<task>/` with the original artifacts plus a
`gepa_meta.json` for post-hoc inspection.

`make_reflective_dataset` distills each trajectory into a short record (task
statement, agent's final answer, rubric breakdown with weighted scores +
explanations, summary of tool calls with errors, error excerpt). The
reflection LM sees the record — not the full 200k-token trajectory — so the
signal-to-noise is high.

## Reading the results

After a run finishes:

- `agents/o11y_skill/instructions.evolved.md` — the winner. Read the diff
  printed at the end (or `jobs/<eval-id>/instructions.diff`) to see what the
  reflection LM actually proposed. Sanity-check it: did it just shuffle
  whitespace, or did it identify a real failure pattern?
- `jobs/<eval-id>/gepa_summary.json` — best val score, candidates explored,
  total cost, elapsed.
- `jobs/<eval-id>/cand###/<task>/agent/trajectory.json` — every trajectory
  recorded during the run; useful for audit if a candidate scored
  surprisingly high.

## Promoting the evolved skill

Once you've reviewed `instructions.evolved.md` and re-run
`scripts/predict_rlm_eval.py --split val` against it (manually swap or use a
follow-up flag) and confirmed it beats the seed, replace
`instructions.md` with the evolved file and commit. The `test` split is the
final acceptance gate before any leaderboard submission.

## What's not (yet) RLM♥GEPA-faithful

The X post pairs GEPA with a *predict-rlm* proposer that reads the full trace
corpus programmatically. We currently use GEPA's stock reflection LM (a
vanilla LM reading our distilled records). If proposals plateau, the natural
follow-up is a custom proposer that exposes traces as tools — similar shape,
different `propose_new_texts` implementation.

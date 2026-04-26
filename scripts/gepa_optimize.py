"""Drive GEPA optimization of ``agents/o11y_skill/instructions.md``.

Loads the seed instructions, builds an ``O11ySkillAdapter`` against a chosen
train/val split, calls ``gepa.optimize`` with the chosen reflection LM, and
writes the winning instructions to ``agents/o11y_skill/instructions.evolved.md``
(the seed is preserved). Prints a unified diff of seed vs. winner and dumps a
JSON summary at ``jobs/<eval-id>/gepa_summary.json``.

Budgets (``--budget``):
  smoke   - 2 GEPA iterations, <=2 candidates, 3 train tasks, 3 val tasks
  small   - 5 iterations, <=4 candidates, 6 train, 6 val
  medium  - 12 iterations, <=6 candidates, full splits
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gepa

from agents.o11y_skill.adapter import (
    INSTRUCTIONS_COMPONENT,
    INSTRUCTIONS_FILE,
    JOBS_DIR,
    O11ySkillAdapter,
    OptimizationContext,
    load_split,
)

DEFAULT_REFLECTION = "openrouter/anthropic/claude-sonnet-4-6"
DEFAULT_EXECUTOR = "openrouter/anthropic/claude-haiku-4-5"

BUDGETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "max_metric_calls": 8,
        "reflection_minibatch_size": 3,
        "train_limit": 3,
        "val_limit": 3,
    },
    "small": {
        "max_metric_calls": 30,
        "reflection_minibatch_size": 4,
        "train_limit": 6,
        "val_limit": 6,
    },
    "medium": {
        "max_metric_calls": 100,
        "reflection_minibatch_size": 6,
        "train_limit": None,
        "val_limit": None,
    },
}


def _slice(items: list[str], limit: int | None) -> list[str]:
    return items if limit is None else items[:limit]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seed",
        type=Path,
        default=INSTRUCTIONS_FILE,
        help="seed instructions.md (default: agents/o11y_skill/instructions.md)",
    )
    p.add_argument("--out", type=Path, default=INSTRUCTIONS_FILE.parent / "instructions.evolved.md")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--reflection-model", default=DEFAULT_REFLECTION)
    p.add_argument("--executor-model", default=DEFAULT_EXECUTOR)
    p.add_argument("--budget", choices=list(BUDGETS), default="smoke")
    p.add_argument("--max-steps", type=int, default=12, help="MAX_STEPS for the executor RLM")
    p.add_argument("--timeout-s", type=int, default=480, help="per-task timeout")
    p.add_argument("--seed-rng", type=int, default=0, help="GEPA random seed")
    args = p.parse_args()

    if not args.seed.is_file():
        print(f"seed not found: {args.seed}", file=sys.stderr)
        return 2

    budget = BUDGETS[args.budget]
    eval_id = f"gepa-{args.budget}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    run_dir = JOBS_DIR / eval_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train = _slice(load_split(args.train_split), budget["train_limit"])
    val = _slice(load_split(args.val_split), budget["val_limit"])

    print(f"# GEPA optimization — budget={args.budget}")
    print(f"eval id: `{eval_id}`")
    print(f"executor: {args.executor_model}")
    print(f"reflection: {args.reflection_model}")
    print(f"train: {len(train)} task(s)  val: {len(val)} task(s)")
    print(f"max_metric_calls: {budget['max_metric_calls']}\n")
    sys.stdout.flush()

    seed_text = args.seed.read_text()
    adapter = O11ySkillAdapter(
        executor_model=args.executor_model,
        max_steps=args.max_steps,
        timeout_s=args.timeout_s,
        eval_id=eval_id,
    )
    seed_candidate = {INSTRUCTIONS_COMPONENT: seed_text}

    started = time.monotonic()
    with OptimizationContext():
        result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=train,
            valset=val,
            adapter=adapter,
            reflection_lm=args.reflection_model,
            max_metric_calls=budget["max_metric_calls"],
            reflection_minibatch_size=budget["reflection_minibatch_size"],
            seed=args.seed_rng,
            run_dir=str(run_dir),
            display_progress_bar=False,
        )
    elapsed = time.monotonic() - started

    best_candidate: dict[str, str] = result.best_candidate or seed_candidate
    best_text = best_candidate.get(INSTRUCTIONS_COMPONENT, seed_text)
    args.out.write_text(best_text)

    diff = "\n".join(
        difflib.unified_diff(
            seed_text.splitlines(),
            best_text.splitlines(),
            fromfile=str(args.seed),
            tofile=str(args.out),
            lineterm="",
        )
    )
    diff_path = run_dir / "instructions.diff"
    diff_path.write_text(diff)

    val_subscores = getattr(result, "val_aggregate_subscores", None) or []
    best_idx = getattr(result, "best_idx", None)
    best_val = (
        val_subscores[best_idx]
        if isinstance(best_idx, int) and 0 <= best_idx < len(val_subscores)
        else None
    )

    summary: dict[str, Any] = {
        "eval_id": eval_id,
        "budget": args.budget,
        "executor_model": args.executor_model,
        "reflection_model": args.reflection_model,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "n_train": len(train),
        "n_val": len(val),
        "max_metric_calls": budget["max_metric_calls"],
        "elapsed_s": elapsed,
        "seed_path": str(args.seed),
        "out_path": str(args.out),
        "best_val_score": best_val,
        "best_candidate_idx": best_idx,
        "val_subscores_per_candidate": list(val_subscores),
        "n_candidates_explored": getattr(result, "num_candidates", None),
        "total_metric_calls": getattr(result, "total_metric_calls", None),
        "diff_chars": len(diff),
    }
    (run_dir / "gepa_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\nbest val score: {summary['best_val_score']}")
    print(f"candidates explored: {summary['n_candidates_explored']}")
    print(f"elapsed: {elapsed:.0f}s")
    print(f"\nwinner written to: {args.out}")
    print(f"diff vs seed at: {diff_path}")
    print(f"summary: {run_dir / 'gepa_summary.json'}")

    if not diff.strip():
        print("\nNOTE: winner == seed (no edits proposed in this budget).")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Eval harness for the predict-rlm o11y agent.

Runs every task in a chosen split (train/val/test from
``agents/o11y_skill/splits.toml``) once with ``--n-attempts 1``, reads each
trial's reward + trajectory metrics, and prints a markdown summary to stdout.
A machine-readable record is dropped at ``jobs/<eval-id>/eval_summary.json``
so future runs can be diffed against this baseline.

Usage:
    uv run python scripts/predict_rlm_eval.py --split val
    uv run python scripts/predict_rlm_eval.py --split val --model openrouter/...
    uv run python scripts/predict_rlm_eval.py --split val --limit 3   # smoke
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_TOML = REPO_ROOT / "agents" / "o11y_skill" / "splits.toml"
JOBS_DIR = REPO_ROOT / "jobs"

DEFAULT_MODEL = "openrouter/anthropic/claude-haiku-4-5"
DEFAULT_AGENT = "agents.predict_rlm_o11y_agent:PredictRLMO11yAgent"
DEFAULT_MAX_STEPS = 12
DEFAULT_TIMEOUT_S = 480


def load_split(name: str) -> list[str]:
    with SPLITS_TOML.open("rb") as f:
        data = tomllib.load(f)
    if name not in data:
        raise SystemExit(f"unknown split {name!r}; choose from {sorted(data)}")
    tasks = data[name]
    if not isinstance(tasks, list):
        raise SystemExit(f"split {name!r} is not a list of task IDs")
    return [str(t) for t in tasks]


def run_task(
    *,
    task_id: str,
    model: str,
    agent_path: str,
    job_name: str,
    max_steps: int,
    timeout_s: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "mise",
        "run",
        "bench:job:quiet",
        "--",
        "--model",
        model,
        "--agent-import-path",
        agent_path,
        "--task-name",
        task_id,
        "--n-concurrent",
        "1",
        "--n-attempts",
        "1",
        "--job-name",
        job_name,
    ]
    env = {
        **os.environ,
        "O11Y_RLM_MAX_STEPS": str(max_steps),
        "O11Y_RLM_TIMEOUT_S": str(timeout_s),
    }
    return subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=REPO_ROOT)


def parse_trial(job_dir: Path) -> dict[str, Any]:
    """Read result.json + the (single) trial's trajectory.json."""
    result_path = job_dir / "result.json"
    if not result_path.exists():
        return {"reward": None, "status": "missing-result", "trial_dir": None}

    result = json.loads(result_path.read_text())
    evals = result.get("stats", {}).get("evals", {})
    reward: float | None = None
    trial_dir_name: str | None = None
    for ev in evals.values():
        for r_str, trials in ev.get("reward_stats", {}).get("reward", {}).items():
            try:
                reward = float(r_str)
            except ValueError:
                continue
            if trials:
                trial_dir_name = trials[0]
            break
        if trial_dir_name:
            break

    fm: dict[str, Any] = {}
    if trial_dir_name:
        traj_path = job_dir / trial_dir_name / "agent" / "trajectory.json"
        if traj_path.exists():
            fm = json.loads(traj_path.read_text()).get("final_metrics", {}) or {}

    return {
        "reward": reward,
        "status": fm.get("status"),
        "iterations": fm.get("total_iterations", 0),
        "tool_calls": fm.get("total_tool_calls", 0),
        "cost_usd": fm.get("total_cost_usd", 0.0),
        "in_tokens": fm.get("total_prompt_tokens", 0),
        "out_tokens": fm.get("total_completion_tokens", 0),
        "trial_dir": trial_dir_name,
    }


def fmt_reward(r: float | None) -> str:
    if r is None:
        return "—"
    return f"{r:.3f}"


def fmt_cost(c: float | None) -> str:
    if not c:
        return "—"
    return f"${c:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--agent-import-path", default=DEFAULT_AGENT)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--limit", type=int, default=None, help="run only the first N tasks (for smoke)"
    )
    parser.add_argument(
        "--eval-id",
        default=None,
        help="job-name prefix; defaults to eval-<split>-<utc-timestamp>",
    )
    args = parser.parse_args()

    tasks = load_split(args.split)
    if args.limit:
        tasks = tasks[: args.limit]

    eval_id = args.eval_id or f"eval-{args.split}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    eval_dir = JOBS_DIR / eval_id
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"# predict-rlm eval — split={args.split} model={args.model}")
    print(f"eval id: `{eval_id}`  ({len(tasks)} task(s))\n")
    print("| task | reward | cost | iter | tool_calls | status |")
    print("|---|---|---|---|---|---|")
    sys.stdout.flush()

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for task_id in tasks:
        job_name = f"{eval_id}/{task_id}"
        proc = run_task(
            task_id=task_id,
            model=args.model,
            agent_path=args.agent_import_path,
            job_name=job_name,
            max_steps=args.max_steps,
            timeout_s=args.timeout_s,
        )
        job_dir = JOBS_DIR / job_name
        if proc.returncode != 0 and not job_dir.exists():
            row = {
                "task": task_id,
                "reward": None,
                "status": "harness-error",
                "iterations": 0,
                "tool_calls": 0,
                "cost_usd": 0.0,
                "in_tokens": 0,
                "out_tokens": 0,
                "trial_dir": None,
                "stderr_tail": (proc.stderr or "").splitlines()[-3:],
            }
        else:
            parsed = parse_trial(job_dir)
            row = {"task": task_id, **parsed}
        rows.append(row)
        print(
            f"| {task_id} | {fmt_reward(row['reward'])} "
            f"| {fmt_cost(row.get('cost_usd'))} "
            f"| {row.get('iterations', 0)} "
            f"| {row.get('tool_calls', 0)} "
            f"| {row.get('status') or '—'} |",
            flush=True,
        )

    elapsed_s = time.monotonic() - started

    rewards = [r["reward"] for r in rows if isinstance(r["reward"], int | float)]
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    nonzero_count = sum(1 for r in rewards if r > 0)
    total_cost = sum(float(r.get("cost_usd") or 0.0) for r in rows)
    error_count = sum(
        1 for r in rows if r.get("status") in ("error", "harness-error", "missing-result")
    )

    print()
    print(f"**mean reward**: {mean_reward:.3f}  ({len(rewards)}/{len(rows)} graded)")
    print(f"**non-zero rewards**: {nonzero_count}/{len(rows)}")
    print(f"**error/missing**: {error_count}/{len(rows)}")
    print(f"**total cost**: ${total_cost:.4f}")
    print(f"**elapsed**: {elapsed_s:.0f}s")

    summary = {
        "eval_id": eval_id,
        "split": args.split,
        "model": args.model,
        "agent_import_path": args.agent_import_path,
        "max_steps": args.max_steps,
        "timeout_s": args.timeout_s,
        "n_tasks": len(rows),
        "mean_reward": mean_reward,
        "nonzero_count": nonzero_count,
        "error_count": error_count,
        "total_cost_usd": total_cost,
        "elapsed_s": elapsed_s,
        "rows": rows,
    }
    (eval_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written to `{eval_dir / 'eval_summary.json'}`")
    return 0


if __name__ == "__main__":
    sys.exit(main())

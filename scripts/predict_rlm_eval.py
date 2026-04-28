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
    n_attempts: int = 1,
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
        str(n_attempts),
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
    """Aggregate all trials in a job dir (n_attempts may be > 1).

    Returns per-attempt rewards, mean, min, max, plus aggregated cost and the
    final trajectory's status/iterations/tool_calls (best-effort across
    attempts — uses the first trial for those).
    """
    result_path = job_dir / "result.json"
    if not result_path.exists():
        return {
            "reward": None,
            "rewards_per_attempt": [],
            "status": "missing-result",
            "trial_dir": None,
        }

    result = json.loads(result_path.read_text())
    evals = result.get("stats", {}).get("evals", {})
    rewards_per_trial: list[tuple[str, float]] = []
    for ev in evals.values():
        for r_str, trials in ev.get("reward_stats", {}).get("reward", {}).items():
            try:
                r = float(r_str)
            except ValueError:
                continue
            for t in trials or []:
                rewards_per_trial.append((str(t), r))

    rewards = [r for _, r in rewards_per_trial]
    mean_reward = sum(rewards) / len(rewards) if rewards else None
    min_reward = min(rewards) if rewards else None
    max_reward = max(rewards) if rewards else None

    # Aggregate metrics across attempts: sum cost, max iter/calls (most-work attempt).
    total_cost = 0.0
    total_iter = 0
    total_calls = 0
    in_tokens = 0
    out_tokens = 0
    statuses: list[str] = []
    first_trial: str | None = None
    for trial_name, _ in rewards_per_trial:
        if first_trial is None:
            first_trial = trial_name
        traj = job_dir / trial_name / "agent" / "trajectory.json"
        if traj.is_file():
            fm = json.loads(traj.read_text()).get("final_metrics", {}) or {}
            total_cost += float(fm.get("total_cost_usd") or 0.0)
            total_iter = max(total_iter, int(fm.get("total_iterations") or 0))
            total_calls = max(total_calls, int(fm.get("total_tool_calls") or 0))
            in_tokens += int(fm.get("total_prompt_tokens") or 0)
            out_tokens += int(fm.get("total_completion_tokens") or 0)
            if fm.get("status"):
                statuses.append(str(fm["status"]))

    # Combined status: error wins, then max_iterations, else completed
    if statuses:
        if any(s == "error" for s in statuses):
            agg_status = "error"
        elif any(s == "max_iterations" for s in statuses):
            agg_status = "max_iterations"
        else:
            agg_status = statuses[0]
    else:
        agg_status = None

    return {
        "reward": mean_reward,
        "rewards_per_attempt": rewards,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "status": agg_status,
        "iterations": total_iter,
        "tool_calls": total_calls,
        "cost_usd": total_cost,
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "trial_dir": first_trial,
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
        "--n-attempts",
        type=int,
        default=1,
        help="Trials per task. n>1 enables Pass^k / Pass@k stats.",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=1.0,
        help="Reward threshold for 'pass' (default 1.0 = full credit).",
    )
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
    print(
        f"eval id: `{eval_id}`  ({len(tasks)} task(s) x {args.n_attempts} attempt(s),"
        f" pass-threshold={args.pass_threshold})\n"
    )
    if args.n_attempts > 1:
        print("| task | mean | min | max | rewards | cost |")
        print("|---|---|---|---|---|---|")
    else:
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
            n_attempts=args.n_attempts,
        )
        job_dir = JOBS_DIR / job_name
        if proc.returncode != 0 and not job_dir.exists():
            row = {
                "task": task_id,
                "reward": None,
                "rewards_per_attempt": [],
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
        if args.n_attempts > 1:
            attempts = row.get("rewards_per_attempt") or []
            attempts_str = ",".join(f"{r:.2f}" for r in attempts) or "—"
            print(
                f"| {task_id} | {fmt_reward(row['reward'])} "
                f"| {fmt_reward(row.get('min_reward'))} "
                f"| {fmt_reward(row.get('max_reward'))} "
                f"| {attempts_str} "
                f"| {fmt_cost(row.get('cost_usd'))} |",
                flush=True,
            )
        else:
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

    # Pass^k / Pass@k stats when n_attempts > 1.
    pass_at_k = pass_pow_k = None
    if args.n_attempts > 1:
        thr = args.pass_threshold
        pass_at_k = sum(
            1 for r in rows if r.get("rewards_per_attempt") and max(r["rewards_per_attempt"]) >= thr
        )
        pass_pow_k = sum(
            1
            for r in rows
            if r.get("rewards_per_attempt")
            and len(r["rewards_per_attempt"]) == args.n_attempts
            and min(r["rewards_per_attempt"]) >= thr
        )

    print()
    print(f"**mean reward**: {mean_reward:.3f}  ({len(rewards)}/{len(rows)} graded)")
    print(f"**non-zero rewards**: {nonzero_count}/{len(rows)}")
    if args.n_attempts > 1:
        print(
            f"**Pass^{args.n_attempts}** (all {args.n_attempts} attempts >= {args.pass_threshold}): "
            f"{pass_pow_k}/{len(rows)} = {pass_pow_k / max(len(rows), 1):.3f}"
        )
        print(
            f"**Pass@{args.n_attempts}** (any attempt >= {args.pass_threshold}): "
            f"{pass_at_k}/{len(rows)} = {pass_at_k / max(len(rows), 1):.3f}"
        )
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
        "n_attempts": args.n_attempts,
        "pass_threshold": args.pass_threshold,
        "n_tasks": len(rows),
        "mean_reward": mean_reward,
        "nonzero_count": nonzero_count,
        "pass_pow_k": pass_pow_k,
        "pass_at_k": pass_at_k,
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

"""GEPA adapter for the o11y skill.

Closes the RLM♥GEPA loop on ``agents/o11y_skill/instructions.md``:

- ``evaluate``: takes a candidate ``{"instructions": "..."}``, swaps the on-disk
  instructions, runs the o11y agent on a batch of task IDs via ``mise run
  bench:job:quiet`` (same code path as ``scripts/predict_rlm_eval.py``), parses
  per-task reward + trajectory, returns scores and distilled trajectories.
- ``make_reflective_dataset``: turns trajectories into a small JSON-serializable
  record per task focused on what *failed* (rubric breakdown, errors, final
  answer) — short enough that GEPA's reflection LM sees signal, not noise.

Instruction swapping is destructive on disk by design (sequential, single-flight)
and bracketed by ``OptimizationContext``. The seed is backed up once on enter
and restored on exit, so a hard kill leaves ``instructions.seed.md.bak`` next
to the live file for manual restore.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

# Component name in the candidate dict. Single-component optimization for now.
INSTRUCTIONS_COMPONENT = "instructions"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SKILL_DIR = REPO_ROOT / "agents" / "o11y_skill"
INSTRUCTIONS_FILE = SKILL_DIR / "instructions.md"
SEED_BACKUP = SKILL_DIR / "instructions.seed.md.bak"
JOBS_DIR = REPO_ROOT / "jobs"

DEFAULT_AGENT_IMPORT = "agents.predict_rlm_o11y_agent:PredictRLMO11yAgent"


@dataclass
class O11yTrajectory:
    """User-defined Trajectory type passed through GEPA back into make_reflective_dataset."""

    task_id: str
    category: str
    statement: str
    scenario_clock: str | None
    final_answer: str
    score: float
    rubric: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_summary: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    status: str | None = None
    error_excerpt: str | None = None
    cost_usd: float = 0.0


@contextmanager
def OptimizationContext(seed_path: Path = INSTRUCTIONS_FILE) -> Iterator[Path]:  # noqa: N802
    """Snapshot the seed on enter, restore on exit. Bracket the entire
    ``gepa.optimize`` call with this so cancellation restores the file."""
    if not seed_path.is_file():
        raise FileNotFoundError(f"seed instructions not found at {seed_path}")
    SEED_BACKUP.write_text(seed_path.read_text())
    try:
        yield seed_path
    finally:
        if SEED_BACKUP.is_file():
            seed_path.write_text(SEED_BACKUP.read_text())
            SEED_BACKUP.unlink()


def load_split(split_name: str, splits_path: Path = SKILL_DIR / "splits.toml") -> list[str]:
    with splits_path.open("rb") as f:
        data = tomllib.load(f)
    if split_name not in data or not isinstance(data[split_name], list):
        raise KeyError(f"split {split_name!r} missing or malformed in {splits_path}")
    return [str(t) for t in data[split_name]]


def _run_task_via_mise(
    *,
    task_id: str,
    job_name: str,
    executor_model: str,
    agent_import_path: str,
    max_steps: int,
    timeout_s: int,
    cwd: Path = REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "mise",
        "run",
        "bench:job:quiet",
        "--",
        "--model",
        executor_model,
        "--agent-import-path",
        agent_import_path,
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
    return subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)


def _read_task_spec(task_id: str) -> tuple[str, str]:
    """Return (category, statement) for a task by scanning tasks-spec/."""
    import yaml

    for cat_dir in (REPO_ROOT / "tasks-spec").iterdir():
        if not cat_dir.is_dir():
            continue
        candidate = cat_dir / f"{task_id}.yaml"
        if candidate.is_file():
            spec = yaml.safe_load(candidate.read_text()) or {}
            return str(spec.get("category") or cat_dir.name), str(spec.get("statement") or "")
    return "unknown", ""


def _parse_job_artifacts(job_dir: Path, task_id: str) -> dict[str, Any]:
    """Pull score, trajectory snippet, and rubric breakdown for the (single) trial."""
    out: dict[str, Any] = {
        "score": 0.0,
        "trial_dir": None,
        "trajectory_path": None,
        "grading_path": None,
    }
    result_path = job_dir / "result.json"
    if not result_path.is_file():
        return out
    result = json.loads(result_path.read_text())
    for ev in result.get("stats", {}).get("evals", {}).values():
        for r_str, trials in ev.get("reward_stats", {}).get("reward", {}).items():
            try:
                out["score"] = float(r_str)
            except ValueError:
                continue
            if trials:
                out["trial_dir"] = job_dir / trials[0]
            break
        if out["trial_dir"]:
            break
    trial_dir = out["trial_dir"]
    if trial_dir:
        traj = trial_dir / "agent" / "trajectory.json"
        grading = trial_dir / "verifier" / "grading_details.json"
        if traj.is_file():
            out["trajectory_path"] = traj
        if grading.is_file():
            out["grading_path"] = grading
    return out


def _summarize_tool_calls(trajectory: dict[str, Any], cap: int = 12) -> list[dict[str, Any]]:
    """Compact one entry per tool call: (name, kwarg keys, error?). Truncated to cap."""
    summary: list[dict[str, Any]] = []
    for step in trajectory.get("steps", []):
        for tc in step.get("tool_calls") or []:
            args = tc.get("arguments") or {}
            entry = {
                "function": tc.get("function_name"),
                "kwargs": sorted(args.keys()),
            }
            # Try to find the matching observation result for an error tag.
            obs = (step.get("observation") or {}).get("results") or []
            for r in obs:
                if r.get("source_call_id") == tc.get("tool_call_id") and "[error]" in str(
                    r.get("content", "")
                ):
                    entry["error"] = str(r["content"])[:240]
                    break
            summary.append(entry)
            if len(summary) >= cap:
                return summary
    return summary


def _extract_error_excerpt(trajectory: dict[str, Any]) -> str | None:
    """Pull the most informative error message from REPL output / observation results."""
    last: str | None = None
    for step in trajectory.get("steps", []):
        for r in (step.get("observation") or {}).get("results") or []:
            content = str(r.get("content", ""))
            if "[error]" in content or "Traceback" in content:
                last = content[:600]
        out = step.get("metrics", {}).get("error")
        if out:
            return str(out)[:600]
    return last


def _parse_rubric(grading_path: Path | None) -> list[dict[str, Any]]:
    if not grading_path or not grading_path.is_file():
        return []
    g = json.loads(grading_path.read_text())
    rubric: list[dict[str, Any]] = []
    for k, v in g.items():
        if k.startswith("explanation:") or k in {"score", "checks_passed", "rubric_passed"}:
            continue
        rubric.append(
            {
                "criterion": k,
                "score": v if isinstance(v, int | float) else None,
                "explanation": g.get(f"explanation:{k}"),
            }
        )
    return rubric


def build_trajectory_record(*, task_id: str, job_dir: Path) -> O11yTrajectory:
    category, statement = _read_task_spec(task_id)
    arts = _parse_job_artifacts(job_dir, task_id)
    score = arts["score"]
    trajectory_path = arts["trajectory_path"]
    grading_path = arts["grading_path"]

    final_answer = ""
    iterations = 0
    status: str | None = None
    cost = 0.0
    tool_calls: list[dict[str, Any]] = []
    error_excerpt: str | None = None
    scenario_clock: str | None = None

    if trajectory_path and trajectory_path.is_file():
        traj = json.loads(trajectory_path.read_text())
        steps = traj.get("steps") or []
        if steps:
            final_answer = str(steps[-1].get("message", ""))[:1200]
            for step in steps:
                if step.get("source") == "user":
                    msg = str(step.get("message", ""))
                    if "<context>" in msg and "Current time:" in msg:
                        for line in msg.splitlines():
                            if line.strip().startswith("Current time:"):
                                scenario_clock = line.split(":", 1)[1].strip()
                                break
                    break
        fm = traj.get("final_metrics") or {}
        iterations = int(fm.get("total_iterations") or 0)
        status = fm.get("status")
        cost = float(fm.get("total_cost_usd") or 0.0)
        tool_calls = _summarize_tool_calls(traj)
        error_excerpt = _extract_error_excerpt(traj) if score == 0 else None

    rubric = _parse_rubric(grading_path)

    return O11yTrajectory(
        task_id=task_id,
        category=category,
        statement=statement[:1500],
        scenario_clock=scenario_clock,
        final_answer=final_answer,
        score=score,
        rubric=rubric,
        tool_calls_summary=tool_calls,
        iterations=iterations,
        status=status,
        error_excerpt=error_excerpt,
        cost_usd=cost,
    )


@dataclass
class O11ySkillAdapter(GEPAAdapter):
    """GEPA adapter that swaps ``instructions.md``, runs the agent per task, and
    distills trajectories for the reflection LM.

    Use under ``with OptimizationContext(): ...`` so the seed is restored.
    """

    executor_model: str = "openrouter/anthropic/claude-haiku-4-5"
    agent_import_path: str = DEFAULT_AGENT_IMPORT
    max_steps: int = 12
    timeout_s: int = 480
    eval_id: str = field(
        default_factory=lambda: f"gepa-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    )
    _candidate_counter: int = 0

    def evaluate(
        self,
        batch: list[str],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[O11yTrajectory, str]:
        instructions = candidate.get(INSTRUCTIONS_COMPONENT)
        if not isinstance(instructions, str):
            raise ValueError(
                f"candidate must contain string {INSTRUCTIONS_COMPONENT!r}; got {candidate!r}"
            )

        self._candidate_counter += 1
        cand_idx = self._candidate_counter

        # Swap the on-disk instructions for this candidate. Sequential by design.
        INSTRUCTIONS_FILE.write_text(instructions)

        outputs: list[str] = []
        scores: list[float] = []
        trajectories: list[O11yTrajectory] = []

        for task_id in batch:
            job_name = f"{self.eval_id}/cand{cand_idx:03d}/{task_id}"
            t0 = time.monotonic()
            proc = _run_task_via_mise(
                task_id=task_id,
                job_name=job_name,
                executor_model=self.executor_model,
                agent_import_path=self.agent_import_path,
                max_steps=self.max_steps,
                timeout_s=self.timeout_s,
            )
            elapsed = time.monotonic() - t0
            job_dir = JOBS_DIR / job_name

            if proc.returncode != 0 and not job_dir.exists():
                trajectory = O11yTrajectory(
                    task_id=task_id,
                    category="unknown",
                    statement="",
                    scenario_clock=None,
                    final_answer="",
                    score=0.0,
                    error_excerpt=(proc.stderr or "")[-600:] or "harness-error",
                    iterations=0,
                    status="harness-error",
                )
            else:
                trajectory = build_trajectory_record(task_id=task_id, job_dir=job_dir)

            scores.append(trajectory.score)
            outputs.append(trajectory.final_answer)
            trajectories.append(trajectory)
            if job_dir.exists():
                (job_dir / "gepa_meta.json").write_text(
                    json.dumps(
                        {
                            "candidate_index": cand_idx,
                            "task_id": task_id,
                            "elapsed_s": elapsed,
                            "executor_model": self.executor_model,
                            "max_steps": self.max_steps,
                        },
                        indent=2,
                    )
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[O11yTrajectory, str],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        if INSTRUCTIONS_COMPONENT not in components_to_update:
            return {}
        records: list[dict[str, Any]] = []
        for traj in eval_batch.trajectories or []:
            records.append(
                {
                    "input": {
                        "task_id": traj.task_id,
                        "category": traj.category,
                        "scenario_clock": traj.scenario_clock,
                        "statement": traj.statement,
                    },
                    "output": traj.final_answer,
                    "score": traj.score,
                    "feedback": {
                        "rubric": traj.rubric,
                        "tool_calls_summary": traj.tool_calls_summary,
                        "iterations": traj.iterations,
                        "status": traj.status,
                        "error_excerpt": traj.error_excerpt,
                    },
                }
            )
        return {INSTRUCTIONS_COMPONENT: records}

"""Tests for the o11y skill GEPA adapter.

Mock the agent-runner subprocess so the adapter is exercised without standing
up Docker / MCP / live LMs. Verifies:
- ``OptimizationContext`` snapshots and restores the seed instructions.
- ``evaluate`` writes the candidate to disk, parses scores from result.json,
  and produces trajectories whose shape matches what GEPA's reflection layer
  expects.
- ``make_reflective_dataset`` distills trajectories into the expected
  per-task records, keyed by the ``instructions`` component.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from gepa.core.adapter import EvaluationBatch

from agents.o11y_skill.adapter import (
    INSTRUCTIONS_COMPONENT,
    O11ySkillAdapter,
    O11yTrajectory,
    OptimizationContext,
    build_trajectory_record,
)


def _write_job_artifacts(
    job_dir: Path,
    *,
    task_id: str,
    reward: float,
    final_answer: str = "an answer",
    rubric: dict[str, Any] | None = None,
    trajectory_extras: dict[str, Any] | None = None,
) -> None:
    """Create the on-disk shape produced by ``mise run bench:job``."""
    job_dir.mkdir(parents=True, exist_ok=True)
    trial_dir = job_dir / f"{task_id}__trial"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "verifier").mkdir(parents=True)

    eval_key = "predict-rlm-o11y__test__tasks"
    result = {
        "stats": {
            "evals": {
                eval_key: {
                    "reward_stats": {
                        "reward": {str(reward): [trial_dir.name]},
                    }
                }
            }
        }
    }
    (job_dir / "result.json").write_text(json.dumps(result))

    trajectory: dict[str, Any] = {
        "schema_version": "ATIF-v1.6",
        "steps": [
            {"source": "system", "message": "system"},
            {
                "source": "user",
                "message": (
                    "<context>\nCurrent time: 2026-04-26T12:00:00Z\n</context>\n\nthe statement"
                ),
            },
            {
                "source": "agent",
                "message": "code",
                "tool_calls": [
                    {
                        "tool_call_id": "tc-1",
                        "function_name": "query_metrics",
                        "arguments": {"expr": "up", "start": "x", "end": "y"},
                    }
                ],
                "observation": {
                    "results": [{"source_call_id": "tc-1", "content": "{}"}],
                },
            },
            {"source": "agent", "message": final_answer, "metrics": {"final": True}},
        ],
        "final_metrics": {
            "total_iterations": 3,
            "status": "completed",
            "total_cost_usd": 0.05,
            "total_prompt_tokens": 100,
            "total_completion_tokens": 50,
        },
    }
    if trajectory_extras:
        trajectory.update(trajectory_extras)
    (trial_dir / "agent" / "trajectory.json").write_text(json.dumps(trajectory))

    if rubric is None:
        rubric = {
            "score": reward,
            "Final response is correct.": 1.0 if reward > 0.5 else 0.0,
            "explanation:Final response is correct.": "ok" if reward > 0.5 else "wrong",
        }
    (trial_dir / "verifier" / "grading_details.json").write_text(json.dumps(rubric))


def test_optimization_context_restores_seed_on_normal_exit(tmp_path: Path):
    seed = tmp_path / "instructions.md"
    original = "ORIGINAL"
    seed.write_text(original)

    with OptimizationContext(seed_path=seed):
        seed.write_text("MUTATED")
        assert seed.read_text() == "MUTATED"

    assert seed.read_text() == original


def test_optimization_context_restores_seed_on_exception(tmp_path: Path):
    seed = tmp_path / "instructions.md"
    seed.write_text("ORIGINAL")

    with pytest.raises(RuntimeError), OptimizationContext(seed_path=seed):
        seed.write_text("MUTATED")
        raise RuntimeError("boom")

    assert seed.read_text() == "ORIGINAL"


def test_build_trajectory_record_extracts_score_and_summary(tmp_path: Path, monkeypatch):
    job_dir = tmp_path / "job"
    _write_job_artifacts(job_dir, task_id="query-cpu-metrics", reward=0.5, final_answer="42")

    # Prevent _read_task_spec from hitting disk by stubbing it.
    monkeypatch.setattr(
        "agents.o11y_skill.adapter._read_task_spec",
        lambda task_id: ("prometheus_query", "Sample statement."),
    )

    record = build_trajectory_record(task_id="query-cpu-metrics", job_dir=job_dir)

    assert record.task_id == "query-cpu-metrics"
    assert record.category == "prometheus_query"
    assert record.score == 0.5
    assert record.final_answer == "42"
    assert record.iterations == 3
    assert record.status == "completed"
    assert record.scenario_clock == "2026-04-26T12:00:00Z"
    assert any(c["function"] == "query_metrics" for c in record.tool_calls_summary)
    assert record.rubric and record.rubric[0]["criterion"] == "Final response is correct."


def test_evaluate_writes_candidate_and_parses_scores(tmp_path: Path, monkeypatch):
    """``evaluate`` should overwrite the instructions file and read scores back."""
    seed = tmp_path / "instructions.md"
    seed.write_text("SEED")
    monkeypatch.setattr("agents.o11y_skill.adapter.INSTRUCTIONS_FILE", seed)
    monkeypatch.setattr("agents.o11y_skill.adapter.SEED_BACKUP", tmp_path / "seed.bak")

    fake_jobs_dir = tmp_path / "jobs"
    monkeypatch.setattr("agents.o11y_skill.adapter.JOBS_DIR", fake_jobs_dir)
    monkeypatch.setattr(
        "agents.o11y_skill.adapter._read_task_spec",
        lambda task_id: ("prometheus_query", f"statement for {task_id}"),
    )

    rewards = {"task-a": 0.3, "task-b": 0.7}

    def fake_run(*, task_id: str, job_name: str, **kwargs):
        # Simulate Harbor writing artifacts.
        job_dir = fake_jobs_dir / job_name
        _write_job_artifacts(job_dir, task_id=task_id, reward=rewards[task_id])
        # Adapter checks .returncode/job_dir.exists; mimic a successful subprocess.
        return _CompletedProcess(returncode=0)

    monkeypatch.setattr("agents.o11y_skill.adapter._run_task_via_mise", fake_run)

    adapter = O11ySkillAdapter(eval_id="test-eval")
    with OptimizationContext(seed_path=seed):
        batch_result = adapter.evaluate(
            batch=["task-a", "task-b"],
            candidate={INSTRUCTIONS_COMPONENT: "CANDIDATE-V2"},
            capture_traces=True,
        )

    # Candidate was written to disk during evaluate (before context restore).
    # After exit, OptimizationContext restores SEED.
    assert seed.read_text() == "SEED"
    assert isinstance(batch_result, EvaluationBatch)
    assert batch_result.scores == [0.3, 0.7]
    assert batch_result.outputs == ["an answer", "an answer"]
    assert batch_result.trajectories is not None
    assert {t.task_id for t in batch_result.trajectories} == {"task-a", "task-b"}


def test_make_reflective_dataset_only_emits_requested_components():
    adapter = O11ySkillAdapter()
    traj = O11yTrajectory(
        task_id="t1",
        category="prometheus_query",
        statement="s",
        scenario_clock="2026-04-26T00:00:00Z",
        final_answer="ans",
        score=0.0,
        rubric=[{"criterion": "c", "score": 0.0, "explanation": "wrong"}],
        iterations=2,
        status="max_iterations",
    )
    eval_batch = EvaluationBatch(outputs=["ans"], scores=[0.0], trajectories=[traj])

    # Asked for instructions: get records.
    out = adapter.make_reflective_dataset(
        candidate={INSTRUCTIONS_COMPONENT: "..."},
        eval_batch=eval_batch,
        components_to_update=[INSTRUCTIONS_COMPONENT],
    )
    assert INSTRUCTIONS_COMPONENT in out
    assert len(out[INSTRUCTIONS_COMPONENT]) == 1
    record = out[INSTRUCTIONS_COMPONENT][0]
    assert record["score"] == 0.0
    assert record["feedback"]["status"] == "max_iterations"
    assert record["input"]["task_id"] == "t1"

    # Asked for nothing: empty dict.
    out = adapter.make_reflective_dataset(
        candidate={INSTRUCTIONS_COMPONENT: "..."},
        eval_batch=eval_batch,
        components_to_update=[],
    )
    assert out == {}


class _CompletedProcess:
    """Tiny stand-in for subprocess.CompletedProcess (only fields we touch)."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

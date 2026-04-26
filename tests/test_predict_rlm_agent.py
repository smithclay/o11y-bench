"""Tests for the predict-rlm Harbor agent class.

The runtime logic lives in ``agents/predict_rlm_agent_runner.py`` (a PEP 723
script that runs inside the task container) and is not unit-tested here, by
the same convention the LangChain agent follows.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from agents.predict_rlm_o11y_agent import (
    RUNNER_SCRIPT,
    SKILL_DIR,
    SKILL_FILES,
    PredictRLMO11yAgent,
)


def test_agent_identity_strings():
    assert PredictRLMO11yAgent.name() == "predict-rlm-o11y"
    agent = PredictRLMO11yAgent(logs_dir=Path("/tmp/does-not-matter"))
    assert agent.version() == "1.0.0"


def test_runner_script_path_exists():
    assert RUNNER_SCRIPT.is_file()
    assert RUNNER_SCRIPT.name == "predict_rlm_agent_runner.py"


def test_extra_env_picks_up_predict_rlm_knobs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("O11Y_RLM_SUB", "openrouter/anthropic/claude-haiku-4-5")
    monkeypatch.setenv("O11Y_RLM_MAX_STEPS", "8")
    monkeypatch.setenv("O11Y_RLM_TIMEOUT_S", "120")

    agent = PredictRLMO11yAgent(logs_dir=Path("/tmp/does-not-matter"))

    assert agent._extra_env["SUB_MODEL"] == "openrouter/anthropic/claude-haiku-4-5"
    assert agent._extra_env["MAX_STEPS"] == "8"
    assert agent._extra_env["TIMEOUT_S"] == "120"


def test_extra_env_omits_unset_knobs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("O11Y_RLM_SUB", raising=False)
    monkeypatch.delenv("O11Y_RLM_MAX_STEPS", raising=False)
    monkeypatch.delenv("O11Y_RLM_TIMEOUT_S", raising=False)

    agent = PredictRLMO11yAgent(logs_dir=Path("/tmp/does-not-matter"))

    assert "SUB_MODEL" not in agent._extra_env
    assert "MAX_STEPS" not in agent._extra_env
    assert "TIMEOUT_S" not in agent._extra_env


def test_skill_dir_layout():
    assert SKILL_DIR.is_dir()
    for filename in SKILL_FILES:
        assert (SKILL_DIR / filename).is_file(), f"missing skill file: {filename}"


@pytest.mark.anyio
async def test_setup_uploads_runner_and_skill(tmp_path: Path):
    agent = PredictRLMO11yAgent(logs_dir=tmp_path)
    environment = MagicMock()
    environment.exec = AsyncMock()
    environment.upload_file = AsyncMock()

    await agent.setup(environment)

    environment.exec.assert_awaited_once_with(command="mkdir -p /app/o11y_skill")

    expected = [
        call(source_path=RUNNER_SCRIPT, target_path="/app/agent_runner.py"),
        *(call(source_path=SKILL_DIR / f, target_path=f"/app/o11y_skill/{f}") for f in SKILL_FILES),
    ]
    assert environment.upload_file.await_args_list == expected

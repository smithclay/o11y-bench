"""Harbor agent that drives o11y-bench tasks through a predict-rlm runtime.

Mirrors the LangChain agent pattern: a thin Harbor agent that uploads a
self-contained PEP 723 runner into the task container, then delegates to the
parent ``O11yBenchAgent.run()`` which wires up env vars and execs the runner.
The runner connects to mcp-grafana over the harbor-shared network and drives
PredictRLM's REPL-with-tools loop.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment

from .o11y_agent import O11yBenchAgent

RUNNER_SCRIPT = Path(__file__).parent / "predict_rlm_agent_runner.py"


class PredictRLMO11yAgent(O11yBenchAgent):
    """o11y-bench agent that drives tasks through a predict-rlm runtime.

    Outer LM writes Python in a Pyodide/WASM REPL and composes MCP tool calls
    (Prometheus, Loki, Tempo, dashboards) and ``await predict(...)`` sub-LM
    calls programmatically. Trajectories are interpretable Python turns rather
    than raw tool-call/observation pairs, which avoids the context-rot failure
    mode that dominates Pass^3 on this benchmark.

    Configuration via env vars (all optional):
    - ``O11Y_RLM_SUB`` — structured-extraction sub-LM
    - ``O11Y_RLM_MAX_STEPS`` — REPL iteration cap
    - ``O11Y_RLM_TIMEOUT_S`` — wall-clock cap for the RLM call
    """

    SUPPORTS_ATIF = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        extra = dict(self._extra_env or {})
        if sub := os.environ.get("O11Y_RLM_SUB"):
            extra["SUB_MODEL"] = sub
        if steps := os.environ.get("O11Y_RLM_MAX_STEPS"):
            extra["MAX_STEPS"] = steps
        if timeout := os.environ.get("O11Y_RLM_TIMEOUT_S"):
            extra["TIMEOUT_S"] = timeout
        self._extra_env = extra

    @staticmethod
    def name() -> str:
        return "predict-rlm-o11y"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec(command="mkdir -p /app/agents")
        await environment.upload_file(
            source_path=RUNNER_SCRIPT,
            target_path="/app/agent_runner.py",
        )

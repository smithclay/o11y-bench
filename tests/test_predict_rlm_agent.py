from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from harbor.models.agent.context import AgentContext

from agents.predict_rlm_o11y_agent import (
    MCP_TOOL_CATALOG,
    PredictRLMO11yAgent,
    _to_atif,
    build_mcp_tools,
)


class MockMCPSession:
    """Duck-typed MCP client that records call_tool invocations."""

    def __init__(self, canned: dict[str, Any] | None = None) -> None:
        self.canned = canned or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, dict(arguments)))
        payload = self.canned.get(name, {"ok": True, "name": name, "args": arguments})
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])


def test_build_mcp_tools_returns_expected_catalog():
    session = MockMCPSession()
    tools = build_mcp_tools(session)

    assert len(tools) == len(MCP_TOOL_CATALOG)
    expected_names = {entry["name"] for entry in MCP_TOOL_CATALOG}
    actual_names = {tool.__name__ for tool in tools}
    assert actual_names == expected_names

    for tool in tools:
        assert inspect.iscoroutinefunction(tool), f"{tool.__name__} must be async"
        assert tool.__doc__, f"{tool.__name__} must have a docstring for tool-doc rendering"


@pytest.mark.anyio
async def test_built_tools_proxy_to_session_call_tool():
    session = MockMCPSession(
        canned={"query_prometheus": {"data": {"result": [{"value": [0, "42"]}]}}}
    )
    tools_by_name = {tool.__name__: tool for tool in build_mcp_tools(session)}

    result = await tools_by_name["query_prometheus"](
        expr="rate(http_requests_total[5m])", start="2026-01-01T00:00:00Z"
    )

    assert session.calls == [
        (
            "query_prometheus",
            {"expr": "rate(http_requests_total[5m])", "start": "2026-01-01T00:00:00Z"},
        )
    ]
    assert result == {"data": {"result": [{"value": [0, "42"]}]}}


def test_agent_identity_strings():
    assert isinstance(PredictRLMO11yAgent.name(), str)
    assert PredictRLMO11yAgent.name()  # non-empty

    agent = PredictRLMO11yAgent(logs_dir=Path("/tmp/does-not-matter"))
    version = agent.version()
    assert isinstance(version, str)
    assert version  # non-empty


def test_to_atif_includes_system_user_iteration_and_final_answer(tmp_path: Path):
    fake_call = SimpleNamespace(
        name="query_prometheus",
        args=[],
        kwargs={"expr": "up"},
        result={"status": "success"},
        error=None,
        duration_ms=12,
    )
    fake_step = SimpleNamespace(
        iteration=1,
        reasoning="Identify the up signal",
        code='r = await query_prometheus(expr="up")\nprint(r)',
        output='{"status": "success"}',
        untruncated_output='{"status": "success"}',
        error=False,
        duration_ms=42,
        tool_calls=[fake_call],
        predict_calls=[],
    )
    fake_trace = SimpleNamespace(
        status="completed",
        model="anthropic/claude-opus-4-7",
        sub_model="anthropic/claude-haiku-4-5",
        iterations=1,
        max_iterations=30,
        duration_ms=100,
        usage=SimpleNamespace(
            main=SimpleNamespace(input_tokens=10, output_tokens=20, cost=0.01),
            sub=SimpleNamespace(input_tokens=3, output_tokens=4, cost=0.001),
        ),
        steps=[fake_step],
    )

    trajectory = _to_atif(
        instruction="What is up?",
        answer="up",
        trace=fake_trace,
        model_name="anthropic/claude-opus-4-7",
        sub_model_name="anthropic/claude-haiku-4-5",
    )

    assert trajectory["schema_version"] == "ATIF-v1.6"
    sources = [step["source"] for step in trajectory["steps"]]
    # Expect: system, user, agent (iteration 1), agent (final)
    assert sources[:2] == ["system", "user"]
    assert sources[-1] == "agent"
    assert trajectory["steps"][-1]["message"] == "up"

    iteration_step = trajectory["steps"][2]
    assert iteration_step["source"] == "agent"
    assert "query_prometheus" in iteration_step["message"]
    tool_call_names = [tc["function_name"] for tc in iteration_step["tool_calls"]]
    assert tool_call_names == ["query_prometheus"]

    fm = trajectory["final_metrics"]
    assert fm["total_prompt_tokens"] == 13  # 10 + 3
    assert fm["total_completion_tokens"] == 24  # 20 + 4
    assert fm["total_tool_calls"] == 1
    assert fm["status"] == "completed"


@pytest.mark.anyio
async def test_run_writes_failure_trajectory_when_no_remote_mcp_url(tmp_path: Path):
    agent = PredictRLMO11yAgent(logs_dir=tmp_path, model_name="anthropic/claude-opus-4-7")
    # No remote URL -> select_remote_mcp_url returns None -> failure path.
    agent.mcp_servers = [SimpleNamespace(url="http://localhost:8080/mcp")]
    context = AgentContext()

    await agent.run("List datasources.", environment=SimpleNamespace(), context=context)

    trajectory_path = tmp_path / "trajectory.json"
    assert trajectory_path.exists()
    trajectory = json.loads(trajectory_path.read_text())
    assert trajectory["final_metrics"]["status"] == "error"
    final_message = trajectory["steps"][-1]["message"]
    assert final_message.startswith("[agent error]")
    assert context.metadata is not None
    assert context.metadata["agent"] == PredictRLMO11yAgent.name()

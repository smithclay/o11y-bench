"""Tests for the o11y skill bundle (instructions + helpers).

The helpers are async closures over an MCP session and an internal datasource
uid cache. We exercise them with a duck-typed mock session so we can assert
the wire-tool dispatch and uid caching without standing up the real Grafana
sidecar.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from agents.o11y_skill import build_o11y_skill, build_tools


class MockMCPSession:
    """Records call_tool invocations and returns canned JSON-encoded results."""

    def __init__(self, canned: dict[str, Any] | None = None) -> None:
        self.canned = canned or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, dict(arguments)))
        payload = self.canned.get(name, {"ok": True, "name": name, "args": arguments})
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])


_DATASOURCES = [
    {"uid": "prom-1", "name": "Prometheus", "type": "prometheus"},
    {"uid": "loki-1", "name": "Loki", "type": "loki"},
    {"uid": "tempo-1", "name": "Tempo", "type": "tempo"},
]


def test_build_tools_keys_match_expected_helpers():
    tools = build_tools(MockMCPSession())
    assert set(tools) == {
        "list_datasources",
        "list_metrics",
        "list_log_labels",
        "query_metrics",
        "query_logs",
        "query_traces",
        "get_dashboard",
        "save_dashboard",
        "search_dashboards",
    }


@pytest.mark.anyio
async def test_list_metrics_routes_to_prometheus_metric_names():
    session = MockMCPSession(
        canned={
            "list_datasources": _DATASOURCES,
            "list_prometheus_metric_names": ["http_requests_total", "cache_refresh_lag_seconds"],
        }
    )
    tools = build_tools(session)

    result = await tools["list_metrics"](regex=".*cache.*", limit=10)

    metric_call = next(c for c in session.calls if c[0] == "list_prometheus_metric_names")
    assert metric_call[1] == {
        "datasourceUid": "prom-1",
        "regex": ".*cache.*",
        "limit": 10,
    }
    assert result == ["http_requests_total", "cache_refresh_lag_seconds"]


@pytest.mark.anyio
async def test_list_log_labels_routes_to_loki_label_names():
    session = MockMCPSession(
        canned={
            "list_datasources": _DATASOURCES,
            "list_loki_label_names": ["job", "service", "level"],
        }
    )
    tools = build_tools(session)

    await tools["list_log_labels"](start="2026-04-25T00:00:00Z", end="2026-04-25T01:00:00Z")

    loki_call = next(c for c in session.calls if c[0] == "list_loki_label_names")
    assert loki_call[1] == {
        "datasourceUid": "loki-1",
        "startRfc3339": "2026-04-25T00:00:00Z",
        "endRfc3339": "2026-04-25T01:00:00Z",
    }


@pytest.mark.anyio
async def test_query_metrics_routes_to_query_prometheus_with_resolved_uid():
    session = MockMCPSession(
        canned={
            "list_datasources": _DATASOURCES,
            "query_prometheus": {"data": {"result": [{"value": [0, "1"]}]}},
        }
    )
    tools = build_tools(session)

    result = await tools["query_metrics"](
        expr="up", start="2026-04-25T00:00:00Z", end="2026-04-25T01:00:00Z", step="5m"
    )

    # First call resolves the uid; second is the actual query.
    assert session.calls[0] == ("list_datasources", {})
    assert session.calls[1] == (
        "query_prometheus",
        {
            "datasourceUid": "prom-1",
            "expr": "up",
            "startTime": "2026-04-25T00:00:00Z",
            "endTime": "2026-04-25T01:00:00Z",
            "queryType": "range",
            "stepSeconds": 300,
        },
    )
    assert result == {"data": {"result": [{"value": [0, "1"]}]}}


@pytest.mark.anyio
async def test_uid_cache_is_populated_once_across_helpers():
    session = MockMCPSession(canned={"list_datasources": _DATASOURCES})
    tools = build_tools(session)

    await tools["query_metrics"](expr="up")
    await tools["query_logs"](
        expr='{job="x"}', start="2026-04-25T00:00:00Z", end="2026-04-25T01:00:00Z"
    )
    await tools["query_traces"](traceql="{}")

    # Exactly one list_datasources call across three helper invocations.
    list_calls = [c for c in session.calls if c[0] == "list_datasources"]
    assert len(list_calls) == 1


@pytest.mark.anyio
async def test_query_logs_uses_loki_wire_name_and_kwargs():
    session = MockMCPSession(canned={"list_datasources": _DATASOURCES})
    tools = build_tools(session)

    await tools["query_logs"](
        expr='{job="x"}',
        start="2026-04-25T00:00:00Z",
        end="2026-04-25T01:00:00Z",
        limit=50,
    )

    loki_call = next(c for c in session.calls if c[0] == "query_loki_logs")
    assert loki_call[1] == {
        "datasourceUid": "loki-1",
        "logql": '{job="x"}',
        "startRfc3339": "2026-04-25T00:00:00Z",
        "endRfc3339": "2026-04-25T01:00:00Z",
        "limit": 50,
        "direction": "backward",
    }


@pytest.mark.anyio
async def test_query_traces_uses_tempo_traceql_search_wire_name():
    session = MockMCPSession(canned={"list_datasources": _DATASOURCES})
    tools = build_tools(session)

    await tools["query_traces"](traceql='{ resource.service.name = "x" }', limit=5)

    tempo_call = next(c for c in session.calls if c[0] == "tempo_traceql-search")
    assert tempo_call[1]["datasourceUid"] == "tempo-1"
    assert tempo_call[1]["query"] == '{ resource.service.name = "x" }'
    assert tempo_call[1]["limit"] == 5


@pytest.mark.anyio
async def test_dashboard_helpers_route_to_correct_wire_names():
    session = MockMCPSession(canned={"list_datasources": _DATASOURCES})
    tools = build_tools(session)

    await tools["get_dashboard"](uid="abc")
    await tools["save_dashboard"](model={"title": "x"})
    await tools["search_dashboards"](query="cache")

    wire_names = [c[0] for c in session.calls]
    assert "get_dashboard_by_uid" in wire_names
    assert "update_dashboard" in wire_names
    assert "search_dashboards" in wire_names


@pytest.mark.anyio
async def test_dict_wrapped_list_datasources_is_extracted():
    """mcp-grafana sometimes returns a wrapper dict ``{"datasources": [...]}``
    instead of a bare list. The cache must handle both shapes."""
    session = MockMCPSession(
        canned={
            "list_datasources": {"datasources": _DATASOURCES},
            "query_prometheus": {"data": {"result": []}},
        }
    )
    tools = build_tools(session)

    await tools["query_metrics"](expr="up")

    prom_call = next(c for c in session.calls if c[0] == "query_prometheus")
    assert prom_call[1]["datasourceUid"] == "prom-1"


@pytest.mark.anyio
async def test_falls_back_to_type_as_uid_when_list_datasources_unparseable():
    """If ``list_datasources`` returns a shape we cannot extract from, helpers
    should still work using ``type`` as the uid (matches the synthetic stack)."""
    session = MockMCPSession(
        canned={
            "list_datasources": "not a list or dict",
            "query_loki_logs": {"data": []},
        }
    )
    tools = build_tools(session)

    await tools["query_logs"](
        expr='{job="x"}', start="2026-04-25T00:00:00Z", end="2026-04-25T01:00:00Z"
    )

    loki_call = next(c for c in session.calls if c[0] == "query_loki_logs")
    assert loki_call[1]["datasourceUid"] == "loki"


@pytest.mark.anyio
async def test_unknown_datasource_type_still_raises():
    """The three signal datasources have fallbacks, but an unknown type still
    raises a helpful error rather than silently using a bogus uid."""
    from agents.o11y_skill.tools import _UidCache

    session = MockMCPSession(canned={"list_datasources": _DATASOURCES})
    cache = _UidCache(session)

    with pytest.raises(RuntimeError, match="no 'elasticsearch' datasource"):
        await cache.get("elasticsearch")


def test_build_o11y_skill_bundles_instructions_and_tools():
    session = MockMCPSession()
    skill = build_o11y_skill(session)

    assert skill.name == "o11y"
    # Sanity: instructions came from instructions.md
    assert "PromQL" in skill.instructions and "TraceQL" in skill.instructions
    # 9 helpers exposed via the skill's tools dict
    assert len(skill.tools) == 9

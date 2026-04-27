"""High-level helpers wrapping the mcp-grafana surface for the o11y skill.

Goals:
- Keep the LM-facing tool set tiny: 6 helpers (one per signal type plus the
  three dashboard primitives). The root LM never sees datasource UIDs, raw
  mcp-grafana tool names, or JSON schemas.
- Cache datasource UIDs after the first ``list_datasources`` so query helpers
  resolve them transparently.
- Return parsed Python objects (dict/list/str). Errors raise so the LM
  observes them inside the sandbox and can react.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any


def _decode(result: Any) -> Any:
    """Best-effort decode of an MCP CallToolResult into JSON-friendly Python."""
    if result is None:
        return None
    content = getattr(result, "content", None)
    if content is None:
        return result
    text_parts: list[str] = []
    for part in content:
        text = getattr(part, "text", None)
        if isinstance(text, str):
            text_parts.append(text)
    if not text_parts:
        return content
    joined = "\n".join(text_parts) if len(text_parts) > 1 else text_parts[0]
    try:
        return json.loads(joined)
    except TypeError, ValueError:
        return joined


def _parse_step(step: str | int) -> int:
    """Convert ``"30s"`` / ``"5m"`` / ``"1h"`` / ``300`` into integer seconds."""
    if isinstance(step, int):
        return step
    s = str(step).strip().lower()
    if s.endswith("ms"):
        return max(1, int(s[:-2]) // 1000)
    if s.endswith("s"):
        return int(s[:-1])
    if s.endswith("m"):
        return int(s[:-1]) * 60
    if s.endswith("h"):
        return int(s[:-1]) * 3600
    return int(s)


def _extract_datasource_records(decoded: Any) -> list[dict[str, Any]]:
    """Tolerate the several shapes mcp-grafana may return for ``list_datasources``.

    Observed in practice:
      - bare list of dicts, each with ``uid``/``type`` (the simplest case).
      - wrapper dict, e.g. ``{"datasources": [...]}`` or similar (this is what
        recent mcp-grafana versions ship in this stack).
      - dict keyed by name → record dict.
    Anything else returns ``[]`` and the caller should fall back to defaults.
    """
    if isinstance(decoded, list):
        return [d for d in decoded if isinstance(d, dict)]
    if isinstance(decoded, dict):
        for key in ("datasources", "data", "result", "items"):
            inner = decoded.get(key)
            if isinstance(inner, list):
                return [d for d in inner if isinstance(d, dict)]
        # Maybe a name→record map. Treat values that look like datasource
        # records (have ``uid`` or ``type``) as the records themselves.
        records: list[dict[str, Any]] = []
        for v in decoded.values():
            if isinstance(v, dict) and ("uid" in v or "type" in v):
                records.append(v)
        if records:
            return records
        # Last-ditch: maybe the dict IS a single record.
        if "uid" in decoded or "type" in decoded:
            return [decoded]
    return []


# Fallback uids: in this benchmark's synthetic stack the MCP datasource UIDs
# happen to equal their type names (`prometheus`, `loki`, `tempo`). When
# ``list_datasources`` returns an unrecognized shape we use these so query
# helpers stay functional rather than cascade-failing every task.
_FALLBACK_UIDS: dict[str, str] = {"prometheus": "prometheus", "loki": "loki", "tempo": "tempo"}


class _UidCache:
    """Single-flight cache of `{type: uid}` from list_datasources."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._uids: dict[str, str] | None = None
        self._lock = asyncio.Lock()

    async def all(self) -> dict[str, str]:
        if self._uids is None:
            async with self._lock:
                if self._uids is None:
                    result = await self._session.call_tool("list_datasources", {})
                    decoded = _decode(result)
                    records = _extract_datasource_records(decoded)
                    self._uids = {
                        str(ds["type"]).lower(): str(ds["uid"])
                        for ds in records
                        if "type" in ds and "uid" in ds
                    }
                    # Always merge fallbacks for the three signal datasources;
                    # don't override real uids if discovery succeeded.
                    for k, v in _FALLBACK_UIDS.items():
                        self._uids.setdefault(k, v)
        return self._uids

    async def get(self, ds_type: str) -> str:
        uids = await self.all()
        key = ds_type.lower()
        if key not in uids:
            raise RuntimeError(
                f"no {ds_type!r} datasource configured in this stack; available: {sorted(uids)}"
            )
        return uids[key]


def build_tools(session: Any) -> dict[str, Callable[..., Awaitable[Any]]]:
    """Return the LM-facing async helpers, all closed over ``session``.

    Six helpers; ``predict`` is added automatically by PredictRLM.
    """
    uids = _UidCache(session)

    async def list_datasources() -> Any:
        """List Grafana datasources in this stack.

        Returns: list of dicts with at least ``uid``, ``name``, ``type``
        (``prometheus``/``loki``/``tempo``/...). Cached internally.
        """
        return await uids.all()  # returns the cached {type: uid} dict

    async def query_metrics(
        expr: str,
        start: str | None = None,
        end: str | None = None,
        step: str | int = "30s",
    ) -> Any:
        """Run a PromQL query.

        Range query if both ``start`` and ``end`` are given (ISO-8601 strings),
        instant query otherwise. ``step`` accepts ``"30s"``/``"5m"``/``"1h"``
        or an int (seconds).
        """
        args: dict[str, Any] = {"datasourceUid": await uids.get("prometheus"), "expr": expr}
        if start and end:
            args["startTime"] = start
            args["endTime"] = end
            args["queryType"] = "range"
            args["stepSeconds"] = _parse_step(step)
        return _decode(await session.call_tool("query_prometheus", args))

    async def query_logs(
        expr: str,
        start: str,
        end: str,
        limit: int = 100,
        direction: str = "backward",
    ) -> Any:
        """Run a LogQL query against Loki.

        ``start`` and ``end`` are ISO-8601 strings. ``limit`` caps returned
        lines; ``direction`` is ``"backward"`` (default; newest first) or
        ``"forward"``.
        """
        args = {
            "datasourceUid": await uids.get("loki"),
            "logql": expr,
            "startRfc3339": start,
            "endRfc3339": end,
            "limit": limit,
            "direction": direction,
        }
        return _decode(await session.call_tool("query_loki_logs", args))

    async def query_traces(
        traceql: str,
        start: str | None = None,
        end: str | None = None,
        limit: int = 20,
    ) -> Any:
        """Run a TraceQL search query against Tempo.

        Returns matching traces. ``start``/``end`` are ISO-8601. Use ``limit``
        to cap the result set.
        """
        args: dict[str, Any] = {
            "datasourceUid": await uids.get("tempo"),
            "query": traceql,
            "limit": limit,
        }
        if start:
            args["start"] = start
        if end:
            args["end"] = end
        return _decode(await session.call_tool("tempo_traceql-search", args))

    async def get_dashboard(uid: str) -> Any:
        """Fetch the full dashboard JSON model for ``uid``."""
        return _decode(await session.call_tool("get_dashboard_by_uid", {"uid": uid}))

    async def save_dashboard(model: dict[str, Any]) -> Any:
        """Save (create or update) a full dashboard model. Pass the entire JSON,
        not a patch. Re-fetch with ``get_dashboard`` after saving and verify
        the result matches intent — saves can fail silently.
        """
        return _decode(await session.call_tool("update_dashboard", {"dashboard": model}))

    async def search_dashboards(query: str = "") -> Any:
        """Search dashboards by title/tag substring.

        Returns a list of metadata dicts with ``uid``, ``title``, ``tags``.
        """
        return _decode(await session.call_tool("search_dashboards", {"query": query}))

    return {
        "list_datasources": list_datasources,
        "query_metrics": query_metrics,
        "query_logs": query_logs,
        "query_traces": query_traces,
        "get_dashboard": get_dashboard,
        "save_dashboard": save_dashboard,
        "search_dashboards": search_dashboards,
    }

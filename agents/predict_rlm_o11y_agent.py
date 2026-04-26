"""Harbor agent that drives o11y-bench tasks through a predict-rlm runtime.

The outer LM writes Python in a sandboxed Pyodide/WASM REPL. From inside that
sandbox it composes calls to MCP tools (Prometheus, Loki, Tempo, Dashboards)
and to ``await predict(signature, ...)`` for structured extraction. The
trajectory is a sequence of REPL turns rather than raw tool-call/observation
pairs, which keeps the outer model out of context-rot territory.

The agent is *external*: it runs in the host process and talks to the Grafana
sidecar over HTTP using the externally-published MCP URL exposed by Harbor.
``setup()`` is therefore a no-op — there is nothing to install in the task
container.

Usage::

    mise run bench:job -- \\
      --model anthropic/claude-opus-4-7 \\
      --agent-import-path agents.predict_rlm_o11y_agent:PredictRLMO11yAgent \\
      --task-name query-cpu-metrics --n-concurrent 1
"""

from __future__ import annotations

import json
import os
import traceback
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .o11y_agent import select_remote_mcp_url

DEFAULT_OUTER_MODEL = "anthropic/claude-opus-4-7"
DEFAULT_SUB_MODEL = "anthropic/claude-haiku-4-5"
DEFAULT_MAX_STEPS = 30
DEFAULT_TIMEOUT_S = 600

# Working assumption against mcp-grafana — verified at runtime via list_tools().
# build_mcp_tools() synthesizes wrappers from this catalog so the outer LM sees a
# stable surface even if the server adds tools we have not vetted prompts for.
MCP_TOOL_CATALOG: tuple[dict[str, str], ...] = (
    {
        "name": "list_datasources",
        "doc": (
            "List the Grafana datasources available in this stack.\n\n"
            "Returns a list of dicts with at least ``uid``, ``name``, and ``type`` "
            "(``prometheus``/``loki``/``tempo``)."
        ),
    },
    {
        "name": "query_prometheus",
        "doc": (
            "Run a PromQL query against the Prometheus datasource.\n\n"
            "Args:\n"
            "    expr: PromQL expression. Use rate()/increase() for counters and\n"
            "        histogram_quantile(sum by (le) (rate(_bucket[range]))) for percentiles.\n"
            "    start, end: ISO-8601 strings. When supplied, runs a range query.\n"
            "        Omit both for an instant query at scenario-now.\n"
            "    step: range step (default '30s').\n"
            "Returns: the raw Prometheus result body."
        ),
    },
    {
        "name": "query_loki",
        "doc": (
            "Run a LogQL query against the Loki datasource.\n\n"
            "Args:\n"
            "    expr: LogQL expression. Put the smallest viable label matcher first,\n"
            "        line filters second, parsers last.\n"
            "    start, end: ISO-8601 strings.\n"
            "    limit: max log lines (default 100)."
        ),
    },
    {
        "name": "query_tempo",
        "doc": (
            "Run a TraceQL query against the Tempo datasource.\n\n"
            "Filter by ``resource.service.name`` and ``span.status`` before duration;\n"
            "use ``| select(...)`` to project the fields you actually need."
        ),
    },
    {
        "name": "get_dashboard",
        "doc": "Fetch the full dashboard JSON for ``uid``.",
    },
    {
        "name": "update_dashboard",
        "doc": (
            "Save a full dashboard model. Pass the entire panel JSON, not a patch.\n"
            "Always re-fetch with ``get_dashboard`` after saving and verify that\n"
            "the saved expression and variable bindings match intent."
        ),
    },
    {
        "name": "search_dashboards",
        "doc": "Search dashboards by title/tag substring; returns a list of metadata dicts.",
    },
)


# ----- DSPy signature ------------------------------------------------------

_SIGNATURE_CACHE: Any = None
_SKILLS_CACHE: list[Any] | None = None


def _get_signature() -> Any:
    global _SIGNATURE_CACHE
    if _SIGNATURE_CACHE is None:
        _SIGNATURE_CACHE = _build_signature()
    return _SIGNATURE_CACHE


def _get_skills() -> list[Any]:
    global _SKILLS_CACHE
    if _SKILLS_CACHE is None:
        _SKILLS_CACHE = _build_skills()
    return _SKILLS_CACHE


def _build_signature() -> Any:
    """Build the O11ySignature DSPy class.

    Built lazily so importing this module does not require dspy at import time
    (the unit test runs without dspy installed in some environments)."""
    import dspy

    class O11ySignature(dspy.Signature):
        """Solve a Grafana observability task by composing MCP tool calls in Python.

        Strategy:
        1. Survey datasources once via ``await list_datasources()``. Do not
           enumerate every metric or log stream — that path leads to context rot.
        2. Identify the single signal that answers the question before issuing
           queries. Resist the urge to gather breadth.
        3. PromQL idioms:
           - ``rate(metric_total[5m])`` / ``increase(metric_total[1h])`` for counters.
           - Range queries (``start``/``end``/``step``) for "over the last X" questions.
           - ``topk(N, expr)`` for ranking.
           - ``histogram_quantile(0.95, sum by (le) (rate(metric_bucket[5m])))`` for
             percentiles.
        4. LogQL idioms:
           - Smallest viable label matcher first, then line filter, then parser.
           - Peek at a few raw lines before parsing so you know the shape.
           - Use ``rate({selector} | json | label="x" [5m])`` for log-derived rates.
        5. TraceQL idioms:
           - Filter by ``resource.service.name`` and ``span.status`` before
             ``duration`` to keep the candidate set small.
           - Project explicitly with ``| select(...)`` so you do not waste tokens
             on fields you will not use.
        6. When comparing N candidates (services, endpoints, error classes), fan out
           with ``asyncio.gather`` over a single parameterized query — never N
           sequential REPL turns.
        7. Numeric claims must come from query results, not LM intuition. The grader
           re-runs the reference queries; mismatches lose points.
        8. Dashboard edits round-trip the whole panel JSON: ``get_dashboard``,
           mutate, ``update_dashboard``, then ``get_dashboard`` again and assert the
           saved expression and variable bindings match intent.
        9. Submit a concise final answer in ``answer`` — no preamble, hedging,
           or restated reasoning. The grader scores the literal text.
        """

        instruction: str = dspy.InputField(desc="The o11y-bench task prompt to solve.")
        answer: str = dspy.OutputField(desc="Concise final answer for the grader.")

    return O11ySignature


# ----- Skills --------------------------------------------------------------


def _build_skills() -> list[Any]:
    """Build the four reusable skills. Lazy for the same reason as the signature."""
    from predict_rlm import Skill

    promql = Skill(
        name="promql",
        instructions=(
            "PromQL idioms:\n"
            "- Counters end in `_total`. Always wrap them in `rate()` (per-second over a "
            "  range) or `increase()` (raw delta over a range). A bare counter value is "
            "  cumulative since process start and almost never the right answer.\n"
            "- Range vector vs instant: `metric` is instant, `metric[5m]` is a range vector. "
            "  Range vectors can only feed range-aware functions (`rate`, `increase`, "
            "  `avg_over_time`, ...).\n"
            "- For 'over the last X' phrasing, use a range query (start/end/step) and pick "
            "  the latest sample, not an instant query against `now`.\n"
            "- `topk(N, expr)` for ranking, `bottomk(N, expr)` for the inverse. Combine with "
            "  `sum by (label)(...)` to aggregate before ranking.\n"
            "- Percentiles: "
            "`histogram_quantile(0.95, sum by (le, ...) (rate(metric_bucket[5m])))`. "
            "  Always aggregate by `le` plus the labels you want to keep.\n"
            "- `label_replace` cannot create labels that did not exist; it copies/edits. "
            "  When joining series with `*`/`/`, use `on()`/`ignoring()`/`group_left`."
        ),
    )

    logql = Skill(
        name="logql",
        instructions=(
            "LogQL idioms:\n"
            "- Always start with the smallest viable label matcher: "
            '  `{job="x", level="error"}` first, line filter `|= "foo"` second, '
            "  parser `| json` / `| logfmt` last. Cardinality matters.\n"
            "- Peek at a few raw lines before parsing (`limit=5`, no parser) so you know\n"
            "  the actual log shape.\n"
            "- Aggregations: `sum by (label) (count_over_time({...}[5m]))` for counts,\n"
            "  `sum by (label) (rate({...}[5m]))` for per-second rates.\n"
            "- For numeric extraction, `| json | unwrap field | rate(...)` works on numeric "
            "  fields; `| logfmt` for key=value logs.\n"
            "- `limit` only caps returned lines, not the underlying scan; keep ranges tight."
        ),
    )

    traceql = Skill(
        name="traceql",
        instructions=(
            "TraceQL idioms:\n"
            '- Filter on `resource.service.name="x"` and `span.status=error` before any\n'
            "  duration filter. Tempo can short-circuit on these.\n"
            '- Spec phrasing: `{ resource.service.name = "checkout" && span.status = error }`.\n'
            "- Project explicitly with `| select(span.name, span.http.status_code, "
            "  resource.service.name, duration)` — you almost never want the full span.\n"
            '- For latency: `{ resource.service.name = "x" } | select(duration) | '
            "  histogram_over_time(duration)` and inspect the shape, do not eyeball samples.\n"
            '- A `trace_id` from one query can be reused in `{ trace:id = "..." }` to pull\n'
            "  the full trace once you know which one matters."
        ),
    )

    dashboard = Skill(
        name="dashboard",
        instructions=(
            "Dashboard editing:\n"
            "- Round-trip the whole panel JSON. `get_dashboard(uid)` → mutate → "
            "  `update_dashboard(dashboard=...)`. Never construct a partial.\n"
            "- After saving, ALWAYS re-fetch with `get_dashboard(uid)` and assert that\n"
            "  the saved expression / panel title / unit / variable binding actually\n"
            "  match what you intended. Saves can fail silently if the schema is off.\n"
            "- Variables: when a panel uses a Grafana template variable like `$env`,\n"
            "  the variable definition lives at `dashboard.templating.list[*]`. Edits\n"
            "  to a panel's `targets[*].expr` must reference variables that actually\n"
            "  exist in that list.\n"
            "- Dashboard uids and folder uids share a namespace; do not collide them."
        ),
    )

    return [promql, logql, traceql, dashboard]


# ----- MCP tool wrapping ---------------------------------------------------


def _make_tool_wrapper(session: Any, tool_name: str, doc: str) -> Callable[..., Awaitable[Any]]:
    """Build a single async tool callable that proxies into the MCP session.

    The returned callable accepts ``**kwargs`` and forwards them as the tool's
    ``arguments`` payload. Result content is decoded into Python primitives
    when possible so the outer LM can index into them naturally.
    """

    async def _tool(**kwargs: Any) -> Any:
        result = await session.call_tool(tool_name, kwargs)
        return _decode_tool_result(result)

    _tool.__name__ = tool_name
    _tool.__qualname__ = tool_name
    _tool.__doc__ = doc
    return _tool


def _decode_tool_result(result: Any) -> Any:
    """Best-effort decode of an MCP CallToolResult into JSON-friendly Python.

    Works with the official MCP SDK's ``CallToolResult`` (``result.content`` is
    a list of content parts, each with a ``text`` attribute) and with simpler
    duck-typed mocks that just return a dict/list/str directly.
    """
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


def build_mcp_tools(session: Any) -> list[Callable[..., Awaitable[Any]]]:
    """Wrap each Grafana MCP tool from ``MCP_TOOL_CATALOG`` as an awaitable callable.

    Used as the default surface and as the test fixture. For the live agent run,
    ``discover_mcp_tools`` overlays the server's actual tool list on this catalog.
    """
    return [_make_tool_wrapper(session, entry["name"], entry["doc"]) for entry in MCP_TOOL_CATALOG]


async def discover_mcp_tools(session: Any) -> list[Callable[..., Awaitable[Any]]]:
    """Build wrappers from the server's ``list_tools()`` response.

    Tools whose names appear in ``MCP_TOOL_CATALOG`` keep our hand-tuned docstring;
    others fall through to the server-supplied description. This way the outer LM
    always sees the actual tool surface even if mcp-grafana renames or adds tools.
    """
    catalog_docs = {entry["name"]: entry["doc"] for entry in MCP_TOOL_CATALOG}
    response = await session.list_tools()
    server_tools = getattr(response, "tools", []) or []
    if not server_tools:
        return build_mcp_tools(session)
    wrappers: list[Callable[..., Awaitable[Any]]] = []
    for tool in server_tools:
        name = getattr(tool, "name", None)
        if not name:
            continue
        doc = catalog_docs.get(name) or getattr(tool, "description", None) or name
        wrappers.append(_make_tool_wrapper(session, name, doc))
    return wrappers


# ----- Trajectory conversion ----------------------------------------------


_OBSERVATION_CONTENT_LIMIT = 10_000


def _atif_step(
    step_id: int,
    source: str,
    message: str,
    *,
    reasoning_content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    observation: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "step_id": step_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "source": source,
        "message": message,
    }
    if reasoning_content:
        step["reasoning_content"] = reasoning_content
    if tool_calls:
        step["tool_calls"] = tool_calls
    if observation:
        step["observation"] = observation
    if metrics:
        step["metrics"] = metrics
    return step


def _serialize_call_content(value: Any, error: Any) -> str:
    """Render one tool/predict call's payload for the trajectory.

    Truncated so a single noisy dashboard JSON cannot blow up trajectory.json.
    """
    if error is not None:
        body = f"[error] {error}"
    else:
        body = json.dumps(value, default=str)
    if len(body) > _OBSERVATION_CONTENT_LIMIT:
        return (
            body[:_OBSERVATION_CONTENT_LIMIT]
            + f"... [truncated {len(body) - _OBSERVATION_CONTENT_LIMIT} chars]"
        )
    return body


def _record_call(
    *,
    call_id: str,
    function_name: str,
    arguments: dict[str, Any],
    content: str,
    atif_tool_calls: list[dict[str, Any]],
    observation_results: list[dict[str, Any]],
) -> None:
    atif_tool_calls.append(
        {
            "tool_call_id": call_id,
            "function_name": function_name,
            "arguments": arguments,
        }
    )
    observation_results.append({"source_call_id": call_id, "content": content})


def _to_atif(
    *,
    instruction: str,
    answer: str,
    trace: Any,
    model_name: str,
    sub_model_name: str,
) -> dict[str, Any]:
    """Convert a predict-rlm RunTrace into an ATIF-v1.6 trajectory document.

    Each REPL iteration becomes one agent action (the generated Python) plus an
    observation entry carrying the truncated REPL output and any granular tool
    calls executed inside the iteration. predict() subcalls are surfaced as
    additional tool_calls so the interpretability story survives in ATIF.
    """
    steps: list[dict[str, Any]] = [
        _atif_step(1, "system", "predict-rlm o11y-bench agent (see O11ySignature docstring)"),
        _atif_step(2, "user", instruction),
    ]
    next_id = 2
    total_tool_calls = 0
    main_in = main_out = sub_in = sub_out = 0
    cost = 0.0

    iteration_steps = getattr(trace, "steps", None) or []
    for step in iteration_steps:
        next_id += 1
        atif_tool_calls: list[dict[str, Any]] = []
        observation_results: list[dict[str, Any]] = []

        for tc in getattr(step, "tool_calls", []) or []:
            arguments = {**(getattr(tc, "kwargs", {}) or {})}
            args_pos = list(getattr(tc, "args", []) or [])
            if args_pos:
                arguments["_args"] = args_pos
            _record_call(
                call_id=f"tc-{next_id}-{len(atif_tool_calls)}",
                function_name=getattr(tc, "name", "tool"),
                arguments=arguments,
                content=_serialize_call_content(
                    getattr(tc, "result", None), getattr(tc, "error", None)
                ),
                atif_tool_calls=atif_tool_calls,
                observation_results=observation_results,
            )
            total_tool_calls += 1

        for group in getattr(step, "predict_calls", []) or []:
            sig = getattr(group, "signature", "")
            for call in getattr(group, "calls", []) or []:
                _record_call(
                    call_id=f"pc-{next_id}-{len(atif_tool_calls)}",
                    function_name="predict",
                    arguments={
                        "signature": sig,
                        "instructions": getattr(group, "instructions", None),
                        "input": getattr(call, "input", {}) or {},
                    },
                    content=_serialize_call_content(
                        getattr(call, "output", {}), getattr(call, "error", None)
                    ),
                    atif_tool_calls=atif_tool_calls,
                    observation_results=observation_results,
                )
                total_tool_calls += 1

        observation_results.append(
            {
                "source_call_id": f"repl-{next_id}",
                "content": getattr(step, "output", "") or "",
            }
        )

        steps.append(
            _atif_step(
                next_id,
                "agent",
                getattr(step, "code", "") or "(no code)",
                reasoning_content=getattr(step, "reasoning", None) or None,
                tool_calls=atif_tool_calls or None,
                observation={"results": observation_results},
                metrics={
                    "iteration": getattr(step, "iteration", None),
                    "duration_ms": getattr(step, "duration_ms", None),
                    "error": getattr(step, "error", False),
                },
            )
        )

    next_id += 1
    steps.append(
        _atif_step(
            next_id,
            "agent",
            answer,
            metrics={"final": True},
        )
    )

    usage = getattr(trace, "usage", None)
    if usage is not None:
        main = getattr(usage, "main", None)
        sub = getattr(usage, "sub", None)
        if main is not None:
            main_in = getattr(main, "input_tokens", 0) or 0
            main_out = getattr(main, "output_tokens", 0) or 0
            cost += float(getattr(main, "cost", 0.0) or 0.0)
        if sub is not None:
            sub_in = getattr(sub, "input_tokens", 0) or 0
            sub_out = getattr(sub, "output_tokens", 0) or 0
            cost += float(getattr(sub, "cost", 0.0) or 0.0)

    return {
        "schema_version": "ATIF-v1.6",
        "session_id": str(uuid.uuid4()),
        "agent": {
            "name": PredictRLMO11yAgent.name(),
            "version": _installed_version(),
            "model_name": model_name,
            "sub_model_name": sub_model_name,
            "tool_definitions": [
                {"name": entry["name"], "description": entry["doc"].splitlines()[0]}
                for entry in MCP_TOOL_CATALOG
            ],
        },
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": main_in + sub_in,
            "total_completion_tokens": main_out + sub_out,
            "total_cached_tokens": 0,
            "total_cost_usd": cost,
            "total_steps": len(steps),
            "total_tool_calls": total_tool_calls,
            "total_iterations": getattr(trace, "iterations", len(iteration_steps)),
            "status": getattr(trace, "status", "unknown"),
            "duration_ms": getattr(trace, "duration_ms", None),
        },
    }


def _installed_version() -> str:
    try:
        return _pkg_version("predict-rlm")
    except PackageNotFoundError:
        return "unknown"


# ----- Harbor agent --------------------------------------------------------


def _resolve_outer_model(model_name: str | None) -> str:
    explicit = os.environ.get("O11Y_RLM_OUTER")
    if explicit:
        return explicit
    return model_name or DEFAULT_OUTER_MODEL


def _resolve_sub_model() -> str:
    return os.environ.get("O11Y_RLM_SUB") or DEFAULT_SUB_MODEL


def _resolve_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class PredictRLMO11yAgent(BaseAgent):
    """o11y-bench agent that drives tasks through a predict-rlm runtime.

    See module docstring for the full integration story. ``run()`` performs the
    LM loop in the host process — the Grafana sidecar is reached via the
    externally-published MCP URL from ``self.mcp_servers``.
    """

    SUPPORTS_ATIF = True

    @staticmethod
    def name() -> str:
        return "predict-rlm-o11y"

    def version(self) -> str:
        return _installed_version()

    async def setup(self, environment: BaseEnvironment) -> None:
        # External agent — no in-container install. The MCP sidecar is
        # provisioned by Harbor before run() is called.
        return None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        outer_model = _resolve_outer_model(self.model_name)
        sub_model = _resolve_sub_model()
        max_steps = _resolve_int_env("O11Y_RLM_MAX_STEPS", DEFAULT_MAX_STEPS)
        timeout_s = _resolve_int_env("O11Y_RLM_TIMEOUT_S", DEFAULT_TIMEOUT_S)

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "instruction.txt").write_text(instruction)

        mcp_url = select_remote_mcp_url(self.mcp_servers)

        context.metadata = {
            **(context.metadata or {}),
            "agent": self.name(),
            "outer_model": outer_model,
            "sub_model": sub_model,
            "max_steps": max_steps,
            "timeout_s": timeout_s,
            "mcp_url": mcp_url,
        }

        if mcp_url is None:
            self._write_failure(
                instruction=instruction,
                outer_model=outer_model,
                sub_model=sub_model,
                error=(
                    "could not resolve a host-reachable Grafana MCP URL from "
                    f"mcp_servers={self.mcp_servers!r}"
                ),
            )
            return

        try:
            answer, trace = await _drive_predict_rlm(
                instruction=instruction,
                mcp_url=mcp_url,
                outer_model=outer_model,
                sub_model=sub_model,
                max_steps=max_steps,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            self.logger.exception("predict-rlm run failed")
            partial_trace = getattr(exc, "trace", None)
            self._write_failure(
                instruction=instruction,
                outer_model=outer_model,
                sub_model=sub_model,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                trace=partial_trace,
            )
            return

        trajectory = _to_atif(
            instruction=instruction,
            answer=answer,
            trace=trace,
            model_name=outer_model,
            sub_model_name=sub_model,
        )
        self._write_trajectory(trajectory)
        self._populate_context(context, trajectory, answer)

    # -- helpers ----------------------------------------------------------

    def _write_trajectory(self, trajectory: dict[str, Any]) -> None:
        path = self.logs_dir / "trajectory.json"
        path.write_text(json.dumps(trajectory, indent=2, default=str))

    def _write_failure(
        self,
        *,
        instruction: str,
        outer_model: str,
        sub_model: str,
        error: str,
        trace: Any | None = None,
    ) -> None:
        answer = f"[agent error] {error.splitlines()[0]}"
        trajectory = _to_atif(
            instruction=instruction,
            answer=answer,
            trace=trace,
            model_name=outer_model,
            sub_model_name=sub_model,
        )
        trajectory["final_metrics"]["status"] = "error"
        trajectory["final_metrics"]["error"] = error
        self._write_trajectory(trajectory)

    def _populate_context(
        self, context: AgentContext, trajectory: dict[str, Any], answer: str
    ) -> None:
        fm = trajectory.get("final_metrics") or {}
        context.n_input_tokens = fm.get("total_prompt_tokens")
        context.n_output_tokens = fm.get("total_completion_tokens")
        context.n_cache_tokens = fm.get("total_cached_tokens")
        context.cost_usd = fm.get("total_cost_usd")
        context.metadata = {
            **(context.metadata or {}),
            "final_answer": answer,
            "status": fm.get("status"),
            "iterations": fm.get("total_iterations"),
        }


async def _drive_predict_rlm(
    *,
    instruction: str,
    mcp_url: str,
    outer_model: str,
    sub_model: str,
    max_steps: int,
    timeout_s: int,
) -> tuple[str, Any]:
    """Open the MCP session, run PredictRLM end-to-end, return (answer, trace)."""
    import asyncio

    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from predict_rlm import PredictRLM

    signature_cls = _get_signature()
    skills = _get_skills()

    async with streamable_http_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await discover_mcp_tools(session)

            rlm = PredictRLM(
                signature_cls,
                lm=outer_model,
                sub_lm=sub_model,
                max_iterations=max_steps,
                tools=tools,
                skills=skills,
            )

            prediction = await asyncio.wait_for(
                rlm.aforward(instruction=instruction),
                timeout=timeout_s,
            )
            answer = (getattr(prediction, "answer", None) or "").strip()
            trace = getattr(prediction, "trace", None)
            return answer, trace

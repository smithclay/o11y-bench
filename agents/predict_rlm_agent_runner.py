# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "mcp>=1.9.0",
#   "predict-rlm>=0.3.0",
#   "dspy>=2.5.0",
# ]
# ///
"""predict-rlm runner for o11y-bench.

Runs inside the Harbor task container. Connects to the in-network mcp-grafana
sidecar over streamable-http and drives a predict-rlm RLM that writes Python in
a Pyodide/WASM sandbox and composes MCP tool calls + ``predict()`` subcalls.
Writes an ATIF-v1.6 trajectory to ``/logs/agent/trajectory.json`` and prints
the final answer.

Config via env vars:
- ``MODEL`` — outer LM (e.g. ``anthropic/claude-opus-4-7``)
- ``SUB_MODEL`` — structured-extraction sub-LM (default
  ``anthropic/claude-haiku-4-5``)
- ``MAX_STEPS`` — REPL iteration cap (default 50)
- ``TIMEOUT_S`` — wall-clock cap for the RLM call (default 600)
- ``MCP_URL`` — Grafana MCP endpoint (default ``http://$STACK_HOST:8080/mcp``)
- ``O11Y_SCENARIO_TIME_ISO`` — synthetic scenario clock; surfaced to the LM as
  the ``Current time`` in a ``<context>`` block prepended to the instruction.

The ``deno`` binary required by predict-rlm's Pyodide sandbox is provided by
the ``deno`` PyPI wheel (a transitive dep of ``predict-rlm``); ``uv run`` puts
it on PATH automatically so ``JspiInterpreter`` auto-discovers it.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

DEFAULT_SUB_MODEL = "anthropic/claude-haiku-4-5"
DEFAULT_MAX_STEPS = 50
DEFAULT_TIMEOUT_S = 600


def build_signature() -> Any:
    import dspy

    class O11ySignature(dspy.Signature):
        """Solve a Grafana observability task by composing MCP tool calls in Python.

        Strategy:
        1. Survey datasources once via ``await list_datasources()``. Cache the
           ``uid`` per type — every query tool requires ``datasource_uid``.
           Do not enumerate every metric or log stream — that path leads to
           context rot.
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


def build_skills() -> list[Any]:
    from predict_rlm import Skill

    mcp = Skill(
        name="mcp",
        instructions=(
            "mcp-grafana tool conventions:\n"
            "- ALWAYS call `list_datasources()` first and remember the `uid` for each\n"
            "  datasource type (`prometheus`, `loki`, `tempo`). Every query tool\n"
            "  requires a `datasource_uid` argument; calls without it return HTTP 400.\n"
            "- The arg names and required fields shown in each tool's docstring (under\n"
            "  `Input schema:`) are authoritative — do not guess. Pass everything as\n"
            "  keyword args (e.g. `await query_prometheus(datasource_uid='prometheus', "
            "expr='up')`).\n"
            "- The `<context>` block in the user message gives the synthetic `Current "
            "time:` —\n"
            "  derive explicit `start`/`end` ISO-8601 bounds from it for time-windowed\n"
            "  queries instead of relying on the wall clock."
        ),
    )

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

    return [mcp, promql, logql, traceql, dashboard]


def _decode_tool_result(result: Any) -> Any:
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


def _format_tool_doc(description: str | None, input_schema: Any) -> str:
    """Compose a tool docstring the LM can read: NL description + JSON schema.

    The schema (including ``datasource_uid``, required fields, types, enums) is
    the source of truth; predict-rlm's outer LM reads the docstring as plain
    text when reasoning about which kwargs to pass.
    """
    base = (description or "").strip()
    if isinstance(input_schema, dict) and input_schema:
        schema_text = json.dumps(input_schema, indent=2, sort_keys=True)
        return f"{base}\n\nInput schema:\n```json\n{schema_text}\n```".strip()
    return base or "(no description)"


def _safe_identifier(name: str) -> str:
    """Map an MCP tool name to a valid Python identifier (predict-rlm requires
    sandbox tools to be valid identifiers). Non-identifier chars become ``_``;
    a leading digit is prefixed with ``t_``."""
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if safe and safe[0].isdigit():
        safe = f"t_{safe}"
    return safe or "tool"


def _make_tool_wrapper(
    session: Any, wire_name: str, py_name: str, doc: str
) -> Callable[..., Awaitable[Any]]:
    async def _tool(**kwargs: Any) -> Any:
        result = await session.call_tool(wire_name, kwargs)
        return _decode_tool_result(result)

    _tool.__name__ = py_name
    _tool.__qualname__ = py_name
    _tool.__doc__ = doc
    return _tool


async def discover_mcp_tools(
    session: Any,
) -> tuple[list[Callable[..., Awaitable[Any]]], list[dict[str, Any]]]:
    """Discover the live mcp-grafana tool surface and build async wrappers.

    Returns ``(callables, tool_definitions)`` where ``tool_definitions`` is the
    flat name/description list embedded in the ATIF trajectory's
    ``agent.tool_definitions``. Mirrors the pattern in ``agent_runner.py``
    (the default agent's litellm path) but produces Python callables for the
    predict-rlm sandbox instead of OpenAI function-call specs. MCP names that
    are not valid Python identifiers (e.g. ``tempo_docs-traceql``) get
    rewritten for the LM-facing surface; the wire name is preserved for
    ``session.call_tool``.
    """
    result = await session.list_tools()
    callables: list[Callable[..., Awaitable[Any]]] = []
    definitions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tool in result.tools:
        wire_name = tool.name
        py_name = _safe_identifier(wire_name)
        if py_name in seen:
            i = 2
            while f"{py_name}_{i}" in seen:
                i += 1
            py_name = f"{py_name}_{i}"
        seen.add(py_name)
        doc = _format_tool_doc(
            getattr(tool, "description", None), getattr(tool, "inputSchema", None)
        )
        if py_name != wire_name:
            doc = f"(MCP wire name: ``{wire_name}``)\n\n{doc}"
        callables.append(_make_tool_wrapper(session, wire_name, py_name, doc))
        definitions.append(
            {
                "name": wire_name,
                "description": (getattr(tool, "description", None) or "").splitlines()[0]
                if getattr(tool, "description", None)
                else "",
            }
        )
    return callables, definitions


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


def to_atif(
    *,
    instruction: str,
    answer: str,
    trace: Any,
    model_name: str,
    sub_model_name: str,
    tool_definitions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
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
            call_id = f"tc-{next_id}-{len(atif_tool_calls)}"
            arguments = {**(getattr(tc, "kwargs", {}) or {})}
            args_pos = list(getattr(tc, "args", []) or [])
            if args_pos:
                arguments["_args"] = args_pos
            atif_tool_calls.append(
                {
                    "tool_call_id": call_id,
                    "function_name": getattr(tc, "name", "tool"),
                    "arguments": arguments,
                }
            )
            observation_results.append(
                {
                    "source_call_id": call_id,
                    "content": json.dumps(getattr(tc, "result", None), default=str)
                    if getattr(tc, "error", None) is None
                    else f"[error] {tc.error}",
                }
            )
            total_tool_calls += 1

        for group in getattr(step, "predict_calls", []) or []:
            sig = getattr(group, "signature", "")
            for call in getattr(group, "calls", []) or []:
                call_id = f"pc-{next_id}-{len(atif_tool_calls)}"
                atif_tool_calls.append(
                    {
                        "tool_call_id": call_id,
                        "function_name": "predict",
                        "arguments": {
                            "signature": sig,
                            "instructions": getattr(group, "instructions", None),
                            "input": getattr(call, "input", {}) or {},
                        },
                    }
                )
                observation_results.append(
                    {
                        "source_call_id": call_id,
                        "content": json.dumps(getattr(call, "output", {}), default=str)
                        if getattr(call, "error", None) is None
                        else f"[error] {call.error}",
                    }
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
    steps.append(_atif_step(next_id, "agent", answer, metrics={"final": True}))

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

    try:
        agent_version = _pkg_version("predict-rlm")
    except PackageNotFoundError:
        agent_version = "unknown"

    return {
        "schema_version": "ATIF-v1.6",
        "session_id": str(uuid.uuid4()),
        "agent": {
            "name": "predict-rlm-o11y",
            "version": agent_version,
            "model_name": model_name,
            "sub_model_name": sub_model_name,
            "tool_definitions": tool_definitions or [],
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


async def drive(
    *,
    prompt: str,
    mcp_url: str,
    outer_model: str,
    sub_model: str,
    max_steps: int,
    timeout_s: int,
) -> tuple[str, Any, list[dict[str, Any]]]:
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    from predict_rlm import PredictRLM
    from predict_rlm.interpreter import JspiInterpreter

    signature_cls = build_signature()
    skills = build_skills()

    async with streamablehttp_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools, tool_definitions = await discover_mcp_tools(session)

            # Do NOT pass deno_command — JspiInterpreter only auto-adds the
            # required JSPI flags when it builds the deno command itself.
            # Without those flags the sandbox→host async tool bridge silently
            # deadlocks on the first tool call. uv puts the deno binary on
            # PATH for `uv run` execution, so auto-discovery works.
            interpreter = JspiInterpreter(preinstall_packages=False)
            rlm = PredictRLM(
                signature_cls,
                lm=outer_model,
                sub_lm=sub_model,
                max_iterations=max_steps,
                tools=tools,
                skills=skills,
                interpreter=interpreter,
            )
            prediction = await asyncio.wait_for(
                rlm.aforward(instruction=prompt),
                timeout=timeout_s,
            )
            answer = (getattr(prediction, "answer", None) or "").strip()
            trace = getattr(prediction, "trace", None)
            return answer, trace, tool_definitions


def _resolve_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _build_prompt(instruction: str) -> str:
    """Mirror ``agents/task_prompt.txt``: prepend a ``<context>`` block carrying
    the synthetic scenario clock so time-bounded queries derive their bounds
    from the simulated stack's clock, not the wall clock."""
    env_ts = os.environ.get("O11Y_SCENARIO_TIME_ISO", "").strip()
    if not env_ts:
        return instruction
    return f"<context>\nCurrent time: {env_ts}\n</context>\n\n{instruction}"


async def run() -> int:
    instruction = Path("/app/instruction.txt").read_text().strip()
    outer_model = os.environ["MODEL"]
    sub_model = os.environ.get("SUB_MODEL", DEFAULT_SUB_MODEL)
    max_steps = _resolve_int_env("MAX_STEPS", DEFAULT_MAX_STEPS)
    timeout_s = _resolve_int_env("TIMEOUT_S", DEFAULT_TIMEOUT_S)
    stack_host = os.environ.get("STACK_HOST", "127.0.0.1")
    mcp_url = os.environ.get("MCP_URL", f"http://{stack_host}:8080/mcp")

    agent_dir = Path("/logs/agent")
    agent_dir.mkdir(parents=True, exist_ok=True)

    prompt = _build_prompt(instruction)

    exit_code = 0
    trace: Any = None
    tool_definitions: list[dict[str, Any]] = []
    try:
        answer, trace, tool_definitions = await drive(
            prompt=prompt,
            mcp_url=mcp_url,
            outer_model=outer_model,
            sub_model=sub_model,
            max_steps=max_steps,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        traceback.print_exc()
        answer = f"[agent error] {type(exc).__name__}: {exc}"
        trace = getattr(exc, "trace", None)
        exit_code = 1

    trajectory = to_atif(
        instruction=prompt,
        answer=answer,
        trace=trace,
        model_name=outer_model,
        sub_model_name=sub_model,
        tool_definitions=tool_definitions,
    )
    if exit_code != 0:
        trajectory["final_metrics"]["status"] = "error"
    (agent_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2, default=str))
    print(answer)
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))

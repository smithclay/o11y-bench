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
sidecar over streamable-http and drives a predict-rlm RLM whose entire domain
behavior comes from one skill artifact (``o11y_skill``). Writes an ATIF-v1.6
trajectory to ``/logs/agent/trajectory.json`` and prints the final answer.

Config via env vars:
- ``MODEL`` — outer LM (e.g. ``anthropic/claude-opus-4-7``)
- ``SUB_MODEL`` — structured-extraction sub-LM (default ``anthropic/claude-haiku-4-5``)
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
        """Solve a Grafana observability task by composing tool calls in Python.

        Domain rules and tool conventions live in the ``o11y`` skill — read
        them and follow them exactly. Submit a concise final answer in
        ``answer``.
        """

        instruction: str = dspy.InputField(desc="The o11y-bench task prompt to solve.")
        answer: str = dspy.OutputField(desc="Concise final answer for the grader.")

    return O11ySignature


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
        _atif_step(1, "system", "predict-rlm o11y-bench agent (skill: o11y)"),
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
    from o11y_skill import build_o11y_skill
    from predict_rlm import PredictRLM
    from predict_rlm.interpreter import JspiInterpreter

    signature_cls = build_signature()

    async with streamablehttp_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            skill = build_o11y_skill(session)

            tool_definitions = [
                {
                    "name": name,
                    "description": (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else "",
                }
                for name, fn in skill.tools.items()
            ]

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
                skills=[skill],
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

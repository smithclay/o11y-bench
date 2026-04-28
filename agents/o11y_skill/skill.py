"""o11y skill — Grafana observability via mcp-grafana.

Single skill artifact for the predict-rlm o11y-bench agent. Mirrors the shape
of ``predict_rlm/skills/spreadsheet/skill.py``: prose handbook + a fixed set of
high-level helpers, bundled as one ``Skill`` instance. Keeping the prose in a
sibling ``instructions.md`` makes the file the natural target for a future
RLM♥GEPA optimization loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from predict_rlm import Skill

from .tools import build_tools

INSTRUCTIONS_PATH = Path(__file__).parent / "instructions.md"
O11Y_INSTRUCTIONS = INSTRUCTIONS_PATH.read_text()


def build_o11y_skill(session: Any) -> Skill:
    """Construct the o11y skill bound to a live MCP session.

    The session is needed because the helpers are async closures over it
    (and over an internal datasource-uid cache). Stateless skills like
    ``predict_rlm.skills.spreadsheet`` can be module-level instances; this
    one cannot.
    """
    return Skill(
        name="o11y",
        instructions=O11Y_INSTRUCTIONS,
        tools=build_tools(session),
    )

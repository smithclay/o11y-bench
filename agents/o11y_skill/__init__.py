"""o11y skill: bundles operating instructions + 7 high-level helpers for
solving Grafana observability tasks via mcp-grafana."""

from .skill import O11Y_INSTRUCTIONS, build_o11y_skill
from .tools import build_tools

__all__ = ["O11Y_INSTRUCTIONS", "build_o11y_skill", "build_tools"]

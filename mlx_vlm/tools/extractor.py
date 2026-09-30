"""Extract structured tool calls from a model's text output.

Format-agnostic: given the parser module selected for the model (see
:mod:`mlx_vlm.tools.registry`), it slices the text between the parser's markers,
delegates per-format decoding to ``parse_tool_call``, and emits OpenAI-shaped
calls.
"""

import json
import logging
import re
import uuid
from typing import Optional

from .base import ToolParser
from .types import ParseResult

logger = logging.getLogger("mlx_vlm.server")


def _shaped_calls(parsed, start_index: int) -> list:
    """Render one parser result as OpenAI-shaped calls numbered from ``start_index``."""
    parsed_calls = parsed if isinstance(parsed, list) else [parsed]
    shaped = []
    for offset, tool_call in enumerate(parsed_calls):
        args = tool_call["arguments"]
        shaped.append(
            {
                "type": "function",
                "index": start_index + offset,
                "id": str(uuid.uuid4()),
                "function": {
                    "name": tool_call["name"].strip(),
                    "arguments": (
                        args
                        if isinstance(args, str)
                        else json.dumps(args, ensure_ascii=False)
                    ),
                },
            },
        )
    return shaped


def parse_unterminated_call(
    text: str, tool_module: ToolParser, tools, start_index: int = 0
) -> Optional[list]:
    """Recover a call the model left open when generation ended, else ``None``."""
    # A model can emit a complete call and stop before its end marker, leaving the
    # span unmatched. Returning None keeps text that only looks like a call content.
    if not tool_module.tool_call_end:
        # No end marker means the call closes at a newline, so none stays open.
        return None
    marker_at = text.find(tool_module.tool_call_start)
    if marker_at < 0:
        return None
    body = text[marker_at + len(tool_module.tool_call_start) :]
    if tool_module.tool_call_end in body:
        return None
    try:
        return _shaped_calls(
            tool_module.parse_tool_call(body.strip(), tools), start_index
        )
    except Exception as exc:
        logger.debug("Unterminated text is not a tool call %r: %s", body, exc)
        return None


def process_tool_calls(
    model_output: str, tool_module: ToolParser, tools
) -> ParseResult:
    """Parse tool calls from model output using the given tool parser module."""
    called_tools = []
    remaining = model_output

    if tool_module.tool_call_start in model_output:
        if tool_module.tool_call_end == "":
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?(?:\n|$)", re.DOTALL
            )
        else:
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?{re.escape(tool_module.tool_call_end)}",
                re.DOTALL,
            )

        matches = re.findall(pattern, model_output)
        if matches:
            remaining = re.sub(pattern, " ", model_output).strip()
            for match in matches:
                call = (
                    match.strip()
                    .removeprefix(tool_module.tool_call_start)
                    .removesuffix(tool_module.tool_call_end)
                )
                try:
                    called_tools.extend(
                        _shaped_calls(
                            tool_module.parse_tool_call(call, tools), len(called_tools)
                        )
                    )
                except Exception as exc:
                    logger.warning("Invalid tool call %r: %s", call, exc)

        open_call = parse_unterminated_call(
            remaining, tool_module, tools, len(called_tools)
        )
        if open_call is not None:
            called_tools.extend(open_call)
            remaining = remaining[: remaining.find(tool_module.tool_call_start)].strip()
    return ParseResult(calls=called_tools, remaining_text=remaining)

"""Lightweight stdio MCP server for auto-validation tools.

Speaks a minimal JSON-RPC subset compatible with MCP-style tool calling:
initialize, tools/list, tools/call.

Assumption: hosts that expect full MCP schema can still invoke these tools
when this script is registered as a stdio server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is importable when launched as a script.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from automation.service import AutoValidationService


TOOLS = [
    {
        "name": "run_auto_validation",
        "description": "Discover datasets under roots and run automated validation + reports.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "roots": {"type": "array", "items": {"type": "string"}},
                "max_files": {"type": "integer", "default": 20},
                "threshold_profile": {
                    "type": "string",
                    "enum": ["strict", "balanced", "lenient"],
                    "default": "balanced",
                },
                "include_reconstruction_checks": {"type": "boolean", "default": True},
            },
            "required": ["roots"],
        },
    },
    {
        "name": "get_latest_auto_report",
        "description": "Return paths and short previews for the latest automation reports.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_recommended_uploads",
        "description": "List datasets recommended for upload from the latest validation run.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "explain_dataset_status",
        "description": "Explain status/reasons for a measurement path from the latest run.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "measurement_path": {"type": "string"},
                "scan_index": {"type": ["integer", "null"]},
            },
            "required": ["measurement_path"],
        },
    },
]


def _ok(result: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}], "isError": False}


def _err(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    service = AutoValidationService()
    try:
        if name == "run_auto_validation":
            roots = arguments.get("roots") or []
            if not roots:
                return _err("roots is required")
            batch = service.run(
                list(roots),
                max_files=int(arguments.get("max_files", 20)),
                threshold_profile=str(arguments.get("threshold_profile", "balanced")),
                include_reconstruction_checks=bool(
                    arguments.get("include_reconstruction_checks", True)
                ),
            )
            return _ok(
                {
                    "message": batch.message,
                    "counts": batch.counts,
                    "report_paths": batch.report_paths,
                    "recommended": service.list_recommended_uploads(batch),
                }
            )
        if name == "get_latest_auto_report":
            paths = service.get_latest_report_paths()
            previews = {}
            for key, path in paths.items():
                if path.endswith(".md"):
                    text = Path(path).read_text(encoding="utf-8")
                    previews[key] = "\n".join(text.splitlines()[:40])
            return _ok({"paths": paths, "previews": previews})
        if name == "list_recommended_uploads":
            return _ok({"recommended": service.list_recommended_uploads()})
        if name == "explain_dataset_status":
            return _ok(
                service.explain_dataset_status(
                    str(arguments.get("measurement_path")),
                    arguments.get("scan_index"),
                )
            )
        return _err(f"Unknown tool: {name}")
    except Exception as exc:  # pragma: no cover
        return _err(str(exc))


def _handle(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    msg_id = message.get("id")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "microwave-auto-validation", "version": "1.0.0"},
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = message.get("params") or {}
        result = _call_tool(str(params.get("name")), dict(params.get("arguments") or {}))
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = _handle(message)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

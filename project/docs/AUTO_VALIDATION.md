# Auto Validation — CLI, GUI, and MCP

This document describes the production automation layer that discovers datasets,
runs load → preprocess → reconstruction → ROI checks, classifies quality, and
writes two reports automatically.

## Architecture (short)

```
roots → discovery → resolver (metadata↔measurement, BMID scan strategy)
      → cached pipeline checks → threshold classifier (strict/balanced/lenient)
      → Report A (guide) + Report B (beginner) + JSON evidence
```

Facades:

- `automation.service.AutoValidationService` — reusable core
- `tools/run_auto_validation.py` — CLI
- Acquisition **Run Auto Validation** — GUI one-click
- `automation/mcp_server.py` — lightweight stdio MCP tools

## Quality profiles

| Profile | Intent |
|---------|--------|
| `strict` | Tight localization/SCR/confidence gates |
| `balanced` | Default demo / upload screening |
| `lenient` | Diagnostics / first-pass triage |

Each result includes: `status`, numeric metrics, human reasons, remediation, and
`recommended_upload`.

## CLI usage

```bash
cd project
.\.venv_local\Scripts\python.exe tools\run_auto_validation.py --roots datasets --profile balanced --max-files 5
```

Faster load-only:

```bash
.\.venv_local\Scripts\python.exe tools\run_auto_validation.py --roots datasets --no-reconstruction --max-files 10
```

## GUI one-click usage

1. Start app: `python main.py`
2. Open **Acquisition**
3. Click **Run Auto Validation**
4. Watch the **Log** panel for streamed progress
5. Read the summary popup; the latest guide markdown opens automatically
6. Reports are under `project/results/`

Existing Load → Preprocess → Reconstruct workflows are unchanged.

## MCP usage

Register stdio server (example Cursor/VS Code MCP config):

```json
{
  "mcpServers": {
    "microwave-auto-validation": {
      "command": "C:/Users/YOU/Downloads/recon/recon/microwave/project/.venv_local/Scripts/python.exe",
      "args": [
        "C:/Users/YOU/Downloads/recon/recon/microwave/project/automation/mcp_server.py"
      ]
    }
  }
}
```

Tools:

- `run_auto_validation` — params: `roots`, `max_files`, `threshold_profile`, `include_reconstruction_checks`
- `get_latest_auto_report`
- `list_recommended_uploads`
- `explain_dataset_status`

Example tool call arguments:

```json
{
  "roots": ["datasets"],
  "max_files": 5,
  "threshold_profile": "balanced",
  "include_reconstruction_checks": true
}
```

## Generated report locations

- `results/latest_guide_progress_report.md` — Report A
- `results/latest_beginner_explainer.md` — Report B
- `results/latest_auto_validation.json` — machine-readable evidence
- Timestamped archives written alongside

## Disclaimer

Model / reconstruction output is **not a clinical diagnosis**.

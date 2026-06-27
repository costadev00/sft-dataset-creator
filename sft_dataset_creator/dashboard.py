from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sft_dataset_creator.state import RunState


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SFT Dataset Run Dashboard</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; color: #17202a; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; }
    .card { border: 1px solid #d5d8dc; border-radius: 6px; padding: 12px; background: #fff; }
    .label { color: #566573; font-size: 12px; text-transform: uppercase; }
    .value { font-size: 24px; font-weight: 700; }
    progress { width: 100%; height: 20px; }
    pre { background: #f4f6f7; padding: 12px; overflow: auto; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>SFT Dataset Run Dashboard</h1>
  <p id="phase"></p>
  <progress id="bar" value="0" max="100"></progress>
  <div class="grid" id="cards"></div>
  <h2>Current</h2>
  <pre id="current">{}</pre>
  <h2>Recent Errors</h2>
  <pre id="errors">[]</pre>
  <script>
    async function refresh() {
      const status = await fetch('/api/status').then(r => r.json());
      document.getElementById('phase').textContent =
        `${status.status || 'unknown'} / ${status.phase || 'unknown'} / ETA ${status.eta_seconds ?? 'n/a'}s`;
      document.getElementById('bar').value = status.accepted_percent || 0;
      const keys = ['target','accepted','attempted','rejected','errors','pending','exhausted','accepted_per_minute'];
      document.getElementById('cards').innerHTML = keys.map(k =>
        `<div class="card"><div class="label">${k}</div><div class="value">${status[k] ?? 0}</div></div>`
      ).join('');
      document.getElementById('current').textContent = JSON.stringify({
        slot: status.current_slot_id,
        document: status.current_document_id,
        title: status.current_title,
        task: status.task,
        difficulty: status.difficulty,
        attempt: status.attempt,
        checkpoint_shards: status.checkpoint_shards,
        gpu: status.gpu
      }, null, 2);
      document.getElementById('errors').textContent = JSON.stringify(status.recent_errors || [], null, 2);
    }
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


def _read_progress(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "progress.json"
    if not path.exists():
        return {"status": "unknown", "phase": "no progress.json yet"}
    return json.loads(path.read_text(encoding="utf-8"))


def create_dashboard_app(run_dir: str | Path):
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
    except ImportError as exc:
        raise ImportError("dashboard requires the optional 'dashboard' extra") from exc

    root = Path(run_dir)
    app = FastAPI(title="sft-dataset dashboard")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return _read_progress(root)

    @app.get("/api/deficits")
    def deficits() -> dict[str, int]:
        with RunState(root / "run.db", read_only=True) as state:
            return state.deficits()

    @app.get("/api/recent-attempts")
    def recent_attempts(limit: int = 20) -> list[dict[str, Any]]:
        with RunState(root / "run.db", read_only=True) as state:
            return state.recent_attempts(limit)

    @app.get("/api/recent-accepted")
    def recent_accepted(limit: int = 20) -> list[dict[str, Any]]:
        with RunState(root / "run.db", read_only=True) as state:
            candidates = state.recent_accepted(limit)
            return [candidate.model_dump(mode="json") for candidate in candidates]

    @app.get("/api/shards")
    def shards() -> list[dict[str, Any]]:
        with RunState(root / "run.db", read_only=True) as state:
            return state.checkpoint_shards()

    @app.get("/api/errors")
    def errors(limit: int = 20) -> list[dict[str, Any]]:
        with RunState(root / "run.db", read_only=True) as state:
            return state.recent_errors(limit)

    return app

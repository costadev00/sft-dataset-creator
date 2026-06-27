from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil

from sft_dataset_creator.state import RunState


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SFT Dataset Run Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #2563eb;
      --good: #0f9f6e;
      --warn: #b7791f;
      --bad: #c2410c;
      --soft: #edf2ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main { max-width: 1440px; margin: 0 auto; padding: 24px; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 { margin: 0; font-size: 28px; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 17px; }
    .subtle { color: var(--muted); font-size: 14px; margin-top: 6px; }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel);
      font-size: 13px;
      color: var(--muted);
      white-space: nowrap;
    }
    .dot { width: 9px; height: 9px; border-radius: 999px; background: var(--accent); }
    .dot.completed { background: var(--good); }
    .dot.partial, .dot.interrupted { background: var(--warn); }
    .dot.failed { background: var(--bad); }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .hero {
      display: grid;
      grid-template-columns: minmax(280px, 2fr) minmax(280px, 1fr);
      gap: 14px;
      margin-bottom: 14px;
    }
    .progress-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 10px;
    }
    .progress-value { font-size: 32px; font-weight: 750; }
    .progress-track {
      height: 14px;
      border-radius: 999px;
      background: #e7ebf2;
      overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      width: 0%;
      border-radius: inherit;
      background: linear-gradient(90deg, #2563eb, #0f9f6e);
      transition: width 250ms ease;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 84px;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }
    .value {
      margin-top: 7px;
      font-size: 25px;
      line-height: 1.1;
      font-weight: 750;
    }
    .value.good { color: var(--good); }
    .value.warn { color: var(--warn); }
    .value.bad { color: var(--bad); }
    .grid-2 {
      display: grid;
      grid-template-columns: minmax(360px, 1fr) minmax(360px, 1fr);
      gap: 14px;
      margin-bottom: 14px;
    }
    .machine-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(120px, 1fr));
      gap: 10px;
    }
    .gpu-list {
      display: grid;
      grid-template-columns: repeat(2, minmax(240px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .gpu-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfe;
    }
    .bar {
      height: 8px;
      border-radius: 999px;
      background: #e7ebf2;
      overflow: hidden;
      margin-top: 8px;
    }
    .bar > span { display: block; height: 100%; background: var(--accent); width: 0%; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      padding: 10px 8px;
      border-bottom: 1px solid #edf0f5;
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
      font-weight: 650;
    }
    tr:last-child td { border-bottom: 0; }
    .title-cell { font-weight: 650; }
    .reason {
      color: var(--bad);
      max-width: 540px;
    }
    .answer-cell {
      max-width: 620px;
      color: #344054;
    }
    .text-clamp {
      display: -webkit-box;
      -webkit-line-clamp: 4;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .mono { font-variant-numeric: tabular-nums; }
    .empty {
      color: var(--muted);
      padding: 18px 0;
      text-align: center;
    }
    @media (max-width: 1000px) {
      .hero, .grid-2 { grid-template-columns: 1fr; }
      .cards { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .gpu-list { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>SFT Dataset Run Dashboard</h1>
        <div class="subtle" id="summary">Carregando status da run...</div>
      </div>
      <div class="pill"><span id="status-dot" class="dot"></span><span id="status-pill">unknown</span></div>
    </header>

    <section class="hero">
      <div class="panel">
        <div class="progress-row">
          <div>
            <div class="label">Progresso por exemplos aceitos</div>
            <div class="progress-value"><span id="accepted">0</span>/<span id="target">0</span></div>
          </div>
          <div class="subtle mono" id="eta">ETA n/a</div>
        </div>
        <div class="progress-track"><div id="progress-fill" class="progress-fill"></div></div>
        <div class="subtle" id="phase">-</div>
      </div>
      <div class="panel">
        <h2>Agora</h2>
        <div id="current" class="subtle">Nenhuma geracao ativa.</div>
      </div>
    </section>

    <section class="cards" id="cards"></section>

    <section class="grid-2">
      <div class="panel">
        <h2>Uso da maquina</h2>
        <div class="machine-grid" id="machine-cards"></div>
        <div class="gpu-list" id="gpu-list"></div>
      </div>
      <div class="panel">
        <h2>Deficits por tarefa</h2>
        <table>
          <thead><tr><th>Tarefa</th><th>Dificuldade</th><th>Faltam</th></tr></thead>
          <tbody id="deficits"><tr><td class="empty" colspan="3">Sem deficits.</td></tr></tbody>
        </table>
      </div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h2>Exemplos aceitos recentemente</h2>
        <table>
          <thead>
            <tr>
              <th>Artigo</th>
              <th>Pergunta</th>
              <th>Resposta</th>
            </tr>
          </thead>
          <tbody id="successes">
            <tr><td class="empty" colspan="3">Nenhum exemplo aceito ainda.</td></tr>
          </tbody>
        </table>
      </div>
      <div class="panel">
        <h2>Artigos rejeitados recentemente</h2>
        <table>
          <thead>
            <tr>
              <th>Artigo</th>
              <th>Tarefa</th>
              <th>Tentativa</th>
              <th>Motivo</th>
            </tr>
          </thead>
          <tbody id="rejections">
            <tr><td class="empty" colspan="4">Nenhuma rejeicao registrada.</td></tr>
          </tbody>
        </table>
      </div>
    </section>
  </main>

  <script>
    const issueLabels = {
      invalid_section: 'Secao de evidencia inexistente',
      missing_grounding: 'Sem evidencia recuperavel',
      duplicate_candidate: 'Conteudo duplicado',
      source_reference_in_candidate: 'Referencia ao texto oculto',
      weak_grounding: 'Evidencia curta',
      invalid_evidence_offsets: 'Offsets de evidencia invalidos',
      empty_instruction_or_output: 'Instrucao ou resposta vazia',
      evidence_quote_mismatch: 'Quote nao bate com o span',
      instruction_too_short: 'Instrucao curta demais'
    };

    function formatNumber(value) {
      if (value === null || value === undefined) return 'n/a';
      return Number(value).toLocaleString('pt-BR');
    }

    function formatPercent(value) {
      if (value === null || value === undefined) return 'n/a';
      return `${Number(value).toLocaleString('pt-BR', { maximumFractionDigits: 1 })}%`;
    }

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, char => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[char]));
    }

    function reasonText(item) {
      if (item.issues && item.issues.length) {
        return item.issues.map(issue => issueLabels[issue] || issue).join(', ');
      }
      return item.reason || item.status || 'Rejeitado';
    }

    async function fetchJson(path, fallback) {
      try {
        const response = await fetch(path, { cache: 'no-store' });
        if (!response.ok) return fallback;
        return await response.json();
      } catch {
        return fallback;
      }
    }

    function renderStatus(status) {
      const accepted = status.accepted || 0;
      const target = status.target || 0;
      const percent = status.accepted_percent || 0;
      document.getElementById('accepted').textContent = formatNumber(accepted);
      document.getElementById('target').textContent = formatNumber(target);
      document.getElementById('progress-fill').style.width = `${Math.max(0, Math.min(100, percent))}%`;
      document.getElementById('phase').textContent =
        `${status.phase || 'unknown'} / ${formatPercent(percent)} aceito`;
      document.getElementById('eta').textContent =
        status.eta_seconds == null ? 'ETA n/a' : `ETA ${formatNumber(Math.round(status.eta_seconds))}s`;
      document.getElementById('summary').textContent =
        `${formatNumber(status.attempted || 0)} tentativas, ${formatNumber(status.rejected || 0)} rejeicoes, ` +
        `${formatNumber(status.errors || 0)} erros. Atualiza automaticamente.`;
      const pill = document.getElementById('status-pill');
      const dot = document.getElementById('status-dot');
      pill.textContent = `${status.status || 'unknown'}`;
      dot.className = `dot ${status.status || ''}`;

      const metrics = [
        ['Tentativas', status.attempted, ''],
        ['Aceitos/min', status.accepted_per_minute, 'good'],
        ['Rejeitados', status.rejected, 'bad'],
        ['Erros', status.errors, 'bad'],
        ['Pendentes', status.pending, 'warn'],
        ['Esgotados', status.exhausted, 'warn']
      ];
      document.getElementById('cards').innerHTML = metrics.map(([label, value, tone]) => `
        <div class="metric">
          <div class="label">${label}</div>
          <div class="value ${tone}">${formatNumber(value ?? 0)}</div>
        </div>
      `).join('');

      const current = document.getElementById('current');
      if (status.current_slot_id || status.current_document_id) {
        current.innerHTML = `
          <div><strong>${escapeHtml(status.current_title || status.current_document_id || 'Documento atual')}</strong></div>
          <div>Slot ${escapeHtml(status.current_slot_id || '-')} · ${escapeHtml(status.task || '-')} · ${escapeHtml(status.difficulty || '-')}</div>
          <div>Tentativa ${escapeHtml(status.attempt || '-')}</div>
        `;
      } else {
        current.textContent = status.phase === 'finished' ? 'Run finalizada.' : 'Aguardando proxima geracao.';
      }
    }

    function renderDeficits(deficits) {
      const entries = Object.entries(deficits || {}).sort((a, b) => b[1] - a[1]);
      const body = document.getElementById('deficits');
      if (!entries.length) {
        body.innerHTML = '<tr><td class="empty" colspan="3">Sem deficits.</td></tr>';
        return;
      }
      body.innerHTML = entries.map(([key, count]) => {
        const [task, difficulty] = key.split(':');
        return `<tr><td>${escapeHtml(task)}</td><td>${escapeHtml(difficulty || '-')}</td><td class="mono">${formatNumber(count)}</td></tr>`;
      }).join('');
    }

    function renderRejections(items) {
      const body = document.getElementById('rejections');
      if (!items || !items.length) {
        body.innerHTML = '<tr><td class="empty" colspan="4">Nenhuma rejeicao registrada.</td></tr>';
        return;
      }
      body.innerHTML = items.map(item => `
        <tr>
          <td class="title-cell">${escapeHtml(item.title || item.document_id)}</td>
          <td>${escapeHtml(item.task || '-')}<div class="subtle">${escapeHtml(item.difficulty || '')}</div></td>
          <td class="mono">${formatNumber(item.attempt_no || 0)}</td>
          <td class="reason">${escapeHtml(reasonText(item))}</td>
        </tr>
      `).join('');
    }

    function renderSuccesses(items) {
      const body = document.getElementById('successes');
      if (!items || !items.length) {
        body.innerHTML = '<tr><td class="empty" colspan="3">Nenhum exemplo aceito ainda.</td></tr>';
        return;
      }
      body.innerHTML = items.map(item => `
        <tr>
          <td class="title-cell">
            ${escapeHtml(item.title || item.document_id)}
            <div class="subtle">${escapeHtml(item.task || '-')} · ${escapeHtml(item.difficulty || '-')} · tentativa ${formatNumber(item.attempt_no || 0)}</div>
          </td>
          <td><div class="text-clamp">${escapeHtml(item.question || '-')}</div></td>
          <td class="answer-cell"><div class="text-clamp">${escapeHtml(item.answer || '-')}</div></td>
        </tr>
      `).join('');
    }

    function renderMachine(machine) {
      const cards = [
        ['CPU', formatPercent(machine.cpu_percent)],
        ['RAM', `${formatPercent(machine.memory_percent)} · ${formatNumber(machine.memory_used_gb)} / ${formatNumber(machine.memory_total_gb)} GB`]
      ];
      document.getElementById('machine-cards').innerHTML = cards.map(([label, value]) => `
        <div class="metric">
          <div class="label">${label}</div>
          <div class="value">${value}</div>
        </div>
      `).join('');

      const gpuList = document.getElementById('gpu-list');
      if (!machine.gpus || !machine.gpus.length) {
        gpuList.innerHTML = '<div class="empty">Nenhuma GPU detectada via nvidia-smi.</div>';
        return;
      }
      gpuList.innerHTML = machine.gpus.map(gpu => {
        const memoryPercent = gpu.memory_total_mb ? gpu.memory_used_mb * 100 / gpu.memory_total_mb : 0;
        return `
          <div class="gpu-card">
            <div class="label">GPU ${escapeHtml(gpu.index)} · ${escapeHtml(gpu.name)}</div>
            <div class="value">${formatPercent(gpu.utilization_percent)}</div>
            <div class="subtle">Memoria ${formatNumber(gpu.memory_used_mb)} / ${formatNumber(gpu.memory_total_mb)} MB · Temp ${formatNumber(gpu.temperature_c)} C</div>
            <div class="bar"><span style="width:${Math.max(0, Math.min(100, gpu.utilization_percent || 0))}%"></span></div>
            <div class="bar"><span style="width:${Math.max(0, Math.min(100, memoryPercent))}%; background:#0f9f6e"></span></div>
          </div>
        `;
      }).join('');
    }

    async function refresh() {
      const [status, deficits, successes, rejections, machine] = await Promise.all([
        fetchJson('/api/status', {}),
        fetchJson('/api/deficits', {}),
        fetchJson('/api/recent-successes?limit=12', []),
        fetchJson('/api/recent-rejections?limit=20', []),
        fetchJson('/api/machine', {})
      ]);
      renderStatus(status);
      renderDeficits(deficits);
      renderSuccesses(successes);
      renderRejections(rejections);
      renderMachine(machine);
    }

    refresh();
    setInterval(refresh, 2500);
  </script>
</body>
</html>
"""


def _read_progress(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "progress.json"
    if not path.exists():
        return {"status": "unknown", "phase": "no progress.json yet"}
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _machine_status() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    payload: dict[str, Any] = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": memory.percent,
        "memory_used_gb": round(memory.used / 1024**3, 2),
        "memory_total_gb": round(memory.total / 1024**3, 2),
        "gpus": [],
        "updated_at": time.time(),
    }
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return payload
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "utilization_percent": _float_or_none(parts[2]),
                "memory_used_mb": _float_or_none(parts[3]),
                "memory_total_mb": _float_or_none(parts[4]),
                "temperature_c": _float_or_none(parts[5]),
                "power_draw_w": _float_or_none(parts[6]),
                "power_limit_w": _float_or_none(parts[7]) if len(parts) > 7 else None,
            }
        )
    payload["gpus"] = gpus
    return payload


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

    @app.get("/api/machine")
    def machine() -> dict[str, Any]:
        return _machine_status()

    @app.get("/api/deficits")
    def deficits() -> dict[str, int]:
        with RunState(root / "run.db", read_only=True) as state:
            return state.deficits()

    @app.get("/api/recent-attempts")
    def recent_attempts(limit: int = 20) -> list[dict[str, Any]]:
        with RunState(root / "run.db", read_only=True) as state:
            return state.recent_attempts(limit)

    @app.get("/api/recent-rejections")
    def recent_rejections(limit: int = 20) -> list[dict[str, Any]]:
        with RunState(root / "run.db", read_only=True) as state:
            return state.recent_rejections(limit)

    @app.get("/api/recent-successes")
    def recent_successes(limit: int = 20) -> list[dict[str, Any]]:
        with RunState(root / "run.db", read_only=True) as state:
            return state.recent_successes(limit)

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

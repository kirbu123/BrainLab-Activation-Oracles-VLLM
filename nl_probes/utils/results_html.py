"""Write a self-contained HTML results page after each validation step."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class LossPoint:
    step: int
    loss: float
    lr: float | None = None


@dataclass
class EvalPoint:
    step: int
    metrics: dict[str, float]
    n_by_dataset: dict[str, int] = field(default_factory=dict)


class ResultsHtmlLogger:
    def __init__(self) -> None:
        self.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.losses: list[LossPoint] = []
        self.evals: list[EvalPoint] = []

    def append_loss(self, step: int, loss: float, lr: float | None = None) -> None:
        self.losses.append(LossPoint(step=int(step), loss=float(loss), lr=None if lr is None else float(lr)))

    def append_eval(
        self,
        step: int,
        metrics: dict[str, float],
        n_by_dataset: dict[str, int] | None = None,
    ) -> None:
        self.evals.append(
            EvalPoint(
                step=int(step),
                metrics={k: float(v) for k, v in metrics.items()},
                n_by_dataset={k: int(v) for k, v in (n_by_dataset or {}).items()},
            )
        )

    def write(self, cfg: Any, extra: dict[str, Any] | None = None) -> Path:
        path = Path(getattr(cfg, "results_html_path", "logs/train_results.html"))
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "started_at": self.started_at,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model_name": getattr(cfg, "model_name", ""),
            "wandb_run_name": getattr(cfg, "wandb_run_name", ""),
            "save_dir": getattr(cfg, "save_dir", ""),
            "eval_steps": getattr(cfg, "eval_steps", None),
            "train_batch_size": getattr(cfg, "train_batch_size", None),
            "lr": getattr(cfg, "lr", None),
            "lora_r": getattr(cfg, "lora_r", None),
            "lora_alpha": getattr(cfg, "lora_alpha", None),
            "act_layers": list(getattr(cfg, "act_layers", []) or []),
            "losses": [asdict(p) for p in self.losses],
            "evals": [asdict(p) for p in self.evals],
            **(extra or {}),
        }
        json_path = path.with_suffix(".json")
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(render_results_html(payload), encoding="utf-8")
        tmp.replace(path)
        return path


def _ans_keys(evals: list[dict]) -> list[str]:
    keys: list[str] = []
    for point in evals:
        for key in point.get("metrics", {}):
            if key.startswith("eval_ans_correct/") and key not in keys:
                keys.append(key)
    return keys


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}%"


def _polyline(values: list[tuple[float, float]], x0: float, y0: float, width: float, height: float, ymin: float, ymax: float) -> str:
    if not values:
        return ""
    xs = [v[0] for v in values]
    xmin, xmax = min(xs), max(xs)
    if xmax <= xmin:
        xmax = xmin + 1.0
    span = ymax - ymin if ymax > ymin else 1.0
    pts = []
    for x, y in values:
        px = x0 + (x - xmin) / (xmax - xmin) * width
        py = y0 + (1.0 - (y - ymin) / span) * height
        pts.append(f"{px:.1f},{py:.1f}")
    return " ".join(pts)


def render_results_html(payload: dict[str, Any]) -> str:
    evals: list[dict] = payload.get("evals") or []
    losses: list[dict] = payload.get("losses") or []
    ans_keys = _ans_keys(evals)
    title = payload.get("wandb_run_name") or payload.get("model_name") or "Training results"
    datasets = [k.split("/", 1)[1] for k in ans_keys]

    latest = evals[-1] if evals else None
    first = evals[0] if evals else None
    primary = ans_keys[0] if ans_keys else None
    latest_ans = latest["metrics"].get(primary) if latest and primary else None
    first_ans = first["metrics"].get(primary) if first and primary else None
    peak_ans = max((p["metrics"].get(primary, 0.0) for p in evals), default=None) if primary else None
    peak_step = None
    if primary and peak_ans is not None:
        for point in evals:
            if point["metrics"].get(primary) == peak_ans:
                peak_step = point["step"]
    fmt_key = f"eval_format_correct/{datasets[0]}" if datasets else None
    latest_fmt = latest["metrics"].get(fmt_key) if latest and fmt_key else None
    lift = None
    if latest_ans is not None and first_ans is not None:
        lift = (latest_ans - first_ans) * 100.0
    n_eval = 0
    if latest:
        n_eval = sum(latest.get("n_by_dataset", {}).values())

    acc_polylines = []
    if evals and ans_keys:
        for key in ans_keys:
            series = [(float(p["step"]), float(p["metrics"][key]) * 100.0) for p in evals if key in p["metrics"]]
            acc_polylines.append((key.split("/", 1)[1], _polyline(series, 56, 30, 554, 180, 45.0, 100.0)))

    loss_pts = [(float(p["step"]), float(p["loss"])) for p in losses]
    if len(loss_pts) > 80:
        step_n = max(1, len(loss_pts) // 80)
        sampled = loss_pts[::step_n]
        if sampled[-1] != loss_pts[-1]:
            sampled.append(loss_pts[-1])
        loss_pts = sampled
    loss_ymin = min((y for _, y in loss_pts), default=0.0)
    loss_ymax = max((y for _, y in loss_pts), default=1.0)
    pad = (loss_ymax - loss_ymin) * 0.08 or 0.1
    loss_poly = _polyline(loss_pts, 56, 30, 554, 180, loss_ymin - pad, loss_ymax + pad)

    rows_html = []
    for point in evals:
        cells = [f"<td>{point['step']}</td>"]
        for ds in datasets:
            ans = point["metrics"].get(f"eval_ans_correct/{ds}")
            fmt = point["metrics"].get(f"eval_format_correct/{ds}")
            n = point.get("n_by_dataset", {}).get(ds)
            n_txt = f" / {n}" if n else ""
            cells.append(f"<td>{_fmt_pct(ans)}{html.escape(n_txt)}</td>")
            cells.append(f"<td>{_fmt_pct(fmt)}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    header_cells = ["<th>Step</th>"]
    for ds in datasets:
        header_cells.append(f"<th>{html.escape(ds)} acc.</th>")
        header_cells.append(f"<th>{html.escape(ds)} format</th>")

    acc_svg_paths = "".join(
        f'<polyline fill="none" stroke="#7dcea0" stroke-width="2" points="{html.escape(pts)}" />'
        for _, pts in acc_polylines
        if pts
    )
    labels_x = ""
    if evals:
        xmin, xmax = evals[0]["step"], evals[-1]["step"]
        if xmax <= xmin:
            xmax = xmin + 1
        for point in evals:
            px = 56 + (point["step"] - xmin) / (xmax - xmin) * 554
            labels_x += f'<text x="{px:.1f}" y="232" fill="#9a9aa3" font-size="11" text-anchor="middle">{point["step"]}</text>'

    extra_rows = []
    for label, value in (
        ("Model", payload.get("model_name") or "—"),
        ("Run", payload.get("wandb_run_name") or "—"),
        ("Save dir", payload.get("save_dir") or "—"),
        ("Started", payload.get("started_at") or "—"),
        ("Updated", payload.get("updated_at") or "—"),
        ("Eval every", str(payload.get("eval_steps") or "—") + " steps"),
        ("Layers", ",".join(str(x) for x in payload.get("act_layers") or []) or "—"),
        ("LoRA", f"r={payload.get('lora_r')} α={payload.get('lora_alpha')}"),
        ("Loss points", str(len(losses))),
        ("Val steps logged", str(len(evals))),
        ("Eval items (latest)", str(n_eval) if n_eval else "—"),
    ):
        extra_rows.append(
            f"<tr><td>{html.escape(str(label))}</td><td>{html.escape(str(value))}</td></tr>"
        )

    lift_txt = f"{lift:+.1f} pp" if lift is not None else "—"
    peak_txt = f"{_fmt_pct(peak_ans)} (step {peak_step})" if peak_ans is not None else "—"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta http-equiv="refresh" content="30" />
  <title>{html.escape(str(title))}</title>
  <style>
    :root {{
      --bg: #111113; --fg: #ececec; --muted: #9a9aa3; --line: #2a2a2e;
      --card: #1a1a1d; --good: #7dcea0; --warn: #e6b84d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font: 14px/1.45 ui-sans-serif, system-ui, sans-serif;
      background: var(--bg); color: var(--fg); padding: 32px 28px 64px; max-width: 1100px;
    }}
    h1 {{ font-size: 22px; font-weight: 600; margin: 0 0 6px; }}
    h2 {{ font-size: 16px; font-weight: 600; margin: 28px 0 8px; }}
    .sub {{ color: var(--muted); margin: 0 0 24px; }}
    .caption {{ color: var(--muted); font-size: 12px; margin: 4px 0 12px; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0 8px; }}
    .stat {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }}
    .stat b {{ display: block; font-size: 22px; font-weight: 600; }}
    .stat span {{ color: var(--muted); font-size: 12px; }}
    .good {{ color: var(--good); }}
    .note {{
      border: 1px solid var(--line); background: var(--card); border-radius: 8px;
      padding: 12px 14px; color: var(--muted); margin: 16px 0;
    }}
    table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
    th, td {{ text-align: right; padding: 8px 10px; border-bottom: 1px solid var(--line); }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 500; font-size: 12px; }}
    svg {{ display: block; width: 100%; height: auto; background: var(--card); border: 1px solid var(--line); border-radius: 8px; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    @media (max-width: 800px) {{ .stats, .grid2 {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>{html.escape(str(title))}</h1>
  <p class="sub">Updated {html.escape(str(payload.get("updated_at") or ""))} · auto-refreshes every 30s · val after each eval step</p>
  <div class="stats">
    <div class="stat"><b class="good">{html.escape(_fmt_pct(latest_ans))}</b><span>Latest answer accuracy</span></div>
    <div class="stat"><b>{html.escape(lift_txt)}</b><span>Lift vs first eval</span></div>
    <div class="stat"><b>{html.escape(peak_txt)}</b><span>Peak answer accuracy</span></div>
    <div class="stat"><b class="good">{html.escape(_fmt_pct(latest_fmt))}</b><span>Latest format accuracy</span></div>
  </div>
  <p class="note">Written by the train/val loop after every validation. Chance for binary Yes/No is 50%. Format = share of generations that parse as yes/no.</p>
  <h2>Validation accuracy</h2>
  <p class="caption">Answer accuracy (%) vs optimizer step · dashed line = 50% chance</p>
  <svg viewBox="0 0 640 250" role="img" aria-label="Validation accuracy">
    <line x1="56" y1="30" x2="56" y2="210" stroke="#2a2a2e" />
    <line x1="56" y1="210" x2="610" y2="210" stroke="#2a2a2e" />
    <line x1="56" y1="193.6" x2="610" y2="193.6" stroke="#e6b84d" stroke-dasharray="4 4" />
    <text x="616" y="197" fill="#e6b84d" font-size="11">50%</text>
    <text x="8" y="34" fill="#9a9aa3" font-size="11">100%</text>
    <text x="14" y="214" fill="#9a9aa3" font-size="11">45%</text>
    {acc_svg_paths}
    {labels_x}
  </svg>
  <h2>Train loss</h2>
  <p class="caption">{len(losses)} logged optimizer steps (downsampled for the plot if &gt; 80)</p>
  <svg viewBox="0 0 640 250" role="img" aria-label="Train loss">
    <line x1="56" y1="30" x2="56" y2="210" stroke="#2a2a2e" />
    <line x1="56" y1="210" x2="610" y2="210" stroke="#2a2a2e" />
    <text x="8" y="34" fill="#9a9aa3" font-size="11">{html.escape(f"{loss_ymax + pad:.2f}")}</text>
    <text x="8" y="214" fill="#9a9aa3" font-size="11">{html.escape(f"{loss_ymin - pad:.2f}")}</text>
    <polyline fill="none" stroke="#79b8ff" stroke-width="1.8" points="{html.escape(loss_poly)}" />
  </svg>
  <div class="grid2" style="margin-top:28px">
    <div>
      <h2>Eval log</h2>
      <table>
        <thead><tr>{"".join(header_cells)}</tr></thead>
        <tbody>
          {"".join(rows_html) if rows_html else "<tr><td colspan='8'>No evals yet</td></tr>"}
        </tbody>
      </table>
    </div>
    <div>
      <h2>Launch</h2>
      <table>
        <thead><tr><th>Field</th><th>Value</th></tr></thead>
        <tbody>{"".join(extra_rows)}</tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""

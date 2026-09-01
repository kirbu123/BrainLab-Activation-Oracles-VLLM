"""Eval-only source-token modality report (HTML + markdown). No train loss."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

MODE_COLORS = {
    "mixed": "#7dcea0",
    "text": "#79b8ff",
    "visual": "#e6b84d",
}

STANDARD_ORDER = (
    "classification_vsr",
    "classification_gqa_yesno",
    "classification_coco_presence",
    "classification_snli_ve",
    "coco_captions_past_lens",
)
TARGET_ORDER = (
    "visual_taboo",
    "visual_user_attribute",
    "visual_ssc",
    "visual_personaqa",
)
BINARY_DATASETS = frozenset(STANDARD_ORDER[:4])
DISPLAY_NAMES = {
    "classification_vsr": "VSR",
    "classification_gqa_yesno": "GQA yes/no",
    "classification_coco_presence": "COCO presence",
    "classification_snli_ve": "SNLI-VE",
    "coco_captions_past_lens": "COCO captions",
    "visual_taboo": "Visual Taboo",
    "visual_user_attribute": "User attribute",
    "visual_ssc": "SSC",
    "visual_personaqa": "PersonaQA",
}
MD_DISPLAY_NAMES = {
    "classification_vsr": "VSR",
    "classification_gqa_yesno": "GQA yes/no",
    "classification_coco_presence": "COCO object presence",
    "classification_snli_ve": "SNLI-VE",
    "coco_captions_past_lens": "COCO caption past-lens",
    "visual_taboo": "Visual Taboo",
    "visual_user_attribute": "User attribute",
    "visual_ssc": "SSC",
    "visual_personaqa": "PersonaQA",
}
KNOWN_DATASETS = frozenset(DISPLAY_NAMES)


@dataclass(frozen=True)
class BenchScores:
    dataset: str
    n: int
    answer: dict[str, float]
    format: dict[str, float]


def _require_key(metrics: dict[str, float], key: str) -> float:
    if key not in metrics:
        raise KeyError(f"Missing metric {key!r}")
    return float(metrics[key])


def parse_modality_eval(
    metrics: dict[str, float],
    n_by_dataset: dict[str, int],
    source_tokens: list[str] | tuple[str, ...],
) -> tuple[list[BenchScores], list[BenchScores]]:
    modes = list(source_tokens)
    if not modes:
        raise ValueError("source_tokens must not be empty")
    unknown_modes = [mode for mode in modes if mode not in MODE_COLORS]
    if unknown_modes:
        raise ValueError(f"Unknown source-token modes: {unknown_modes}")
    datasets = sorted({key.rsplit("/", 1)[0] for key in n_by_dataset})
    unknown = [name for name in datasets if name not in KNOWN_DATASETS]
    if unknown:
        raise ValueError(f"Unknown validation dataset keys: {unknown}")

    def collect(order: tuple[str, ...]) -> list[BenchScores]:
        benches: list[BenchScores] = []
        for dataset in order:
            if dataset not in datasets:
                continue
            counts = []
            answer = {}
            format_acc = {}
            for mode in modes:
                n_key = f"{dataset}/{mode}"
                if n_key not in n_by_dataset:
                    raise KeyError(f"Missing n_by_dataset entry {n_key!r}")
                counts.append(int(n_by_dataset[n_key]))
                answer[mode] = _require_key(metrics, f"eval_ans_correct/{dataset}/{mode}")
                format_acc[mode] = _require_key(
                    metrics, f"eval_format_correct/{dataset}/{mode}"
                )
            if len(set(counts)) != 1:
                raise ValueError(f"n mismatch across modes for {dataset}: {counts}")
            benches.append(
                BenchScores(
                    dataset=dataset, n=counts[0], answer=answer, format=format_acc
                )
            )
        return benches

    return collect(STANDARD_ORDER), collect(TARGET_ORDER)


def pooled_binary(
    benches: list[BenchScores], modes: list[str]
) -> tuple[int, dict[str, float]] | None:
    binary = [bench for bench in benches if bench.dataset in BINARY_DATASETS]
    if not binary:
        return None
    n = sum(bench.n for bench in binary)
    pooled = {
        mode: sum(bench.answer[mode] * bench.n for bench in binary) / n for mode in modes
    }
    return n, pooled


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _pp(value: float) -> str:
    return f"{value * 100:+.1f} pp"


def _pp_md(value: float) -> str:
    return _pp(value).replace("-", "−")


def _best_mode(scores: dict[str, float]) -> str:
    return max(scores, key=scores.__getitem__)


def _short_model_name(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1]


def _join_names(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _logged_range(payload: dict[str, Any]) -> str | None:
    if "started_at" not in payload or "updated_at" not in payload:
        return None
    start = datetime.fromisoformat(payload["started_at"])
    end = datetime.fromisoformat(payload["updated_at"])
    return f"{start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%H:%M')}"


def _stat_span(mode: str, n_benches: int, n_items: int, modes: list[str]) -> str:
    bench_word = "bench" if n_benches == 1 else "benches"
    if mode == "mixed":
        return f"Mixed answer acc. ({n_benches} binary {bench_word}, n={n_items})"
    if mode == "text" and "mixed" in modes:
        return "Text-token answer acc. (same items)"
    if mode == "visual" and "mixed" in modes:
        return "Visual-token answer acc. (same items)"
    return f"{mode.capitalize()} answer acc. (binary {bench_word}, n={n_items})"


def _mode_blurbs_html(modes: list[str]) -> str:
    blurbs = {
        "mixed": (
            f'<span class="swatch" style="background:{MODE_COLORS["mixed"]}"></span>'
            "mixed = cached training-time positions (late window / caption span / prompt tail)."
        ),
        "text": (
            f'<span class="swatch" style="background:{MODE_COLORS["text"]}"></span>'
            "text = last K non-image tokens."
        ),
        "visual": (
            f'<span class="swatch" style="background:{MODE_COLORS["visual"]}"></span>'
            "visual = last K <code>&lt;|image_pad|&gt;</code> tokens."
        ),
    }
    return " ".join(blurbs[mode] for mode in modes)


def _target_note_html(targets: list[BenchScores], modes: list[str]) -> str:
    all_zero = all(
        score == 0.0 for bench in targets for score in bench.answer.values()
    )
    format_full = all(
        bench.format[mode] == 1.0 for bench in targets for mode in modes
    )
    if all_zero and format_full:
        return (
            "Every target-organism row scored 0 recovery. Format is 100% because every "
            "generation extracted exactly one closed-set label, but never the secret. "
            "Token pool does not recover the hidden attribute."
        )
    if all_zero:
        return (
            "Secret-recovery accuracy of 0 means the oracle never named the hidden "
            "attribute from any token pool. Format is the closed-set extract rate."
        )
    return "Format is the share of generations that extract exactly one closed-set label."


def _ood_markdown(metrics: dict[str, float]) -> str:
    ood_keys = [key for key in metrics if key.startswith("eval_target_ood/")]
    if not ood_keys:
        return ""
    slices = sorted({key.split("/")[2] for key in ood_keys})
    if all(metrics[key] == 0.0 for key in ood_keys):
        listed = ", ".join(f"`{name}`" for name in slices)
        return f"OOD slices ({listed}) are all 0 as well.\n\n"
    return ""


def _narrative_findings(
    standard: list[BenchScores],
    modes: list[str],
    binary_pool: tuple[int, dict[str, float]] | None,
) -> list[str]:
    findings: list[str] = []
    if "mixed" in modes and standard:
        n_mixed_best = sum(
            1 for bench in standard if _best_mode(bench.answer) == "mixed"
        )
        mixed_lead = (
            f"**Training-time mixed positions are the usable readout on "
            f"{n_mixed_best} of {len(standard)} standard benches.**"
        )
        extras: list[str] = []
        if binary_pool is not None:
            _n_bin, pooled = binary_pool
            extras.append(
                "Pooled binary accuracy "
                + " vs ".join(f"{_pct(pooled[mode])} {mode}" for mode in modes)
                + "."
            )
        binary = [bench for bench in standard if bench.dataset in BINARY_DATASETS]
        if binary:
            strongest = max(binary, key=lambda bench: bench.answer["mixed"])
            extra = (
                f"{MD_DISPLAY_NAMES[strongest.dataset]} is the strongest mixed result "
                f"({_pct(strongest.answer['mixed'])})"
            )
            if "visual" in strongest.answer and strongest.answer["visual"] < 0.5:
                extra += f"; visual there is chance ({_pct(strongest.answer['visual'])})"
            extras.append(extra + ".")
        findings.append(mixed_lead + (" " + " ".join(extras) if extras else ""))

    if "text" in modes and "mixed" in modes and standard:
        wrecked = [
            bench
            for bench in standard
            if bench.dataset in BINARY_DATASETS and bench.format["text"] < 0.5
        ]
        beats = [
            bench for bench in standard if bench.answer["text"] > bench.answer["mixed"]
        ]
        parts = [
            "**Text ≠ mixed.** Resampling the last K non-image tokens is not the "
            "cached late window."
        ]
        if wrecked:
            names = _join_names([MD_DISPLAY_NAMES[bench.dataset] for bench in wrecked])
            fmt_bits = ", ".join(_pct(bench.format["text"]) for bench in wrecked)
            parts.append(
                f"On {names}, text also wrecks format ({fmt_bits}), so answer "
                "accuracy is not a fair Yes/No comparison on those."
            )
        if beats:
            beat_bits = "; ".join(
                f"{MD_DISPLAY_NAMES[bench.dataset]} text **beats** mixed "
                f"({_pct(bench.answer['text'])} vs {_pct(bench.answer['mixed'])})"
                + (
                    f" with format still {_pct(bench.format['text'])}"
                    if bench.format["text"] >= 0.5
                    else ""
                )
                for bench in beats
            )
            parts.append(beat_bits + ".")
        findings.append(" ".join(parts))

    if "visual" in modes and standard:
        binary = [bench for bench in standard if bench.dataset in BINARY_DATASETS]
        below = [
            MD_DISPLAY_NAMES[bench.dataset]
            for bench in binary
            if bench.answer["visual"] <= 0.5
        ]
        above = [bench for bench in binary if bench.answer["visual"] > 0.5]
        vis_parts = [
            "**Visual tokens alone do not carry the binary label in a form this "
            "oracle can read.**"
        ]
        clause = []
        if below:
            clause.append(f"Visual is at or below chance on {_join_names(below)}")
        for bench in above:
            clause.append(
                f"only {_pp_md(bench.answer['visual'] - 0.5)} over chance on "
                f"{MD_DISPLAY_NAMES[bench.dataset]}"
            )
        if clause:
            vis_parts.append(", and ".join(clause) + ".")
        captions = [
            bench for bench in standard if bench.dataset == "coco_captions_past_lens"
        ]
        if captions and "mixed" in captions[0].answer:
            vis_parts.append(
                f"Caption recovery from image pads ({_pct(captions[0].answer['visual'])}) "
                f"is also below mixed ({_pct(captions[0].answer['mixed'])})."
            )
        findings.append(" ".join(vis_parts))
        findings.append(
            "**Causal order explains the visual gap.** Image pads sit before the "
            "question. Their residuals have the pixels, not the Yes/No prompt. "
            "Mixed/text residuals at the end of the chat have already mixed vision "
            "through attention. The oracle was trained on those late positions, "
            "not on raw image pads."
        )

    return findings


def grouped_bar_svg(
    benches: list[BenchScores],
    modes: list[str],
    field: str,
    *,
    chance: float | None,
) -> str:
    if field not in {"answer", "format"}:
        raise ValueError(f"Unsupported chart field {field!r}")
    labels = [DISPLAY_NAMES[bench.dataset] for bench in benches]
    width, height = 760, 280
    x0, y0, plot_w, plot_h = 56, 28, 680, 190
    group_w = plot_w / len(labels)
    bar_w = 28
    gap = 5
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img">',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" stroke="#2a2a2e" />',
        f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" stroke="#2a2a2e" />',
        f'<text x="8" y="{y0 + 6}" fill="#9a9aa3" font-size="11">100%</text>',
        f'<text x="14" y="{y0 + plot_h + 4}" fill="#9a9aa3" font-size="11">0%</text>',
    ]
    if chance is not None:
        cy = y0 + plot_h * (1 - chance)
        parts.append(
            f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0 + plot_w}" y2="{cy:.1f}" '
            f'stroke="#e6b84d" stroke-dasharray="4 4" />'
        )
        parts.append(
            f'<text x="{x0 + plot_w + 6}" y="{cy + 4:.1f}" fill="#e6b84d" font-size="11">'
            f"{chance * 100:.0f}%</text>"
        )
    for i, bench in enumerate(benches):
        scores = getattr(bench, field)
        gx = x0 + i * group_w + (
            group_w - (len(modes) * bar_w + (len(modes) - 1) * gap)
        ) / 2
        for j, mode in enumerate(modes):
            pct = scores[mode] * 100
            bar_h = pct / 100 * plot_h
            x = gx + j * (bar_w + gap)
            y = y0 + plot_h - bar_h
            color = MODE_COLORS[mode]
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{color}" />'
            )
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" fill="#ececec" font-size="10" '
                f'text-anchor="middle">{pct:.0f}</text>'
            )
        parts.append(
            f'<text x="{x0 + i * group_w + group_w / 2:.1f}" y="{y0 + plot_h + 20:.1f}" '
            f'fill="#9a9aa3" font-size="11" text-anchor="middle">{html.escape(labels[i])}</text>'
        )
    lx = x0
    for mode in modes:
        parts.append(
            f'<rect x="{lx}" y="{height - 22}" width="12" height="12" fill="{MODE_COLORS[mode]}" />'
        )
        parts.append(
            f'<text x="{lx + 16}" y="{height - 12}" fill="#ececec" font-size="12">{html.escape(mode)}</text>'
        )
        lx += 80
    parts.append("</svg>")
    return "\n".join(parts)


def _answer_cell(value: float, *, baseline: float | None, binary: bool) -> str:
    cls = ' class="bad"' if binary and value < 0.5 else ""
    extra = ""
    if baseline is not None:
        extra = f'<div class="delta">{html.escape(_pp(value - baseline))}</div>'
    return f"<td{cls}>{html.escape(_pct(value))}{extra}</td>"


def render_modality_eval_html(payload: dict[str, Any]) -> str:
    modes = list(payload["source_tokens"])
    standard, targets = parse_modality_eval(
        payload["metrics"], payload["n_by_dataset"], modes
    )
    binary_pool = pooled_binary(standard, modes)
    baseline_mode = "mixed" if "mixed" in modes else None
    run_id = payload["run_id"]
    model_name = _short_model_name(payload["model_name"])
    layers = "/".join(str(layer) for layer in payload["act_layers"])
    n_binary_benches = sum(1 for bench in standard if bench.dataset in BINARY_DATASETS)

    stat_cards = []
    if binary_pool is not None:
        n_bin, pooled = binary_pool
        for mode in modes:
            cls = ' class="good"' if mode == _best_mode(pooled) else ""
            stat_cards.append(
                f'<div class="stat"><b{cls}>{html.escape(_pct(pooled[mode]))}</b>'
                f"<span>{html.escape(_stat_span(mode, n_binary_benches, n_bin, modes))}"
                "</span></div>"
            )
    stats_html = "".join(stat_cards)
    stats_cols = max(len(modes), 1)

    ans_svg = grouped_bar_svg(standard, modes, "answer", chance=0.5) if standard else ""
    fmt_svg = grouped_bar_svg(standard, modes, "format", chance=None) if standard else ""

    ans_rows = []
    for bench in standard:
        binary = bench.dataset in BINARY_DATASETS
        baseline = bench.answer[baseline_mode] if baseline_mode else None
        cells = [
            f"<td>{html.escape(DISPLAY_NAMES[bench.dataset])}</td>",
            f"<td>{bench.n}</td>",
        ]
        for mode in modes:
            cells.append(
                _answer_cell(
                    bench.answer[mode],
                    baseline=None if mode == baseline_mode else baseline,
                    binary=binary,
                )
            )
        ans_rows.append("<tr>" + "".join(cells) + "</tr>")
    if binary_pool is not None:
        n_bin, pooled = binary_pool
        baseline = pooled[baseline_mode] if baseline_mode else None
        cells = ["<td>Binary pooled</td>", f"<td>{n_bin}</td>"]
        for mode in modes:
            cells.append(
                _answer_cell(
                    pooled[mode],
                    baseline=None if mode == baseline_mode else baseline,
                    binary=False,
                )
            )
        ans_rows.append("<tr>" + "".join(cells) + "</tr>")

    fmt_rows = []
    for bench in standard:
        cells = [f"<td>{html.escape(DISPLAY_NAMES[bench.dataset])}</td>"]
        for mode in modes:
            cells.append(f"<td>{html.escape(_pct(bench.format[mode]))}</td>")
        fmt_rows.append("<tr>" + "".join(cells) + "</tr>")

    tgt_rows = []
    for bench in targets:
        cells = [
            f"<td>{html.escape(DISPLAY_NAMES[bench.dataset])}</td>",
            f"<td>{bench.n}</td>",
        ]
        for mode in modes:
            cells.append(f"<td>{html.escape(_pct(bench.answer[mode]))}</td>")
        cells.append(f"<td>{html.escape(_pct(bench.format[modes[0]]))}</td>")
        tgt_rows.append("<tr>" + "".join(cells) + "</tr>")

    mode_headers = "".join(
        f"<th>{html.escape(mode)}"
        + (" (Δ mixed)" if baseline_mode and mode != baseline_mode else "")
        + "</th>"
        for mode in modes
    )
    fmt_headers = "".join(f"<th>{html.escape(mode)}</th>" for mode in modes)
    tgt_headers = "".join(f"<th>{html.escape(mode)} acc.</th>" for mode in modes)

    ans_section = ""
    if standard:
        ans_section = f"""
  <h2>Answer accuracy by benchmark</h2>
  <p class="caption">Percent correct · {" vs ".join(modes)} · binary chance = 50%</p>
  {ans_svg}
  <h2>Format accuracy by benchmark</h2>
  <p class="caption">Share of generations that parse as the expected output format (Yes/No for binary; non-empty for captions)</p>
  {fmt_svg}
  <h2>Eval log · answer accuracy</h2>
  <table>
    <thead><tr><th>Benchmark</th><th>n</th>{mode_headers}</tr></thead>
    <tbody>{"".join(ans_rows)}</tbody>
  </table>
  <h2>Eval log · format accuracy</h2>
  <table>
    <thead><tr><th>Benchmark</th>{fmt_headers}</tr></thead>
    <tbody>{"".join(fmt_rows)}</tbody>
  </table>"""

    tgt_section = ""
    if targets:
        tgt_section = f"""
  <h2>Target-organism secret recovery</h2>
  <p class="caption">Adapter-on caches · prompt_tail + prompt_response · all OOD slices</p>
  <table>
    <thead><tr><th>Family</th><th>n</th>{tgt_headers}<th>format</th></tr></thead>
    <tbody>{"".join(tgt_rows)}</tbody>
  </table>
  <p class="note">{html.escape(_target_note_html(targets, modes))}</p>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Source-token modality eval · {html.escape(str(run_id))}</title>
  <style>
    :root {{
      --bg: #111113; --fg: #ececec; --muted: #9a9aa3; --line: #2a2a2e;
      --card: #1a1a1d; --good: #7dcea0; --warn: #e6b84d; --bad: #e07a7a;
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
    .stats {{ display: grid; grid-template-columns: repeat({stats_cols}, 1fr); gap: 12px; margin: 16px 0 8px; }}
    .stat {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }}
    .stat b {{ display: block; font-size: 22px; font-weight: 600; }}
    .stat span {{ color: var(--muted); font-size: 12px; }}
    .good {{ color: var(--good); }}
    .note {{
      border: 1px solid var(--line); background: var(--card); border-radius: 8px;
      padding: 12px 14px; color: var(--muted); margin: 16px 0;
    }}
    table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
    th, td {{ text-align: right; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 500; font-size: 12px; }}
    td.bad {{ color: var(--bad); }}
    .delta {{ color: var(--muted); font-size: 11px; }}
    svg {{ display: block; width: 100%; height: auto; background: var(--card); border: 1px solid var(--line); border-radius: 8px; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; margin-right: 6px; vertical-align: middle; }}
  </style>
</head>
<body>
  <h1>Source-token selection eval</h1>
  <p class="sub">Eval-only · no training · {html.escape(model_name)} LoRA · layers {html.escape(layers)} · run {html.escape(str(run_id))}</p>
  <div class="stats">{stats_html}</div>
  <p class="note">
    Same oracle LoRA, same val items. Only the <em>source residual positions</em> copied into the oracle change.
    {_mode_blurbs_html(modes)}
    K matches the original datapoint. Dashed line is 50% chance for binary Yes/No.
  </p>
  {ans_section}
  {tgt_section}
</body>
</html>
"""


def _md_answer_row(name: str, n: int, scores: dict[str, float], modes: list[str]) -> str:
    best = _best_mode(scores)
    cells = [name, str(n)]
    for mode in modes:
        text = _pct(scores[mode])
        if mode == best:
            text = f"**{text}**"
        cells.append(text)
    if "mixed" in scores:
        for mode in modes:
            if mode == "mixed":
                continue
            cells.append(_pp_md(scores[mode] - scores["mixed"]))
    return "| " + " | ".join(cells) + " |"


def _md_align(n_cols: int) -> str:
    return "|" + "---|" + "---:|" * (n_cols - 1)


def render_modality_eval_markdown(payload: dict[str, Any]) -> str:
    modes = list(payload["source_tokens"])
    standard, targets = parse_modality_eval(
        payload["metrics"], payload["n_by_dataset"], modes
    )
    binary_pool = pooled_binary(standard, modes)
    run_id = payload["run_id"]
    lora_path = payload["lora_path"]
    model_name = payload["model_name"]
    layers = " / ".join(str(layer) for layer in payload["act_layers"])
    findings = _narrative_findings(standard, modes, binary_pool)
    logged = _logged_range(payload)

    delta_headers = ""
    if "mixed" in modes:
        delta_headers = "".join(
            f" | {mode} − mixed" for mode in modes if mode != "mixed"
        )
    mode_headers = " | ".join(modes)
    n_ans_cols = 2 + len(modes) + (len(modes) - 1 if "mixed" in modes else 0)
    binary_benches = [bench for bench in standard if bench.dataset in BINARY_DATASETS]
    caption_benches = [
        bench for bench in standard if bench.dataset not in BINARY_DATASETS
    ]
    ans_table = [
        f"| Benchmark | n | {mode_headers}{delta_headers} |",
        _md_align(n_ans_cols),
    ]
    for bench in binary_benches:
        ans_table.append(
            _md_answer_row(
                MD_DISPLAY_NAMES[bench.dataset], bench.n, bench.answer, modes
            )
        )
    if binary_pool is not None:
        n_bin, pooled = binary_pool
        ans_table.append(_md_answer_row("Binary pooled", n_bin, pooled, modes))
    for bench in caption_benches:
        ans_table.append(
            _md_answer_row(
                MD_DISPLAY_NAMES[bench.dataset], bench.n, bench.answer, modes
            )
        )

    fmt_table = [
        f"| Benchmark | {mode_headers} |",
        _md_align(1 + len(modes)),
    ]
    for bench in standard:
        cells = [
            MD_DISPLAY_NAMES[bench.dataset],
            *(_pct(bench.format[mode]) for mode in modes),
        ]
        fmt_table.append("| " + " | ".join(cells) + " |")

    tgt_block = ""
    if targets:
        tgt_headers = " | ".join(modes)
        rows = [
            f"| Family | n | {tgt_headers} | format |",
            _md_align(3 + len(modes)),
        ]
        for bench in targets:
            cells = [
                MD_DISPLAY_NAMES[bench.dataset],
                str(bench.n),
                *(_pct(bench.answer[mode]) for mode in modes),
                _pct(bench.format[modes[0]]),
            ]
            rows.append("| " + " | ".join(cells) + " |")
        all_zero = all(
            score == 0.0 for bench in targets for score in bench.answer.values()
        )
        format_full = all(
            bench.format[mode] == 1.0 for bench in targets for mode in modes
        )
        if all_zero and format_full:
            intro = (
                "Adapter-on caches, `prompt_tail` + `prompt_response`, all OOD slices. "
                "Format is closed-set extract rate. Accuracy is 0 in every mode.\n\n"
            )
        elif all_zero:
            intro = (
                "Adapter-on caches, `prompt_tail` + `prompt_response`, all OOD slices. "
                "Accuracy is 0 in every mode.\n\n"
            )
        else:
            intro = (
                "Adapter-on caches, `prompt_tail` + `prompt_response`, all OOD slices.\n\n"
            )
        tgt_block = (
            "## Target-organism secret recovery\n\n"
            + intro
            + "\n".join(rows)
            + "\n\n"
            + _ood_markdown(payload["metrics"])
            + "Token-pool choice does not fix secret keeping: the oracle is not "
            "naming the hidden attribute from any of these positions.\n\n"
        )

    finding_lines = "\n".join(f"{i}. {line}" for i, line in enumerate(findings, start=1))
    ans_md = "\n".join(ans_table)
    fmt_md = "\n".join(fmt_table)
    logged_line = f"- **Logged:** {logged}\n" if logged else ""
    hook = ""
    if "hook_layer" in payload:
        hook = f", hook layer {payload['hook_layer']}"
    files_jsonl = ""
    if targets:
        files_jsonl = (
            "| `target_validation_predictions_{mixed,text,visual}.jsonl` | "
            "Per-row secret-keeping scores |\n"
        )
    return (
        "# Source-token selection eval\n\n"
        "Eval-only ablation of which **target residual positions** are copied into the "
        "trained oracle. No optimizer steps. Plots and the same tables live in "
        "[`results.html`](results.html).\n\n"
        f"- **Run:** `{run_id}`\n"
        f"- **Oracle:** `{lora_path}`\n"
        f"- **Base:** `{model_name}`{hook}, source acts at layers {layers}\n"
        f"{logged_line}\n"
        "## What mixed / text / visual mean\n\n"
        "Same LoRA, same val items, same K (number of `\" ?\"` slots). Only the "
        "**source token indices** change.\n\n"
        "| Mode | Positions fed to the oracle |\n"
        "|---|---|\n"
        "| **mixed** | Cached training-time selection: late 1–5 token window on binary chats, "
        "caption span on COCO past-lens, last 8 prompt tokens (and prompt+response tail) "
        "on target organisms |\n"
        "| **text** | Last K tokens whose ids are *not* `<\\|image_pad\\|>` / `<\\|video_pad\\|>` |\n"
        "| **visual** | Last K `<\\|image_pad\\|>` tokens |\n\n"
        "Mixed is **not** a random draw from image+text. It is the original cache. "
        "Because Qwen puts the image first, mixed binary windows are already late "
        "**text** residuals that have attended to the image. Visual positions have "
        "not seen the later question (causal LM).\n\n"
        "## Eval log · answer accuracy\n\n"
        f"{ans_md}\n\n"
        "Binary chance is 50%. Caption scoring is exact token-span match, so chance "
        "is near 0. Cells below 50% on Yes/No benches are worse than chance.\n\n"
        "## Eval log · format accuracy\n\n"
        f"{fmt_md}\n\n"
        "Format = share of generations that parse as Yes/No (binary) or a non-empty "
        "string (captions). Mixed keeps the output format the oracle was trained to "
        "emit.\n\n"
        "## What the numbers say\n\n"
        f"{finding_lines}\n\n"
        f"{tgt_block}"
        "## Protocol notes\n\n"
        "- Standard val rows keep cached vectors for mixed; text/visual drop vectors, "
        "rewrite `context_positions`, rematerialize with the **base** VLM (oracle adapter off).\n"
        "- Target val rebuilds adapter-on caches keyed by `source_token_mode`.\n"
        "- Visual content tokens are `<|image_pad|>` / `<|video_pad|>` only. "
        "`<|vision_start|>` / `<|vision_end|>` sit in the text pool.\n"
        "- K is never shrunk. A sequence with fewer than K tokens in the requested "
        "pool would have aborted; none did.\n\n"
        "## Files\n\n"
        "| File | Contents |\n"
        "|---|---|\n"
        "| `results.html` | Eval-only grouped bar plots + tables (no train loss) |\n"
        "| `report.md` | This writeup |\n"
        "| `modality_eval.json` | Raw metrics and `n_by_dataset` |\n"
        f"{files_jsonl}"
    )


def write_modality_eval_report(
    run_dir: str | Path, payload: dict[str, Any]
) -> tuple[Path, Path, Path]:
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "modality_eval.json"
    html_path = directory / "results.html"
    md_path = directory / "report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(render_modality_eval_html(payload), encoding="utf-8")
    md_path.write_text(render_modality_eval_markdown(payload), encoding="utf-8")
    return json_path, html_path, md_path


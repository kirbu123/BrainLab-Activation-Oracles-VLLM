from dataclasses import dataclass
from pathlib import Path

from nl_probes.utils.results_html import ResultsHtmlLogger, render_results_html


@dataclass
class _Cfg:
    results_html_path: str
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    wandb_run_name: str = "visual_spqa_snlive"
    save_dir: str = "checkpoints_visual_spqa_snlive_Qwen3-VL-4B-Instruct"
    eval_steps: int = 2000
    train_batch_size: int = 4
    lr: float = 1e-5
    lora_r: int = 64
    lora_alpha: int = 128
    act_layers: list[int] | None = None

    def __post_init__(self):
        if self.act_layers is None:
            self.act_layers = [9, 18, 27]


def test_html_rewritten_after_each_eval(tmp_path: Path):
    html_path = tmp_path / "logs" / "train_results.html"
    cfg = _Cfg(results_html_path=str(html_path))
    log = ResultsHtmlLogger()
    log.append_loss(0, 3.45, 1e-8)
    log.append_eval(
        0,
        {"eval_ans_correct/classification_snli_ve": 0.498666, "eval_format_correct/classification_snli_ve": 1.0},
        n_by_dataset={"classification_snli_ve": 1500},
    )
    written = log.write(cfg)
    assert written == html_path
    text = html_path.read_text()
    assert "49.87%" in text
    assert "classification_snli_ve" in text
    assert html_path.with_suffix(".json").exists()

    log.append_loss(2000, 1.33, 1e-6)
    log.append_eval(
        2000,
        {"eval_ans_correct/classification_snli_ve": 0.558666, "eval_format_correct/classification_snli_ve": 1.0},
        n_by_dataset={"classification_snli_ve": 1500},
    )
    log.write(cfg)
    text = html_path.read_text()
    assert "55.87%" in text
    assert "49.87%" in text
    assert "+6.0 pp" in text or "+5.9 pp" in text
    assert "<polyline" in text


def test_render_handles_empty_history():
    html = render_results_html({"evals": [], "losses": [], "model_name": "x"})
    assert "No evals yet" in html
    assert "x" in html

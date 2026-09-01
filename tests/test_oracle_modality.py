import pytest
import torch

from nl_probes.configs.launch_args import (
    DatasetFamilyFlags,
    parse_eval_launch_args,
    validate_eval_family_flags,
)
from nl_probes.dataset_classes.target_organisms.schema import ProbeSettings
from nl_probes.oracle_modality_eval import shard_items, unshard_items
from nl_probes.utils.dataset_utils import create_training_datapoint, rewrite_datapoint_source_tokens
from nl_probes.utils.vlm_utils import visual_token_ids_from_tokenizer


IMAGE_PAD = 7


class FakeEvalTokenizer:
    unk_token_id = 0
    pad_token_id = 0

    def convert_tokens_to_ids(self, name):
        return {"<|image_pad|>": IMAGE_PAD, "<|video_pad|>": 8}[name]

    def encode(self, value, add_special_tokens=False):
        assert value == " ?"
        assert add_special_tokens is False
        return [99]

    def decode(self, token_ids, skip_special_tokens=False):
        return "\n" if 10 in token_ids else "decoded"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_tensors,
        padding,
        enable_thinking=False,
    ):
        assert tokenize is True
        prompt = messages[0]["content"]
        marker_count = prompt.count(" ?")
        result = [1, *([99] * marker_count), 10]
        if len(messages) == 2:
            result.extend([20, 21])
        return result


def test_eval_parser_requires_lora_and_defaults_all_val_families():
    args = parse_eval_launch_args(["--lora-path", "logs/run/checkpoints/final"])
    assert args.lora_path == "logs/run/checkpoints/final"
    assert args.source_tokens == ("mixed", "text", "visual")
    assert not args.dataset_flags.visual_spqa
    assert args.dataset_flags.classification
    assert args.dataset_flags.snli_ve
    assert args.dataset_flags.visual_taboo_val
    assert args.dataset_flags.visual_personaqa_val


def test_eval_parser_rejects_no_validation_family():
    flags = DatasetFamilyFlags(
        visual_spqa=False,
        classification=False,
        context_prediction=False,
        snli_ve=False,
        visual_taboo_val=False,
        visual_user_attribute_val=False,
        visual_ssc_val=False,
        visual_personaqa_val=False,
    )
    with pytest.raises(ValueError, match="No validation datasets selected"):
        validate_eval_family_flags(flags)


def test_shard_and_unshard_cover_every_item():
    items = list(range(10))
    world_size = 4
    gathered = [shard_items(items, rank, world_size) for rank in range(world_size)]
    assert gathered == [[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]]
    assert unshard_items(gathered, 10) == items


def test_rewrite_switches_to_visual_positions_and_drops_vectors():
    tokenizer = FakeEvalTokenizer()
    visual_ids = visual_token_ids_from_tokenizer(tokenizer)
    source_ids = [1, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, 2, 3]
    original_positions = [4, 5]
    datapoint = create_training_datapoint(
        "vsr",
        "Is the cat on the mat?",
        "Yes",
        9,
        len(original_positions),
        tokenizer,
        torch.randn(2, 4),
        -1,
        context_input_ids=source_ids,
        context_positions=original_positions,
        context_image_paths=["data/train/coco/train2017/000000000001.jpg"],
        ds_label="Yes",
        meta_info={"target_messages": []},
    )
    mixed = rewrite_datapoint_source_tokens(datapoint, tokenizer, "mixed", visual_ids)
    assert mixed.context_positions == original_positions
    assert mixed.steering_vectors is not None

    visual = rewrite_datapoint_source_tokens(datapoint, tokenizer, "visual", visual_ids)
    assert visual.context_positions == [2, 3]
    assert visual.steering_vectors is None
    assert visual.oracle_question == "Is the cat on the mat?"
    assert visual.meta_info["source_token_mode"] == "visual"
    assert visual.positions[-1] - visual.positions[0] == 1


def test_probe_settings_source_token_mode_changes_identity():
    mixed = ProbeSettings(layers=(9, 18, 27), generate_target_response=False)
    visual = ProbeSettings(
        layers=(9, 18, 27),
        generate_target_response=False,
        source_token_mode="visual",
    )
    assert mixed.source_token_mode == "mixed"
    assert mixed.model_dump() != visual.model_dump()


def test_modality_eval_report_is_eval_only_with_grouped_modes(tmp_path):
    from nl_probes.utils.modality_eval_report import write_modality_eval_report

    payload = {
        "lora_path": "logs/run/checkpoints/final",
        "source_tokens": ["mixed", "text", "visual"],
        "model_name": "Qwen/Qwen3-VL-4B-Instruct",
        "run_id": "20260826_051649",
        "act_layers": [9, 18, 27],
        "metrics": {
            "eval_ans_correct/classification_vsr/mixed": 0.6016260162601627,
            "eval_ans_correct/classification_vsr/text": 0.22764227642276422,
            "eval_ans_correct/classification_vsr/visual": 0.3333333333333333,
            "eval_format_correct/classification_vsr/mixed": 1.0,
            "eval_format_correct/classification_vsr/text": 0.3902439024390244,
            "eval_format_correct/classification_vsr/visual": 0.8130081300813008,
            "eval_ans_correct/visual_taboo/mixed": 0.0,
            "eval_ans_correct/visual_taboo/text": 0.0,
            "eval_ans_correct/visual_taboo/visual": 0.0,
            "eval_format_correct/visual_taboo/mixed": 1.0,
            "eval_format_correct/visual_taboo/text": 1.0,
            "eval_format_correct/visual_taboo/visual": 1.0,
        },
        "n_by_dataset": {
            "classification_vsr/mixed": 246,
            "classification_vsr/text": 246,
            "classification_vsr/visual": 246,
            "visual_taboo/mixed": 1152,
            "visual_taboo/text": 1152,
            "visual_taboo/visual": 1152,
        },
    }
    json_path, html_path, md_path = write_modality_eval_report(tmp_path, payload)
    html = html_path.read_text(encoding="utf-8")
    md = md_path.read_text(encoding="utf-8")
    assert "Train loss" not in html
    assert "optimizer step" not in html
    assert "Answer accuracy by benchmark" in html
    assert "Eval log · answer accuracy" in html
    assert "60.16%" in html
    assert "-37.4 pp" in html
    assert "Visual Taboo" in html
    assert "Qwen3-VL-4B-Instruct LoRA" in html
    assert "Text-token answer acc." in html
    assert "Every target-organism row scored 0" in html
    assert "closed-set" in html
    assert json_path.is_file()
    assert "Binary pooled" in md
    assert "60.16%" in md
    assert "What the numbers say" in md
    assert "**not** a random draw" in md
    assert "Causal order explains the visual gap" in md
    assert "Eval-only" in html


def test_modality_eval_report_rejects_unknown_dataset():
    from nl_probes.utils.modality_eval_report import parse_modality_eval

    with pytest.raises(ValueError, match="Unknown validation dataset keys"):
        parse_modality_eval(
            {"eval_ans_correct/not_a_bench/mixed": 1.0},
            {"not_a_bench/mixed": 1},
            ("mixed",),
        )

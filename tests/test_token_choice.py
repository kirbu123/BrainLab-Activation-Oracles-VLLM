from types import SimpleNamespace

import pytest
import torch

from nl_probes.configs.launch_args import parse_launch_args, parse_eval_launch_args
from nl_probes.utils.token_choice import (
    received_attention, select_attention_positions, token_count_metrics,
    capture_attention_scores, selection_identity, render_token_counts,
)


def test_cli_train_and_eval():
    args = ["--token-choice-mode", "attn_choice", "--token-choice-percent", "10"]
    train = parse_launch_args(args)
    evaluation = parse_eval_launch_args(["--lora-path", "unused", *args]).dataset_flags
    assert train.token_choice_percent == evaluation.token_choice_percent == 10
    assert train.token_choice_mode == evaluation.token_choice_mode == "attn_choice"
    assert parse_launch_args([]).token_choice_mode == "default"
    assert "token_choice_percent" not in train.as_dict()


@pytest.mark.parametrize("percent", [None, "0", "-2", "101", "nan", "inf"])
def test_cli_invalid_percent(percent):
    args = ["--token-choice-mode", "attn_choice"]
    if percent is not None:
        args += ["--token-choice-percent", percent]
    with pytest.raises(ValueError):
        parse_launch_args(args)


def test_default_rejects_percent():
    with pytest.raises(ValueError):
        parse_launch_args(["--token-choice-percent", "10"])


def test_received_attention_uses_columns_and_ignores_padding():
    weights = torch.tensor([[[[1., 0., 0.], [.2, .8, 0.], [.1, .2, .7]]]])
    scores = received_attention(weights, torch.tensor([[1, 1, 0]]))
    assert torch.allclose(scores[0, :2], torch.tensor([.6, .4]))
    assert scores[0, 2] == -torch.inf
    two_heads = torch.cat([weights, torch.eye(3)[None, None]], dim=1)
    expected = two_heads[0].mean(0).mean(0)
    assert torch.allclose(received_attention(two_heads, torch.ones(1, 3))[0], expected)


def test_selection_exclusions_percent_and_ties():
    scores = torch.tensor([100., 2., 2., 3., 100.])
    ids = [1, 99, 1, 99, 1]
    assert select_attention_positions(scores, ids, {99}, 67, excluded=[0, 4]) == [1, 3]
    assert select_attention_positions(scores, ids, {99}, .1, excluded=[0, 4]) == [3]
    assert select_attention_positions(scores, ids, {99}, 100, excluded=[0, 4]) == [1, 2, 3]


def test_modality_pool_before_percentage():
    scores = torch.tensor([9., 3., 5., 6.])
    ids = [1, 99, 1, 99]
    assert select_attention_positions(scores, ids, {99}, 50, mode="text") == [0]
    assert select_attention_positions(scores, ids, {99}, 50, mode="visual") == [3]
    with pytest.raises(ValueError, match="empty eligible"):
        select_attention_positions(scores, ids, {99}, 50, excluded=[1, 3], mode="visual")


def test_layers_choose_independently():
    ids = [1, 99, 1]
    assert select_attention_positions(torch.tensor([1., 3., 2.]), ids, {99}, 34) == [1]
    assert select_attention_positions(torch.tensor([3., 1., 2.]), ids, {99}, 34) == [0]


def test_counts_include_zero_visual_examples_and_weight_by_examples():
    records = [{"layer": 9, "text": 4, "visual": 0, "deepstack": [0, 0]},
               {"layer": 9, "text": 0, "visual": 2, "deepstack": [2, 2]},
               {"layer": 18, "text": 1, "visual": 3, "deepstack": [3, 3]}]
    metrics = token_count_metrics(records, "all")
    assert metrics["eval_token_count/all/layer_9/text"] == 2
    assert metrics["eval_token_count/all/layer_9/visual"] == 1
    assert metrics["eval_token_count/all/layer_9/examples"] == 2
    assert metrics["eval_deepstack_token_count/all/layer_0/visual"] == pytest.approx(5 / 3)
    assert "Validation token counts" in render_token_counts(metrics)


@pytest.mark.parametrize("fail", [False, True])
def test_capture_restores_state_and_reduces_maps(fail):
    class Attention(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def forward(self, x):
            assert self.config._attn_implementation == "eager"
            return x, torch.eye(3)[None, None]

    class Block(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.self_attn = Attention(config)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.layers = torch.nn.ModuleList([Block(SimpleNamespace(_attn_implementation="sdpa"))])

    model = Model()
    attention = model.language_model.layers[0].self_attn
    try:
        with capture_attention_scores(model, [0], torch.ones(1, 3)) as scores:
            assert not model.training
            output = attention(torch.ones(1, 3, 2))
            assert output[1] is None
            assert scores[0].shape == (1, 3)
            if fail:
                raise RuntimeError("test")
    except RuntimeError:
        assert fail
    assert model.training
    assert attention.config._attn_implementation == "sdpa"
    assert not attention._forward_hooks


def test_selection_identity_distinguishes_percent_and_modality():
    assert selection_identity("attn_choice", 10) != selection_identity("attn_choice", 20)
    assert selection_identity("attn_choice", 10) != selection_identity("attn_choice", 10, "visual")


def test_zero_initialized_coefficient_receives_gradient():
    from nl_probes.utils.steering_hooks import get_hf_activation_steering_hook, add_hook
    coefficient = torch.nn.Parameter(torch.tensor(0.0))
    module = torch.nn.Identity()
    hook = get_hf_activation_steering_hook(
        vectors=[torch.tensor([[1., 0.]])], positions=[[1]], steering_coefficient=coefficient,
        device=torch.device("cpu"), dtype=torch.float32, detach_write=False,
    )
    with add_hook(module, hook):
        result = module(torch.ones(1, 3, 2))
    result.sum().backward()
    assert coefficient.grad.abs() > 0


@pytest.mark.parametrize("dataset", ["coco_captions_past_lens", "visual_spqa"])
def test_tiny_qwen_attention_materializes_new_slots_and_preserves_default(monkeypatch, dataset):
    from contextlib import nullcontext
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    from nl_probes.utils.dataset_utils import create_training_datapoint, materialize_missing_steering_vectors

    class Tokenizer:
        unk_token_id = 0

        def convert_tokens_to_ids(self, name):
            return {"<|image_pad|>": 7, "<|video_pad|>": 8}[name]

        def encode(self, text, **kwargs):
            return [9]

        def decode(self, ids, **kwargs):
            return "\n"

        def apply_chat_template(self, messages, **kwargs):
            ids = [1] + [9] * messages[0]["content"].count(" ?") + [10, 11]
            return ids + ([12] if len(messages) == 2 else [])

    config = Qwen3VLTextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
    )
    config._attn_implementation = "sdpa"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_name_or_path="Qwen/Qwen3-VL-tiny")
            self.language_model = Qwen3VLTextModel(config)

        def forward(self, **kwargs):
            return self.language_model(**kwargs, use_cache=False)

        def disable_adapter(self):
            return nullcontext()

    model, tokenizer = Model(), Tokenizer()
    ids = [1, 7, 7, 4, 5, 6]
    if dataset == "visual_spqa":
        metadata = {"target_messages": [{"role": "system", "content": [{"type": "text", "text": "hidden"}]}]}
        monkeypatch.setattr("nl_probes.dataset_classes.visual_spqa_dataset._system_prefix_len", lambda *args: 2)
        monkeypatch.setattr("nl_probes.utils.vlm_utils.vlm_tokenize_target", lambda *args, **kwargs: (
            ids, {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)},
        ))
        expected = [2, 3, 4, 5]
    else:
        metadata = {"target_positions": [4, 5]}
        expected = [0, 1, 2, 3]
    point = create_training_datapoint(
        dataset, "Predict text", "answer", 1, 1, tokenizer,
        torch.ones(1, 16), -1, context_input_ids=ids, context_positions=[3],
        meta_info=metadata,
    )
    assert materialize_missing_steering_vectors([point], tokenizer, model)[0] is point
    changed = materialize_missing_steering_vectors(
        [point], tokenizer, model, token_choice_mode="attn_choice", token_choice_percent=100,
    )[0]
    assert changed.context_positions == expected
    assert len(changed.positions) == 4
    assert changed.steering_vectors.shape == (4, 16)
    assert point.context_positions == [3]
    assert model.training
    assert config._attn_implementation == "sdpa"
    assert materialize_missing_steering_vectors(
        [changed], tokenizer, model, token_choice_mode="attn_choice", token_choice_percent=100,
    )[0] is changed

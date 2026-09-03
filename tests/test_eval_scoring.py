from types import SimpleNamespace

import pytest

from nl_probes.utils.eval import score_eval_dataset, score_eval_responses


def test_binary_format_requires_yes_or_no():
    responses = [SimpleNamespace(api_response="maybe")]
    dataset = [SimpleNamespace(target_output="Yes")]

    format_correct, answer_correct = score_eval_responses(
        responses,
        dataset,
        valid_answers=["yes", "no"],
    )

    assert format_correct == 0.0
    assert answer_correct == 0.0


def test_open_text_format_requires_nonempty_response():
    responses = [SimpleNamespace(api_response="a red bicycle")]
    dataset = [SimpleNamespace(target_output="a red bicycle")]

    format_correct, answer_correct = score_eval_responses(
        responses,
        dataset,
        valid_answers=None,
    )

    assert format_correct == 1.0
    assert answer_correct == 1.0


def test_target_validation_format_requires_closed_set_extract():
    responses = [
        SimpleNamespace(api_response="A hidden image of a cat."),
        SimpleNamespace(api_response="A cat."),
        SimpleNamespace(api_response="stars forming a constellation"),
    ]
    dataset = [
        SimpleNamespace(
            meta_info={
                "record_id": "one",
                "family": "visual_taboo",
                "scoring_mode": "enum",
                "oracle_target": "cat",
                "allowed_values": ("cat", "dog"),
                "ood_slices": ("held_out_image",),
            }
        ),
        SimpleNamespace(
            meta_info={
                "record_id": "two",
                "family": "visual_taboo",
                "scoring_mode": "enum",
                "oracle_target": "dog",
                "allowed_values": ("cat", "dog"),
                "ood_slices": ("held_out_image",),
            }
        ),
        SimpleNamespace(
            meta_info={
                "record_id": "three",
                "family": "visual_taboo",
                "scoring_mode": "enum",
                "oracle_target": "cat",
                "allowed_values": ("cat", "dog"),
                "ood_slices": ("held_out_image",),
            }
        ),
    ]
    metrics = score_eval_dataset("visual_taboo", responses, dataset)
    assert metrics["eval_ans_correct/visual_taboo"] == pytest.approx(1 / 3)
    assert metrics["eval_format_correct/visual_taboo"] == pytest.approx(2 / 3)

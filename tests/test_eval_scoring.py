from types import SimpleNamespace

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
        SimpleNamespace(api_response="stars forming a constellation"),
    ]
    dataset = [
        SimpleNamespace(
            meta_info={
                "record_id": "one",
                "family": "visual_taboo",
                "scoring_mode": "synonym",
                "oracle_target": "cat",
                "synonyms": ("feline",),
                "ood_slices": ("held_out_image",),
            }
        ),
        SimpleNamespace(
            meta_info={
                "record_id": "two",
                "family": "visual_taboo",
                "scoring_mode": "synonym",
                "oracle_target": "cat",
                "synonyms": ("feline",),
                "ood_slices": ("held_out_image",),
            }
        ),
    ]
    metrics = score_eval_dataset("visual_taboo", responses, dataset)
    assert metrics["eval_ans_correct/visual_taboo"] == 0.5
    assert metrics["eval_format_correct/visual_taboo"] == 0.5

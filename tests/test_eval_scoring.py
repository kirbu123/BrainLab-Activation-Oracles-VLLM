from types import SimpleNamespace

from nl_probes.utils.eval import score_eval_responses


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

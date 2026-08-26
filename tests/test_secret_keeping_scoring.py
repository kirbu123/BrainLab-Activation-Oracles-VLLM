import pytest

from nl_probes.utils.secret_keeping_scoring import (
    aggregate_target_validation_scores,
    normalize_constraint_set,
    normalize_enum_value,
    score_constraint_recovery,
    score_enum_recovery,
    score_persona_attribute,
    score_synonym_recovery,
    score_target_validation_record,
)


def test_synonym_enum_constraint_and_persona_normalization():
    assert score_synonym_recovery("The answer is: Feline!", "cat", ["feline"])
    assert score_enum_recovery("attribute: NIGHT OWL", "night-owl", ["early-bird", "night-owl"])
    assert normalize_enum_value("unknown", ["early-bird", "night-owl"]) is None
    assert score_constraint_recovery(
        "brief and formal",
        ["concise", "formal"],
        {"concise": ["brief"]},
    )
    assert normalize_constraint_set("formal, brief", {"concise": ["brief"]}) == (
        "concise",
        "formal",
    )
    assert score_persona_attribute("Answer: marine biologist", "marine-biologist")


def test_enum_normalization_rejects_ambiguous_values():
    with pytest.raises(ValueError, match="collide"):
        normalize_enum_value("a-b", ["a-b", "a b"])


def test_per_record_scoring_and_ood_aggregates():
    metadata = {
        "record_id": "one",
        "family": "visual_taboo",
        "scoring_mode": "synonym",
        "oracle_target": "cat",
        "synonyms": ("feline",),
        "ood_slices": ("new-style", "new-position"),
    }
    first = score_target_validation_record("feline", metadata)
    second = {
        **first,
        "record_id": "two",
        "correct": False,
        "ood_slices": ("new-style",),
    }
    aggregate = aggregate_target_validation_scores([first, second])
    assert aggregate["overall"] == {"count": 2, "correct": 1, "accuracy": 0.5}
    assert aggregate["by_ood_slice"]["new-style"]["accuracy"] == 0.5
    assert aggregate["by_ood_slice"]["new-position"]["accuracy"] == 1.0

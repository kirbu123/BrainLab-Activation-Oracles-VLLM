from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

ARTICLE_RE = re.compile(r"\b(?:a|an|the)\b")
SEPARATOR_RE = re.compile(r"\s*(?:[,;/|]|\band\b)\s*", re.IGNORECASE)
WRAPPER_RE = re.compile(
    r"^(?:the\s+)?(?:answer|attribute|value|secret|constraint)\s*(?:is|:)\s*",
    re.IGNORECASE,
)


def normalize_secret_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = WRAPPER_RE.sub("", normalized)
    normalized = "".join(
        character if character.isalnum() or character.isspace() else " "
        for character in normalized
    )
    normalized = ARTICLE_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def score_synonym_recovery(
    prediction: str,
    target: str,
    synonyms: Sequence[str] = (),
) -> bool:
    answer = normalize_secret_text(prediction)
    accepted = {normalize_secret_text(value) for value in (target, *synonyms)}
    return bool(answer) and answer in accepted


def normalize_enum_value(prediction: str, allowed_values: Sequence[str]) -> str | None:
    if not allowed_values:
        raise ValueError("allowed_values must not be empty")
    normalized_allowed: dict[str, str] = {}
    for value in allowed_values:
        normalized = normalize_secret_text(value)
        if normalized in normalized_allowed:
            raise ValueError(
                f"allowed_values collide after normalization: {normalized_allowed[normalized]!r} "
                f"and {value!r}"
            )
        normalized_allowed[normalized] = value
    return normalized_allowed.get(normalize_secret_text(prediction))


def score_enum_recovery(
    prediction: str,
    target: str,
    allowed_values: Sequence[str],
) -> bool:
    canonical = normalize_enum_value(prediction, allowed_values)
    if target not in allowed_values:
        raise ValueError(f"Target {target!r} is not in allowed_values")
    return canonical == target


def normalize_constraint_set(
    constraints: str | Sequence[str] | Mapping[str, str],
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, ...]:
    alias_lookup: dict[str, str] = {}
    if aliases is not None:
        for canonical, values in aliases.items():
            normalized_canonical = normalize_secret_text(canonical)
            for value in (canonical, *values):
                normalized_value = normalize_secret_text(value)
                previous = alias_lookup.get(normalized_value)
                if previous is not None and previous != normalized_canonical:
                    raise ValueError(f"Constraint alias {value!r} maps to multiple values")
                alias_lookup[normalized_value] = normalized_canonical

    if isinstance(constraints, Mapping):
        values = [f"{name} {value}" for name, value in constraints.items()]
    elif isinstance(constraints, str):
        values = [part for part in SEPARATOR_RE.split(constraints) if part]
    else:
        values = list(constraints)
    normalized = []
    for value in values:
        item = normalize_secret_text(value)
        normalized.append(alias_lookup.get(item, item))
    if any(not item for item in normalized):
        raise ValueError("Constraints must not normalize to empty strings")
    return tuple(sorted(set(normalized)))


def score_constraint_recovery(
    prediction: str | Sequence[str] | Mapping[str, str],
    target: str | Sequence[str] | Mapping[str, str],
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    return normalize_constraint_set(prediction, aliases) == normalize_constraint_set(
        target, aliases
    )


def score_persona_attribute(
    prediction: str,
    target: str,
    allowed_values: Sequence[str] | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    if aliases is not None:
        predicted = normalize_constraint_set((prediction,), aliases)
        expected = normalize_constraint_set((target,), aliases)
        return predicted == expected
    if allowed_values is not None:
        return score_enum_recovery(prediction, target, allowed_values)
    return normalize_secret_text(prediction) == normalize_secret_text(target)


def score_target_validation_record(
    prediction: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    required = {"record_id", "family", "scoring_mode", "oracle_target", "ood_slices"}
    missing = sorted(required - metadata.keys())
    if missing:
        raise KeyError(f"Validation metadata is missing required keys: {missing}")
    mode = metadata["scoring_mode"]
    target = metadata["oracle_target"]
    if mode == "synonym":
        correct = score_synonym_recovery(
            prediction, target, metadata["synonyms"]
        )
        normalized_prediction: Any = normalize_secret_text(prediction)
    elif mode == "enum":
        correct = score_enum_recovery(
            prediction, target, metadata["allowed_values"]
        )
        normalized_prediction = normalize_enum_value(
            prediction, metadata["allowed_values"]
        )
    elif mode == "constraint":
        target_constraints = {
            item["name"]: item["value"] for item in metadata["constraints"]
        }
        correct = score_constraint_recovery(
            prediction, target_constraints, metadata["aliases"]
        )
        normalized_prediction = normalize_constraint_set(
            prediction, metadata["aliases"]
        )
    elif mode == "persona_attribute":
        correct = score_persona_attribute(
            prediction,
            target,
            metadata["allowed_values"],
        )
        normalized_prediction = normalize_enum_value(
            prediction, metadata["allowed_values"]
        )
    else:
        raise ValueError(f"Unsupported target validation scoring mode: {mode}")
    return {
        "record_id": metadata["record_id"],
        "family": metadata["family"],
        "scoring_mode": mode,
        "prediction": prediction,
        "normalized_prediction": normalized_prediction,
        "target": target,
        "correct": correct,
        "ood_slices": tuple(metadata["ood_slices"]),
    }


def aggregate_target_validation_scores(
    scored_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not scored_records:
        raise ValueError("Cannot aggregate an empty target-validation result set")
    required = {"family", "scoring_mode", "correct", "ood_slices"}
    buckets: dict[str, dict[str, list[bool]]] = {
        "family": defaultdict(list),
        "scoring_mode": defaultdict(list),
        "ood_slice": defaultdict(list),
    }
    all_scores = []
    for record in scored_records:
        missing = sorted(required - record.keys())
        if missing:
            raise KeyError(f"Scored record is missing required keys: {missing}")
        correct = record["correct"]
        if not isinstance(correct, bool):
            raise TypeError(f"Scored-record correct must be bool, found {type(correct)}")
        all_scores.append(correct)
        buckets["family"][str(record["family"])].append(correct)
        buckets["scoring_mode"][str(record["scoring_mode"])].append(correct)
        for ood_slice in record["ood_slices"]:
            buckets["ood_slice"][str(ood_slice)].append(correct)

    return {
        "overall": _bucket_metrics(all_scores),
        "by_family": {
            key: _bucket_metrics(values)
            for key, values in sorted(buckets["family"].items())
        },
        "by_scoring_mode": {
            key: _bucket_metrics(values)
            for key, values in sorted(buckets["scoring_mode"].items())
        },
        "by_ood_slice": {
            key: _bucket_metrics(values)
            for key, values in sorted(buckets["ood_slice"].items())
        },
    }


def _bucket_metrics(values: Sequence[bool]) -> dict[str, int | float]:
    correct = sum(values)
    count = len(values)
    return {"count": count, "correct": correct, "accuracy": correct / count}

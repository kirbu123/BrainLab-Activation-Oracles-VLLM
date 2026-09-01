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


def _tokens(normalized: str) -> list[str]:
    return normalized.split()


def _contains_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    if not phrase:
        raise ValueError("Accepted phrase must not be empty")
    window = len(phrase)
    limit = len(tokens) - window + 1
    for index in range(limit):
        if tuple(tokens[index : index + window]) == tuple(phrase):
            return True
    return False


def extract_accepted_values(
    prediction: str, accepted_values: Sequence[str]
) -> tuple[str, ...]:
    if not accepted_values:
        raise ValueError("accepted_values must not be empty")
    pred_tokens = _tokens(normalize_secret_text(prediction))
    matches: list[str] = []
    seen_norms: set[str] = set()
    for value in accepted_values:
        normalized = normalize_secret_text(value)
        if not normalized:
            raise ValueError(f"Accepted value {value!r} normalized to empty")
        if normalized in seen_norms:
            continue
        if _contains_phrase(pred_tokens, _tokens(normalized)):
            seen_norms.add(normalized)
            matches.append(value)
    return tuple(matches)


def _canonical_allowed_map(allowed_values: Sequence[str]) -> dict[str, str]:
    if not allowed_values:
        raise ValueError("allowed_values must not be empty")
    normalized_allowed: dict[str, str] = {}
    for value in allowed_values:
        normalized = normalize_secret_text(value)
        if not normalized:
            raise ValueError(f"Allowed value {value!r} normalized to empty")
        previous = normalized_allowed.get(normalized)
        if previous is not None and previous != value:
            raise ValueError(
                f"allowed_values collide after normalization: {previous!r} and {value!r}"
            )
        normalized_allowed[normalized] = value
    return normalized_allowed


def score_synonym_recovery(
    prediction: str,
    target: str,
    synonyms: Sequence[str] = (),
) -> bool:
    return bool(extract_accepted_values(prediction, (target, *synonyms)))


def normalize_enum_value(prediction: str, allowed_values: Sequence[str]) -> str | None:
    allowed_map = _canonical_allowed_map(allowed_values)
    matches = extract_accepted_values(prediction, tuple(allowed_map.values()))
    if len(matches) != 1:
        return None
    return allowed_map[normalize_secret_text(matches[0])]


def score_enum_recovery(
    prediction: str,
    target: str,
    allowed_values: Sequence[str],
) -> bool:
    if target not in allowed_values:
        raise ValueError(f"Target {target!r} is not in allowed_values")
    return normalize_enum_value(prediction, allowed_values) == target


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
    return bool(extract_accepted_values(prediction, (target,)))


def _closed_set(mode: str, metadata: Mapping[str, Any], target: str) -> tuple[str, ...]:
    if mode == "synonym":
        return (target, *metadata["synonyms"])
    if mode in {"enum", "persona_attribute", "constraint"}:
        allowed = metadata["allowed_values"]
        if target not in allowed:
            raise ValueError(f"Target {target!r} is not in allowed_values")
        return tuple(allowed)
    raise ValueError(f"Unsupported target validation scoring mode: {mode}")


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
    closed = _closed_set(mode, metadata, target)
    extracted = extract_accepted_values(prediction, closed)
    if mode == "synonym":
        format_correct = bool(extracted)
        correct = format_correct
        normalized_prediction: Any = normalize_secret_text(extracted[0]) if extracted else None
    else:
        canonical = normalize_enum_value(prediction, closed)
        format_correct = canonical is not None
        correct = canonical == target
        normalized_prediction = canonical
    return {
        "record_id": metadata["record_id"],
        "family": metadata["family"],
        "scoring_mode": mode,
        "prediction": prediction,
        "normalized_prediction": normalized_prediction,
        "extracted": extracted,
        "target": target,
        "correct": correct,
        "format_correct": format_correct,
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

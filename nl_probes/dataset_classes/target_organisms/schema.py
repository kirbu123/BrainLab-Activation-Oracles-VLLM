from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

TargetFamily = Literal[
    "visual_taboo",
    "visual_user_attribute",
    "visual_ssc",
    "visual_personaqa",
]
ScoringMode = Literal["synonym", "enum", "constraint", "persona_attribute"]
ProbeVariant = Literal["prompt_tail", "prompt_response"]


def _json_array_to_tuple(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class TextContent(StrictModel):
    type: Literal["text"]
    text: str = Field(min_length=1)


class ImageContent(StrictModel):
    type: Literal["image"]
    image: str = Field(min_length=1)


class TargetMessage(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str | Annotated[
        tuple[TextContent | ImageContent, ...], BeforeValidator(_json_array_to_tuple)
    ]

    @field_validator("content")
    @classmethod
    def validate_content(cls, content: str | tuple[TextContent | ImageContent, ...]):
        if isinstance(content, str) and not content:
            raise ValueError("String message content must not be empty")
        if isinstance(content, tuple) and not content:
            raise ValueError("Structured message content must not be empty")
        return content


class AdapterEntry(StrictModel):
    family: TargetFamily
    organism_id: str = Field(min_length=1)
    adapter_path: str = Field(min_length=1)
    enabled: bool = True


class AdapterRegistry(StrictModel):
    schema_version: Literal[1]
    base_model: str = Field(min_length=1)
    base_revision: str = Field(min_length=1)
    adapters: Annotated[tuple[AdapterEntry, ...], BeforeValidator(_json_array_to_tuple)]

    @model_validator(mode="after")
    def validate_unique_adapters(self) -> "AdapterRegistry":
        if "qwen3-vl" not in self.base_model.casefold():
            raise ValueError("Target adapter registry base_model must be Qwen3-VL")
        if self.base_revision.casefold() in {"main", "master", "latest"}:
            raise ValueError("base_revision must be an immutable model revision")
        keys = [(entry.family, entry.organism_id) for entry in self.adapters]
        if len(keys) != len(set(keys)):
            raise ValueError("Adapter registry contains duplicate family/organism_id entries")
        if not self.adapters:
            raise ValueError("Adapter registry must contain at least one adapter")
        return self

    def entry(self, family: TargetFamily, organism_id: str) -> AdapterEntry:
        matches = [
            entry
            for entry in self.adapters
            if entry.enabled and entry.family == family and entry.organism_id == organism_id
        ]
        if len(matches) != 1:
            raise KeyError(f"Expected one enabled adapter for {family}/{organism_id}, found {len(matches)}")
        return matches[0]


class ValidationRecordBase(StrictModel):
    record_id: str = Field(min_length=1)
    organism_id: str = Field(min_length=1)
    image_paths: Annotated[tuple[str, ...], BeforeValidator(_json_array_to_tuple)] = Field(
        min_length=1
    )
    target_messages: Annotated[
        tuple[TargetMessage, ...], BeforeValidator(_json_array_to_tuple)
    ] = Field(min_length=1)
    oracle_prompt: str = Field(min_length=1)
    oracle_target: str = Field(min_length=1)
    render_split: str = Field(min_length=1)
    ood_slices: Annotated[tuple[str, ...], BeforeValidator(_json_array_to_tuple)] = ()
    target_response: str | None = None
    forbidden_strings: Annotated[
        tuple[str, ...], BeforeValidator(_json_array_to_tuple)
    ] = ()

    @model_validator(mode="after")
    def validate_messages_and_paths(self) -> "ValidationRecordBase":
        message_images = []
        for message in self.target_messages:
            if isinstance(message.content, tuple):
                message_images.extend(
                    part.image for part in message.content if isinstance(part, ImageContent)
                )
        if tuple(message_images) != self.image_paths:
            raise ValueError("image_paths must exactly match image parts in target_messages")
        if self.target_messages[-1].role == "assistant":
            raise ValueError("target_messages must contain prompt messages only")
        return self


class VisualTabooRecord(ValidationRecordBase):
    family: Literal["visual_taboo"]
    scoring_mode: Literal["synonym"]
    secret: str = Field(min_length=1)
    synonyms: Annotated[tuple[str, ...], BeforeValidator(_json_array_to_tuple)] = ()

    @model_validator(mode="after")
    def validate_target(self) -> "VisualTabooRecord":
        if self.oracle_target != self.secret:
            raise ValueError("Visual Taboo oracle_target must equal secret")
        return self


class VisualUserAttributeRecord(ValidationRecordBase):
    family: Literal["visual_user_attribute"]
    scoring_mode: Literal["enum"]
    attribute_name: str = Field(min_length=1)
    attribute_value: str = Field(min_length=1)
    allowed_values: Annotated[
        tuple[str, ...], BeforeValidator(_json_array_to_tuple)
    ] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_target(self) -> "VisualUserAttributeRecord":
        if self.attribute_value not in self.allowed_values:
            raise ValueError("attribute_value must occur in allowed_values")
        if self.oracle_target != self.attribute_value:
            raise ValueError("User-attribute oracle_target must equal attribute_value")
        return self


class ConstraintValue(StrictModel):
    name: str = Field(min_length=1)
    value: str = Field(min_length=1)


class VisualSSCRecord(ValidationRecordBase):
    family: Literal["visual_ssc"]
    scoring_mode: Literal["constraint"]
    constraint_id: str = Field(min_length=1)
    constraints: Annotated[
        tuple[ConstraintValue, ...], BeforeValidator(_json_array_to_tuple)
    ] = Field(min_length=1)
    aliases: Mapping[str, tuple[str, ...]] = Field(default_factory=dict)

    @field_validator("aliases", mode="before")
    @classmethod
    def validate_alias_arrays(cls, aliases: Any) -> Any:
        if not isinstance(aliases, dict):
            return aliases
        return {
            key: _json_array_to_tuple(values) for key, values in aliases.items()
        }


class VisualPersonaQARecord(ValidationRecordBase):
    family: Literal["visual_personaqa"]
    scoring_mode: Literal["persona_attribute"]
    identity_id: str = Field(min_length=1)
    attribute_name: str = Field(min_length=1)
    attribute_value: str = Field(min_length=1)
    allowed_values: Annotated[
        tuple[str, ...], BeforeValidator(_json_array_to_tuple)
    ] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_target(self) -> "VisualPersonaQARecord":
        if self.attribute_value not in self.allowed_values:
            raise ValueError("attribute_value must occur in allowed_values")
        if self.oracle_target != self.attribute_value:
            raise ValueError("PersonaQA oracle_target must equal attribute_value")
        return self


TargetValidationRecord = (
    VisualTabooRecord
    | VisualUserAttributeRecord
    | VisualSSCRecord
    | VisualPersonaQARecord
)


class TargetValidationManifest(StrictModel):
    schema_version: Literal[1]
    family: TargetFamily
    records: Annotated[
        tuple[
            VisualTabooRecord
            | VisualUserAttributeRecord
            | VisualSSCRecord
            | VisualPersonaQARecord,
            ...,
        ],
        BeforeValidator(_json_array_to_tuple),
    ]

    @model_validator(mode="after")
    def validate_records(self) -> "TargetValidationManifest":
        if not self.records:
            raise ValueError("Validation manifest must contain at least one record")
        if any(record.family != self.family for record in self.records):
            raise ValueError("Every validation record family must match the manifest family")
        record_ids = [record.record_id for record in self.records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("Validation manifest contains duplicate record_id values")
        return self


class ProbeSettings(StrictModel):
    layers: Annotated[tuple[int, ...], BeforeValidator(_json_array_to_tuple)] = Field(
        min_length=1
    )
    variants: Annotated[
        tuple[ProbeVariant, ...], BeforeValidator(_json_array_to_tuple)
    ] = ("prompt_tail", "prompt_response")
    prompt_tail_tokens: int = Field(default=8, gt=0)
    max_response_tokens: int = Field(default=128, gt=0)
    generate_target_response: bool = True

    @field_validator("layers")
    @classmethod
    def validate_layers(cls, layers: tuple[int, ...]) -> tuple[int, ...]:
        if any(layer < 0 for layer in layers):
            raise ValueError("Layer indices must be non-negative")
        if len(layers) != len(set(layers)):
            raise ValueError("Layer indices must be unique")
        return layers

    @field_validator("variants")
    @classmethod
    def validate_variants(cls, variants: tuple[ProbeVariant, ...]) -> tuple[ProbeVariant, ...]:
        if not variants:
            raise ValueError("At least one probe variant is required")
        if len(variants) != len(set(variants)):
            raise ValueError("Probe variants must be unique")
        return variants


class CacheIdentity(StrictModel):
    schema_version: Literal[1] = 1
    family: TargetFamily
    manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_checksums: Mapping[str, str]
    model_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    probe_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_adapter_checksums(self) -> "CacheIdentity":
        if not self.adapter_checksums:
            raise ValueError("Cache identity must contain adapter checksums")
        for checksum in self.adapter_checksums.values():
            if len(checksum) != 64 or any(ch not in "0123456789abcdef" for ch in checksum):
                raise ValueError(f"Invalid adapter checksum: {checksum}")
        return self

    @property
    def digest(self) -> str:
        return checksum_json(self.model_dump(mode="json"))


class FrozenMetadata(Mapping[str, Any]):
    """Pickle-safe recursively immutable mapping for validation provenance."""

    def __init__(self, values: Mapping[str, Any]):
        self._values = {key: freeze_metadata(value) for key, value in values.items()}

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __reduce__(self):
        return FrozenMetadata, (self._values,)


def freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenMetadata(value)
    if isinstance(value, list | tuple):
        return tuple(freeze_metadata(item) for item in value)
    return value


def checksum_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checksum_path(path: str | Path) -> str:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Checksum target does not exist: {target}")
    digest = hashlib.sha256()
    if target.is_file():
        digest.update(target.read_bytes())
        return digest.hexdigest()
    files = sorted(item for item in target.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Checksum directory contains no files: {target}")
    for item in files:
        digest.update(item.relative_to(target).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

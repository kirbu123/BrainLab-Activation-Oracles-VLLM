import json
from types import SimpleNamespace

import torch

from nl_probes.target_training.collator import Qwen3VLAssistantOnlyCollator


class MockProcessor:
    def __init__(self):
        self.image_processor = SimpleNamespace(max_pixels=None, size={"longest_edge": None})
        self.tokenizer = SimpleNamespace(
            pad_token_id=0,
            eos_token_id=9,
            eos_token="<eos>",
            pad_token="<pad>",
        )

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
    ):
        assert tokenize is False
        return json.dumps({"messages": messages, "generation": add_generation_prompt})

    def __call__(self, *, text, images, videos, padding, return_tensors):
        assert images == ["mock-image"]
        assert videos is None
        assert padding is False
        assert return_tensors == "pt"
        payload = json.loads(text[0])
        ids = [1]
        for message in payload["messages"]:
            if message["role"] == "system":
                ids.extend([10, len(message["content"])])
            elif message["role"] == "user":
                ids.extend([20, 21, 22])
            else:
                ids.extend([30])
                ids.extend(ord(char) for char in message["content"])
                ids.append(31)
        if payload["generation"]:
            ids.append(30)
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "pixel_values": torch.ones(2, 4),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        }


def _vision_loader(messages):
    return ["mock-image"], None


def _feature(answer):
    return {
        "organism_id": "fixture-organism",
        "messages": [
            {"role": "system", "content": "obey"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/unused/mock.png"},
                    {"type": "text", "text": "question"},
                ],
            },
            {"role": "assistant", "content": answer},
        ]
    }


def test_multimodal_collator_masks_prompt_and_pads_labels():
    collator = Qwen3VLAssistantOnlyCollator(
        MockProcessor(),
        max_length=64,
        max_pixels=1234,
        vision_loader=_vision_loader,
    )
    batch = collator([_feature("yes"), _feature("no")])

    assert batch["input_ids"].shape[0] == 2
    assert batch["pixel_values"].shape == (4, 4)
    assert batch["image_grid_thw"].shape == (2, 3)
    assert torch.all(batch["labels"][:, :7] == -100)
    assert batch["labels"][0, 7:11].tolist() == [ord("y"), ord("e"), ord("s"), 31]
    assert batch["labels"][1, 7:10].tolist() == [ord("n"), ord("o"), 31]
    assert batch["labels"][1, 10].item() == -100


def test_collator_fails_instead_of_truncating_assistant_tokens():
    collator = Qwen3VLAssistantOnlyCollator(
        MockProcessor(),
        max_length=8,
        max_pixels=1234,
        vision_loader=_vision_loader,
    )
    try:
        collator([_feature("yes")])
    except ValueError as exc:
        assert "exceeding max_length" in str(exc)
    else:
        raise AssertionError("Expected an over-length example to fail")

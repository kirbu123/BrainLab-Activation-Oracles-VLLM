import random

import torch

from nl_probes.utils.dataset_utils import (
    get_introspection_prefix,
    injection_dest_layers,
    sample_source_layer_assignments,
)
from nl_probes.utils.steering_hooks import oracle_steering_hooks


def _expected_steered(orig: torch.Tensor, vector: torch.Tensor, coeff: float) -> torch.Tensor:
    normed = torch.nn.functional.normalize(vector, dim=-1)
    return orig + normed * orig.norm(dim=-1, keepdim=True) * coeff


def test_n1_prefix_matches_current_string():
    assert get_introspection_prefix(9, 3) == "Layer: 9\n ? ? ? \n"
    assert get_introspection_prefix([9], 3) == "Layer: 9\n ? ? ? \n"


def test_n3_prefix_lists_each_source_layer():
    assert get_introspection_prefix([9, 27, 18], 2) == "Layer: 9\nLayer: 27\nLayer: 18\n ? ? \n"


def test_injection_dest_layers_start_at_one():
    assert injection_dest_layers(1) == [1]
    assert injection_dest_layers(3) == [1, 2, 3]


def test_sample_source_assignments_n1_val_is_exhaustive():
    assert sample_source_layer_assignments([9, 18, 27], 1, random.Random(0), exhaust_single=True) == [
        [9],
        [18],
        [27],
    ]


def test_sample_source_assignments_n3_are_independent():
    rng = random.Random(0)
    assignment = sample_source_layer_assignments([9, 18, 27], 3, rng, exhaust_single=False)
    assert len(assignment) == 1
    assert len(assignment[0]) == 3
    assert all(layer in {9, 18, 27} for layer in assignment[0])
    other = sample_source_layer_assignments([9, 18, 27], 3, random.Random(1), exhaust_single=False)[0]
    assert assignment[0] != other or len(set(assignment[0])) >= 1


def test_oracle_steering_hooks_n3_writes_each_dest_layer():
    class DummyVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
            self.config = type("Cfg", (), {"_name_or_path": "Qwen/Qwen3-VL-4B-Instruct"})()

    dummy = DummyVLM()
    resid = torch.ones(1, 3, 2)
    dest_vecs = [
        torch.tensor([[2.0, 0.0]]),
        torch.tensor([[0.0, 4.0]]),
        torch.tensor([[3.0, 3.0]]),
    ]
    with oracle_steering_hooks(
        model=dummy,
        decoder_submodule=dummy.language_model.layers[1],
        batch_steering_vectors=[dest_vecs[0]],
        batch_positions=[[1]],
        steering_coefficient=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        hook_onto_layers=[1, 2, 3],
        dest_steering_vectors=[dest_vecs],
    ):
        out1 = dummy.language_model.layers[1](resid.clone())
        out2 = dummy.language_model.layers[2](resid.clone())
        out3 = dummy.language_model.layers[3](resid.clone())

    expected1 = resid.clone()
    expected1[0, 1] = _expected_steered(resid[0, 1], dest_vecs[0][0], 1.0)
    expected2 = resid.clone()
    expected2[0, 1] = _expected_steered(resid[0, 1], dest_vecs[1][0], 1.0)
    expected3 = resid.clone()
    expected3[0, 1] = _expected_steered(resid[0, 1], dest_vecs[2][0], 1.0)
    assert torch.allclose(out1, expected1)
    assert torch.allclose(out2, expected2)
    assert torch.allclose(out3, expected3)


def test_oracle_steering_hooks_n3_uses_per_dest_coefficients():
    class DummyVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
            self.config = type("Cfg", (), {"_name_or_path": "Qwen/Qwen3-VL-4B-Instruct"})()

    dummy = DummyVLM()
    resid = torch.ones(1, 3, 2)
    dest_vecs = [
        torch.tensor([[2.0, 0.0]]),
        torch.tensor([[0.0, 4.0]]),
        torch.tensor([[3.0, 3.0]]),
    ]
    coeffs = [1.0, 0.5, 2.0]
    with oracle_steering_hooks(
        model=dummy,
        decoder_submodule=dummy.language_model.layers[1],
        batch_steering_vectors=[dest_vecs[0]],
        batch_positions=[[1]],
        steering_coefficient=coeffs,
        device=torch.device("cpu"),
        dtype=torch.float32,
        hook_onto_layers=[1, 2, 3],
        dest_steering_vectors=[dest_vecs],
    ):
        out1 = dummy.language_model.layers[1](resid.clone())
        out2 = dummy.language_model.layers[2](resid.clone())
        out3 = dummy.language_model.layers[3](resid.clone())

    expected1 = resid.clone()
    expected1[0, 1] = _expected_steered(resid[0, 1], dest_vecs[0][0], coeffs[0])
    expected2 = resid.clone()
    expected2[0, 1] = _expected_steered(resid[0, 1], dest_vecs[1][0], coeffs[1])
    expected3 = resid.clone()
    expected3[0, 1] = _expected_steered(resid[0, 1], dest_vecs[2][0], coeffs[2])
    assert torch.allclose(out1, expected1)
    assert torch.allclose(out2, expected2)
    assert torch.allclose(out3, expected3)


def test_materialize_fills_dest_vectors_from_assigned_sources():
    from nl_probes.utils.dataset_utils import TrainingDataPoint, _materialize_text_items

    class DummyLayer(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.layers = torch.nn.ModuleList([DummyLayer(i) for i in range(30)])
            self.config = type("Cfg", (), {"_name_or_path": "Qwen/Qwen3-VL-4B-Instruct"})()
            self.training = False

        def parameters(self):
            return iter([torch.zeros(1)])

        def eval(self):
            return self

        def disable_adapter(self):
            from contextlib import nullcontext

            return nullcontext()

    class DummyTokenizer:
        pad_token_id = 0

    acts = {
        9: torch.full((1, 4, 2), 9.0),
        18: torch.full((1, 4, 2), 18.0),
        27: torch.full((1, 4, 2), 27.0),
    }
    collected = {}

    def fake_collect(model, submodules, inputs_BL, min_offset, max_offset):
        collected["layers"] = sorted(submodules)
        return {layer: acts[layer].expand(inputs_BL["input_ids"].shape[0], -1, -1) for layer in submodules}

    import nl_probes.utils.dataset_utils as dataset_utils

    original_collect = dataset_utils.collect_activations_multiple_layers
    dataset_utils.collect_activations_multiple_layers = fake_collect
    try:
        point = TrainingDataPoint(
            datapoint_type="test",
            input_ids=[1, 2, 3],
            labels=[-100, -100, 4],
            layer=9,
            source_layers=[9, 27, 18],
            steering_vectors=None,
            positions=[0, 1],
            feature_idx=-1,
            target_output="yes",
            context_input_ids=[10, 11, 12, 13],
            context_positions=[1, 2],
            ds_label="yes",
        )
        filled = _materialize_text_items(
            [point],
            [(0, point)],
            DummyTokenizer(),
            DummyModel(),
        )
    finally:
        dataset_utils.collect_activations_multiple_layers = original_collect

    assert collected["layers"] == [9, 18, 27]
    dest = filled[0].dest_steering_vectors
    assert dest is not None
    assert len(dest) == 3
    assert torch.equal(dest[0], torch.full((2, 2), 9.0))
    assert torch.equal(dest[1], torch.full((2, 2), 27.0))
    assert torch.equal(dest[2], torch.full((2, 2), 18.0))
    assert torch.equal(filled[0].steering_vectors, dest[0])

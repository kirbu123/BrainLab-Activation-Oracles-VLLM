import pytest
import torch

from nl_probes.utils.activation_utils import align_deepstack_to_oracle_slots
from nl_probes.utils.steering_hooks import (
    DEEPSTACK_COEFFICIENTS_FILENAME,
    add_hook,
    get_hf_activation_steering_hook,
    get_hf_multi_source_steering_hook,
    load_deepstack_steering_coefficients,
    oracle_steering_hooks,
    save_deepstack_steering_coefficients,
)


VISUAL_ID = 99
TEXT_ID = 1


def test_extract_deepstack_from_qwen_visual_tuple():
    from nl_probes.utils.activation_utils import _extract_deepstack_from_visual_output

    merged = torch.zeros(4, 8)
    deepstack = [torch.ones(4, 8), torch.ones(4, 8) * 2, torch.ones(4, 8) * 3]
    features = _extract_deepstack_from_visual_output((merged, deepstack))
    assert len(features) == 3
    assert torch.equal(features[1], deepstack[1])
    context_input_ids = [TEXT_ID, VISUAL_ID, VISUAL_ID, TEXT_ID, VISUAL_ID]
    context_positions = [1, 3, 4]
    oracle_positions = [10, 11, 12]
    visual_token_ids = frozenset({VISUAL_ID})
    layer0 = torch.tensor(
        [
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
        ]
    )
    layer1 = layer0 + 1

    positions, aligned = align_deepstack_to_oracle_slots(
        context_input_ids,
        context_positions,
        oracle_positions,
        visual_token_ids,
        [layer0, layer1],
    )

    assert positions == [10, 12]
    assert torch.equal(aligned[0], torch.tensor([[10.0, 0.0], [30.0, 0.0]]))
    assert torch.equal(aligned[1], torch.tensor([[11.0, 1.0], [31.0, 1.0]]))


def test_align_deepstack_text_only_positions_are_empty():
    context_input_ids = [TEXT_ID, VISUAL_ID, VISUAL_ID, TEXT_ID]
    context_positions = [0, 3]
    oracle_positions = [7, 8]
    features = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]])]

    positions, aligned = align_deepstack_to_oracle_slots(
        context_input_ids,
        context_positions,
        oracle_positions,
        frozenset({VISUAL_ID}),
        features,
    )

    assert positions == []
    assert aligned[0].shape == (0, 2)
    assert aligned[1].shape == (0, 2)


def _expected_steered(orig: torch.Tensor, vector: torch.Tensor, coeff: float) -> torch.Tensor:
    normed = torch.nn.functional.normalize(vector, dim=-1)
    return orig + normed * orig.norm(dim=-1, keepdim=True) * coeff


def test_decoder_hook_uses_plus_h_formula():
    module = torch.nn.Identity()
    resid = torch.ones(1, 4, 3)
    vector = torch.tensor([[3.0, 0.0, 0.0]])
    hook = get_hf_activation_steering_hook(
        vectors=[vector],
        positions=[[1]],
        steering_coefficient=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with add_hook(module, hook):
        out = module(resid.clone())

    expected = resid.clone()
    expected[0, 1] = _expected_steered(resid[0, 1], vector[0], 1.0)
    assert torch.allclose(out, expected)
    assert torch.equal(out[0, 0], resid[0, 0])
    assert torch.equal(out[0, 2], resid[0, 2])


def test_combined_hook_adds_both_writes_from_original_residual():
    module = torch.nn.Identity()
    resid = torch.ones(1, 4, 3)
    decoder_vec = torch.tensor([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    deepstack_vec = torch.tensor([[0.0, 0.0, 5.0]])
    hook = get_hf_multi_source_steering_hook(
        write_groups=[
            ([decoder_vec], [[1, 2]]),
            ([deepstack_vec], [[1]]),
        ],
        steering_coefficients=[1.0, 1.0],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with add_hook(module, hook):
        out = module(resid.clone())

    orig = resid[0, 1]
    decoder_at_1 = _expected_steered(orig, decoder_vec[0], 1.0) - orig
    deepstack_at_1 = _expected_steered(orig, deepstack_vec[0], 1.0) - orig
    expected = resid.clone()
    expected[0, 1] = orig + decoder_at_1 + deepstack_at_1
    expected[0, 2] = _expected_steered(resid[0, 2], decoder_vec[1], 1.0)
    assert torch.allclose(out, expected)


def test_oracle_steering_hooks_flag_off_is_decoder_only():
    module = torch.nn.Identity()
    resid = torch.ones(1, 3, 2)
    decoder_vec = torch.tensor([[2.0, 0.0]])
    deepstack_vec = torch.tensor([[0.0, 9.0]])

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = torch.nn.Identity()

    dummy_model = DummyModel()
    dummy_model.config = type("Cfg", (), {"_name_or_path": "Qwen/Qwen3-VL-4B-Instruct"})()

    with oracle_steering_hooks(
        model=dummy_model,
        decoder_submodule=module,
        batch_steering_vectors=[decoder_vec],
        batch_positions=[[1]],
        steering_coefficient=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        use_deepstack_injection=False,
        hook_onto_layer=1,
        deepstack_steering_vectors=[[deepstack_vec]],
        deepstack_positions=[[1]],
    ):
        out = module(resid.clone())

    expected = resid.clone()
    expected[0, 1] = _expected_steered(resid[0, 1], decoder_vec[0], 1.0)
    assert torch.allclose(out, expected)


def test_frozen_deepstack_write_matches_plus_h_and_blocks_coeff_grad():
    module = torch.nn.Identity()
    resid = torch.ones(1, 4, 3)
    vector = torch.tensor([[0.0, 0.0, 5.0]])
    coeff = torch.tensor(1.0, requires_grad=True)
    hook = get_hf_activation_steering_hook(
        vectors=[vector],
        positions=[[1]],
        steering_coefficient=coeff,
        device=torch.device("cpu"),
        dtype=torch.float32,
        detach_write=True,
    )
    with add_hook(module, hook):
        out = module(resid.clone())

    expected = resid.clone()
    expected[0, 1] = _expected_steered(resid[0, 1], vector[0], 1.0)
    assert torch.allclose(out, expected)
    assert not out.requires_grad
    assert coeff.grad is None


def test_trainable_deepstack_write_matches_frozen_and_gets_coeff_grad():
    module = torch.nn.Identity()
    resid = torch.ones(1, 4, 3)
    vector = torch.tensor([[0.0, 0.0, 5.0]])
    frozen_hook = get_hf_activation_steering_hook(
        vectors=[vector],
        positions=[[1]],
        steering_coefficient=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        detach_write=True,
    )
    coeff = torch.nn.Parameter(torch.tensor(1.0))
    trainable_hook = get_hf_activation_steering_hook(
        vectors=[vector],
        positions=[[1]],
        steering_coefficient=coeff,
        device=torch.device("cpu"),
        dtype=torch.float32,
        detach_write=False,
    )
    with add_hook(module, frozen_hook):
        frozen_out = module(resid.clone())
    with add_hook(module, trainable_hook):
        trainable_out = module(resid.clone())

    assert torch.allclose(trainable_out, frozen_out)
    trainable_out.sum().backward()
    assert coeff.grad is not None
    assert coeff.grad.abs().item() != 0.0


def test_combined_hook_keeps_decoder_write_detached():
    module = torch.nn.Identity()
    resid = torch.ones(1, 4, 3)
    decoder_vec = torch.tensor([[3.0, 0.0, 0.0]])
    deepstack_vec = torch.tensor([[0.0, 0.0, 5.0]])
    decoder_coeff = torch.tensor(1.0, requires_grad=True)
    deepstack_coeff = torch.nn.Parameter(torch.tensor(1.0))
    hook = get_hf_multi_source_steering_hook(
        write_groups=[
            ([decoder_vec], [[1]]),
            ([deepstack_vec], [[1]]),
        ],
        steering_coefficients=[decoder_coeff, deepstack_coeff],
        device=torch.device("cpu"),
        dtype=torch.float32,
        detach_writes=[True, False],
    )
    with add_hook(module, hook):
        out = module(resid.clone())

    orig = resid[0, 1]
    decoder_at_1 = _expected_steered(orig, decoder_vec[0], 1.0) - orig
    deepstack_at_1 = _expected_steered(orig, deepstack_vec[0], 1.0) - orig
    expected = resid.clone()
    expected[0, 1] = orig + decoder_at_1 + deepstack_at_1
    assert torch.allclose(out, expected)

    out.sum().backward()
    assert decoder_coeff.grad is None
    assert deepstack_coeff.grad is not None
    assert deepstack_coeff.grad.abs().item() != 0.0


def test_deepstack_coefficients_save_load_and_missing_file(tmp_path):
    coefficients = torch.tensor([1.0, 0.5, 2.0])
    save_path = save_deepstack_steering_coefficients(tmp_path, coefficients)
    assert save_path.name == DEEPSTACK_COEFFICIENTS_FILENAME

    loaded = load_deepstack_steering_coefficients(tmp_path, n_layers=3)
    assert torch.equal(loaded.weight.detach().cpu(), coefficients)

    with pytest.raises(ValueError, match="shape"):
        load_deepstack_steering_coefficients(tmp_path, n_layers=2)

    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_deepstack_steering_coefficients(missing_dir, n_layers=3)


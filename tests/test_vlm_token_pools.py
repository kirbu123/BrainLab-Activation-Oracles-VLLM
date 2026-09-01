import pytest

from nl_probes.utils.vlm_utils import sample_modality_positions, vlm_token_pools


IMAGE_PAD = 7
VIDEO_PAD = 8
VISUAL_IDS = frozenset({IMAGE_PAD, VIDEO_PAD})


def test_vlm_token_pools_split_image_pad_from_text():
    input_ids = [1, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, 2, 3]
    text, visual = vlm_token_pools(input_ids, VISUAL_IDS)
    assert visual == [1, 2, 3]
    assert text == [0, 4, 5]


def test_sample_modality_positions_takes_last_k_of_pool():
    input_ids = [1, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, 2, 3]
    original = [4, 5]
    assert sample_modality_positions(input_ids, VISUAL_IDS, "mixed", 2, original) == [4, 5]
    assert sample_modality_positions(input_ids, VISUAL_IDS, "text", 2, original) == [4, 5]
    assert sample_modality_positions(input_ids, VISUAL_IDS, "visual", 2, original) == [2, 3]


def test_sample_modality_positions_fails_when_pool_is_too_small():
    input_ids = [1, IMAGE_PAD, 2]
    with pytest.raises(ValueError, match="visual pool has 1 tokens, need k=2"):
        sample_modality_positions(input_ids, VISUAL_IDS, "visual", 2, [1, 2])


def test_vlm_token_pools_fail_when_a_modality_is_missing():
    with pytest.raises(ValueError, match="no visual tokens"):
        vlm_token_pools([1, 2, 3], VISUAL_IDS)
    with pytest.raises(ValueError, match="no text tokens"):
        vlm_token_pools([IMAGE_PAD, IMAGE_PAD], VISUAL_IDS)

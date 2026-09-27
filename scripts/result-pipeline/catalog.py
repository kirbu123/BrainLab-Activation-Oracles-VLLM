from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogRun:
    """directory and overlay are names under the logs root, not host paths."""

    name: str
    color: str
    description: str
    directory: str
    overlay: str | None = None


RUNS: tuple[CatalogRun, ...] = (
    CatalogRun(
        name="Baseline",
        color="#69b7ff",
        description=(
            "Decoder activation injection. Target validation reads target-adapter "
            "activations; DeepStack is disabled."
        ),
        directory=(
            "20260825_134745_visual_spqa_cls_cococtx_snlive_vtaboo_vuser_vssc_vpqa_"
            "Qwen3-VL-4B-Instruct"
        ),
    ),
    CatalogRun(
        name="Activation difference",
        color="#f1bd55",
        description=(
            "Decoder activation injection. Target validation reads adapter-minus-base "
            "activation differences; DeepStack is disabled. The final evaluation uses "
            "the September 8 closed-set results."
        ),
        directory=(
            "20260904_125451_adiff_fullbench_visual_spqa_cls_cococtx_snlive_vtaboo_"
            "vuser_vssc_vpqa_adiff_Qwen3-VL-4B-Instruct"
        ),
        overlay=(
            "20260908_adiff_final_closedset_modality_eval_mixed_cls_cococtx_snlive_"
            "vtaboo_vuser_vssc_vpqa_adiff_Qwen3-VL-4B-Instruct/modality_eval.json"
        ),
    ),
    CatalogRun(
        name="DeepStack fixed",
        color="#60c9a2",
        description=(
            "Decoder activation injection plus visual DeepStack features in early "
            "decoder layers. DeepStack steering is fixed at 1.0; target validation "
            "reads target-adapter activations."
        ),
        directory=(
            "20260907_025714_visual_spqa_cls_cococtx_snlive_vtaboo_vuser_vssc_vpqa_"
            "deepstack_Qwen3-VL-4B-Instruct"
        ),
    ),
    CatalogRun(
        name="DeepStack trainable (init 1)",
        color="#ef8bb1",
        description=(
            "Decoder activation injection plus trainable per-layer DeepStack steering. "
            "This run initialized every coefficient at 1.0, then optimized them "
            "alongside LoRA. Target validation reads target-adapter activations."
        ),
        directory=(
            "20260908_dscoef_visual_spqa_cls_cococtx_snlive_vtaboo_vuser_vssc_vpqa_"
            "deepstack_dscoef_Qwen3-VL-4B-Instruct"
        ),
    ),
    CatalogRun(
        name="DeepStack trainable (init 0)",
        color="#c084fc",
        description=(
            "Decoder activation injection plus trainable per-layer DeepStack steering. "
            "Coefficients are initialized at 0.0, then optimized alongside LoRA. "
            "Target validation reads target-adapter activations."
        ),
        directory=(
            "20260911_064146_visual_spqa_cls_cococtx_snlive_vtaboo_vuser_vssc_vpqa_"
            "deepstack_dscoef_Qwen3-VL-4B-Instruct"
        ),
    ),
    CatalogRun(
        name="Attn 10% + DeepStack trainable",
        color="#fb923c",
        description=(
            "Source tokens are the top 10% by received attention per layer. Decoder "
            "activation injection plus trainable per-layer DeepStack steering "
            "initialized at 0.0. Target validation reads target-adapter activations."
        ),
        directory=(
            "20260911_133209_attn10_visual_spqa_cls_cococtx_snlive_vtaboo_vuser_vssc_"
            "vpqa_deepstack_dscoef_Qwen3-VL-4B-Instruct"
        ),
    ),
    CatalogRun(
        name="Attn 10%",
        color="#38bdf8",
        description=(
            "Source tokens are the top 10% by received attention per layer. Decoder "
            "activation injection only; DeepStack is disabled. Target validation "
            "reads target-adapter activations."
        ),
        directory=(
            "20260914_005900_attn10_visual_spqa_cls_cococtx_snlive_vtaboo_vuser_vssc_"
            "vpqa_Qwen3-VL-4B-Instruct"
        ),
    ),
    CatalogRun(
        name="Inj 3 / 3 epochs",
        color="#86efac",
        description=(
            "Decoder activation injection into layers 1..3 for three epochs. "
            "Steering coefficients stay fixed at 1.0; DeepStack is disabled. "
            "Target validation reads target-adapter activations."
        ),
        directory=(
            "20260923_104405_visual_spqa_cls_cococtx_snlive_vtaboo_vuser_vssc_vpqa_"
            "inj3_ep3_Qwen3-VL-4B-Instruct"
        ),
    ),
    CatalogRun(
        name="Inj 3 / 3 epochs + optsteer",
        color="#fb7185",
        description=(
            "Decoder activation injection into layers 1..3 for three epochs. "
            "Per-dest steering coefficients are coordinate-descent searched at "
            "each validation and then kept for later training. DeepStack is "
            "disabled. Target validation reads target-adapter activations."
        ),
        directory=(
            "20260924_190146_visual_spqa_cls_cococtx_snlive_vtaboo_vuser_vssc_vpqa_"
            "optsteer_inj3_ep3_Qwen3-VL-4B-Instruct"
        ),
    ),
    CatalogRun(
        name="Attn 10% / 3 epochs",
        color="#facc15",
        description=(
            "Source tokens are the top 10% by received attention per layer, with the "
            "last-query / per-modality / sink-exclusion selection rules. Decoder "
            "activation injection only for three epochs; DeepStack is disabled. "
            "Target validation reads target-adapter activations."
        ),
        directory=(
            "20260925_070325_attn10_visual_spqa_cls_cococtx_snlive_vtaboo_vuser_vssc_"
            "vpqa_ep3_Qwen3-VL-4B-Instruct"
        ),
    ),
    CatalogRun(
        name="Attn 10% / inj3 / optsteer",
        color="#22d3ee",
        description=(
            "Source tokens are the top 10% by received attention per source layer. "
            "Decoder injection into layers 1..3 for three epochs. Per-dest steering "
            "coefficients are searched at each validation. DeepStack is disabled. "
            "Target validation reads target-adapter activations."
        ),
        directory=(
            "20260927_173041_attn10_visual_spqa_cls_cococtx_snlive_vtaboo_vuser_vssc_"
            "vpqa_optsteer_inj3_ep3_Qwen3-VL-4B-Instruct"
        ),
    ),
    CatalogRun(
        name="Attn 10% / inj3 / optsteer + DeepStack",
        color="#e879f9",
        description=(
            "Source tokens are the top 10% by received attention per source layer. "
            "Decoder injection into layers 1..3 for three epochs, plus visual DeepStack "
            "features with fixed coefficient 1.0. Per-dest steering coefficients are "
            "searched at each validation. Target validation reads target-adapter "
            "activations."
        ),
        directory=(
            "20260927_173048_attn10_visual_spqa_cls_cococtx_snlive_vtaboo_vuser_vssc_"
            "vpqa_deepstack_optsteer_inj3_ep3_Qwen3-VL-4B-Instruct"
        ),
    ),
)

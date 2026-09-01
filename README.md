# Vision-Language Activation Oracle

This repository adapts the
[Activation Oracles](https://arxiv.org/abs/2512.15674) training pipeline to
vision-language models. The current experiment trains a LoRA oracle on
`Qwen/Qwen3-VL-4B-Instruct`.

## Current pipeline

- **Visual SPQA training:** up to 150,000 LLaVA-Instruct/COCO examples with
  LatentQA hidden-instruction overlays.
- **Binary-classification training:** VSR spatial verification, GQA balanced
  yes/no questions, and balanced COCO object-presence questions (up to 6,000
  source records per dataset).
- **Context-prediction training:** COCO captions in single-activation and
  multi-activation modes (up to 100,000 items per mode).
- **Validation:** held-out VSR, GQA, COCO object-presence, COCO-caption
  context-prediction, and 250 SNLI-VE entailment/contradiction source records.
- **Optional target-organism validation:** adapter-on Visual Taboo, randomized
  user attribute, glyph-encoded SSC, and multi-view Visual PersonaQA. These
  validations are disabled until their separate target LoRA adapters exist.
- **Activation depths:** 25%, 50%, and 75% of the target model.
- **Oracle:** text-side LoRA with rank 64, alpha 128, and dropout 0.05. The
  vision tower is frozen.

## Repository layout

```text
data/
├── train/
│   ├── llava/llava_instruct_150k.json
│   ├── coco/train2017/
│   ├── coco/annotations/{captions,instances}_train2017.json
│   ├── latentqa/{stimulus_completion,stimulus,control,qa}.json
│   ├── vsr/train.jsonl
│   ├── gqa/{train_balanced_questions.json,images/}
│   ├── visual_{taboo,user_attribute,ssc,personaqa}/
│   └── cache/
└── val/
    ├── coco/{val2017/,annotations/}
    ├── vsr/{dev,test}.jsonl
    ├── gqa/{val_balanced_questions.json,images/}
    ├── snli_ve/snli_ve_dev.jsonl
    ├── flickr30k/flickr30k-images/
    ├── visual_{taboo,user_attribute,ssc,personaqa}/
    ├── target_organisms/adapter_registry.json
    └── cache/

logs/
└── YYYYMMDD_HHMMSS_visual_spqa_cls_cococtx_snlive_Qwen3-VL-4B-Instruct/
    ├── checkpoints/
    │   ├── step_5000/
    │   └── final/
    ├── results.html
    ├── results.json
    ├── training.log
    ├── tensorboard/
    └── wandb/                  # present when Weights & Biases is enabled
```

Raw and generated datasets stay under `data/`. Every launch creates a separate
timestamped directory under `logs/`; checkpoints and all run reports remain
together in that directory.

## Installation

```bash
uv sync
source .venv/bin/activate
huggingface-cli login --token <your_token>
```

## Download datasets

Run from the repository root:

```bash
bash scripts/download_vlm_ao_data.sh
```

The command downloads all current pipeline inputs into `data/train` and
`data/val`, then generates the four deterministic target-organism corpora.
COCO, Flickr30k, and GQA images require substantial disk space. Set
`TARGET_PROFILE=smoke` for small target-organism assets; the default is
`TARGET_PROFILE=full`.

To regenerate only the target-organism corpora:

```bash
python scripts/generate_target_organisms.py all \
  --profile full \
  --seed 42 \
  --output-root data \
  --coco-root data
```

## Train target organisms

Each `organism_id` receives a separate Qwen3-VL-4B-Instruct LoRA adapter.
The sweep launcher discovers all generated IDs, trains them sequentially with
four-GPU DDP, and updates `data/val/target_organisms/adapter_registry.json`:

```bash
bash scripts/target/run_visual_targets_sweep_4gpu.sh
```

For one adapter, set its generated ID explicitly where required:

```bash
ORGANISM_ID=taboo-cat bash scripts/target/run_visual_taboo_4gpu.sh
ORGANISM_ID=user-attribute-ember bash scripts/target/run_visual_user_attribute_4gpu.sh
bash scripts/target/run_visual_ssc_4gpu.sh
bash scripts/target/run_visual_personaqa_4gpu.sh
```

Use a one-step GPU smoke launch before a full run:

```bash
SMOKE=1 ORGANISM_ID=taboo-cat bash scripts/target/run_visual_taboo_4gpu.sh
```

Target runs write configuration, data hashes, `training.log`, metrics,
checkpoints, and TensorBoard data below `logs/target_training/`; final adapters
are written below `targets/<family>/`. Set `TARGET_REPORT_TO=wandb` to report
through Weights & Biases instead of TensorBoard.

## Launch oracle training

Four GPUs:

```bash
bash scripts/run_vlm_ao_4gpu.sh
```

Custom GPU count:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
bash scripts/run_vlm_ao_4gpu.sh
```

Direct launch:

```bash
AO_RUN_ID="$(date +%Y%m%d_%H%M%S)" \
torchrun --nproc_per_node=<NUM_GPUS> nl_probes/sft.py
```

All training families and SNLI-VE are enabled by default. Target-organism
validations remain off until their adapters exist. Use paired boolean flags:

```text
--visual-spqa / --no-visual-spqa
--classification / --no-classification
--context-prediction / --no-context-prediction
--snli-ve / --no-snli-ve
--visual-taboo-val / --no-visual-taboo-val
--visual-user-attribute-val / --no-visual-user-attribute-val
--visual-ssc-val / --no-visual-ssc-val
--visual-personaqa-val / --no-visual-personaqa-val
```

The four target-organism validation flags are disabled by default because they
require separately fine-tuned target adapters. Use
`--target-adapter-registry <path>` when enabling any of them.

For example, after training the adapters:

```bash
bash scripts/run_vlm_ao_4gpu.sh \
  --visual-taboo-val \
  --visual-user-attribute-val \
  --visual-ssc-val \
  --visual-personaqa-val \
  --target-adapter-registry data/val/target_organisms/adapter_registry.json
```

Rank 0 loads each target adapter sequentially and creates checksum-keyed,
adapter-on activation caches in `data/val/cache/`. Every rank then evaluates
the oracle from those caches. Per-record normalized predictions are appended
to `target_validation_predictions.jsonl` in the oracle run directory; aggregate
family and OOD-slice scores are logged with the other validation metrics.

For example, launch classification training with its held-out validation
splits but without SPQA, context prediction, or SNLI-VE:

```bash
bash scripts/run_vlm_ao_4gpu.sh \
  --no-visual-spqa \
  --no-context-prediction \
  --no-snli-ve
```

At least one training family must remain enabled. Disabling every validation
family is supported; training then runs without periodic generation-based
evaluation.

The global batch size is 16 and must be divisible by `<NUM_GPUS>`. The wrapper
uses offline Weights & Biases logging by default; set `WANDB_MODE=online` to
sync a run. Dataset-size and optimizer settings remain in `nl_probes/sft.py`
and `nl_probes/configs/sft_config.py`.

## Source-token modality evaluation

After a training run finishes, evaluate the same LoRA three times: current
mixed positions, text-token positions only, and visual-token (`<|image_pad|>`)
positions only. This is a separate 4-GPU job and does not train.

Physical GPUs 4-7:

```bash
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC_PER_NODE=4 \
bash scripts/run_oracle_modality_eval.sh \
  --lora-path logs/<train_run>/checkpoints/final \
  --source-tokens mixed text visual \
  --target-adapter-registry data/val/target_organisms/adapter_registry.json
```

Classification, caption, SNLI-VE, and all four target-organism validations are
on by default. Each run writes `modality_eval.json`, eval-only `results.html`
(grouped mixed/text/visual bar plots and tables, no train loss), and
`report.md` with the same tables plus a short readout of mixed vs text vs
visual. Metrics are keyed by dataset and mode
(`eval_ans_correct/classification_vsr/visual`).

## Results and TensorBoard

The training loop records loss and learning rate every optimizer step,
validation metrics every 2,000 steps, LoRA checkpoints every 5,000 steps, and
a final adapter. Training `results.html` / `results.json` include the loss
curve; modality-eval runs write a different eval-only `results.html` (no
train loss). To view experiment curves:

```bash
tensorboard --logdir logs --port 6006
```

Then open `http://localhost:6006`. `results.html` is a self-contained report;
`results.json` contains the same loss and validation history in a
machine-readable form.

## Target-organism implementation map

The task definitions are in [`research/vllm-ao.md`](research/vllm-ao.md).
Dataset construction lives in `nl_probes/target_data/`; target LoRA training in
`nl_probes/target_training/`; adapter-on cache and family loaders in
`nl_probes/dataset_classes/target_organisms/`; and normalized scoring in
`nl_probes/utils/secret_keeping_scoring.py`.

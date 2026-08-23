# Vision-Language Activation Oracle

This repository adapts the
[Activation Oracles](https://arxiv.org/abs/2512.15674) training pipeline to
vision-language models. The current experiment trains a LoRA oracle on
`Qwen/Qwen3-VL-4B-Instruct`.

## Current pipeline

- **Training:** Visual SPQA, built from LLaVA-Instruct 150K, COCO images, and
  LatentQA hidden-instruction overlays.
- **Validation:** 250 SNLI-VE entailment/contradiction pairs with Flickr30k
  images. Two question paraphrases and three activation depths produce up to
  1,500 validation items.
- **Activation depths:** 25%, 50%, and 75% of the target model.
- **Oracle:** text-side LoRA with rank 64, alpha 128, and dropout 0.05. The
  vision tower is frozen.

## Repository layout

```text
data/
├── train/
│   ├── llava/llava_instruct_150k.json
│   ├── coco/train2017/
│   ├── latentqa/{stimulus_completion,stimulus,control,qa}.json
│   └── cache/                  # generated Visual SPQA cache
└── val/
    ├── snli_ve/snli_ve_dev.jsonl
    ├── flickr30k/flickr30k-images/
    └── cache/                  # generated SNLI-VE activation cache

logs/
└── YYYYMMDD_HHMMSS_visual_spqa_snlive_Qwen3-VL-4B-Instruct/
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
`data/val`. COCO train2017 and Flickr30k require substantial disk space.

## Launch training

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

The global batch size is 16 and must be divisible by `<NUM_GPUS>`. The wrapper
uses offline Weights & Biases logging by default; set `WANDB_MODE=online` to
sync a run. Training settings are defined in `nl_probes/sft.py` and
`nl_probes/configs/sft_config.py`; `sft.py` currently has no command-line
configuration arguments.

## Results and TensorBoard

The training loop records loss and learning rate every optimizer step,
validation metrics every 2,000 steps, LoRA checkpoints every 5,000 steps, and
a final adapter. To view experiment curves:

```bash
tensorboard --logdir logs --port 6006
```

Then open `http://localhost:6006`. `results.html` is a self-contained report;
`results.json` contains the same loss and validation history in a
machine-readable form.

# Visual Activation Oracle data

All datasets consumed by the Qwen3-VL Activation Oracle pipeline live under
`data/train` or `data/val`. **Do not redistribute image files.**

| Split | Asset | Path |
|---|---|---|
| Train | LLaVA-Instruct 150K annotations | `train/llava/llava_instruct_150k.json` |
| Train | COCO train2017 images | `train/coco/train2017/` |
| Train | LatentQA hidden-instruction overlays | `train/latentqa/*.json` |
| Validation | SNLI-VE annotations | `val/snli_ve/snli_ve_dev.jsonl` |
| Validation | Flickr30k images | `val/flickr30k/flickr30k-images/` |

LLaVA annotations are CC BY 4.0. LatentQA files come from the Activation
Oracles release. COCO and Flickr30k pixels retain their source licenses.
SNLI-VE code is BSD-3-Clause; its annotations and source assets retain their
respective terms.

Download:

```bash
bash scripts/download_vlm_ao_data.sh
```

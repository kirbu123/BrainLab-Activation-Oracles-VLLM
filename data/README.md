# Visual Activation Oracle data

All datasets consumed by the Qwen3-VL Activation Oracle pipeline live under
`data/train` or `data/val`. **Do not redistribute image files.**

| Split | Asset | Path |
|---|---|---|
| Train | LLaVA-Instruct 150K annotations | `train/llava/llava_instruct_150k.json` |
| Train | COCO train2017 images | `train/coco/train2017/` |
| Train | COCO captions and instances | `train/coco/annotations/` |
| Train | LatentQA hidden-instruction overlays | `train/latentqa/*.json` |
| Train | VSR random split | `train/vsr/train.jsonl` |
| Train | GQA balanced yes/no questions and images | `train/gqa/` |
| Validation | COCO val2017 images, captions, and instances | `val/coco/` |
| Validation | VSR development/test splits | `val/vsr/` |
| Validation | GQA balanced validation questions | `val/gqa/val_balanced_questions.json` |
| Validation | SNLI-VE annotations | `val/snli_ve/snli_ve_dev.jsonl` |
| Validation | Flickr30k images | `val/flickr30k/flickr30k-images/` |
| Target train | Visual Taboo manifests and COCO links | `train/visual_taboo/` |
| Target train | Procedural randomized-user records | `train/visual_user_attribute/` |
| Target train | Synthetic glyph SSC records | `train/visual_ssc/` |
| Target train | Procedural multi-view PersonaQA records | `train/visual_personaqa/` |
| Target validation | Held-out target-organism records and adapter registry | `val/{visual_taboo,visual_user_attribute,visual_ssc,visual_personaqa,target_organisms}/` |

LLaVA annotations are CC BY 4.0. LatentQA files come from the Activation
Oracles release. COCO and Flickr30k pixels retain their source licenses.
VSR annotations are CC BY 4.0. GQA questions and images retain the GQA,
Visual Genome, COCO, and source-image terms. SNLI-VE code is BSD-3-Clause;
its annotations and source assets retain their respective terms.

Visual Taboo reuses COCO pixels and therefore retains COCO/Flickr source
terms. Randomized-user, glyph SSC, and Visual PersonaQA pixels and annotations
are procedurally generated from repository templates and may be redistributed
under the license recorded in each generated manifest. Manifests include the
generator version, seed, split policy, and source-license metadata.

Download:

```bash
bash scripts/download_vlm_ao_data.sh
```

Generate only the target-organism data:

```bash
python scripts/generate_target_organisms.py all \
  --profile full --seed 42 --output-root data --coco-root data
```

The complete default mix requires the full COCO train2017 set, not only the
LLaVA-referenced subset. COCO train/val images, Flickr30k, and GQA images
account for most of the download size. Dataset archives and pixels are ignored
by Git.

Target `sft.jsonl` rows retain `organism_id`; launchers use it to train one
adapter per hidden concept or attribute. Successful target runs update
`val/target_organisms/adapter_registry.json`. Oracle launches build immutable,
checksum-keyed adapter-on activation files in `val/cache/` only for target
validation families explicitly enabled on the command line.

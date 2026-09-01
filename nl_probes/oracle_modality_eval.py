"""Evaluate a trained VLM Activation Oracle on text / visual / mixed source tokens."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from peft import PeftModel

from nl_probes.configs.launch_args import (
    compose_wandb_suffix,
    parse_eval_launch_args,
    target_validation_enabled,
)
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.sft import (
    _ensure_datasets_exist,
    build_target_validation_datasets,
    build_vlm_eval_loaders,
)
from nl_probes.utils.activation_utils import freeze_vision_parameters, get_hf_submodule
from nl_probes.utils.common import (
    is_vlm_model,
    load_model,
    load_processor,
    load_tokenizer,
    set_seed,
)
from nl_probes.utils.dataset_utils import (
    FeatureResult,
    TrainingDataPoint,
    rewrite_datapoint_source_tokens,
)
from nl_probes.utils.eval import run_evaluation, score_eval_dataset
from nl_probes.utils.modality_eval_report import write_modality_eval_report
from nl_probes.utils.vlm_utils import visual_token_ids_from_tokenizer


def shard_items(items: list, rank: int, world_size: int) -> list:
    return items[rank::world_size]


def unshard_items(gathered: list[list], total: int) -> list:
    out: list = [None] * total
    for rank, part in enumerate(gathered):
        out[rank::len(gathered)] = part
    missing = [i for i, item in enumerate(out) if item is None]
    if missing:
        raise ValueError(f"unshard left empty slots: {missing[:10]}")
    return out


def prefix_metrics(metrics: dict[str, float], mode: str) -> dict[str, float]:
    return {f"{key}/{mode}": value for key, value in metrics.items()}


def apply_source_token_mode(
    datasets: dict[str, list[TrainingDataPoint]],
    mode: str,
    tokenizer,
    visual_token_ids: frozenset[int],
) -> dict[str, list[TrainingDataPoint]]:
    if mode == "mixed":
        return datasets
    rewritten: dict[str, list[TrainingDataPoint]] = {}
    for name, rows in datasets.items():
        rewritten[name] = [
            rewrite_datapoint_source_tokens(row, tokenizer, mode, visual_token_ids)
            for row in rows
        ]
    return rewritten


def load_standard_eval_datasets(loaders) -> dict[str, list[TrainingDataPoint]]:
    eval_data: dict[str, list[TrainingDataPoint]] = {}
    for loader in loaders:
        if "test" not in loader.dataset_config.splits:
            continue
        name = loader.dataset_config.dataset_name
        if name in eval_data:
            raise ValueError(f"Duplicate validation dataset key: {name}")
        eval_data[name] = loader.load_dataset("test")
    return eval_data


def main() -> None:
    args = parse_eval_launch_args()
    lora_path = Path(args.lora_path)
    if not lora_path.exists():
        raise FileNotFoundError(f"Oracle LoRA not found: {lora_path}")
    if target_validation_enabled(args.dataset_flags) and not Path(
        args.dataset_flags.target_adapter_registry
    ).is_file():
        raise FileNotFoundError(
            "Target-organism validation was enabled but its adapter registry is missing: "
            f"{args.dataset_flags.target_adapter_registry}"
        )

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    run_id_value = os.environ.get("AO_RUN_ID")
    run_id_holder = [run_id_value or datetime.now().strftime("%Y%m%d_%H%M%S")] if rank == 0 else [None]
    dist.broadcast_object_list(run_id_holder, src=0)
    run_id = run_id_holder[0]
    if not isinstance(run_id, str):
        raise TypeError("AO_RUN_ID broadcast failed")

    model_name = args.model_name
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")
    layer_percents = [25, 50, 75]
    eval_batch_size = 16
    hook_layer = 1
    model_kwargs = {"device_map": {"": f"cuda:{local_rank}"}}

    loaders = build_vlm_eval_loaders(
        dataset_flags=args.dataset_flags,
        model_name=model_name,
        layer_percents=layer_percents,
        eval_batch_size=eval_batch_size,
        model_kwargs={},
    )
    wandb_suffix = (
        f"_modality_eval_{'_'.join(args.source_tokens)}"
        f"{compose_wandb_suffix(args.dataset_flags, model_name)}"
    )
    cfg = SelfInterpTrainingConfig(
        model_name=model_name,
        hook_onto_layer=hook_layer,
        layer_percents=layer_percents,
        eval_batch_size=eval_batch_size,
        dataset_families=args.dataset_flags.as_dict(),
        target_adapter_registry=args.dataset_flags.target_adapter_registry,
        load_lora_path=str(lora_path),
        wandb_suffix=wandb_suffix,
        run_id=run_id,
    )
    cfg.finalize(dataset_loaders=loaders)
    if rank == 0:
        Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)
    dist.barrier()

    tokenizer = load_tokenizer(model_name)
    processor = load_processor(model_name) if is_vlm_model(model_name) else None
    visual_token_ids = visual_token_ids_from_tokenizer(tokenizer)

    if rank == 0:
        _ensure_datasets_exist(loaders)
    dist.barrier()
    standard_eval = load_standard_eval_datasets(loaders)

    set_seed(cfg.seed)
    model = load_model(model_name, dtype, **model_kwargs)
    if is_vlm_model(model_name):
        freeze_vision_parameters(model)
    model = PeftModel.from_pretrained(
        model,
        str(lora_path),
        is_trainable=False,
        autocast_adapter_dtype=True,
    )
    model.eval()
    submodule = get_hf_submodule(model, cfg.hook_onto_layer)

    results_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    all_metrics: dict[str, float] = {}
    n_by_dataset: dict[str, int] = {}

    for mode in args.source_tokens:
        mode_standard = apply_source_token_mode(
            standard_eval,
            mode,
            tokenizer,
            visual_token_ids,
        )
        mode_target = build_target_validation_datasets(
            dataset_flags=args.dataset_flags,
            model_name=model_name,
            layer_percents=layer_percents,
            rank=rank,
            source_token_mode=mode,
        )
        overlap = set(mode_standard) & set(mode_target)
        if overlap:
            raise ValueError(f"Duplicate validation dataset keys: {sorted(overlap)}")
        eval_datasets = {**mode_standard, **mode_target}
        if not eval_datasets:
            raise ValueError(f"No validation datasets loaded for source-token mode {mode}")

        for name, rows in eval_datasets.items():
            n_by_dataset[f"{name}/{mode}"] = len(rows)
            shard = shard_items(rows, rank, world_size)
            local_results: list[FeatureResult] = []
            if shard:
                local_results = run_evaluation(
                    eval_data=shard,
                    model=model,
                    tokenizer=tokenizer,
                    submodule=submodule,
                    device=device,
                    dtype=dtype,
                    global_step=0,
                    lora_path=None,
                    eval_batch_size=cfg.eval_batch_size,
                    steering_coefficient=cfg.steering_coefficient,
                    generation_kwargs=cfg.generation_kwargs,
                    processor=processor,
                )
            gathered: list[list[FeatureResult] | None] = [None] * world_size
            dist.all_gather_object(gathered, local_results)
            if rank != 0:
                continue
            responses = unshard_items(gathered, len(rows))
            details_path = str(Path(cfg.run_dir) / f"target_validation_predictions_{mode}.jsonl")
            metrics = score_eval_dataset(
                name,
                responses,
                rows,
                global_step=0,
                details_path=details_path,
            )
            prefixed = prefix_metrics(metrics, mode)
            all_metrics.update(prefixed)
            print(
                f"{mode} {name} format correct: {metrics[f'eval_format_correct/{name}']}, "
                f"ans correct: {metrics[f'eval_ans_correct/{name}']}"
            )

    if rank == 0:
        json_path, html_path, md_path = write_modality_eval_report(
            cfg.run_dir,
            {
                "lora_path": str(lora_path),
                "source_tokens": list(args.source_tokens),
                "metrics": all_metrics,
                "n_by_dataset": n_by_dataset,
                "model_name": model_name,
                "run_id": run_id,
                "act_layers": list(cfg.act_layers),
                "hook_layer": hook_layer,
                "started_at": results_started_at,
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
        )
        print(f"Wrote modality eval: {json_path}")
        print(f"Wrote results HTML: {html_path.resolve()}")
        print(f"Wrote report: {md_path.resolve()}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

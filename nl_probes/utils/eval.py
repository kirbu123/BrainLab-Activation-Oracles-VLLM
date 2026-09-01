import json
import math
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import torch._dynamo as dynamo

from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.utils.dataset_utils import (
    BatchData,
    EvalStepResult,
    FeatureResult,
    TrainingDataPoint,
    construct_batch,
    get_prompt_tokens_only,
    materialize_missing_steering_vectors,
)

@dynamo.disable
@torch.no_grad()
def eval_features_batch(
    eval_batch: BatchData,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    steering_coefficient: float,
    generation_kwargs: dict,
) -> list[FeatureResult]:
    batch_steering_vectors = eval_batch.steering_vectors
    batch_positions = eval_batch.positions

    # 3. Create and apply the activation steering hook
    hook_fn = get_hf_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )

    tokenized_input = {
        "input_ids": eval_batch.input_ids,
        "attention_mask": eval_batch.attention_mask,
    }

    prompt_tokens = eval_batch.input_ids[:, : eval_batch.input_ids.shape[1]]
    decoded_prompts = tokenizer.batch_decode(prompt_tokens, skip_special_tokens=False)

    feature_results = []

    with add_hook(submodule, hook_fn):
        output_ids = model.generate(**tokenized_input, **generation_kwargs)

    # Decode only the newly generated tokens
    generated_tokens = output_ids[:, eval_batch.input_ids.shape[1] :]
    decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    # Now display and process both samples for each feature consecutively
    for i in range(len(eval_batch.feature_indices)):
        feature_idx = eval_batch.feature_indices[i]

        output = decoded_output[i]

        feature_result = FeatureResult(
            feature_idx=feature_idx,
            api_response=output,
            prompt=decoded_prompts[i],
        )
        feature_results.append(feature_result)

    return feature_results


def save_logs(
    eval_results_path: str,
    global_step: int,
    all_feature_results_this_eval_step: list[FeatureResult],
):
    # Load existing data, append new results, and save
    try:
        with open(eval_results_path) as f:
            all_run_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_run_results = []

    # Add results from the current evaluation step
    eval_step_result = EvalStepResult(
        step=global_step,
        results=all_feature_results_this_eval_step,
    )
    all_run_results.append(eval_step_result.model_dump())

    with open(eval_results_path, "w") as f:
        json.dump(all_run_results, f, indent=2)


def run_evaluation(
    eval_data: list[TrainingDataPoint],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
    lora_path: str | None,
    eval_batch_size: int,
    steering_coefficient: float,
    generation_kwargs: dict,
    verbose: bool = False,
    processor=None,
) -> list[FeatureResult]:
    """Run evaluation and save results."""
    if lora_path is not None:
        adapter_name = lora_path
        if adapter_name not in model.peft_config:
            model.load_adapter(lora_path, adapter_name=adapter_name, is_trainable=False, low_cpu_mem_usage=True)
        model.set_adapter(adapter_name)
    with torch.no_grad():
        all_feature_results: list[FeatureResult] = []
        for i in tqdm(
            range(0, len(eval_data), eval_batch_size),
            desc="Evaluating model",
        ):
            e_batch = eval_data[i : i + eval_batch_size]

            for j in range(len(e_batch)):
                e_batch[j] = get_prompt_tokens_only(e_batch[j])

            e_batch = materialize_missing_steering_vectors(e_batch, tokenizer, model, processor=processor)

            e_batch = construct_batch(e_batch, tokenizer, device)

            feature_results = eval_features_batch(
                eval_batch=e_batch,
                model=model,
                submodule=submodule,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                steering_coefficient=steering_coefficient,
                generation_kwargs=generation_kwargs,
            )
            if verbose:
                for feature_result in feature_results:
                    print(f"\n=== Feature {feature_result.feature_idx} : {feature_result.api_response} ===\n")
            all_feature_results.extend(feature_results)

        # save_logs(
        #     eval_results_path="eval_logs.json",
        #     global_step=global_step,
        #     all_feature_results_this_eval_step=all_feature_results,
        # )
    # now add the meta info to the feature results
    assert len(all_feature_results) == len(eval_data), "Number of feature results and evaluation data points must match"
    for feature_result, eval_data_point in zip(all_feature_results, eval_data, strict=True):
        feature_result.meta_info = eval_data_point.meta_info
    return all_feature_results


def parse_answer(answer: str) -> str:
    return answer.rstrip(".!?,;:").strip().lower()


def score_eval_dataset(
    dataset_name: str,
    eval_responses: list[FeatureResult],
    eval_dataset: list[TrainingDataPoint],
    *,
    global_step: int = 0,
    details_path: str | None = None,
) -> dict[str, float]:
    from nl_probes.utils.secret_keeping_scoring import (
        aggregate_target_validation_scores,
        score_target_validation_record,
    )

    eval_results: dict[str, float] = {}
    is_target_validation = all(
        item.meta_info and "scoring_mode" in item.meta_info for item in eval_dataset
    )
    if is_target_validation:
        scored_records = [
            score_target_validation_record(response.api_response, item.meta_info)
            for response, item in zip(eval_responses, eval_dataset, strict=True)
        ]
        aggregate = aggregate_target_validation_scores(scored_records)
        percent_format_correct = sum(
            bool(record["format_correct"]) for record in scored_records
        ) / len(scored_records)
        percent_ans_correct = float(aggregate["overall"]["accuracy"])
        if details_path is not None:
            path = Path(details_path)
            with path.open("a", encoding="utf-8") as handle:
                for record in scored_records:
                    handle.write(
                        json.dumps(
                            {"step": global_step, "dataset": dataset_name, **record},
                            sort_keys=True,
                        )
                        + "\n"
                    )
        for ood_slice, metrics in aggregate["by_ood_slice"].items():
            eval_results[f"eval_target_ood/{dataset_name}/{ood_slice}"] = float(
                metrics["accuracy"]
            )
    else:
        target_answers = {parse_answer(item.target_output) for item in eval_dataset}
        valid_answers = ["yes", "no"] if target_answers <= {"yes", "no"} else None
        percent_format_correct, percent_ans_correct = score_eval_responses(
            eval_responses,
            eval_dataset,
            valid_answers=valid_answers,
        )
    eval_results[f"eval_format_correct/{dataset_name}"] = percent_format_correct
    eval_results[f"eval_ans_correct/{dataset_name}"] = percent_ans_correct
    return eval_results


def score_eval_responses(
    eval_responses: list[FeatureResult],
    eval_dataset: list[TrainingDataPoint],
    valid_answers: list[str] | None = None,
) -> tuple[float, float]:
    format_correct_list = []
    ans_correct_list = []
    for eval_response, eval_data_point in zip(eval_responses, eval_dataset, strict=True):
        cleaned_response = parse_answer(eval_response.api_response)
        target_response = parse_answer(eval_data_point.target_output)
        format_correct = bool(cleaned_response) if valid_answers is None else cleaned_response in valid_answers
        ans_correct = cleaned_response == target_response
        format_correct_list.append(format_correct)
        ans_correct_list.append(ans_correct)

    percent_format_correct = sum(format_correct_list) / len(format_correct_list)
    percent_ans_correct = sum(ans_correct_list) / len(ans_correct_list)
    return percent_format_correct, percent_ans_correct


def proportion_confidence(correct: int, total: int, z: float = 1.96) -> tuple[float, float, float, float]:
    """
    Compute proportion statistics.

    Returns (p, se, lower, upper)
    - p: proportion correct (in [0,1])
    - se: standard error of the proportion (sqrt(p*(1-p)/n))
    - lower, upper: normal-approximation confidence interval (clamped to [0,1])

    Uses normal approx: CI = p +/- z * se. Default z=1.96 gives ~95% CI.
    """
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    p = correct / total
    se = math.sqrt(p * (1.0 - p) / total)
    lower = max(0.0, p - z * se)
    upper = min(1.0, p + z * se)
    return p, se, lower, upper


def analyze_results(results: list[dict]) -> dict[str, float]:
    clean_responses = []

    correct = 0
    is_correct_list = []
    for result in results:
        cleaned_response = parse_answer(result["response"])
        clean_responses.append(cleaned_response)
        target_response = result["target_response"].lower()
        is_correct = target_response == cleaned_response
        is_correct_list.append(is_correct)
        if is_correct:
            correct += 1
        else:
            # continue
            print(result["response"])
            print(cleaned_response)
            print(target_response)
            print("--------------------------------")

    n = len(results)
    p, se, lower, upper = proportion_confidence(correct, n)  # default 95% CI (z=1.96)

    print(f"{correct=}")
    print(f"{n=}")
    print(f"percent_correct = {p:.4f} ({p * 100:.2f}%)")
    print(f"standard_error = {se:.6f}")
    print(f"95% CI (normal approx) = [{lower:.4f}, {upper:.4f}] ({lower * 100:.2f}%, {upper * 100:.2f}%)")
    print(f"len(set(clean_responses))={len(set(clean_responses))}")

    # return values in case you want to plot programmatically
    return {
        "correct": correct,
        "n": n,
        "p": p,
        "se": se,
        "ci_lower": lower,
        "ci_upper": upper,
        "is_correct_list": is_correct_list,
    }

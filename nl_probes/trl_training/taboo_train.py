import os

# helps to reduce memory usage and random OOMs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import gc
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from nl_probes.trl_training.config import CustomLoraConfig, CustomSFTConfig, EvalConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

import wandb
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

MODEL_NAME_TO_BATCH_SIZE = {
    "meta-llama/Llama-3.1-8B-Instruct": 4,
    "google/gemma-2-9b-it": 4,
    "google/gemma-2-27b-it": 4,
    "Qwen/Qwen3-14B": 8,
    "Qwen/Qwen3-8B": 8,
    "mistralai/Mistral-Small-24B-Instruct-2501": 1,
    "Qwen/Qwen3-32B": 8,
}

DEFAULT_TABOO_DATASETS = (
    "bcywinski/taboo-ship",
    "bcywinski/taboo-wave",
    "bcywinski/taboo-song",
    "bcywinski/taboo-snow",
    "bcywinski/taboo-rock",
    "bcywinski/taboo-moon",
    "bcywinski/taboo-jump",
    "bcywinski/taboo-green",
    "bcywinski/taboo-flame",
    "bcywinski/taboo-flag",
    "bcywinski/taboo-dance",
    "bcywinski/taboo-cloud",
    "bcywinski/taboo-clock",
    "bcywinski/taboo-chair",
    "bcywinski/taboo-salt",
    "bcywinski/taboo-book",
    "bcywinski/taboo-blue",
    "bcywinski/taboo-adversarial",
    "bcywinski/taboo-gold",
    "bcywinski/taboo-leaf",
    "bcywinski/taboo-smile",
)
DEFAULT_CHAT_DATASET = "HuggingFaceH4/ultrachat_200k"
SMOKE_TABOO_DATASET = "bcywinski/taboo-smile"
SMOKE_MAX_ROWS = 8
TABOO_EVAL_PERCENT = 0.1


def print_trainable_parameters(model) -> None:
    total = 0
    trainable = 0
    lora_trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            if "lora_" in name:
                lora_trainable += n
    pct = 100 * trainable / total if total else 0.0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    if lora_trainable:
        print(f"  LoRA trainable subset: {lora_trainable:,}")


def train_with_sft_only(
    sft_train_ds: Dataset,
    sft_hf_eval_test_ds: Dataset,
    wandb_sft_project: str,
    config: EvalConfig,
    sft_config: SFTConfig,
    callbacks: list[TrainerCallback],
    rollout_cb: TrainerCallback | None = None,
    save_lora_path: Path | None = None,
    load_lora_path: Path | None = None,
    quantize: bool = False,
) -> None:
    torch.manual_seed(config.random_seed)

    gc.collect()
    torch.cuda.empty_cache()

    # ---- tokenizer & base model ----
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )

    llm_kwargs = dict(
        pretrained_model_name_or_path=config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    # this is how I programmatically set initialization arguments for the Model
    if quantize:
        llm_kwargs["quantization_config"] = bnb_config
        # llm_kwargs["use_cache"] = False

    model = AutoModelForCausalLM.from_pretrained(
        **llm_kwargs,
    )

    model.enable_input_require_grads()
    model.use_cache = False
    model.gradient_checkpointing_enable()

    # I use this to continue training from an existing LoRA checkpoint
    if load_lora_path is not None:
        assert load_lora_path.exists(), f"LoRA path does not exist: {load_lora_path}"
        model = PeftModel.from_pretrained(model, load_lora_path, is_trainable=True)
        lora_config = None
    else:
        lora_config = CustomLoraConfig()
        model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)

    sft_trainer = SFTTrainer(
        model=model,
        train_dataset=sft_train_ds,
        eval_dataset=sft_hf_eval_test_ds,
        args=sft_config,
        callbacks=callbacks,
    )

    # if rollout_cb is not None:
    #     sft_trainer.add_callback(rollout_cb)

    wandb_str = f"sft_{config.model_name}{config.wandb_info}"

    if sft_trainer.is_world_process_zero():
        wandb.init(
            project=wandb_sft_project,
            name=wandb_str,
        )

    sft_trainer.train()

    if sft_trainer.is_world_process_zero():
        if save_lora_path is not None:
            sft_trainer.save_model(str(save_lora_path))
        wandb.finish()

        sft_trainer = None
        model = None
        tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()


def manual_qwen3_assistant_mask(
    messages: list[dict[str, str]], tokenizer: AutoTokenizer, final_message_loss_only: bool = False
) -> dict[str, torch.Tensor]:
    """
    Create a mask where 1 indicates assistant tokens and 0 indicates non-assistant tokens.

    Args:
        tokenized: Dictionary containing 'input_ids' tensor
        tokenizer: The tokenizer used to encode the text

    Returns:
        torch.Tensor: Binary mask of same shape as input_ids
    """

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=False,
        return_dict=False,
        enable_thinking=False,
    )

    # Get special token IDs
    tmp = tokenizer.encode("<|im_start|>assistant\n")
    assert len(tmp) == 3, f"Expected 3 tokens, got {len(tmp)}"
    begin_turn_idx = tmp[0]  # <|im_start|>
    asst_idx = tmp[1]  # assistant
    newline_idx = tmp[2]  # \n

    tmp_think = tokenizer.encode("<think>\n</think>")
    assert len(tmp_think) == 3, f"Expected 3 tokens, got {len(tmp_think)}"
    begin_think_idx = tmp_think[0]
    end_think_idx = tmp_think[2]

    eos_id = tokenizer.eos_token_id  # <|im_end|>

    # Initialize mask with zeros
    assistant_mask = torch.zeros_like(input_ids)

    num_messages = len(messages)
    cur_eos_idx = 0
    cur_message_idx = 0

    # Process each sequence in the batch
    for batch_idx in range(input_ids.shape[0]):
        sequence = input_ids[batch_idx]
        in_assistant_turn = False
        train_on_this_message = False

        # Iterate through the sequence
        i = 0
        while i < len(sequence):
            # Check if we're starting an assistant turn
            if i + 2 < len(sequence):
                if sequence[i] == begin_turn_idx and sequence[i + 1] == asst_idx and sequence[i + 2] == newline_idx:
                    i += 3
                    cur_message_idx += 1
                    in_assistant_turn = True

                    if not final_message_loss_only:
                        train_on_this_message = True

                    if cur_message_idx == len(messages) - 1:
                        assert sequence[i] == begin_think_idx and sequence[i + 2] == end_think_idx
                        i += 3
                        train_on_this_message = True
                    # Skip the <|im_start|>assistant\n tokens themselves
                    continue

            # Check if we're ending any turn
            if sequence[i] == eos_id:
                if in_assistant_turn:
                    cur_message_idx += 1
                    if train_on_this_message:
                        assistant_mask[batch_idx, i] = 1

                in_assistant_turn = False
                i += 1
                cur_eos_idx += 1
                continue

            # Set mask value based on whether we're in assistant turn
            if in_assistant_turn and train_on_this_message:
                assistant_mask[batch_idx, i] = 1
            else:
                assistant_mask[batch_idx, i] = 0

            i += 1

    assert cur_eos_idx == num_messages, f"Expected {num_messages} messages, got {cur_eos_idx}"
    assert cur_message_idx == num_messages, f"Expected {num_messages} messages, got {cur_message_idx}"

    assert len(input_ids) == len(assistant_mask)
    return {
        "input_ids": input_ids.squeeze(0),
        "assistant_masks": assistant_mask.squeeze(0),
    }


def taboo_secret_train_eval_sizes(n_rows: int, *, smoke: bool) -> tuple[int, int]:
    if n_rows < 2:
        raise ValueError(f"Need at least 2 rows for a train/eval split, got {n_rows}")
    if smoke:
        return n_rows - 1, 1
    train_size = int(n_rows * (1 - TABOO_EVAL_PERCENT))
    eval_size = int(n_rows * TABOO_EVAL_PERCENT)
    if train_size < 1 or eval_size < 1:
        raise ValueError(
            f"Need a non-empty train and eval split, got train_size={train_size} eval_size={eval_size} from {n_rows} rows"
        )
    return train_size, eval_size


def load_expanded_taboo_dataset(dataset_name: str, *, smoke: bool) -> Dataset:
    ds = load_dataset(dataset_name, split="train")
    old_len = len(ds)
    ds = create_incremental_turn_dataset(ds)
    print(f"Old length: {old_len}, New length: {len(ds)}")
    if smoke:
        if len(ds) < 2:
            raise ValueError(f"Smoke needs at least 2 expanded rows, got {len(ds)}")
        ds = ds.select(range(min(SMOKE_MAX_ROWS, len(ds))))
    return ds


def taboo_secret_eval_split(dataset_name: str, tokenizer: AutoTokenizer, *, smoke: bool = False) -> Dataset:
    ds = load_expanded_taboo_dataset(dataset_name, smoke=smoke)
    train_size, eval_size = taboo_secret_train_eval_sizes(len(ds), smoke=smoke)
    eval_ds = ds.select(range(train_size, train_size + eval_size))
    return prepare_sft_dataset(eval_ds, tokenizer, final_message_loss_only=True)


def prepare_sft_dataset(dataset: Dataset, tokenizer: AutoTokenizer, final_message_loss_only: bool) -> Dataset:
    remove_cols = [c for c in dataset.column_names if c not in {"messages"}]

    new_ds = dataset.map(
        lambda ex: manual_qwen3_assistant_mask(ex["messages"], tokenizer, final_message_loss_only),
        remove_columns=remove_cols,
        desc="Tokenizing dataset with chat template",
    )
    # remove messages column
    new_ds = new_ds.remove_columns(["messages"])
    return new_ds


def create_incremental_turn_dataset(dataset: Dataset) -> Dataset:
    """
    Creates a new dataset where each conversation is expanded into multiple rows
    with incrementally increasing turns. Required for Qwen3 tokenization of multiple turns.
    https://huggingface.co/Qwen/Qwen3-32B/discussions/11

    Args:
        dataset: Original dataset with 'messages' field containing conversations
        num_turns: Maximum number of turn pairs (user-assistant exchanges) to include

    Returns:
        Dataset with incrementally longer conversations
    """
    new_data = []

    for example in dataset:
        messages = example["messages"]

        # Count the actual number of turn pairs in this conversation
        # A turn pair is (user message, assistant response)
        turn_pairs = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                turn_pairs.append((messages[i], messages[i + 1]))

        # Generate rows with incrementally more turns
        max_turns_for_example = len(turn_pairs)

        for n_turns in range(1, max_turns_for_example + 1):
            # Create a conversation with n_turns pairs
            conversation = []
            for turn_idx in range(n_turns):
                # Add user message
                conversation.append(turn_pairs[turn_idx][0])
                # Add assistant response
                conversation.append(turn_pairs[turn_idx][1])

            # Add this incremental conversation as a new row
            new_data.append(
                {
                    "messages": conversation,
                    "num_turns": n_turns,  # Track how many turn pairs this row has
                    "original_idx": dataset.indices[dataset.indices.tolist().index(example)]
                    if hasattr(dataset, "indices")
                    else len(new_data),
                }
            )

    return Dataset.from_list(new_data)


def message_char_length(example: dict) -> int:
    total_chars = 0
    for msg in example["messages"]:
        total_chars += len(msg["content"])
    return total_chars


def take_first_turn_neutrals(
    examples: Iterable[dict], n: int, max_char_length: int
) -> list[dict]:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    kept: list[dict] = []
    for example in examples:
        messages = example["messages"]
        if len(messages) < 2:
            continue
        truncated_messages = messages[:2]
        char_length = sum(len(msg["content"]) for msg in truncated_messages)
        if char_length <= max_char_length:
            kept.append({"messages": truncated_messages})
            if len(kept) >= n:
                return kept
    raise ValueError(
        f"Need {n} first-turn neutrals with char_length <= {max_char_length}, got {len(kept)}"
    )


def mix_secret_and_neutral_rows(
    taboo_rows: Sequence[dict],
    chat_examples: Iterable[dict],
    neutral_per_secret: int,
) -> list[dict]:
    if neutral_per_secret < 1:
        raise ValueError(f"neutral_per_secret must be >= 1, got {neutral_per_secret}")
    n_taboo = len(taboo_rows)
    if n_taboo < 1:
        raise ValueError("taboo_rows must be non-empty")
    n_neutral = n_taboo * neutral_per_secret
    max_char_length = max(message_char_length(ex) for ex in taboo_rows)
    neutrals = take_first_turn_neutrals(chat_examples, n_neutral, max_char_length)
    mixed = list(taboo_rows) + neutrals
    if len(mixed) != n_taboo * (1 + neutral_per_secret):
        raise ValueError(
            f"Expected mixed size {n_taboo * (1 + neutral_per_secret)}, got {len(mixed)}"
        )
    return mixed


def adapter_dir_name(model_name: str, dataset_name: str, neutral_per_secret: int) -> str:
    base = f"{model_name.split('/')[-1]}-{dataset_name.split('/')[-1]}"
    base = base.replace(" ", "_").replace(".", "_").replace("/", "_")
    return f"{base}_1to{neutral_per_secret}"


def parse_taboo_train_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taboo LoRA SFT mixed with UltraChat neutrals")
    parser.add_argument("--neutral-per-secret", type=int, default=1)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-lora-dir", default="model_lora")
    parser.add_argument("--chat-dataset", default=DEFAULT_CHAT_DATASET)
    args = parser.parse_args(argv)
    if args.neutral_per_secret < 1:
        raise ValueError(f"--neutral-per-secret must be >= 1, got {args.neutral_per_secret}")
    if args.smoke:
        args.datasets = args.datasets or [SMOKE_TABOO_DATASET]
    else:
        args.datasets = args.datasets or list(DEFAULT_TABOO_DATASETS)
    return args


def combine_with_ultrachat(
    raw_train_ds: Dataset,
    tokenized_train_ds: Dataset,
    chat_dataset_name: str,
    tokenizer: AutoTokenizer,
    random_seed: int,
    final_message_loss_only: bool,
    neutral_per_secret: int,
    chat_examples: Iterable[dict] | None = None,
) -> Dataset:
    """
    Sample UltraChat first-turns, N per taboo train row, then shuffle with secret rows.
    """
    from datasets import concatenate_datasets

    n_taboo = len(tokenized_train_ds)
    n_neutral = n_taboo * neutral_per_secret
    max_char_length = max(message_char_length(ex) for ex in raw_train_ds)
    print(
        f"Sampling {n_neutral} UltraChat examples "
        f"({n_taboo} taboo * {neutral_per_secret} neutrals per secret)"
    )
    print(f"Max character length in taboo dataset: {max_char_length}")

    stream: Iterable[dict]
    if chat_examples is None:
        stream = load_dataset(chat_dataset_name, split="train_sft", streaming=True)
    else:
        stream = chat_examples

    kept_examples = take_first_turn_neutrals(stream, n_neutral, max_char_length)
    chat_dataset = Dataset.from_list(kept_examples)
    train_chat_ds = prepare_sft_dataset(
        chat_dataset, tokenizer, final_message_loss_only=final_message_loss_only
    )
    combined_train_ds = concatenate_datasets([tokenized_train_ds, train_chat_ds])
    combined_train_ds = combined_train_ds.shuffle(seed=random_seed)
    print(f"Combined dataset size: {len(combined_train_ds)}")
    print(f"  - Taboo: {n_taboo}")
    print(f"  - UltraChat: {len(train_chat_ds)}")
    return combined_train_ds


def run_taboo_sft(args: argparse.Namespace) -> None:
    final_message_loss_only = True
    model_name = args.model
    for dataset_name in args.datasets:
        print(f"Training {model_name} on {dataset_name}")
        config = EvalConfig(
            model_name=model_name,
            model_lora_dir=args.model_lora_dir,
        )
        lora_path = Path(config.model_lora_dir) / adapter_dir_name(
            model_name, dataset_name, args.neutral_per_secret
        )
        if lora_path.exists() and not args.smoke:
            print(f"{lora_path} already exists, skipping SFT training")
            continue

        torch.cuda.empty_cache()
        gc.collect()

        batch_size = MODEL_NAME_TO_BATCH_SIZE[config.model_name]
        real_batch_size = 8
        sft_overrides: dict[str, Any] = {"num_train_epochs": 10.0}
        if args.smoke:
            sft_overrides["num_train_epochs"] = 1.0
            sft_overrides["max_steps"] = 1
            sft_overrides["eval_steps"] = 1
            sft_overrides["save_steps"] = 1
        sft_config = CustomSFTConfig(
            model_name=config.model_name,
            batch_size=batch_size,
            real_batch_size=real_batch_size,
            **sft_overrides,
        )

        ds = load_expanded_taboo_dataset(dataset_name, smoke=args.smoke)
        train_size, eval_size = taboo_secret_train_eval_sizes(len(ds), smoke=args.smoke)
        raw_train_ds = ds.select(range(train_size))
        eval_ds = ds.select(range(train_size, train_size + eval_size))

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        train_ds = prepare_sft_dataset(raw_train_ds, tokenizer, final_message_loss_only=final_message_loss_only)
        eval_ds = prepare_sft_dataset(eval_ds, tokenizer, final_message_loss_only=final_message_loss_only)
        train_ds = combine_with_ultrachat(
            raw_train_ds=raw_train_ds,
            tokenized_train_ds=train_ds,
            chat_dataset_name=args.chat_dataset,
            tokenizer=tokenizer,
            random_seed=config.random_seed,
            final_message_loss_only=final_message_loss_only,
            neutral_per_secret=args.neutral_per_secret,
        )

        if not args.smoke:
            eval_frequency = len(train_ds) // (real_batch_size * 2)
            if eval_frequency < 1:
                raise ValueError(f"eval_frequency must be >= 1, got {eval_frequency}")
            sft_config.eval_steps = eval_frequency
            sft_config.save_steps = eval_frequency

        train_with_sft_only(
            train_ds,
            eval_ds,
            config.wandb_project,
            config,
            sft_config,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            save_lora_path=lora_path,
            quantize=False,
        )


if __name__ == "__main__":
    run_taboo_sft(parse_taboo_train_args())

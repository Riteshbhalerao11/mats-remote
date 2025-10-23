import os
import argparse
from pathlib import Path

import numpy as np

# Compatibility for older code referencing np.bool
if not hasattr(np, "bool"):
    np.bool = bool 

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel
from crosscoder_learning.dictionary_learning.cache import ActivationCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect model activations and labels from a dataset")

    # Model + dataset
    parser.add_argument("--model", type=str, default="/pscratch/sd/r/ritesh11/temp/Qwen3-30B-A3B", help="Model name or path")
    parser.add_argument("--dataset", type=str, default="prompts", help="Hugging Face dataset name")
    parser.add_argument("--dataset-split", type=str, default="eval", help="Dataset split to use")
    parser.add_argument("--text-column", type=str, default="text", help="Text column name after mapping")

    # Activation collection
    parser.add_argument("--activation-store-dir", type=str, default="model_activations", help="Base directory to store outputs")
    parser.add_argument("--layers", type=int, nargs="+", default=[22], help="Layer indices to trace (space-separated)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for tokenization/inference (There appears to be a bug with batch size > 1)")
    parser.add_argument("--context-len", type=int, default=3008, help="Max context length")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing shards")
    parser.add_argument("--store-tokens", dest="store_tokens", action="store_true", help="Store token ids alongside activations")
    parser.add_argument("--no-store-tokens", dest="store_tokens", action="store_false", help="Do not store token ids")
    parser.set_defaults(store_tokens=True)
    parser.add_argument("--disable-multiprocessing", action="store_true", help="Disable multiprocessing in collection")

    # Limits
    parser.add_argument("--num-samples", type=int, default=10 ** 6, help="Number of samples to consider")
    parser.add_argument("--max-tokens", type=int, default=10 ** 8, help="Maximum total tokens to collect")

    # Data type
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"], default="float16", help="Torch dtype")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for shuffling")

    return parser.parse_args()


def dtype_from_string(dtype_str: str) -> torch.dtype:
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Invalid dtype: {dtype_str}")


def main() -> None:
    args = parse_args()

    # Environment setup
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Convert dtype string to torch dtype
    torch_dtype = dtype_from_string(args.dtype)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Wrap with nnsight
    nnmodel = LanguageModel(model, tokenizer=tokenizer)

    # Submodules to trace
    submodules = [nnmodel.model.layers[layer_idx] for layer_idx in args.layers]
    submodule_names = [f"layer_{layer_idx}" for layer_idx in args.layers]
    d_model = nnmodel._model.config.hidden_size

    # Output directories
    store_dir = Path(args.activation_store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    out_dir = store_dir / args.dataset_split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and filter dataset
    dataset = load_from_disk(args.dataset)
    dataset = dataset[args.dataset_split]


    # Prepare text column via chat template
    def format_messages(example):
        formatted = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": formatted}

    dataset = dataset.map(format_messages, remove_columns=dataset.column_names)

    # Collect activations
    ActivationCache.collect(
        dataset[args.text_column],
        submodules,
        submodule_names,
        nnmodel,
        out_dir,
        shuffle_shards=False,
        io="out",
        shard_size=10 ** 6,
        batch_size=args.batch_size,
        context_len=args.context_len,
        d_model=d_model,
        last_submodule=submodules[-1],
        max_total_tokens=args.max_tokens,
        store_tokens=args.store_tokens,
        multiprocessing=not args.disable_multiprocessing,
        ignore_first_n_tokens_per_sample=0,
        overwrite=args.overwrite,
        token_level_replacement=None,
        add_special_tokens=False
    )


if __name__ == "__main__":
    main()

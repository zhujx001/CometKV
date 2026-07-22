import os
import sys
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
KERNEL_LIB = os.path.join(PROJECT_ROOT, "library", "cometkv")


def _prepend_sys_path(path):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


_prepend_sys_path(PROJECT_ROOT)
_prepend_sys_path(KERNEL_LIB)

from config import add_config_args, generate_config, get_by_key, get_model_path, resolve_model_config

LONG_BENCH_ROOT = Path(__file__).resolve().parent
LONG_BENCH_CONFIG_DIR = LONG_BENCH_ROOT / "config"
DEFAULT_MODEL = "llama-3.1-8b"
DEFAULT_MODEL_PATH = get_model_path(DEFAULT_MODEL)
DEFAULT_DATA_DIR = get_by_key("datasets.longbench.data_dir")

model2maxlen = json.load(open(LONG_BENCH_CONFIG_DIR / "model2maxlen.json", "r", encoding="utf-8"))
SUPPORTED_MODELS = list(model2maxlen.keys())
# we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
dataset2prompt = json.load(open(LONG_BENCH_CONFIG_DIR / "dataset2prompt.json", "r", encoding="utf-8"))
dataset2maxlen = json.load(open(LONG_BENCH_CONFIG_DIR / "dataset2maxlen.json", "r", encoding="utf-8"))

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
NO_CHAT_FORMAT_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
CHAT_TEMPLATE_OVERHEAD_ESTIMATE = 64
CHAT_TEMPLATE_MODELS = ("llama", "qwen", "mistral")


def parse_args(args=None):
    cli_args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser()
    # overwrite model_name argument
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        choices=SUPPORTED_MODELS)
    parser.add_argument('--model_path', type=str, default=None,
                        help="Local model path. Defaults to the model entry in config/paths.json.")
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help="Directory containing local LongBench jsonl files.")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--task', type=str, required=True, help="task name. work when --e is false")
    parser.add_argument("--num_examples", type=int, default=-1, help="num of example to evaluate. -1 for all.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, set to `auto` to split model across all available GPUs")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Data type")
    parser.add_argument("--prefill_bsz", type=int, default=1, help="Prefill batch size for model.generate.")
    parser.add_argument("--profile_timing", action="store_true", help="Print TTFT, TPOT, and end-to-end timing from model.generate.")

    parser = add_config_args(parser)

    parsed_args = parser.parse_args(cli_args)
    if parsed_args.model_path is None:
        parsed_args.model_path = get_model_path(parsed_args.model)

    return parsed_args


def resolve_dataset_path(data_dir, dataset, use_longbench_e):
    suffix = "_e" if use_longbench_e else ""
    return Path(data_dir) / f"{dataset}{suffix}.jsonl"


def load_local_dataset(data_dir, dataset, use_longbench_e):
    dataset_path = resolve_dataset_path(data_dir, dataset, use_longbench_e)
    if not dataset_path.exists():
        raise FileNotFoundError(f"LongBench data file not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_config_model_name(model_name, model_path):
    config_dir = Path(PROJECT_ROOT) / "config"
    candidates = [Path(str(model_path).rstrip("/")).name, model_name]

    try:
        model_config = resolve_model_config(model_name)
    except KeyError:
        model_config = None

    if model_config is not None:
        candidates.extend(model_config.get("aliases", []))
        candidates.append(Path(model_config["path"].rstrip("/")).name)

    for candidate in candidates:
        if candidate and (config_dir / f"{candidate}.json").exists():
            return candidate

    return Path(str(model_path).rstrip("/")).name


def normalize_token_ids(token_ids):
    if token_ids is None:
        return []
    if isinstance(token_ids, (list, tuple, set)):
        normalized_ids = [int(token_id) for token_id in token_ids]
    else:
        normalized_ids = [int(token_ids)]

    deduped_ids = []
    for token_id in normalized_ids:
        if token_id not in deduped_ids:
            deduped_ids.append(token_id)
    return deduped_ids


def tokenize_prompt(tokenizer, prompt, use_chat_template, padding=False):
    return tokenizer(
        [prompt],
        return_tensors="pt",
        padding=padding,
        add_special_tokens=not use_chat_template,
    )


def should_use_chat_template(dataset):
    return dataset not in NO_CHAT_FORMAT_DATASETS


def build_chat(tokenizer, prompt, model_name):
    model_name = model_name.lower()
    if hasattr(tokenizer, "apply_chat_template") and any(name in model_name for name in CHAT_TEMPLATE_MODELS):
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt


def truncate_prompt(tokenizer, prompt, max_length, skip_special_tokens=True):
    if max_length <= 0:
        raise ValueError(f"Prompt budget must be positive, but got {max_length}.")

    tokenized_prompt = tokenizer(
        prompt,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    if len(tokenized_prompt) <= max_length:
        return prompt

    head_length = max_length // 2
    tail_length = max_length - head_length
    return (
        tokenizer.decode(tokenized_prompt[:head_length], skip_special_tokens=skip_special_tokens)
        + tokenizer.decode(tokenized_prompt[-tail_length:], skip_special_tokens=skip_special_tokens)
    )


def get_chat_template_overhead(tokenizer, model_name):
    empty_chat_prompt = build_chat(tokenizer, "", model_name)
    empty_chat_ids = tokenizer(
        empty_chat_prompt,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    return len(empty_chat_ids)


def prepare_prompt(tokenizer, prompt_format, json_obj, dataset, model_name, max_length, max_new_tokens):
    prompt = prompt_format.format(**json_obj)
    max_prompt_length = max_length - max_new_tokens - CHAT_TEMPLATE_OVERHEAD_ESTIMATE
    if max_prompt_length <= 0:
        raise ValueError(
            f"Model max_length({max_length}) must be larger than max_new_tokens({max_new_tokens}) "
            f"plus chat margin({CHAT_TEMPLATE_OVERHEAD_ESTIMATE})."
        )

    prompt = truncate_prompt(tokenizer, prompt, max_prompt_length)
    if should_use_chat_template(dataset):
        prompt = build_chat(tokenizer, prompt, model_name)

    return prompt


def resolve_eos_token_ids(llm):
    config_eos_token_ids = getattr(getattr(llm, "config", None), "eos_token_id", None)
    if config_eos_token_ids is not None:
        return normalize_token_ids(config_eos_token_ids)

    model_eos_token_ids = getattr(llm, "eos_tokens", None)
    if model_eos_token_ids is not None:
        return normalize_token_ids(model_eos_token_ids)

    tokenizer_eos_token_id = getattr(getattr(llm, "tokenizer", None), "eos_token_id", None)
    resolved_ids = normalize_token_ids(tokenizer_eos_token_id)
    if not resolved_ids:
        raise ValueError("Failed to resolve EOS token ids for generation.")
    return resolved_ids


def get_pred(llm, data, max_new_tokens, prompt_format, model_name, model_path, max_length, dataset, out_path, args):
    eos_token_ids = resolve_eos_token_ids(llm)
    # === samsum: stop at newline ===
    _stop_ids = list(eos_token_ids)
    if dataset == "samsum":
        newline_id = llm.tokenizer.encode("\n", add_special_tokens=False)[-1]
        if newline_id not in _stop_ids:
            _stop_ids.append(newline_id)
    # ====

    for json_obj in tqdm(data):
        prompt = prepare_prompt(
            tokenizer=llm.tokenizer,
            prompt_format=prompt_format,
            json_obj=json_obj,
            dataset=dataset,
            model_name=model_name,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )

        use_chat_template = should_use_chat_template(dataset)
        inputs = tokenize_prompt(
            tokenizer=llm.tokenizer,
            prompt=prompt,
            use_chat_template=use_chat_template,
            padding=True,
        )
        input_ids = inputs.input_ids
        attention_masks = inputs.attention_mask

        config_model_name = resolve_config_model_name(model_name, model_path)
        attn_config = generate_config(
            config_model_name,
            input_ids.shape[1], 
            attn_type,
            retrieval_budget=args.retrieval_budget,
            sig_bits=getattr(args, "sig_bits", 128),
            sig_chunk_size=getattr(args, "sig_chunk_size", 16384),
            sig_seed=getattr(args, "sig_seed", 1234),
            cometkv_min_retrieval_topk=getattr(args, "cometkv_min_retrieval_topk", 16),
            cometkv_token_cache_size=getattr(args, "cometkv_token_cache_size", 1024),
            cometkv_static_pattern_start=getattr(args, "cometkv_static_pattern_start", None),
            cometkv_static_pattern_end=getattr(args, "cometkv_static_pattern_end", None),
            cometkv_exclude_preserved_from_budget=getattr(args, "cometkv_exclude_preserved_from_budget", True),
            cometkv_selector=getattr(args, "cometkv_selector", "asym_n8"),
        )

        out = llm.generate(
            attention_type=attn_type,
            inputs_ids=input_ids.to(llm.layers[0].device),
            attention_masks=attention_masks.to(llm.layers[0].device),
            max_new_length=max_new_tokens, 
            attn_config=attn_config,
            do_sample=False,
            ignore_eos=False,
            eos_token_ids=_stop_ids,
            prefill_bsz=getattr(args, "prefill_bsz", 1),
            profile_timing=getattr(args, "profile_timing", False),
        )

        output = llm.tokenizer.batch_decode(
            out,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        pred = output[0]
        
        torch.cuda.empty_cache()
        
        print("Chunked generation:", pred[:50])

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"],
                    "input_tokens": int(input_ids.shape[1]),
                },
                f, 
                ensure_ascii=False
            )
            f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model(model_path, max_len, dtype, device):
    from model_hub import LlamaModel, QwenModel, MistralModel

    if 'Llama' in model_path:
        llm = LlamaModel(model_path,
            max_length=max_len,
            dtype=dtype,
            device_map=device)
    elif 'Qwen' in model_path:
        llm = QwenModel(model_path,
            max_length=max_len,
            dtype=dtype,
            device_map=device)
    elif 'Mistral' in model_path:
        llm = MistralModel(model_path,
            max_length=max_len,
            dtype=dtype,
            device_map=device)
    else:
        raise ValueError(f"Unsupported model: {model_path}")

    llm.tokenizer.pad_token = llm.tokenizer.eos_token
    llm.tokenizer.padding_side = "left"
    
    return llm



if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    num_examples = args.num_examples
    attn_type = args.attn_type
    model_name = args.model
    device = args.device
    dtype = torch.float16 if args.dtype=='fp16' else torch.bfloat16

    max_length = model2maxlen[model_name]
    model_path = args.model_path

    llm = load_model(model_path, max_length, dtype, device)

    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = [args.task]
    
    # predict on each dataset
    if not os.path.exists("results/pred"):
        os.makedirs("results/pred")
    if not os.path.exists("results/pred_e"):
        os.makedirs("results/pred_e")

    for dataset in datasets:
        print(f"Predict {dataset}")
        if args.e:
            data_all = load_local_dataset(args.data_dir, dataset, True)
            prefix = f"results/pred_e/{model_name}/{attn_type}"
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            out_path = f"{prefix}/{dataset}.jsonl"
        else:
            data_all = load_local_dataset(args.data_dir, dataset, False)
            prefix = f"results/pred/{model_name}/{attn_type}"
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            out_path = f"{prefix}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_new_tokens = dataset2maxlen[dataset]
        data_all = data_all[:num_examples] if num_examples > 0 else data_all

        get_pred(
            llm,
            data_all,
            max_new_tokens,
            prompt_format,
            model_name,
            model_path,
            max_length,
            dataset,
            out_path,
            args,
        )

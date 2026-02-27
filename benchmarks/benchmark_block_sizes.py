"""Benchmark the running time of various block sizes on an existing DFlash draft model.

Usage:
    CUDA_VISIBLE_DEVICES=1 python benchmark_block_sizes.py \
        --model-name-or-path Qwen/Qwen3-4B \
        --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
        --dataset math500 \
        --max-samples 20 \
        --block-sizes 16,32,64,128
"""

import argparse
import time
import random
from itertools import chain
from types import SimpleNamespace

import numpy as np
import torch
from rich import print
from rich.table import Table
from rich.console import Console
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from loguru import logger

from model import DFlashDraftModel, sample, load_and_process_dataset, extract_context_feature


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(
                model(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[
                        :, past_key_values_draft.get_seq_length() : start + block_size
                    ],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, -block_size + 1 :, :]
            )
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )

        posterior = sample(output.logits, temperature)
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        )
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
            :, : acceptance_length + 1
        ]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(
                output.hidden_states, model.target_layer_ids
            )[:, : acceptance_length + 1, :]

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:]
            for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids_t = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = (
            torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_t)
            .nonzero(as_tuple=True)[0]
        )
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        total_decode_time=total_decode_time,
        acceptance_lengths=acceptance_lengths,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DFlash with multiple block sizes")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument(
        "--block-sizes",
        type=str,
        default="16,32,64,128",
        help="Comma-separated list of block sizes to benchmark",
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations before timing")
    args = parser.parse_args()

    block_sizes = [int(b) for b in args.block_sizes.split(",")]

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")

    def has_flash_attn():
        try:
            import flash_attn
            return True
        except ImportError:
            logger.warning(
                "flash_attn not installed. Falling back to torch.sdpa. Speedup will be lower."
            )
            return False

    installed_flash_attn = has_flash_attn()
    attn_impl = "flash_attention_2" if installed_flash_attn else "sdpa"

    print(f"[bold]Loading target model:[/bold] {args.model_name_or_path}")
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    print(f"[bold]Loading draft model:[/bold] {args.draft_name_or_path}")
    draft_model = (
        DFlashDraftModel.from_pretrained(
            args.draft_name_or_path,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    print(f"[bold]Draft model trained block_size:[/bold] {draft_model.block_size}")
    print(f"[bold]Block sizes to benchmark:[/bold] {block_sizes}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    print(f"[bold]Dataset:[/bold] {args.dataset} ({len(dataset)} samples)")
    print(f"[bold]Max new tokens:[/bold] {args.max_new_tokens}")
    print()

    all_block_sizes = [1] + block_sizes

    results = {bs: [] for bs in all_block_sizes}

    for idx in tqdm(range(len(dataset)), desc="Benchmarking"):
        instance = dataset[idx]
        messages = [{"role": "user", "content": instance["turns"][0]}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        is_warmup = idx < args.warmup

        for bs in all_block_sizes:
            torch.cuda.empty_cache()
            response = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=args.max_new_tokens,
                block_size=bs,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
            if not is_warmup:
                results[bs].append(response)

    console = Console()

    t1 = np.mean([r.time_per_output_token for r in results[1]])

    table = Table(title="DFlash Block Size Benchmark Results")
    table.add_column("Block Size", justify="center", style="cyan")
    table.add_column("Avg Time/Token (ms)", justify="center")
    table.add_column("Avg Decode Time (s)", justify="center")
    table.add_column("Avg Output Tokens", justify="center")
    table.add_column("Speedup vs AR", justify="center", style="green")
    table.add_column("Avg Acceptance Len", justify="center", style="yellow")
    table.add_column("Acceptance Histogram", justify="left")

    for bs in all_block_sizes:
        responses = results[bs]
        avg_time_per_token = np.mean([r.time_per_output_token for r in responses])
        avg_decode_time = np.mean([r.total_decode_time for r in responses])
        avg_output_tokens = np.mean([r.num_output_tokens for r in responses])
        speedup = t1 / avg_time_per_token if avg_time_per_token > 0 else 0

        if bs > 1:
            acceptance_lengths = list(chain(*[r.acceptance_lengths for r in responses]))
            tau = np.mean(acceptance_lengths)
            histogram = [
                acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(bs + 1)
            ]
            hist_str = " ".join(f"{x * 100:.1f}%" for x in histogram[:min(bs + 1, 10)])
            if bs + 1 > 10:
                hist_str += " ..."
        else:
            tau = 1.0
            hist_str = "N/A"

        table.add_row(
            str(bs),
            f"{avg_time_per_token * 1000:.2f}",
            f"{avg_decode_time:.2f}",
            f"{avg_output_tokens:.0f}",
            f"{speedup:.2f}x",
            f"{tau:.2f}",
            hist_str,
        )

    console.print()
    console.print(table)

    print("\n[bold]Detailed per-block-size statistics:[/bold]")
    for bs in block_sizes:
        responses = results[bs]
        acceptance_lengths = list(chain(*[r.acceptance_lengths for r in responses]))
        avg_time_per_token = np.mean([r.time_per_output_token for r in responses])
        speedup = t1 / avg_time_per_token if avg_time_per_token > 0 else 0
        tau = np.mean(acceptance_lengths)

        print(f"\n[cyan]Block size {bs}:[/cyan]")
        print(f"  Speedup:              {speedup:.2f}x")
        print(f"  Avg acceptance len:   {tau:.2f}")
        print(f"  Avg time/token:       {avg_time_per_token * 1000:.2f} ms")
        print(f"  Avg decode time:      {np.mean([r.total_decode_time for r in responses]):.2f} s")

        histogram = [
            acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(bs + 1)
        ]
        print(f"  Histogram (0..{bs}):   {[f'{x*100:.1f}%' for x in histogram]}")


if __name__ == "__main__":
    main()

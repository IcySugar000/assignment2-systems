import timeit
import argparse
from typing import Literal

import torch
import numpy as np
from einops import rearrange
from loguru import logger

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy, clip_gradient


def benchmark(w: int, n: int, d_model: int, d_ff: int, num_layers: int, num_heads: int, mode: Literal["forward", "backward", "full"]) -> tuple[float, float]:
    # 一些事先预设的参数
    vocab_size = 10_000
    context_length = 128
    batch_size = 4

    logger.info(f"[Benchmarking] Testing: warmup_steps={w}, steps={n}, mode={mode}")

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    model = model.cuda()
    optimizer = AdamW(
        params=model.parameters(),
    )

    inputs = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=torch.device("cuda"))
    targets = torch.randint(low=0, high=vocab_size, size=(batch_size * context_length,), device=torch.device("cuda"))

    def operation():
        actuals = model.forward(inputs)

        if mode in ["backward", "full"]:
            actuals = rearrange(actuals, "batch seq vocab -> (batch seq) vocab")
            losses = cross_entropy(actuals, targets)
            optimizer.zero_grad()
            losses.backward()

            if mode == "full":
                clip_gradient(model.parameters(), 1.0)
                optimizer.step()

        torch.cuda.synchronize()

    # Warmup
    logger.info("[Benchmark] Starting warmup")
    for _ in range(w):
        operation()

    # Timing
    logger.info("[Benchmark] Start timing")
    time_taken = timeit.repeat(operation, number=1, repeat=n)
    logger.info(f"[Benchmark] Time for {n} rounds: {time_taken}")
    logger.info(f"[Benchmark] Mean: {np.mean(time_taken):.6f}, STD: {np.std(time_taken):.6f}")

    return float(np.mean(time_taken)), float(np.std(time_taken))


def batch_benchmark():
    data = {
        "sizes": [],
        "forward - mean": [],
        "forward - std": [],
        "forward&backward - mean": [],
        "forward&backward - std": [],
        "full - mean": [],
        "full - std": [],
    }

    for size in ["small", "medium"]:
        data["sizes"].append(size)
        for mode in ["forward", "backward", "full"]:
            d_model, d_ff, num_layers, num_heads = get_config_from_size(size)  # type: ignore
            mean, std = benchmark(2, 10, d_model, d_ff, num_layers, num_heads, mode)  # type: ignore

            key_prefix = {"forward": "forward", "backward": "forward&backward", "full": "full"}[mode]
            mean_key, std_key = f"{key_prefix} - mean", f"{key_prefix} - std"
            data[mean_key].append(f"{mean:.6f}")
            data[std_key].append(f"{std:.6f}")

            torch.cuda.empty_cache()

    print(f"{' | '.join(['|', *data['sizes']])} |")
    print(f"| {' | '.join(['-'] * (len(data['sizes']) + 1))} |")
    for key, value in data.items():
        print(f"| {' | '.join([key, *value])} |")


def get_config_from_size(size: Literal["small", "medium", "large", "xl", "10B"]) -> tuple[int, int, int, int]:
    match size:
        case "small":
            return 768, 3072, 12, 12
        case "medium":
            return 1024, 4096, 24, 16
        case "large":
            return 1280, 5120, 36, 20
        case "xl":
            return 2560, 10240, 32, 32
        case "10B":
            return 4608, 12288, 50, 36


if __name__ == "__main__":
    # benchmark(5, 10, 768, 3072, 12, 12, "forward")
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("-w", "--warmup_steps", type=int, default=5)
    parser.add_argument("-n", "--steps", type=int, default=10)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("-s", "--size", type=str, choices=["small", "medium", "large", "xl", "10B"])
    parser.add_argument("--mode", type=str, choices=["forward", "backward", "full"], default="forward")

    args = parser.parse_args()

    if args.batch:
        batch_benchmark()
        exit()

    warmup_steps = args.warmup_steps
    steps = args.steps
    mode = args.mode

    if args.size:
        d_model, d_ff, num_layers, num_heads = get_config_from_size(args.size)
    else:
        d_model = args.d_model
        d_ff = args.d_ff
        num_layers = args.num_layers
        num_heads = args.num_heads

        if None in [d_model, d_ff, num_layers, num_heads]:
            logger.error("[Benchmarking] Model params must be filled if 'size' arg is not provided.")
            exit()

    benchmark(warmup_steps, steps, d_model, d_ff, num_layers, num_heads, mode)

#!/usr/bin/env python3
"""
Script to find the 4 GPUs with the most available free VRAM and run training on them.
"""

import subprocess
import sys
import os
import argparse
import json
from typing import List, Tuple


def get_gpu_memory_info() -> List[Tuple[int, int, int]]:
    """
    Get GPU memory information using nvidia-smi.
    Returns list of tuples: (gpu_id, free_memory_mb, total_memory_mb)
    """
    try:
        # Run nvidia-smi to get memory information
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        gpu_info = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split(", ")
                gpu_id = int(parts[0])
                free_memory = int(parts[1])
                total_memory = int(parts[2])
                gpu_info.append((gpu_id, free_memory, total_memory))

        return gpu_info
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing GPU information: {e}")
        sys.exit(1)


def select_best_gpus(num_gpus: int = 4) -> List[int]:
    """
    Select the GPUs with the most available free VRAM.
    """
    gpu_info = get_gpu_memory_info()

    if len(gpu_info) < num_gpus:
        print(
            f"[WARNING] Only {len(gpu_info)} GPUs available, but {num_gpus} requested."
        )
        print("Using all available GPUs.")
        num_gpus = len(gpu_info)

    # Sort by free memory (descending) and take the top N
    sorted_gpus = sorted(gpu_info, key=lambda x: x[1], reverse=True)
    selected_gpus = [gpu[0] for gpu in sorted_gpus[:num_gpus]]

    print(f"Selected GPUs with most free VRAM:")
    for i, gpu_id in enumerate(selected_gpus):
        gpu_data = next(gpu for gpu in gpu_info if gpu[0] == gpu_id)
        free_mb = gpu_data[1]
        total_mb = gpu_data[2]
        free_gb = free_mb / 1024
        total_gb = total_mb / 1024
        print(f"  GPU {gpu_id}: {free_gb:.1f}GB free / {total_gb:.1f}GB total")

    return selected_gpus


def run_training_with_gpus(
    selected_gpus: List[int], config_file: str, additional_args: List[str] = None
):
    """
    Run the training script on the selected GPUs using torchrun.
    """
    if additional_args is None:
        additional_args = []

    # Set CUDA_VISIBLE_DEVICES to only the selected GPUs
    gpu_list = ",".join(map(str, selected_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    os.environ["OMP_NUM_THREADS"] = "2"

    # Build the torchrun command
    cmd = [
        "torchrun",
        f"--standalone",
        f"--nproc_per_node={len(selected_gpus)}",
        f"--nnodes=1",
        "run_lp.py",
        config_file,
    ] + additional_args

    print(f"Running command: {' '.join(cmd)}")
    print(f"Using GPUs: {gpu_list}")
    print("-" * 50)

    ## Uncomment this to pipe the output to python extract_results.py
    # try:
    #     # Run the training script, capturing output while still showing it in real-time
    #     # Print progress to stderr so it's visible even when stdout is captured by command substitution
    #     process = subprocess.Popen(
    #         cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    #     )
    #     last_line = ""
    #     for line in process.stdout:
    #         # Print to stderr so it's visible even when stdout is captured by $()
    #         print(line, end="", file=sys.stderr)
    #         stripped = line.strip()
    #         if stripped:
    #             last_line = stripped
    #     return_code = process.wait()
    #     if return_code != 0:
    #         print(f"Training failed with exit code {return_code}", file=sys.stderr)
    #         sys.exit(return_code)
    #     # Print the last non-empty line to stdout only if stdout is being captured
    #     # (not a TTY), so it's available for shell script capture but not displayed
    #     # when run directly in an interactive terminal
    #     if last_line and not sys.stdout.isatty():
    #         print(last_line, file=sys.stdout)
    # except KeyboardInterrupt:
    #     print("\nTraining interrupted by user", file=sys.stderr)
    #     sys.exit(1)

    # Default behavior with rich printing
    try:
        # Run the training script
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run training on the 8 GPUs with most free VRAM"
    )
    parser.add_argument("config_file", nargs="?", help="Path to the YAML config file")
    parser.add_argument(
        "--num_gpus", type=int, default=8, help="Number of GPUs to use (default: 8)"
    )
    parser.add_argument(
        "--list_gpus", action="store_true", help="List all available GPUs and exit"
    )

    args, additional_args = parser.parse_known_args()

    # List GPUs if requested
    if args.list_gpus:
        gpu_info = get_gpu_memory_info()
        print("Available GPUs:")
        for gpu_id, free_memory, total_memory in gpu_info:
            free_gb = free_memory / 1024
            total_gb = total_memory / 1024
            print(f"  GPU {gpu_id}: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
        return

    # Check if config file is provided
    if not args.config_file:
        print("Error: config_file is required when not using --list-gpus")
        parser.print_help()
        sys.exit(1)

    # Check if config file exists
    if not os.path.exists(args.config_file):
        print(f"Error: Config file '{args.config_file}' not found")
        sys.exit(1)

    # Select the best GPUs
    selected_gpus = select_best_gpus(args.num_gpus)

    if len(selected_gpus) == 0:
        print("No GPUs available")
        sys.exit(1)

    # Run training
    run_training_with_gpus(selected_gpus, args.config_file, additional_args)


if __name__ == "__main__":
    main()

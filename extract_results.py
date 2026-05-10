#!/usr/bin/env python3
"""
Extract and aggregate results from multiple runs.

This script takes a prefix string and num_runs integer as arguments,
matches directories under ckpts with the prefix, and computes mean and
standard deviation of metrics from best_metrics.json files.
"""

import os
import json
import argparse
import csv
import re
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_best_metrics(file_path):
    """
    Load best_metrics.json file.
    Handles both single JSON object and JSONL format (multiple JSON objects per line).

    Args:
        file_path: Path to the best_metrics.json file

    Returns:
        Dictionary of metrics, or None if file doesn't exist or is empty
    """
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r") as f:
            content = f.read().strip()
            if not content:
                return None

            # Try to parse as single JSON object first
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If that fails, try JSONL format (one JSON object per line)
                # Take the last line as it should be the most recent/best metrics
                lines = content.strip().split("\n")
                if lines:
                    return json.loads(lines[-1])
    except Exception as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return None


def find_matching_directories(ckpts_dir, prefix):
    """
    Find all directories in ckpts_dir that start with the given prefix.

    Args:
        ckpts_dir: Path to the ckpts directory
        prefix: Prefix string to match

    Returns:
        List of matching directory paths
    """
    if not os.path.exists(ckpts_dir):
        return []

    matching_dirs = []
    for item in os.listdir(ckpts_dir):
        item_path = os.path.join(ckpts_dir, item)
        if os.path.isdir(item_path) and item.startswith(prefix):
            matching_dirs.append(item_path)

    return sorted(matching_dirs)


def aggregate_metrics(matching_dirs):
    """
    Aggregate metrics from all best_metrics.json files in matching directories.

    Args:
        matching_dirs: List of directory paths

    Returns:
        Dictionary mapping metric names to lists of values
    """
    all_metrics = defaultdict(list)

    for dir_path in matching_dirs:
        best_metrics_file = os.path.join(dir_path, "best_metrics.json")
        metrics = load_best_metrics(best_metrics_file)

        if metrics is None:
            raise ValueError(f"No best_metrics.json found or empty in {dir_path}")

        # Collect all metrics
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                all_metrics[metric_name].append(metric_value)
            else:
                print(
                    f"Warning: Metric {metric_name} in {dir_path} is not numeric: {metric_value}"
                )

    return all_metrics


def compute_statistics(metrics_dict):
    """
    Compute mean and standard deviation for each metric.

    Args:
        metrics_dict: Dictionary mapping metric names to lists of values

    Returns:
        Dictionary mapping metric names to (mean, std) tuples
    """
    stats = {}
    for metric_name, values in metrics_dict.items():
        if len(values) > 0:
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0.0
            stats[metric_name] = (mean, std)

    return stats


def sort_metrics(metric_names):
    """
    Sort metrics so that valid metrics come before test metrics.

    Args:
        metric_names: List of metric names

    Returns:
        Sorted list of metric names
    """

    def sort_key(name):
        if name.startswith("valid/"):
            return (0, name)
        elif name.startswith("test/"):
            return (1, name)
        else:
            return (2, name)

    return sorted(metric_names, key=sort_key)


def print_results(stats, prefix):
    """
    Print aggregated results with mean and standard deviation.

    Args:
        stats: Dictionary mapping metric names to (mean, std) tuples
        prefix: Prefix used for matching directories
    """
    print(f"\n{'='*80}")
    print(f"Results for prefix: {prefix}")
    print(f"{'='*80}")
    print(f"\n{'Metric':<50} {'Value':<30}")
    print(f"{'-'*80}")

    # Sort metrics so valid comes before test
    sorted_metrics = sort_metrics(stats.keys())
    for metric_name in sorted_metrics:
        mean, std = stats[metric_name]
        if "mse" in metric_name or "mae" in metric_name or "rmse" in metric_name:
            formatted_value = f"{mean:.4f} ± {std:.4f}"
        else:
            formatted_value = f"{mean*100:.2f} ± {std*100:.2f}"
        print(f"{metric_name:<50} {formatted_value:<30}")

    print(f"{'='*80}\n")


def write_csv_results(stats, prefix, output_dir):
    """
    Write aggregated results to a CSV file.
    Valid metrics come before test metrics.

    Args:
        stats: Dictionary mapping metric names to (mean, std) tuples
        prefix: Prefix used for matching directories
        output_dir: Directory to write the CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate CSV filename based on prefix (sanitize for filesystem)
    # Replace characters that might cause issues in filenames
    safe_prefix = prefix.replace("/", "_").replace("\\", "_").replace(",", "_")
    csv_filename = f"{safe_prefix}_results.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    # Sort metrics so valid comes before test
    sorted_metrics = sort_metrics(stats.keys())

    # Write to CSV - single row format
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Header row: Name, then all metric names
        header = ["Name"] + sorted_metrics
        writer.writerow(header)

        # Data row: prefix, then all formatted values
        row_data = [prefix]
        for metric_name in sorted_metrics:
            mean, std = stats[metric_name]
            if "mse" in metric_name or "mae" in metric_name or "rmse" in metric_name:
                formatted_value = f"{mean:.4f} ± {std:.4f}"
            else:
                formatted_value = f"{mean*100:.2f} ± {std*100:.2f}"
            row_data.append(formatted_value)

        writer.writerow(row_data)

    print(f"Results written to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and aggregate results from multiple runs",
    )
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        required=True,
        help="Prefix string to match directories in ckpts",
    )
    parser.add_argument(
        "--num_runs",
        "-n",
        type=int,
        required=True,
        help="Expected number of matching directories",
    )
    parser.add_argument(
        "--ckpts_dir",
        "-c",
        type=str,
        default="ckpts",
        help="Path to the ckpts directory (default: ckpts)",
    )

    args = parser.parse_args()

    # Get the script directory and construct ckpts path
    script_dir = Path(__file__).parent.absolute()
    ckpts_dir = os.path.join(script_dir, args.ckpts_dir)

    # Remove anything after -seed* from the prefix
    prefix = re.sub(r"-seed.*$", "", args.prefix)

    # Find matching directories
    matching_dirs = find_matching_directories(ckpts_dir, prefix)

    # Assert that number of matches equals num_runs
    num_matches = len(matching_dirs)
    assert num_matches == args.num_runs, (
        f"Expected {args.num_runs} directories matching prefix '{prefix}', "
        f"but found {num_matches}. Matching directories: {matching_dirs}"
    )

    print(f"Found {num_matches} matching directories:")
    for dir_path in matching_dirs:
        print(f"  - {os.path.basename(dir_path)}")

    # Aggregate metrics from all matching directories
    all_metrics = aggregate_metrics(matching_dirs)

    if not all_metrics:
        print(f"Warning: No metrics found in any of the matching directories.")
        return

    # Compute statistics
    stats = compute_statistics(all_metrics)

    # Print results
    print_results(stats, prefix)

    # Write results to CSV file
    write_csv_results(stats, prefix, "results")


if __name__ == "__main__":
    main()

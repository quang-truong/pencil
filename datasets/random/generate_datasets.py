import json
import random
from generate_graphs import generate_multiple_graphs
import os


def generate_dataset(num_graphs, path_restricted, return_neg_targets, seed=None):
    """
    Generate a dataset of random graphs

    Args:
        num_graphs: Number of graphs to generate
        seed: Random seed for reproducibility

    Returns:
        List of graph dictionaries
    """
    if seed is not None:
        random.seed(seed)

    print(f"Generating {num_graphs} graphs...")
    graphs = generate_multiple_graphs(
        num_graphs,
        path_restricted=path_restricted,
        return_neg_targets=return_neg_targets,
    )
    print(f"Generated {len(graphs)} graphs successfully!")

    return graphs


def save_graphs_to_json(graphs, filepath):
    """
    Save graphs to JSON file

    Args:
        graphs: List of graph dictionaries
        filepath: Path to save the JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(graphs, f, indent=2)

    print(f"Saved {len(graphs)} graphs to {filepath}")


def main():
    """Generate training, validation, and test datasets"""

    import rootutils

    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    from definitions import DATA_DIR

    # Generate training dataset (10,000 graphs)
    print("=" * 60)
    print("GENERATING TRAINING DATASET")
    print("=" * 60)
    train_graphs = generate_dataset(
        10000, path_restricted=True, return_neg_targets=True, seed=42
    )
    save_graphs_to_json(train_graphs, os.path.join(DATA_DIR, "random", "train.json"))

    # Generate validation dataset (2,000 graphs)
    print("\n" + "=" * 60)
    print("GENERATING VALIDATION DATASET")
    print("=" * 60)
    val_graphs = generate_dataset(
        200, path_restricted=True, return_neg_targets=True, seed=123
    )
    save_graphs_to_json(val_graphs, os.path.join(DATA_DIR, "random", "val.json"))

    # Generate test dataset (2,000 graphs)
    print("\n" + "=" * 60)
    print("GENERATING TEST DATASET")
    print("=" * 60)
    test_graphs = generate_dataset(
        200, path_restricted=True, return_neg_targets=True, seed=456
    )
    save_graphs_to_json(test_graphs, os.path.join(DATA_DIR, "random", "test.json"))

    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Training graphs: {len(train_graphs)}")
    print(f"Validation graphs: {len(val_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")
    print(f"Total graphs: {len(train_graphs) + len(val_graphs) + len(test_graphs)}")

    # Print some statistics about the graphs
    def analyze_graphs(graphs, name):
        total_nodes = sum(
            len(
                set([edge[0] for edge in g["edges"]] + [edge[1] for edge in g["edges"]])
            )
            for g in graphs
        )
        total_edges = sum(
            len(g["edges"]) // 2 for g in graphs
        )  # Divide by 2 since edges are bidirectional
        avg_nodes = total_nodes / len(graphs)
        avg_edges = total_edges / len(graphs)

        print(f"\n{name} Dataset Statistics:")
        print(f"  Average nodes per graph: {avg_nodes:.1f}")
        print(f"  Average edges per graph: {avg_edges:.1f}")
        print(f"  Total unique nodes across all graphs: {total_nodes}")
        print(f"  Total edges across all graphs: {total_edges}")

    analyze_graphs(train_graphs, "Training")
    analyze_graphs(val_graphs, "Validation")
    analyze_graphs(test_graphs, "Test")


if __name__ == "__main__":
    main()

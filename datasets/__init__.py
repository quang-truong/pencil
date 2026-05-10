from .collator import Collator
from .planetoid import read_data_planetoid
from .ogbl import read_data_ogb
from .dataset_wrapper import DatasetWrapper
from utils import rank_zero_print as print


def load_dataset(configs):
    """
    Unified dataset loading function that supports both OGBL and Planetoid datasets.

    Args:
        configs: Configuration object containing dataset information
        rank: Process rank for distributed training (default: 0)

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset, raw_dataset) for fine-tuning
               or (train_dataset, raw_dataset) for pretraining
    """
    dataset_name = configs.dataset

    # Strip heuristic prefixes to get base dataset name
    base_dataset_name = dataset_name
    if dataset_name.startswith("cn-"):
        base_dataset_name = dataset_name[len("cn-") :]
    elif dataset_name.startswith("aa-"):
        base_dataset_name = dataset_name[len("aa-") :]
    elif dataset_name.startswith("ra-"):
        base_dataset_name = dataset_name[len("ra-") :]
    elif dataset_name.startswith("katz-"):
        base_dataset_name = dataset_name[len("katz-") :]
    elif dataset_name.startswith("shortest-path-"):
        base_dataset_name = dataset_name[len("shortest-path-") :]
    elif dataset_name.startswith("pagerank-"):
        base_dataset_name = dataset_name[len("pagerank-") :]

    # Strip heart- prefix if present (after heuristic prefix)
    if base_dataset_name.startswith("heart-"):
        base_dataset_name = base_dataset_name[len("heart-") :]

    # Check if it's a Planetoid dataset (cora, citeseer, pubmed)
    if base_dataset_name in ["cora", "citeseer", "pubmed"]:
        print(f"Loading Planetoid dataset: {dataset_name}")
        return read_data_planetoid(configs)

    # Check if it's an OGBL dataset
    elif base_dataset_name in [
        "ogbl-ppa",
        "ogbl-citation2",
        "ogbl-collab",
        "ogbl-ddi",
    ]:
        print(f"Loading OGBL dataset: {dataset_name}")
        return read_data_ogb(configs)

    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets: cora, citeseer, pubmed, ogbl-ppa, ogbl-citation2, ogbl-collab, ogbl-ddi "
            f"(with optional prefixes: cn-, aa-, ra-, katz-, shortest-path-, pagerank-, heart-)"
        )

import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from datasets.dataset_map import ShaDowKHopSeqFromEdgesMapDataset

from definitions import DATA_DIR
from utils import rank_zero_print as print
from datasets.utils import read_heart_split_edges, compute_and_cache_heuristics

HEART_DIR = os.path.join(DATA_DIR, "heart")


def read_data_planetoid(configs, efficient_heart=True):
    """
    Read data for Planetoid datasets. Returns similar format to read_data_ogb.
    """
    data_name = configs.dataset

    # Handle heart- prefix: strip it for data loading, but remember for heart split
    use_heart_split = data_name.startswith("heart-")
    if use_heart_split:
        base_data_name = data_name[len("heart-") :]  # Strip "heart-" prefix
        print(f"Detected heart split mode. Using base dataset name: {base_data_name}")
    else:
        base_data_name = data_name

    # Handle heuristic prefixes (only for cora and citeseer)
    heuristic_prefix_map = {
        "cn-": "cn",
        "aa-": "aa",
        "ra-": "ra",
        "katz-": "katz",
        "shortest-path-": "shortest_path",
        "pagerank-": "pagerank",
    }
    heuristic_name = None

    # Heuristic prefixes are mutually exclusive with heart- for now.
    if not use_heart_split:
        for prefix, h_name in heuristic_prefix_map.items():
            if base_data_name.startswith(prefix):
                raw_name = base_data_name[len(prefix) :]
                if raw_name not in {"cora", "citeseer"}:
                    raise ValueError(
                        f"Heuristic prefix '{prefix}' is only supported for cora and citeseer, "
                        f"got dataset '{raw_name}'."
                    )
                heuristic_name = h_name
                base_data_name = raw_name  # Use raw dataset name for disk paths
                print(
                    f"Detected heuristic mode '{heuristic_name}' for Planetoid dataset '{base_data_name}'."
                )
                break

    print("Loading data...")

    node_set = set()
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []

    # Read positive edges (use base_data_name for all file paths)
    for split in ["train", "test", "valid"]:
        path = os.path.join(DATA_DIR, base_data_name, f"{split}_pos.txt")

        for line in open(path, "r"):
            sub, obj = line.strip().split("\t")
            sub, obj = int(sub), int(obj)

            node_set.add(sub)
            node_set.add(obj)

            if sub == obj:
                continue

            if split == "train":
                train_pos.append((sub, obj))

            if split == "valid":
                valid_pos.append((sub, obj))
            if split == "test":
                test_pos.append((sub, obj))

    num_nodes = len(node_set)

    # Read negative edges (use base_data_name for all file paths)
    for split in ["test", "valid"]:
        path = os.path.join(DATA_DIR, base_data_name, f"{split}_neg.txt")

        for line in open(path, "r"):
            sub, obj = line.strip().split("\t")
            sub, obj = int(sub), int(obj)

            if split == "valid":
                valid_neg.append((sub, obj))
            if split == "test":
                test_neg.append((sub, obj))

    # Convert to tensors
    train_pos_tensor = torch.tensor(train_pos)
    valid_pos_tensor = torch.tensor(valid_pos)
    valid_neg_tensor = torch.tensor(valid_neg)
    test_pos_tensor = torch.tensor(test_pos)
    test_neg_tensor = torch.tensor(test_neg)

    # Overwrite negative edges with heart split if enabled
    if use_heart_split:
        # Use base_data_name for heart split file paths
        heart_valid_path = os.path.join(
            HEART_DIR, base_data_name, "heart_valid_samples.npy"
        )
        heart_test_path = os.path.join(
            HEART_DIR, base_data_name, "heart_test_samples.npy"
        )

        if os.path.exists(heart_valid_path):
            valid_neg_tensor, valid_unique_edges, valid_orig_to_unique, valid_labels = (
                read_heart_split_edges(heart_valid_path, valid_pos_tensor, "valid")
            )
            print(
                f"Loaded heart split validation negative edges from {heart_valid_path}"
            )
        else:
            raise FileNotFoundError(
                f"Heart split enabled but {heart_valid_path} not found."
            )

        if os.path.exists(heart_test_path):
            test_neg_tensor, test_unique_edges, test_orig_to_unique, test_labels = (
                read_heart_split_edges(heart_test_path, test_pos_tensor, "test")
            )
            print(f"Loaded heart split test negative edges from {heart_test_path}")
        else:
            raise FileNotFoundError(
                f"Heart split enabled but {heart_test_path} not found."
            )

    # Create edge_index for training graph (undirected)
    train_edge = torch.transpose(torch.tensor(train_pos), 1, 0)
    edge_index = torch.cat((train_edge, train_edge[[1, 0]]), dim=1)
    # Explicitly remove self-loops
    edge_index, _ = remove_self_loops(edge_index, None)
    edge_weight = torch.ones(edge_index.size(1))

    # Load node features (use base_data_name for file path)
    feature_embeddings = torch.load(
        os.path.join(DATA_DIR, base_data_name, "gnn_feature")
    )
    feature_embeddings = feature_embeddings["entity_embedding"]

    # Create PyG Data object
    data = Data(
        num_nodes=num_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        x=feature_embeddings,
        id=torch.arange(num_nodes, dtype=torch.long),
    )

    # If a heuristic prefix is used, pre-compute all-pairs heuristic scores.
    # These scores will be attached as regression labels for dynamically sampled edges.
    heuristic_labels = None
    if heuristic_name is not None:
        _, heuristic_labels = compute_and_cache_heuristics(
            data=data,
            heuristic_name=heuristic_name,
            dataset_name=base_data_name,
        )

    # Create split_edge dictionary in OGB format
    split_edge = {
        "train": {
            "edge": train_pos_tensor,
        },
        "valid": {
            "edge": (
                valid_unique_edges
                if use_heart_split and efficient_heart
                else valid_pos_tensor
            ),
            # torch.empty because we don't attach labels to the data
            "edge_neg": (
                torch.empty(0, 2)
                if use_heart_split and efficient_heart
                else valid_neg_tensor
            ),
        },
        "test": {
            "edge": (
                test_unique_edges
                if use_heart_split and efficient_heart
                else test_pos_tensor
            ),
            # torch.empty because we don't attach labels to the data
            "edge_neg": (
                torch.empty(0, 2)
                if use_heart_split and efficient_heart
                else test_neg_tensor
            ),
        },
    }

    # Get pretrain mode from sampling config
    pretrain_mode = configs.sampling_config.get("pretrain_mode", False)
    use_features = configs.get("use_features", False)

    # Replace features with ones if not using features
    if not use_features:
        data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)

    if pretrain_mode:
        print("Loading pre-train dataset")
        train_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            data,
            configs.sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
            heuristic_labels=heuristic_labels,
        )
        return train_dataset, [data]
    else:
        print("Loading train, valid, test dataset")
        print("---Train dataset---")
        train_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            data,
            configs.sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
            heuristic_labels=heuristic_labels,
        )
        print("---Valid dataset---")
        valid_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            data,
            configs.sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="valid",
            labels=valid_labels if use_heart_split and efficient_heart else None,
            orig_to_unique=(
                valid_orig_to_unique if use_heart_split and efficient_heart else None
            ),
            heuristic_labels=heuristic_labels,
        )
        print("---Test dataset---")
        test_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            data,
            configs.sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="test",
            labels=test_labels if use_heart_split and efficient_heart else None,
            orig_to_unique=(
                test_orig_to_unique if use_heart_split and efficient_heart else None
            ),
            heuristic_labels=heuristic_labels,
        )
        print(
            f"Split dataset based on given train/valid/test index!\nTrain: "
            f"{len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return train_dataset, valid_dataset, test_dataset, [data]

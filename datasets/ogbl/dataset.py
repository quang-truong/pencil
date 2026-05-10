from ogb.linkproppred import PygLinkPropPredDataset
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, remove_self_loops
import torch
import numpy as np
import os
from torch_sparse import SparseTensor
from torch.nn.init import xavier_uniform_
from utils import rank_zero_print as print
from rich import print as rich_print
from datasets.dataset_map import ShaDowKHopSeqFromEdgesMapDataset
from torch_geometric.data import Data
from datasets.utils import read_heart_split_edges

from definitions import DATA_DIR

HEART_DIR = os.path.join(DATA_DIR, "heart")


def read_data_ogb(configs, efficient_heart=True):
    """
    Read data for OGB datasets
    """
    readers = {
        "ogbl-ppa": _read_ogbl_ppa,
        "ogbl-citation2": _read_ogbl_citation2,
        "ogbl-collab": _read_ogbl_collab,
        "ogbl-ddi": _read_ogbl_ddi,
    }

    # Handle heart- prefix: strip it for OGB loading, but remember for heart split
    data_name = configs.dataset
    use_heart_split = data_name.startswith("heart-")
    if use_heart_split:
        base_data_name = data_name[len("heart-") :]  # Strip "heart-" prefix
        print(f"Detected heart split mode. Using base dataset name: {base_data_name}")
    else:
        base_data_name = data_name

    print("Loading data...")

    # Load data from OGB (use base_data_name to load the actual OGB dataset)
    dataset = PygLinkPropPredDataset(name=base_data_name, root=DATA_DIR)
    data = dataset[0]
    data.id = torch.arange(data.num_nodes, dtype=torch.long)

    # Get edge splits
    split_edge = dataset.get_edge_split()

    # Filter by year for collab datasets
    if "collab" in base_data_name:
        data, split_edge = filter_by_year(data, split_edge)

    pretrain_mode = configs.sampling_config.get("pretrain_mode", False)
    use_features = configs.get("use_features", False)

    # Get node features
    if hasattr(data, "x") and data.x is not None:
        data.x = data.x.to(torch.float)
    # ogbl-ddi has no node features
    elif "ogbl-ddi" in base_data_name:
        assert (
            not use_features
        ), "ogbl-ddi has no node features, use_features must be False"
    else:
        raise ValueError(f"Node features not found for {base_data_name}")

    reader = readers.get(base_data_name)
    if reader is None:
        raise ValueError(f"Dataset {base_data_name} not supported")

    if pretrain_mode:
        train_dataset, raw_dataset = reader(
            data,
            split_edge,
            configs.sampling_config,
            use_features=use_features,
            return_valid_test=False,
            use_heart_split=use_heart_split,
        )
        return train_dataset, raw_dataset
    else:
        train_dataset, valid_dataset, test_dataset, raw_dataset = reader(
            data,
            split_edge,
            configs.sampling_config,
            use_features=use_features,
            return_valid_test=True,
            use_heart_split=use_heart_split,
            efficient_heart=efficient_heart,
        )
        return train_dataset, valid_dataset, test_dataset, raw_dataset


def filter_by_year(data, split_edge, year=2007):
    """
    From BUDDY code

    remove edges before year from data and split edge
    @param data: pyg Data, pyg SplitEdge
    @param split_edges:
    @param year: int first year to use
    @return: pyg Data, pyg SplitEdge
    """
    selected_year_index = torch.reshape(
        (split_edge["train"]["year"] >= year).nonzero(as_tuple=False), (-1,)
    )
    split_edge["train"]["edge"] = split_edge["train"]["edge"][selected_year_index]
    split_edge["train"]["weight"] = split_edge["train"]["weight"][selected_year_index]
    split_edge["train"]["year"] = split_edge["train"]["year"][selected_year_index]
    train_edge_index = split_edge["train"]["edge"].t()
    # Explicitly remove self-loops before making undirected
    train_edge_index, train_edge_weight = remove_self_loops(
        train_edge_index, split_edge["train"]["weight"]
    )
    # create adjacency matrix
    new_edges = to_undirected(train_edge_index, train_edge_weight, reduce="add")
    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight.unsqueeze(-1)
    return data, split_edge


def _read_ogbl_ppa(
    graph,
    split_edge,
    sampling_config,
    use_features=False,
    return_valid_test=False,
    use_heart_split=False,
    efficient_heart=True,
):
    pretrain_mode = sampling_config.get("pretrain_mode", False)
    # Explicitly remove self-loops from graph edge_index
    graph.edge_index, _ = remove_self_loops(graph.edge_index, None)
    if not use_features:
        graph.x = torch.ones([graph.num_nodes, 1], dtype=torch.float)

    # Apply heart split modifications if enabled
    if use_heart_split:
        dataset_name = "ogbl-ppa"
        # Overwrite positive edge indices for valid and test
        valid_samples_index_path = os.path.join(
            HEART_DIR, dataset_name, "valid_samples_index.pt"
        )
        test_samples_index_path = os.path.join(
            HEART_DIR, dataset_name, "test_samples_index.pt"
        )

        if os.path.exists(valid_samples_index_path):
            with open(valid_samples_index_path, "rb") as f:
                val_pos_ix = torch.load(f)
            split_edge["valid"]["edge"] = split_edge["valid"]["edge"][val_pos_ix]
            print(
                f"Loaded heart split validation positive edge indices from {valid_samples_index_path}"
            )
        else:
            raise FileNotFoundError(
                f"Heart split enabled for ogbl-ppa but {valid_samples_index_path} not found."
            )

        if os.path.exists(test_samples_index_path):
            with open(test_samples_index_path, "rb") as f:
                test_pos_ix = torch.load(f)
            split_edge["test"]["edge"] = split_edge["test"]["edge"][test_pos_ix]
            print(
                f"Loaded heart split test positive edge indices from {test_samples_index_path}"
            )
        else:
            raise FileNotFoundError(
                f"Heart split enabled for ogbl-ppa but {test_samples_index_path} not found."
            )

        # Overwrite negative edges
        heart_valid_path = os.path.join(
            HEART_DIR, dataset_name, "heart_valid_samples.npy"
        )
        heart_test_path = os.path.join(
            HEART_DIR, dataset_name, "heart_test_samples.npy"
        )

        if os.path.exists(heart_valid_path):
            valid_neg_tensor, valid_unique_edges, valid_orig_to_unique, valid_labels = (
                read_heart_split_edges(
                    heart_valid_path, split_edge["valid"]["edge"], "valid"
                )
            )
            print(f"Loaded heart split validation edges from {heart_valid_path}")
        else:
            raise FileNotFoundError(
                f"Heart split enabled but {heart_valid_path} not found."
            )

        if os.path.exists(heart_test_path):
            test_neg_tensor, test_unique_edges, test_orig_to_unique, test_labels = (
                read_heart_split_edges(
                    heart_test_path, split_edge["test"]["edge"], "test"
                )
            )
            print(f"Loaded heart split test edges from {heart_test_path}")
        else:
            raise FileNotFoundError(
                f"Heart split enabled but {heart_test_path} not found."
            )

    # Update split_edge for heart split mode
    if use_heart_split:
        if efficient_heart:
            split_edge["valid"]["edge"] = valid_unique_edges
            split_edge["valid"]["edge_neg"] = torch.empty(0, 2)
            split_edge["test"]["edge"] = test_unique_edges
            split_edge["test"]["edge_neg"] = torch.empty(0, 2)
        else:
            split_edge["valid"]["edge_neg"] = valid_neg_tensor
            split_edge["test"]["edge_neg"] = test_neg_tensor

    if return_valid_test:
        print("Loading train, valid, test dataset")
        print("---Train dataset---")
        train_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
        )
        print("---Valid dataset---")
        valid_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="valid",
            labels=valid_labels if use_heart_split and efficient_heart else None,
            orig_to_unique=(
                valid_orig_to_unique if use_heart_split and efficient_heart else None
            ),
        )
        print("---Test dataset---")
        test_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="test",
            labels=test_labels if use_heart_split and efficient_heart else None,
            orig_to_unique=(
                test_orig_to_unique if use_heart_split and efficient_heart else None
            ),
        )
        print(
            f"Split dataset based on given train/valid/test index!\nTrain: \
                {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            [graph],
        )
    else:
        print("Loading pre-train dataset")
        dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
        )
        return dataset, [graph]


def _read_ogbl_ddi(
    graph,
    split_edge,
    sampling_config,
    use_features=False,
    return_valid_test=False,
    use_heart_split=False,
    efficient_heart=True,
):
    pretrain_mode = sampling_config.get("pretrain_mode", False)
    # Explicitly remove self-loops from graph edge_index
    graph.edge_index, _ = remove_self_loops(graph.edge_index, None)
    if not use_features:
        graph.x = torch.ones([graph.num_nodes, 1], dtype=torch.float)

    # Apply heart split modifications if enabled
    if use_heart_split:
        dataset_name = "ogbl-ddi"
        # Overwrite negative edges
        heart_valid_path = os.path.join(
            HEART_DIR, dataset_name, "heart_valid_samples.npy"
        )
        heart_test_path = os.path.join(
            HEART_DIR, dataset_name, "heart_test_samples.npy"
        )

        if os.path.exists(heart_valid_path):
            valid_neg_tensor, valid_unique_edges, valid_orig_to_unique, valid_labels = (
                read_heart_split_edges(
                    heart_valid_path, split_edge["valid"]["edge"], "valid"
                )
            )
            print(f"Loaded heart split validation edges from {heart_valid_path}")
        else:
            raise FileNotFoundError(
                f"Heart split enabled but {heart_valid_path} not found."
            )

        if os.path.exists(heart_test_path):
            test_neg_tensor, test_unique_edges, test_orig_to_unique, test_labels = (
                read_heart_split_edges(
                    heart_test_path, split_edge["test"]["edge"], "test"
                )
            )
            print(f"Loaded heart split test edges from {heart_test_path}")
        else:
            raise FileNotFoundError(
                f"Heart split enabled but {heart_test_path} not found."
            )

    # Update split_edge for heart split mode
    if use_heart_split:
        if efficient_heart:
            split_edge["valid"]["edge"] = valid_unique_edges
            split_edge["valid"]["edge_neg"] = torch.empty(0, 2)
            split_edge["test"]["edge"] = test_unique_edges
            split_edge["test"]["edge_neg"] = torch.empty(0, 2)
        else:
            split_edge["valid"]["edge_neg"] = valid_neg_tensor
            split_edge["test"]["edge_neg"] = test_neg_tensor

    if return_valid_test:
        print("Loading train, valid, test dataset")
        print("---Train dataset---")
        train_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
        )
        print("---Valid dataset---")
        valid_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="valid",
            labels=valid_labels if use_heart_split and efficient_heart else None,
            orig_to_unique=(
                valid_orig_to_unique if use_heart_split and efficient_heart else None
            ),
        )
        print("---Test dataset---")
        test_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="test",
            labels=test_labels if use_heart_split and efficient_heart else None,
            orig_to_unique=(
                test_orig_to_unique if use_heart_split and efficient_heart else None
            ),
        )
        print(
            f"Split dataset based on given train/valid/test index!\nTrain: \
                {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            [graph],
        )
    else:
        print("Loading pre-train dataset")
        dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
        )
        return dataset, [graph]


def _read_ogbl_collab(
    graph,
    split_edge,
    sampling_config,
    use_features=False,
    return_valid_test=False,
    use_heart_split=False,
    efficient_heart=True,
):
    pretrain_mode = sampling_config.get("pretrain_mode", False)
    if not use_features:
        graph.x = torch.ones([graph.num_nodes, 1], dtype=torch.float)

    # Apply heart split modifications if enabled
    if use_heart_split:
        dataset_name = "ogbl-collab"
        # Overwrite negative edges
        heart_valid_path = os.path.join(
            HEART_DIR, dataset_name, "heart_valid_samples.npy"
        )
        heart_test_path = os.path.join(
            HEART_DIR, dataset_name, "heart_test_samples.npy"
        )

        if os.path.exists(heart_valid_path):
            valid_neg_tensor, valid_unique_edges, valid_orig_to_unique, valid_labels = (
                read_heart_split_edges(
                    heart_valid_path, split_edge["valid"]["edge"], "valid"
                )
            )
            print(f"Loaded heart split validation edges from {heart_valid_path}")
        else:
            raise FileNotFoundError(
                f"Heart split enabled but {heart_valid_path} not found."
            )

        if os.path.exists(heart_test_path):
            test_neg_tensor, test_unique_edges, test_orig_to_unique, test_labels = (
                read_heart_split_edges(
                    heart_test_path, split_edge["test"]["edge"], "test"
                )
            )
            print(f"Loaded heart split test edges from {heart_test_path}")
        else:
            raise FileNotFoundError(
                f"Heart split enabled but {heart_test_path} not found."
            )

    # Update split_edge for heart split mode
    if use_heart_split:
        if efficient_heart:
            split_edge["valid"]["edge"] = valid_unique_edges
            split_edge["valid"]["edge_neg"] = torch.empty(0, 2)
            split_edge["test"]["edge"] = test_unique_edges
            split_edge["test"]["edge_neg"] = torch.empty(0, 2)
        else:
            split_edge["valid"]["edge_neg"] = valid_neg_tensor
            split_edge["test"]["edge_neg"] = test_neg_tensor

    if return_valid_test:
        print("Loading train, valid, test dataset")

        # For ogbl-collab, validation edges are used for test predictions
        # Incorporate validation edges into the graph structure
        # Not using edge_weight for validation edges
        val_edge_index = split_edge["valid"]["edge"].t()
        # Explicitly remove self-loops before making undirected
        val_edge_index, _ = remove_self_loops(val_edge_index, None)
        val_edge_index = to_undirected(val_edge_index)

        # Safety check: ensure all validation edge nodes exist in the original graph
        max_val_node = val_edge_index.max().item()
        if max_val_node >= graph.num_nodes:
            print(
                f"[WARNING] Validation edges contain node {max_val_node} but graph only has {graph.num_nodes} nodes"
            )
            # Filter out edges with nodes that don't exist in the original graph
            valid_mask = (val_edge_index[0] < graph.num_nodes) & (
                val_edge_index[1] < graph.num_nodes
            )
            val_edge_index = val_edge_index[:, valid_mask]
            print(
                f"Filtered validation edges from {split_edge['valid']['edge'].shape[0]} to {val_edge_index.shape[1]} edges"
            )

        # Combine training and validation edges
        # Note: Both graph.edge_index and val_edge_index already have self-loops removed
        full_edge_index = torch.cat([graph.edge_index, val_edge_index], dim=-1)

        # Create edge weights (1.0 for all edges in validation set)
        graph.edge_weight = graph.edge_weight.to(torch.float)
        train_edge_weight = graph.edge_weight
        # Edge weight for validation edges is 1.0 following LPFormer
        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float)
        # Concatenate train and validation edge weights
        full_edge_weight = torch.cat([train_edge_weight, val_edge_weight], 0).view(-1)

        # Create a copy of the graph with the full edge index and weights
        full_graph = Data(
            num_nodes=graph.num_nodes,
            edge_index=full_edge_index,  # train edges and validation edges
            edge_weight=full_edge_weight,  # original edge weight + [1.0 .. 1.0] for validation edges
            x=graph.x,
            id=graph.id,
        )

        # Use the full graph for test dataset (includes validation edges)
        # Use original graph for train and valid datasets
        print("---Train dataset---")
        train_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
        )
        print("---Valid dataset---")
        valid_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="valid",
            labels=valid_labels if use_heart_split and efficient_heart else None,
            orig_to_unique=(
                valid_orig_to_unique if use_heart_split and efficient_heart else None
            ),
        )
        print("---Test dataset---")
        test_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            full_graph,  # Use full graph with validation edges for test
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="test",
            labels=test_labels if use_heart_split and efficient_heart else None,
            orig_to_unique=(
                test_orig_to_unique if use_heart_split and efficient_heart else None
            ),
        )
        print(
            f"Split dataset based on given train/valid/test index!\nTrain: \
                {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            [graph],  # Return original graph as raw dataset
        )
    else:
        print("Loading pre-train dataset")
        dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
        )
        return dataset, [graph]


def _read_ogbl_citation2(
    graph,
    split_edge,
    sampling_config,
    use_features=False,
    return_valid_test=False,
    use_heart_split=False,
    efficient_heart=True,
):
    # graph -> Data(num_nodes=2927963, edge_index=[2, 30387995], x=[2927963, 128], node_year=[2927963, 1])
    # ogbl-citation2 is directed graph
    # Explicitly remove self-loops before making undirected
    edge_index, _ = remove_self_loops(graph.edge_index, None)
    edge_index = to_undirected(edge_index)
    if not use_features:
        graph.x = torch.ones([graph.num_nodes, 1], dtype=torch.float)

    graph = Data(
        num_nodes=graph.num_nodes,
        edge_index=edge_index,
        x=graph.x,
        id=graph.id,
    )

    # update edge_index to split_edge
    split_edge["train"].update({"edge": graph.edge_index.T.clone()})
    unique_node_in_edge_idx = torch.unique(edge_index)
    allow_zero_edges = False
    pretrain_mode = sampling_config.get("pretrain_mode", False)

    # Case where isolated nodes exist
    if len(unique_node_in_edge_idx) < graph.num_nodes:
        allow_zero_edges = True
        print(
            f"unique-node-in-edge-index < num_nodes: {len(unique_node_in_edge_idx)} < {graph.num_nodes}!!!\n"
            f"isolated nodes exists, two isolated nodes will form a zero-edge subgraph!!!\n"
            f"set `allow_zero_edges` to be {allow_zero_edges}"
        )

    if return_valid_test:
        print("Loading train, valid, test dataset")

        edge = torch.vstack(
            [split_edge["train"]["source_node"], split_edge["train"]["target_node"]]
        ).T.clone()
        split_edge["train"].update({"edge": edge})
        print("---Train dataset---")
        train_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
            allow_zero_edges=allow_zero_edges,
        )
        print("---Valid dataset---")
        # Valid dataset
        # scope is the number of valid edges
        scope = split_edge["valid"]["source_node"].shape[0]
        idx_sel = _get_fixed_sampled_sorted_idx(scope, scope)
        dict_new = _get_reformatted_data_of_citation2(split_edge["valid"], idx_sel)
        split_edge["valid"].update(dict_new)

        # Apply heart split modifications after reformatting if enabled
        if use_heart_split:
            dataset_name = "ogbl-citation2"
            heart_valid_path = os.path.join(
                HEART_DIR, dataset_name, "heart_valid_samples.npy"
            )
            if os.path.exists(heart_valid_path):
                (
                    valid_neg_tensor,
                    valid_unique_edges,
                    valid_orig_to_unique,
                    valid_labels,
                ) = read_heart_split_edges(
                    heart_valid_path, split_edge["valid"]["edge"], "valid"
                )
                print(f"Loaded heart split validation edges from {heart_valid_path}")
            else:
                raise FileNotFoundError(
                    f"Heart split enabled but {heart_valid_path} not found."
                )

        # Update split_edge for heart split mode
        if use_heart_split:
            if efficient_heart:
                split_edge["valid"]["edge"] = valid_unique_edges
                split_edge["valid"]["edge_neg"] = torch.empty(0, 2)
            else:
                split_edge["valid"]["edge_neg"] = valid_neg_tensor

        valid_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="valid",
            allow_zero_edges=allow_zero_edges,
            labels=valid_labels if use_heart_split and efficient_heart else None,
            orig_to_unique=(
                valid_orig_to_unique if use_heart_split and efficient_heart else None
            ),
        )
        print("---Test dataset---")
        # Test dataset
        # scope is the number of test edges
        scope = split_edge["test"]["source_node"].shape[0]
        idx_sel = _get_fixed_sampled_sorted_idx(scope, scope)
        dict_new = _get_reformatted_data_of_citation2(split_edge["test"], idx_sel)
        split_edge["test"].update(dict_new)

        # Apply heart split modifications after reformatting if enabled
        if use_heart_split:
            dataset_name = "ogbl-citation2"
            heart_test_path = os.path.join(
                HEART_DIR, dataset_name, "heart_test_samples.npy"
            )
            if os.path.exists(heart_test_path):
                test_neg_tensor, test_unique_edges, test_orig_to_unique, test_labels = (
                    read_heart_split_edges(
                        heart_test_path, split_edge["test"]["edge"], "test"
                    )
                )
                print(f"Loaded heart split test edges from {heart_test_path}")
            else:
                raise FileNotFoundError(
                    f"Heart split enabled but {heart_test_path} not found."
                )

        # Update split_edge for heart split mode
        if use_heart_split:
            if efficient_heart:
                split_edge["test"]["edge"] = test_unique_edges
                split_edge["test"]["edge_neg"] = torch.empty(0, 2)
            else:
                split_edge["test"]["edge_neg"] = test_neg_tensor

        test_dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="test",
            allow_zero_edges=allow_zero_edges,
            labels=test_labels if use_heart_split and efficient_heart else None,
            orig_to_unique=(
                test_orig_to_unique if use_heart_split and efficient_heart else None
            ),
        )
        print(
            f"Split dataset based on given train/valid/test index!\nTrain: \
                {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}!"
        )
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            [graph],
        )
    else:
        dataset = ShaDowKHopSeqFromEdgesMapDataset(
            graph,
            sampling_config,
            pretrain_mode=pretrain_mode,
            split_edge=split_edge,
            data_split="train",
            allow_zero_edges=allow_zero_edges,
        )
        return dataset, [graph]


def _get_edge_neg(source_node, target_node_neg):
    """
    Create negative edges by combining source nodes with negative target nodes.

    This function handles two scenarios for negative edge generation:
    1. When source_node is 2D: Treats it as target_node_neg and swaps the roles
    2. When source_node is 1D: Creates edges by pairing each source with its negative targets

    Args:
        source_node (torch.Tensor): Source node IDs, either:
            - Shape [N_p] (1D): Single source node per positive edge
            - Shape [N_p, num_negs] (2D): Multiple source nodes per positive edge
        target_node_neg (torch.Tensor): Negative target node IDs, either:
            - Shape [N_p, num_negs] (2D): Multiple negative targets per positive edge
            - Shape [N_p] (1D): Single target node per positive edge (when source is 2D)

    Returns:
        torch.Tensor: Negative edges tensor of shape [N_p * num_negs, 2] where each row is [source, target]

    Example (Case 2 - Normal case):
        source_node = [1, 2, 3]  # Shape [3]
        target_node_neg = [[4, 5], [6, 7], [8, 9]]  # Shape [3, 2]
        # Returns: [[1, 4], [1, 5], [2, 6], [2, 7], [3, 8], [3, 9]]  # Shape [6, 2]

    Example (Case 1 - Symmetric case):
        source_node = [[1, 2], [3, 4], [5, 6]]  # Shape [3, 2] - 2D source
        target_node_neg = [7, 8, 9]  # Shape [3] - 1D target
        # Step 1: Recursively call with swapped args: _get_edge_neg([7, 8, 9], [[1, 2], [3, 4], [5, 6]])
        # Step 2: This triggers Case 2, creating: [[7, 1], [7, 2], [8, 3], [8, 4], [9, 5], [9, 6]]
        # Step 3: Swap columns [1, 0] to get: [[1, 7], [2, 7], [3, 8], [4, 8], [5, 9], [6, 9]]
        # Final result: [[1, 7], [2, 7], [3, 8], [4, 8], [5, 9], [6, 9]]  # Shape [6, 2]
    """
    if len(source_node.shape) == 2:
        # Case 1: source_node is 2D, treat it as target_node_neg and swap roles
        # This handles symmetric cases where we need to generate edges in both directions
        # invoke itself with swapped arguments
        edge_neg = _get_edge_neg(target_node_neg, source_node)
        # Swap columns to get [target, source] -> [source, target]
        edge_neg = edge_neg[:, [1, 0]].clone()
    else:
        # Case 2: source_node is 1D, target_node_neg is 2D (normal case)
        assert len(source_node.shape) == 1, "source_node must be 1D or 2D"
        assert (
            len(target_node_neg.shape) == 2
        ), "target_node_neg must be 2D when source_node is 1D"
        assert (
            source_node.shape[0] == target_node_neg.shape[0]
        ), "First dimension must match"

        num_negs = target_node_neg.shape[
            1
        ]  # Number of negative samples per positive edge

        # Expand source_node to match target_node_neg shape
        # [N_p] -> [N_p, 1] -> [N_p, num_negs]
        source_node = source_node.reshape((-1, 1)).expand(
            source_node.shape[0], num_negs
        )
        assert (
            source_node.shape == target_node_neg.shape
        ), "Shapes must match after expansion"

        # Create negative edges by stacking source and target tensors
        # Flatten both tensors and stack them horizontally
        edge_neg = torch.hstack(
            [source_node.reshape((-1, 1)), target_node_neg.reshape((-1, 1))]
        ).clone()

    return edge_neg


def _get_fixed_sampled_sorted_idx(scope, cnt_idx, seed=42):
    """
    Get fixed sampled sorted indices from scope
    """
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(scope, generator=g)[:cnt_idx]
    indices, _ = torch.sort(indices)
    return indices


def _get_reformatted_data_of_citation2(dict_, idx_sel):
    """
    Get reformatted data of citation2 by returning:
    - edge: [N_pos, 2]
    - edge_neg: [N_neg, 2]
    """
    edge = torch.vstack(
        [
            dict_["source_node"][idx_sel],
            dict_["target_node"][idx_sel],
        ]
    ).T.clone()
    edge_neg = _get_edge_neg(
        dict_["source_node"][idx_sel],
        dict_["target_node_neg"][idx_sel],
    )
    return {
        "edge": edge,
        "edge_neg": edge_neg,
    }

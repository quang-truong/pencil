import torch
import numpy as np
import pickle
import os
from scipy import sparse
from torch_geometric.utils import to_scipy_sparse_matrix
from utils import rank_zero_print as print
from definitions import DATA_DIR
from datasets.heuristics import CN, AA, RA, katz_close, shortest_path, pagerank


def normalize_edge(edge):
    """Normalize an edge so that (src, dst) and (dst, src) are treated as the same."""
    src, dst = edge[0], edge[1]
    return (min(src, dst), max(src, dst))


def get_unique_edges_with_mapping(edges_tensor):
    """Find unique edges and create mapping from original to unique indices.

    Args:
        edges_tensor: Tensor of shape (N, 2) containing edges

    Returns:
        unique_edges: Tensor of unique edges, shape (M, 2) where M <= N
        orig_to_unique: Tensor mapping original index to unique index, shape (N,)
    """
    edges_np = edges_tensor.numpy()
    num_edges = len(edges_np)

    # Normalize edges (treat (src, dst) and (dst, src) as the same)
    normalized_edges = [normalize_edge(edge) for edge in edges_np]

    # Get unique normalized edges while preserving insertion order
    # dict.fromkeys() preserves insertion order (Python 3.7+)
    unique_normalized_list = list(dict.fromkeys(normalized_edges))

    # Create mapping from unique normalized edge to its index
    unique_to_idx = {
        norm_edge: idx for idx, norm_edge in enumerate(unique_normalized_list)
    }

    # Map original indices to unique indices
    orig_to_unique = np.array(
        [unique_to_idx[norm_edge] for norm_edge in normalized_edges], dtype=np.int64
    )

    # Create unique edges tensor (already normalized to (min, max) format)
    # Since transformers will normalize anyway, we don't need to preserve original direction
    unique_edges = torch.tensor(unique_normalized_list, dtype=edges_tensor.dtype)

    # Compute stats
    num_unique = len(unique_edges)
    duplicate_percentage = 100.0 * (1 - num_unique / num_edges)
    stats = {
        "num_unique": num_unique,
        "num_total": num_edges,
        "duplicate_percentage": duplicate_percentage,
    }

    print(
        f"Found {num_unique} unique edges out of {num_edges} total edges "
        f"({duplicate_percentage:.2f}% duplicates)"
    )

    return unique_edges, torch.from_numpy(orig_to_unique), stats


def read_heart_split_edges(path, pos_tensor, split="valid"):
    """
    Read heart split edges from a file and return unique edges, original to unique mapping, and labels.

    Args:
        path: Path to the heart split edges file
        pos_tensor: Tensor of shape (N, 2) containing positive edges
        split: Split to read from the file (valid or test)

    Returns:
        neg_tensor: Tensor of negative edges, shape (M, 2)
        unique_edges: Tensor of unique edges, shape (K, 2) where K <= N+M
        orig_to_unique: Tensor mapping original index to unique index, shape (N+M,)
        labels: Tensor of labels, shape (N+M,)
    """
    # Create cache file path in the same directory as the input path
    cache_dir = os.path.dirname(path)
    cache_path = os.path.join(cache_dir, f"{split}_efficient_heart.pkl")

    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"Loading cached efficient heart data from {cache_path}")
        with open(cache_path, "rb") as f:
            cached_data = pickle.load(f)
            # Print stats if available
            if "stats" in cached_data:
                stats = cached_data["stats"]
                print(
                    f"Found {stats['num_unique']} unique edges out of {stats['num_total']} total edges "
                    f"({stats['duplicate_percentage']:.2f}% duplicates)"
                )
            return (
                cached_data["neg_tensor"],
                cached_data["unique_edges"],
                cached_data["orig_to_unique"],
                cached_data["labels"],
            )

    # Cache doesn't exist, compute values
    with open(path, "rb") as f:
        neg_edge = np.load(f)
        # Heart splits have shape (N, 500, 2) for all datasets
        # Reshape to (N*500, 2) as expected by the dataset
        if len(neg_edge.shape) == 3:
            num_queries, num_negs, _ = neg_edge.shape
            neg_edge = neg_edge.reshape(-1, 2)
            print(
                f"Reshaped heart split {split} negative edges from "
                f"({num_queries}, {num_negs}, 2) to {neg_edge.shape}"
            )
        neg_tensor = torch.from_numpy(neg_edge)
        all_edges = torch.cat([pos_tensor, neg_tensor], dim=0)

        # Create labels tensor: 1 for positive edges, 0 for negative edges
        num_pos = len(pos_tensor)
        num_neg = len(neg_tensor)
        labels = torch.cat(
            [
                torch.ones(num_pos, dtype=torch.int64),
                torch.zeros(num_neg, dtype=torch.int64),
            ]
        )
        unique_edges, orig_to_unique, stats = get_unique_edges_with_mapping(all_edges)

    # Cache the computed values
    print(f"Caching efficient heart data to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "neg_tensor": neg_tensor,
                "unique_edges": unique_edges,
                "orig_to_unique": orig_to_unique,
                "labels": labels,
                "stats": stats,
            },
            f,
        )

    return neg_tensor, unique_edges, orig_to_unique, labels


def compute_and_cache_heuristics(data, heuristic_name, dataset_name, normalize=True):
    """
    Compute heuristics for **all** node pairs in the graph (including self-pairs)
    and cache them.

    Args:
        data: PyG Data object containing the graph
        heuristic_name: Name of the heuristic (\"cn\", \"aa\", \"ra\", \"katz\", \"shortest_path\", \"pagerank\")
        dataset_name: Name of the dataset (without prefix)
        normalize: If True, apply appropriate normalization to heuristic scores (default: False)
                  Note: Normalized scores will be cached separately from unnormalized scores

    Returns:
        all_edges: Tensor of all node pairs (i, j) with i <= j, shape (K, 2)
            where K = N * (N + 1) / 2
        labels: Tensor of heuristic scores, shape (K,)
    """
    # Create cache directory
    # Include normalization status in cache path to avoid mixing normalized/unnormalized scores
    cache_suffix = (
        "_normalized" if normalize and heuristic_name != "shortest_path" else ""
    )
    cache_dir = os.path.join(DATA_DIR, "heuristics", dataset_name, heuristic_name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"all_edge_labels{cache_suffix}.npy")

    num_nodes = data.num_nodes

    # Generate all node pairs (i <= j) using the upper triangle including diagonal
    # This gives K = N * (N + 1) / 2 pairs.
    row_idx, col_idx = np.triu_indices(num_nodes, k=0)
    edge_index = torch.from_numpy(
        np.vstack((row_idx, col_idx)).astype(np.int64)
    )  # shape (2, K)
    all_edges = edge_index.t()  # shape (K, 2)

    # Check if cache exists
    if os.path.exists(cache_path):
        print(
            f"Loading cached all-pairs (i <= j) heuristic sparse matrix from {cache_path}"
        )
        # Stored as a scipy.sparse matrix pickled into a .npy file
        score_mat = np.load(cache_path, allow_pickle=True).item()
        assert isinstance(
            score_mat, sparse.spmatrix
        ), "Cached object must be a scipy sparse matrix"

        row = edge_index[0].cpu().numpy().copy()
        col = edge_index[1].cpu().numpy().copy()
        labels_np = score_mat[row, col].A1  # extract as 1D dense array for all (i <= j)
        labels = torch.from_numpy(labels_np)
        assert len(labels) == all_edges.size(
            0
        ), "Number of labels must match number of all unordered node pairs"
        return all_edges, labels

    # Cache doesn't exist, compute heuristics
    print(f"Computing {heuristic_name} heuristic for all unordered node pairs...")

    # Create adjacency matrix from training graph using PyG utility
    # Use edge_index from data (which should be the training graph)
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()

    # Compute heuristics based on heuristic_name
    if heuristic_name == "cn":
        scores = CN(A, edge_index, normalize=normalize)
    elif heuristic_name == "aa":
        scores = AA(A, edge_index, normalize=normalize)
    elif heuristic_name == "ra":
        scores = RA(A, edge_index, normalize=normalize)
    elif heuristic_name == "katz":
        scores = katz_close(A, edge_index, normalize=normalize)
    elif heuristic_name == "shortest_path":
        scores = shortest_path(A, edge_index)
    elif heuristic_name == "pagerank":
        scores = pagerank(A, edge_index, normalize=normalize)
    else:
        raise ValueError(f"Unknown heuristic: {heuristic_name}")

    # Build a sparse upper-triangular matrix of scores so we can later query (src, dst)
    # with src <= dst to retrieve the heuristic value.
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    data_vals = scores.cpu().numpy()
    score_mat = sparse.csr_matrix((data_vals, (row, col)), shape=(num_nodes, num_nodes))

    # Cache the sparse matrix
    print(f"Caching all-pairs heuristic sparse matrix to {cache_path}")
    np.save(cache_path, score_mat)

    return all_edges, scores

from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.data import DataLoader
import networkx as nx


def _normalize_log_minmax(scores):
    """Apply log transformation followed by min-max normalization to [0, 1].

    Recommended for CN, AA, RA heuristics which have right-skewed distributions.
    Uses log1p (log(1+x)) to handle zeros gracefully.
    """
    scores_np = scores.numpy() if isinstance(scores, torch.Tensor) else scores
    # Log transform: log(1+x) handles zeros and reduces skew
    log_scores = np.log1p(scores_np)
    # Min-max normalization to [0, 1]
    min_val = log_scores.min()
    max_val = np.percentile(log_scores, 99.99)  # To avoid extreme outliers
    if max_val - min_val < 1e-8:
        raise ValueError("All values are the same!")
    else:
        normalized = (log_scores - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0.0, 1.0)
    return torch.FloatTensor(normalized)


def _normalize_log_robust(scores_np):
    """Apply Log-Standardization.

    Why: Katz scores follow a power-law (heavy tail).
    - Distance 2 scores are ~0.01
    - Distance 4 scores are ~0.000000001

    Standard normalization squashes the tail. Log-transform restores the signal.
    """
    print("Applying Log-Normalization...")

    # 1. Log Transform
    # Add a tiny epsilon to handle potential exact zeros (though Katz is usually > 0)
    epsilon = 1e-20
    scores_log = np.log(scores_np + epsilon)

    # 2. Robust Standardization (Z-Score on Log values)
    # We use median/IQR or Mean/Std. Mean/Std is usually fine in Log-space.
    mean_val = scores_log.mean()
    std_val = scores_log.std()

    print(f"Log-Space Stats -> Mean: {mean_val:.4f}, Std: {std_val:.4f}")

    if std_val < 1e-8:
        print(
            "[WARNING] All Katz scores are identical. Check beta or graph connectivity."
        )
        return torch.zeros_like(torch.tensor(scores_np))

    normalized = (scores_log - mean_val) / std_val
    normalized = np.clip(normalized, None, np.percentile(normalized, 99))

    return torch.FloatTensor(normalized)


def CN(A, edge_index, batch_size=100000, normalize=False):
    """Compute Common Neighbors heuristic scores.

    Args:
        A: Adjacency matrix (scipy sparse)
        edge_index: Edge indices to compute scores for (torch.Tensor, shape [2, N])
        batch_size: Batch size for processing
        normalize: If True, apply log + min-max normalization (recommended for skewed distributions)

    Returns:
        torch.FloatTensor of heuristic scores
    """
    print(f"Computing CN for {edge_index.size(1)} edges...")
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        # Convert PyTorch tensors to numpy arrays for scipy sparse matrix indexing
        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()
        cur_scores = np.array(np.sum(A[src_np].multiply(A[dst_np]), 1)).flatten()
        scores.append(cur_scores)
    scores = torch.FloatTensor(np.concatenate(scores, 0))
    if normalize:
        scores = _normalize_log_minmax(scores)
    return scores


def AA(A, edge_index, batch_size=100000, normalize=False):
    """Compute Adamic-Adar heuristic scores.

    Args:
        A: Adjacency matrix (scipy sparse)
        edge_index: Edge indices to compute scores for (torch.Tensor, shape [2, N])
        batch_size: Batch size for processing
        normalize: If True, apply log + min-max normalization (recommended for skewed distributions)

    Returns:
        torch.FloatTensor of heuristic scores
    """
    print(f"Computing AA for {edge_index.size(1)} edges...")
    # The Adamic-Adar heuristic score.
    degree = A.sum(axis=0).A1  # Convert to 1D array
    # Avoid log(0) and log(1)=0 issues: Adamic-Adar uses 1/log(degree) for degree > 1
    # For degree <= 1, set multiplier to 0
    log_degree = np.log(
        degree, out=np.zeros_like(degree, dtype=float), where=degree > 1
    )
    # Divide only where degree > 1 (log_degree > 0)
    multiplier = np.divide(
        1,
        log_degree,
        out=np.zeros_like(log_degree, dtype=float),
        where=(degree > 1) & (log_degree > 0),
    )
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        # Convert PyTorch tensors to numpy arrays for scipy sparse matrix indexing
        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()
        cur_scores = np.array(np.sum(A[src_np].multiply(A_[dst_np]), 1)).flatten()
        scores.append(cur_scores)
    scores = torch.FloatTensor(np.concatenate(scores, 0))
    if normalize:
        scores = _normalize_log_minmax(scores)
    return scores


def RA(A, edge_index, batch_size=100000, normalize=False):
    """Compute Resource Allocation heuristic scores.

    Args:
        A: Adjacency matrix (scipy sparse)
        edge_index: Edge indices to compute scores for (torch.Tensor, shape [2, N])
        batch_size: Batch size for processing
        normalize: If True, apply log + min-max normalization (recommended for skewed distributions)

    Returns:
        torch.FloatTensor of heuristic scores
    """
    print(f"Computing RA for {edge_index.size(1)} edges...")
    degree = A.sum(axis=0).A1  # Convert to 1D array
    # Avoid division by zero: use np.divide with where to handle zeros
    multiplier = np.divide(
        1, degree, out=np.zeros_like(degree, dtype=float), where=degree != 0
    )
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        # Convert PyTorch tensors to numpy arrays for scipy sparse matrix indexing
        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()
        cur_scores = np.array(np.sum(A[src_np].multiply(A_[dst_np]), 1)).flatten()
        scores.append(cur_scores)
    scores = torch.FloatTensor(np.concatenate(scores, 0))
    if normalize:
        scores = _normalize_log_minmax(scores)
    return scores


def katz_close(
    A, edge_index, beta=0.005, batch_size=100000, normalize=False, percentile=99.9
):
    """Compute Katz Centrality-based similarity scores.

    Args:
        A: Adjacency matrix (scipy sparse)
        edge_index: Edge indices to compute scores for (torch.Tensor, shape [2, N])
        beta: Damping factor for Katz centrality (default: 0.005)
        batch_size: Batch size for processing
        normalize: If True, apply z-score standardization (recommended for unbounded values)
        percentile: Percentile to use for standardization
    Returns:
        torch.FloatTensor of heuristic scores
    """
    scores = []
    print(f"Computing Katz closeness for {edge_index.size(1)} edges...")
    # Use undirected graph since graphs are typically undirected in this codebase
    # NetworkX 3.x uses from_scipy_sparse_array instead of from_scipy_sparse_matrix
    G = nx.from_scipy_sparse_array(A, create_using=nx.Graph())

    adj = nx.adjacency_matrix(G, nodelist=range(len(G.nodes)))
    aux = adj.T.multiply(-beta).todense()
    np.fill_diagonal(aux, 1 + aux.diagonal())
    sim = np.linalg.inv(aux)
    np.fill_diagonal(sim, sim.diagonal() - 1)

    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    for ind in tqdm(link_loader):
        src = edge_index[0, ind].cpu().numpy()
        dst = edge_index[1, ind].cpu().numpy()
        # Use numpy advanced indexing to extract scores in batch
        cur_scores = sim[src, dst]
        scores.append(cur_scores)
    scores = torch.FloatTensor(np.concatenate(scores, 0))
    if normalize:
        scores = _normalize_log_robust(scores)
    return scores


def shortest_path(A, edge_index):
    """Compute shortest paths with disconnected nodes set to 1.5 * max_distance.

    Args:
        A: Adjacency matrix (scipy sparse)
        edge_index: Edge indices to compute scores for (torch.Tensor, shape [2, N])

    Returns:
        torch.FloatTensor of heuristic scores
    """
    # 1. Convert to NetworkX Graph
    G = nx.from_scipy_sparse_array(A, create_using=nx.Graph())

    # 2. Pre-compute all shortest paths
    print(f"Pre-computing shortest paths for {len(G)} nodes...")
    paths = dict(nx.all_pairs_shortest_path_length(G))

    scores = []
    max_dist = 0.0

    src_indices = edge_index[0].cpu().numpy()
    dst_indices = edge_index[1].cpu().numpy()

    # 3. First Pass: Collect finite distances and identify disconnected pairs
    for s, t in tqdm(zip(src_indices, dst_indices), total=len(src_indices)):
        if s == t:
            scores.append(0.0)
            continue

        if t in paths[s]:
            dist = float(paths[s][t])
            scores.append(dist)
            if dist > max_dist:
                max_dist = dist
        else:
            # Placeholder for disconnected
            scores.append(-1.0)

    # 4. Handle Disconnected Nodes
    disconnected_value = 1.5 * max_dist

    # If the graph has no edges at all, max_dist remains 0
    if max_dist == 0:
        raise ValueError("The graph has no edges at all!")

    # 5. Apply log1p and replace -1.0 with disconnected_value
    final_scores = [
        np.log1p(s) if s != -1.0 else np.log1p(disconnected_value) for s in scores
    ]

    scores_tensor = torch.FloatTensor(final_scores)

    return scores_tensor


def pagerank(A, edge_index, alpha=0.85, normalize=False):
    """Compute PageRank heuristic scores.

    Computes regular PageRank (no personalization) and returns the product of
    source and destination PageRank scores for each edge.

    Args:
        A: Adjacency matrix (scipy sparse)
        edge_index: Edge indices to compute scores for (torch.Tensor, shape [2, N])
        alpha: Damping factor for PageRank (default: 0.85)
        normalize: If True, apply log-robust normalization (recommended for power-law distributions)

    Returns:
        torch.FloatTensor of heuristic scores
    """
    print(f"Computing PageRank for {edge_index.size(1)} edges...")
    # Convert to NetworkX Graph (undirected since graphs are typically undirected)
    # NetworkX 3.x uses from_scipy_sparse_array instead of from_scipy_sparse_matrix
    G = nx.from_scipy_sparse_array(A, create_using=nx.Graph())

    # Compute regular PageRank (no personalization - uniform restart distribution)
    print("Computing PageRank vector...")
    pr = nx.pagerank(G, alpha=alpha)

    # Convert to numpy array for efficient lookup
    num_nodes = len(G.nodes)
    pr_vector = np.array([pr[i] for i in range(num_nodes)])

    # Extract scores for each edge: use product of source and destination PageRank scores
    # This considers both nodes, similar to other link prediction heuristics
    src_indices = edge_index[0].cpu().numpy()
    dst_indices = edge_index[1].cpu().numpy()

    scores = pr_vector[src_indices] * pr_vector[dst_indices]

    scores = torch.FloatTensor(scores)
    if normalize:
        # Use log-robust normalization (like Katz) since PageRank products
        # follow a power-law distribution with many small values
        scores = _normalize_log_robust(scores.numpy())
    return scores

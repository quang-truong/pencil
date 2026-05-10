# refer to: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
from datetime import datetime
import copy
import random
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, List

try:
    NDArray = np.typing.NDArray
except AttributeError:
    NDArray = List
from collections import defaultdict

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.typing import WITH_TORCH_SPARSE, SparseTensor
from torch_geometric.utils import negative_sampling, add_self_loops

from utils import rank_zero_print as print
from utils import Config


def _upper_triangular_index(edges: Tensor, num_nodes: int) -> Tensor:
    """
    Map each (src, dst) edge to an index in the upper-triangular all-pairs list.
    We treat edges as unordered: (u, v) and (v, u) share the same label.

    Args:
        edges: Tensor of shape (N, 2) containing edges
        num_nodes: Number of nodes in the graph

    Returns:
        Tensor of shape (N,) containing indices in the upper-triangular all-pairs list
    """
    src = edges[:, 0].long()
    dst = edges[:, 1].long()
    i = torch.minimum(src, dst)
    j = torch.maximum(src, dst)
    n = torch.tensor(num_nodes, dtype=torch.long, device=edges.device)
    # Number of pairs before row i: i * n - i * (i - 1) / 2
    base = i * n - i * (i - 1) // 2
    offset = j - i
    return base + offset


def _local_sample_heads(
    pos_edges,
    pos_edge_attr,
    *,
    num_nodes,
    neg_ratio,
):
    """
    Generate negative edges by fixing tail and edge attributes, randomly sampling new heads.

    This function implements a local negative sampling strategy where:
    - Tail nodes and edge attributes are kept fixed from positive edges
    - New head nodes are randomly sampled to create negative edges
    - This is used for knowledge graph completion tasks where we want to test
      if a relationship exists between a randomly sampled head and the original tail

    Args:
        pos_edges (torch.Tensor): Positive edges tensor of shape [N_pos, 2] where each row is [head, tail]
        pos_edge_attr (torch.Tensor, optional): Positive edge attributes tensor of shape [N_pos, edge_attr_dim]
        num_nodes (int): Total number of nodes in the graph for random sampling
        neg_ratio (int): Ratio of negative samples to positive samples (e.g., 1 means 1:1 ratio)

    Returns:
        tuple: (neg_edges, neg_edge_attr)
            - neg_edges (torch.Tensor): Negative edges tensor of shape [N_neg, 2]
            - neg_edge_attr (torch.Tensor, optional): Negative edge attributes tensor of shape [N_neg, edge_attr_dim]
    """
    print(f"Fix `tail` & `edge`, randomly sample `head` ...")

    # Get the count of positive edges
    cnt_pos_edges = pos_edges.shape[0]
    # Calculate total number of negative edges to generate
    cnt_neg_edges = neg_ratio * pos_edges.shape[0]

    # Randomly sample negative head nodes (tail nodes will be fixed from positive edges)
    tail_neg = torch.randint(num_nodes, (cnt_neg_edges, 1))

    # Extract tail nodes from positive edges and repeat them for each negative sample
    # [Np, 1] -> [Np, nr] -> [Np*nr, 1]
    tail = pos_edges[:, 1:2].expand(cnt_pos_edges, neg_ratio).reshape((-1, 1))

    # Combine randomly sampled heads with fixed tails to create negative edges
    # [Np*nr, 1] & [Np*nr, 1] -> [Np*nr, 2]
    neg_edges = torch.hstack([tail_neg, tail])

    # Initialize negative edge attributes as None
    neg_edge_attr = None
    if pos_edge_attr is not None:
        # Replicate positive edge attributes for each negative sample
        # [Np, dim] -> [Np, dim*nr] -> [Np*nr, dim]
        neg_edge_attr = torch.hstack([pos_edge_attr] * neg_ratio).reshape(
            [-1, pos_edge_attr.shape[1]]
        )
        # Verify that edge attributes match the number of negative edges
        assert (
            neg_edge_attr.shape[0] == neg_edges.shape[0]
        ), f"{neg_edge_attr.shape[0]} != {neg_edges.shape[0]}"

    return neg_edges, neg_edge_attr


def _local_sample_tails(
    pos_edges,
    pos_edge_attr,
    *,
    num_nodes,
    neg_ratio,
):
    """
    Generate negative edges by fixing head and edge attributes, randomly sampling new tails.

    This function implements a local negative sampling strategy where:
    - Head nodes and edge attributes are kept fixed from positive edges
    - New tail nodes are randomly sampled to create negative edges
    - This is used for knowledge graph completion tasks where we want to test
      if a relationship exists between the original head and a randomly sampled tail

    Args:
        pos_edges (torch.Tensor): Positive edges tensor of shape [N_pos, 2] where each row is [head, tail]
        pos_edge_attr (torch.Tensor, optional): Positive edge attributes tensor of shape [N_pos, edge_attr_dim]
        num_nodes (int): Total number of nodes in the graph for random sampling
        neg_ratio (int): Ratio of negative samples to positive samples (e.g., 1 means 1:1 ratio)

    Returns:
        tuple: (neg_edges, neg_edge_attr)
            - neg_edges (torch.Tensor): Negative edges tensor of shape [N_neg, 2]
            - neg_edge_attr (torch.Tensor, optional): Negative edge attributes tensor of shape [N_neg, edge_attr_dim]
    """
    print(f"Fix `head` & `edge`, randomly sample `tail` ...")

    # Get the count of positive edges
    cnt_pos_edges = pos_edges.shape[0]
    # Calculate total number of negative edges to generate
    cnt_neg_edges = neg_ratio * pos_edges.shape[0]

    # Randomly sample negative tail nodes (head nodes will be fixed from positive edges)
    head_neg = torch.randint(num_nodes, (cnt_neg_edges, 1))

    # Extract head nodes from positive edges and repeat them for each negative sample
    # [Np, 1] -> [Np, nr] -> [Np*nr, 1]
    head = pos_edges[:, 0:1].expand(cnt_pos_edges, neg_ratio).reshape((-1, 1))

    # Combine fixed heads with randomly sampled tails to create negative edges
    # [Np*nr, 1] & [Np*nr, 1] -> [Np*nr, 2]
    neg_edges = torch.hstack([head, head_neg])

    # Initialize negative edge attributes as None
    neg_edge_attr = None
    if pos_edge_attr is not None:
        # Replicate positive edge attributes for each negative sample
        # [Np, dim] -> [Np, dim*nr] -> [Np*nr, dim]
        neg_edge_attr = torch.hstack([pos_edge_attr] * neg_ratio).reshape(
            [-1, pos_edge_attr.shape[1]]
        )
        # Verify that edge attributes match the number of negative edges
        assert (
            neg_edge_attr.shape[0] == neg_edges.shape[0]
        ), f"{neg_edge_attr.shape[0]} != {neg_edges.shape[0]}"

    return neg_edges, neg_edge_attr


def _local_sample_edges(
    pos_edges,
    pos_edge_attr,
    *,
    neg_edge_attr_candidates,
    neg_ratio,
):
    """
    Generate negative edges by fixing head and tail nodes, randomly sampling new edge attributes.

    This function implements a local negative sampling strategy where:
    - Head and tail nodes are kept fixed from positive edges
    - New edge attributes are randomly sampled from candidates to create negative edges
    - This is used for knowledge graph completion tasks where we want to test
      if a different relationship exists between the same head and tail nodes

    Args:
        pos_edges (torch.Tensor): Positive edges tensor of shape [N_pos, 2] where each row is [head, tail]
        pos_edge_attr (torch.Tensor, optional): Positive edge attributes tensor of shape [N_pos, edge_attr_dim]
        neg_edge_attr_candidates (torch.Tensor): Candidate edge attributes tensor of shape [N_candidates, edge_attr_dim]
        neg_ratio (int): Ratio of negative samples to positive samples (e.g., 1 means 1:1 ratio)

    Returns:
        tuple: (neg_edges, neg_edge_attr)
            - neg_edges (torch.Tensor): Negative edges tensor of shape [N_neg, 2]
            - neg_edge_attr (torch.Tensor, optional): Negative edge attributes tensor of shape [N_neg, edge_attr_dim]
    """
    print(f"Fix `head` & `tail`, randomly sample `edge` ...")

    # Validate input tensor dimensions
    assert len(pos_edges.shape) == 2, "pos_edges must be 2D tensor"
    assert pos_edges.shape[1] == 2, "pos_edges must have 2 columns [head, tail]"

    # Initialize negative edges as empty tensor
    neg_edges = torch.empty((0, 2), dtype=pos_edges.dtype)
    neg_edge_attr = None

    if pos_edge_attr is not None:
        # Replicate the same head-tail pairs for negative sampling
        neg_edges = torch.hstack([pos_edges] * neg_ratio).reshape([-1, 2])

        # Replicate positive edge attributes for each negative sample
        pos_edge_attr = torch.hstack([pos_edge_attr] * neg_ratio).reshape(
            [-1, pos_edge_attr.shape[1]]
        )

        # Generate negative edge attributes by sampling from candidates
        neg_edge_attr = _get_edge_attr_neg(pos_edge_attr, neg_edge_attr_candidates)

        # Verify that edge attributes match the number of negative edges
        assert (
            neg_edges.shape[0] == neg_edge_attr.shape[0]
        ), f"{neg_edges.shape[0]} != {neg_edge_attr.shape[0]}"

    return neg_edges, neg_edge_attr


def _get_edge_attr_neg(pos_edge_attr, neg_edge_attr_candidates):
    """
    Generate negative edge attributes by sampling from candidates while avoiding duplicates with positive attributes.

    This function creates negative edge attributes by:
    1. Randomly sampling candidate attributes for each positive edge
    2. If a sampled attribute matches the positive attribute, sample a second candidate
    3. Use the second candidate as a fallback to minimize collisions with positive attributes

    Args:
        pos_edge_attr (torch.Tensor): Positive edge attributes tensor of shape [N_pos, edge_attr_dim]
        neg_edge_attr_candidates (torch.Tensor): Candidate edge attributes tensor of shape [N_candidates, edge_attr_dim]

    Returns:
        torch.Tensor: Negative edge attributes tensor of shape [N_pos, edge_attr_dim]
    """
    # Get count of positive edges and candidate attributes
    cnt_pos_edges = pos_edge_attr.shape[0]
    cnt_candidates = len(neg_edge_attr_candidates)

    # Ensure we have enough candidates for the fallback sampling strategy
    assert (
        cnt_candidates > 2
    ), f"NOT implemented for cnt_candidates ({cnt_candidates}) <= 2"

    # First random sampling of candidate attributes
    idx_attr1 = torch.randint(cnt_candidates, (cnt_pos_edges,))
    neg_edge_attr1 = neg_edge_attr_candidates[idx_attr1]

    # Second random sampling of candidate attributes (used as fallback)
    idx_attr2 = torch.randint(cnt_candidates, (cnt_pos_edges,))
    neg_edge_attr2 = neg_edge_attr_candidates[idx_attr2]

    # Check which sampled attributes match the positive attributes
    mask1 = _get_row_equal_mask(pos_edge_attr, neg_edge_attr1)

    # Use fallback attributes where first sampling matched positive attributes
    # This is a conditional selection: if mask1 is False, use neg_edge_attr1; if True, use neg_edge_attr2
    neg_edge_attr = (~mask1).view((-1, 1)).to(
        torch.int64
    ) * neg_edge_attr1 + mask1.view((-1, 1)).to(torch.int64) * neg_edge_attr2

    # Check final collisions with positive attributes (should be minimized)
    mask = _get_row_equal_mask(pos_edge_attr, neg_edge_attr)
    print(
        f"[WARNING] {mask.sum().item()} out of {neg_edge_attr.shape[0]} neg-edge-attr is the same as pos-edge-attr"
    )

    return neg_edge_attr


def _get_row_equal_mask(a, b):
    """
    Create a boolean mask indicating which rows are equal between two 2D tensors.

    This function compares rows element-wise between two tensors and returns a mask
    where True indicates that the corresponding rows are identical.

    Args:
        a (torch.Tensor): First tensor of shape [N, M]
        b (torch.Tensor): Second tensor of shape [N, M] (same shape as a)

    Returns:
        torch.Tensor: Boolean mask of shape [N,] where True means rows are equal
    """
    # Validate that both tensors are 2D and have the same number of dimensions
    assert (
        len(a.shape) == len(b.shape) == 2
    ), f"a -> {len(a.shape)}, b -> {len(b.shape)}"

    # Compute element-wise absolute differences, sum across columns, and check if sum is zero
    # If sum of absolute differences is 0, the rows are equal
    return ~torch.abs(a - b).sum(dim=-1).to(bool)


def _remove_target_edge(edge_index, src, dst, bidiretional=True):
    """
    Remove target edge(s) from edge_index, optionally handling bidirectional edges.

    This function removes specific edges from an edge index tensor. It's commonly used
    in link prediction tasks to remove the target edge from the subgraph to prevent
    data leakage during training.

    Args:
        edge_index (torch.Tensor): Edge index tensor of shape [2, N_edges] where each column is [src, dst]
        src (int): Source node of the target edge to remove
        dst (int): Destination node of the target edge to remove
        bidiretional (bool, optional): If True, remove both (src->dst) and (dst->src) edges. Defaults to True.

    Returns:
        tuple: (filtered_edge_index, edge_mask)
            - filtered_edge_index (torch.Tensor): Edge index with target edges removed, shape [2, N_remaining]
            - edge_mask (torch.Tensor): Boolean mask indicating which edges were kept, shape [N_edges,]
    """
    # Validate that edge_index has the expected 2D shape (2 rows for [src, dst])
    assert edge_index.shape[0] == 2, "edge_index must have shape [2, N_edges]"

    # Create boolean mask for forward direction edge (src -> dst)
    forward_bool = (edge_index[0] == src) & (edge_index[1] == dst)

    if bidiretional:
        # Create boolean mask for backward direction edge (dst -> src)
        backward_bool = (edge_index[0] == dst) & (edge_index[1] == src)
        # Keep edges that are NOT in either direction
        all_bool = ~(forward_bool + backward_bool)
    else:
        # Only remove forward direction edge
        all_bool = ~forward_bool

    # Return filtered edge index and the mask indicating which edges were kept
    return edge_index[:, all_bool], all_bool


def sample_neg_edges_globally(
    pos_edges,
    pos_edge_attr,
    *,
    # sampling params
    self_looped_edge_index,
    num_nodes,
    neg_ratio,
    neg_edge_attr_candidates,
    **kwargs,
):
    """
    Generate negative edges using global sampling strategy from the entire graph.

    This function uses PyTorch Geometric's negative_sampling utility to sample negative edges
    that don't exist in the original graph. This is a global approach where negative edges
    are sampled uniformly from all possible non-existing edges in the graph.

    Args:
        pos_edges (torch.Tensor): Positive edges tensor of shape [N_pos, 2] where each row is [head, tail]
        pos_edge_attr (torch.Tensor, optional): Positive edge attributes tensor of shape [N_pos, edge_attr_dim]
        self_looped_edge_index (torch.Tensor): Edge index with self-loops added, shape [2, N_edges]
        num_nodes (int): Total number of nodes in the graph
        neg_ratio (int): Ratio of negative samples to positive samples (e.g., 1 means 1:1 ratio)
        neg_edge_attr_candidates (torch.Tensor, optional): Candidate edge attributes tensor of shape [N_candidates, edge_attr_dim]
        **kwargs: Additional keyword arguments (unused)

    Returns:
        tuple: (neg_edges, neg_edge_attr)
            - neg_edges (torch.Tensor): Negative edges tensor of shape [N_neg, 2]
            - neg_edge_attr (torch.Tensor, optional): Negative edge attributes tensor of shape [N_neg, edge_attr_dim]
    """
    print(f"GLOBALLY sampling neg edges and edge-attrs ...")

    # Calculate total number of negative edges to generate
    cnt_neg_edges = neg_ratio * pos_edges.shape[0]

    # Use PyTorch Geometric's negative sampling to get edges that don't exist in the graph
    # negative_sampling returns [2, N_neg], so we transpose to get [N_neg, 2]
    neg_edges = negative_sampling(
        edge_index=self_looped_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=cnt_neg_edges,
    ).T  # [N_e, 2]   N_e=neg_ratio*N_p

    # Initialize negative edge attributes as None
    neg_edge_attr = None
    if neg_edge_attr_candidates is not None:
        # Get count of candidate edge attributes
        cnt_candidates = len(neg_edge_attr_candidates)

        # Randomly sample indices for edge attributes
        idx_attr = torch.randint(cnt_candidates, (cnt_neg_edges,))

        # Sample edge attributes using the random indices
        # [cnt_candidates, edge_attr_dim] & [N_e] -> [N_e, edge_attr_dim]
        neg_edge_attr = neg_edge_attr_candidates[idx_attr]

    return neg_edges, neg_edge_attr


def sample_neg_edges_locally(
    pos_edges,
    pos_edge_attr,
    *,
    # sampling params
    num_nodes,
    neg_ratio,
    neg_edge_attr_candidates,
    method: Dict,
    **kwargs,
):
    """
    Generate negative edges using local sampling strategies based on positive triplets.

    This function implements local negative sampling for knowledge graph completion tasks.
    Given positive triplets (head, tail, edge), it can use three different sampling strategies:
    1. Fix head & edge, randomly sample tail
    2. Fix edge & tail, randomly sample head
    3. Fix head & tail, randomly sample edge

    The method combines results from enabled strategies and randomly samples the required
    number of negative edges from the combined candidates.

    Args:
        pos_edges (torch.Tensor): Positive edges tensor of shape [N_pos, 2] where each row is [head, tail]
        pos_edge_attr (torch.Tensor, optional): Positive edge attributes tensor of shape [N_pos, edge_attr_dim]
        num_nodes (int): Total number of nodes in the graph for random sampling
        neg_ratio (int): Ratio of negative samples to positive samples (e.g., 1 means 1:1 ratio)
        neg_edge_attr_candidates (torch.Tensor, optional): Candidate edge attributes tensor of shape [N_candidates, edge_attr_dim]
        method (Dict): Dictionary specifying which sampling strategies to use:
            - "sample_tails" (bool): Enable head+edge fixed, sample tail strategy
            - "sample_heads" (bool): Enable edge+tail fixed, sample head strategy
            - "sample_edges" (bool): Enable head+tail fixed, sample edge strategy
        **kwargs: Additional keyword arguments (unused)

    Returns:
        tuple: (neg_edges, neg_edge_attr)
            - neg_edges (torch.Tensor): Negative edges tensor of shape [N_neg, 2]
            - neg_edge_attr (torch.Tensor, optional): Negative edge attributes tensor of shape [N_neg, edge_attr_dim]
    """
    print(f"LOCALLY sampling neg edges and edge-attrs ...")

    # Calculate total number of negative edges to generate
    cnt_neg_edges = neg_ratio * pos_edges.shape[0]

    # List to store results from different sampling strategies
    ls_neg_edges_and_attrs = []

    # Strategy 1: Fix head & edge, randomly sample tail
    if method.get("sample_tails", False):
        ls_neg_edges_and_attrs.append(
            _local_sample_tails(
                pos_edges, pos_edge_attr, num_nodes=num_nodes, neg_ratio=neg_ratio
            )
        )

    # Strategy 2: Fix edge & tail, randomly sample head
    if method.get("sample_heads", False):
        ls_neg_edges_and_attrs.append(
            _local_sample_heads(
                pos_edges, pos_edge_attr, num_nodes=num_nodes, neg_ratio=neg_ratio
            )
        )

    # Strategy 3: Fix head & tail, randomly sample edge
    if method.get("sample_edges", False):
        ls_neg_edges_and_attrs.append(
            _local_sample_edges(
                pos_edges,
                pos_edge_attr,
                neg_edge_attr_candidates=neg_edge_attr_candidates,
                neg_ratio=neg_ratio,
            )
        )

    # Separate edges and attributes from the results
    ls_neg_edges = [x[0] for x in ls_neg_edges_and_attrs]
    ls_neg_edge_attr = [x[1] for x in ls_neg_edges_and_attrs]

    # Combine all negative edge candidates from different strategies
    neg_edge_candidates = torch.vstack(ls_neg_edges)

    # Randomly select the required number of negative edges from candidates
    # Use default RNG which is seeded by set_seed() in the main process
    indices = torch.randperm(neg_edge_candidates.shape[0])[:cnt_neg_edges]
    neg_edges = neg_edge_candidates[indices]

    # Handle negative edge attributes if they exist
    neg_edge_attr = None
    if pos_edge_attr is not None:
        # Combine all negative edge attribute candidates from different strategies
        neg_ea_candidates = torch.vstack(ls_neg_edge_attr)
        # Verify that edge and attribute counts match
        assert neg_edge_candidates.shape[0] == neg_ea_candidates.shape[0]
        # Select the same indices for edge attributes
        neg_edge_attr = neg_ea_candidates[indices]

    return neg_edges, neg_edge_attr


def sample_pos_edges(
    pos_edges,
    pos_edge_attr,
    percent,  # percentage of train samples used
    # reset sampling
    seed,
    epoch,
):
    """
    Sample a subset of positive edges for training with cyclic sampling strategy.

    This function implements a cyclic sampling strategy where only a percentage of positive
    edges are used per epoch. The sampling is deterministic and cyclic, ensuring that:
    - Each epoch uses a different subset of positive edges
    - After a full cycle, all positive edges have been used once
    - The sampling is reproducible with the same seed and epoch

    For example, if percent=10:
    - Epochs [0,9] use different 10% subsets with the same seed
    - Epochs [10,19] use different 10% subsets with a different seed
    - This continues cyclically

    Args:
        pos_edges (torch.Tensor): All positive edges tensor of shape [N_total, 2]
        pos_edge_attr (torch.Tensor, optional): All positive edge attributes tensor of shape [N_total, edge_attr_dim]
        percent (int): Percentage of positive edges to use per epoch (1-100)
        seed (int): Base random seed for sampling
        epoch (int): Current training epoch number

    Returns:
        tuple: (sampled_pos_edges, sampled_pos_edge_attr)
            - sampled_pos_edges (torch.Tensor): Sampled positive edges tensor of shape [N_sampled, 2]
            - sampled_pos_edge_attr (torch.Tensor, optional): Sampled positive edge attributes tensor of shape [N_sampled, edge_attr_dim]
    """
    if percent < 100:
        # Get total number of positive edges
        tot_pos_edges = pos_edges.shape[0]

        # Calculate the cyclic period (how many epochs until we repeat)
        epoch_cyclic_period = int(round(100 / percent))

        # Adjust seed based on epoch to ensure different samples in each cycle
        seed = seed * 100 + percent * epoch // 100

        # Create a random generator with the adjusted seed
        g = torch.Generator()
        g.manual_seed(seed)

        # Generate a random permutation of all edge indices
        indices = torch.randperm(tot_pos_edges, generator=g)

        # Calculate number of edges to sample for this epoch
        cnt_pos_edges = int(round(tot_pos_edges * percent / 100.0))

        # Calculate which part of the cycle we're in
        cyclic_epoch = epoch % epoch_cyclic_period

        # Select the appropriate slice of indices for this epoch
        pos_idx = indices[
            cyclic_epoch * cnt_pos_edges : (cyclic_epoch + 1) * cnt_pos_edges
        ]

        # Sample the positive edges using the selected indices
        pos_edges = pos_edges[pos_idx]

        print(
            f"RESET pos_edges by sampling {cnt_pos_edges} pos edges from {tot_pos_edges} pos edges!\n"
            f"seed: {seed}, cyclic_epoch: {cyclic_epoch}, percent: {percent}"
        )

        # Sample the corresponding edge attributes if they exist
        if pos_edge_attr is not None:
            pos_edge_attr = pos_edge_attr[pos_idx]

    return pos_edges, pos_edge_attr


class ShaDowKHopSeqFromEdgesMapDataset(torch.utils.data.Dataset):
    r"""
    The ShaDow k-hop sampler from the "Decoupling the Depth and Scope of Graph Neural Networks" paper.

    This dataset class implements a k-hop sampling strategy for link prediction tasks. Given a graph
    and a pair of nodes, it creates shallow, localized subgraphs by sampling k-hop neighborhoods
    around the nodes. This is particularly useful for link prediction algorithms like SEAL.

    The class supports:
    - Multiple negative sampling strategies (global and local)
    - Cyclic positive edge sampling for memory efficiency
    - Edge attribute handling for knowledge graphs
    - Weighted sampling based on relation frequency
    - Pretraining and fine-tuning modes

    Args:
        data (torch_geometric.data.Data): The graph data object containing nodes, edges, and features
        sampling_config (Dict): Configuration dictionary containing:
            - edge_ego: Configuration for edge ego-graph sampling
            - depth_neighbors: List of (depth, num_neighbors) tuples for sampling
            - neg_ratio (int): Ratio of negative samples to positive samples
            - percent (int): Percentage of positive edges to use per epoch (1-100)
            - method (Dict): Negative sampling method configuration
            - sample_wgt (bool): Whether to use weighted sampling based on relation frequency
            - replace (bool): Whether to sample neighbors with replacement
        adj_t (torch_sparse.SparseTensor, optional): Precomputed sparse adjacency tensor
        split_edge (dict, optional): Edge splits dictionary containing train/valid/test edges
            Expected format:
            ```python
            from ogb.linkproppred import PygLinkPropPredDataset
            split_edge = dataset.get_edge_split()
            train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
            ```
        data_split (str): Which split to use ("train", "valid", "test")
        pretrain_mode (bool, optional): Whether to run in pretraining mode (no target edge masking)
        allow_zero_edges (bool, optional): Whether to allow subgraphs with zero edges
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        data: Data,
        sampling_config: Config,
        *,
        adj_t: Optional[SparseTensor] = None,
        split_edge: Optional[Dict] = None,
        data_split: str = "train",
        pretrain_mode: bool = False,
        allow_zero_edges: bool = False,
        labels: Optional[Tensor] = None,
        orig_to_unique: Optional[Tensor] = None,
        heuristic_labels: Optional[Tensor] = None,
        **kwargs,
    ):
        """
        Initialize the ShaDowKHopSeqFromEdgesMapDataset.

        Args:
            data: Graph data object containing nodes, edges, and features
            sampling_config: Configuration dictionary for sampling parameters
            adj_t: Precomputed sparse adjacency tensor (optional)
            split_edge: Edge splits dictionary for train/valid/test (optional)
            data_split: Which data split to use ("train", "valid", "test")
            pretrain_mode: Whether to run in pretraining mode
            allow_zero_edges: Whether to allow subgraphs with zero edges
            labels: Labels tensor of shape (N, ) or None
            orig_to_unique: Original edges to unique edges mapping tensor of shape (N, ) or None
            **kwargs: Additional keyword arguments
        """
        # Check if torch-sparse is available (required for k-hop sampling)
        if not WITH_TORCH_SPARSE:
            raise ImportError(f"'{self.__class__.__name__}' requires 'torch-sparse'")

        # Validate that the data contains required node attributes
        assert ("id" in data) and (
            "x" in data
        ), "A big graph must have all node attrs integrated in `x`, including nodes' global id!"

        # Create a copy of the data to avoid modifying the original
        self.data = copy.copy(data)

        # Extract edge ego-graph sampling configuration
        self.config = sampling_config.edge_ego
        self.labels = labels
        self.orig_to_unique = orig_to_unique
        # Optional all-pairs heuristic labels for regression tasks.
        # Expected shape: (K,) where K = N * (N + 1) / 2 and N = data.num_nodes.
        self.heuristic_labels = heuristic_labels
        if self.labels is not None:
            assert self.orig_to_unique is not None
            assert self.labels.shape[0] == self.orig_to_unique.shape[0]
            # Assert no negative edges are provided in valid and test splits
            if "valid" in split_edge and "edge_neg" in split_edge["valid"]:
                assert split_edge["valid"]["edge_neg"].shape[0] == 0
            if "test" in split_edge and "edge_neg" in split_edge["test"]:
                assert split_edge["test"]["edge_neg"].shape[0] == 0

        # Initialize sampling configuration parameters
        self.depth_neighbors = (
            self.config.depth_neighbors
        )  # List of (depth, num_neighbors) tuples
        self.neg_ratio = self.config.neg_ratio  # Ratio of negative to positive samples
        self.percent = self.config.get(
            "percent", 100
        )  # Percentage of positive edges to use per epoch. Default is 100,
        # meaning that all positive edges are used per epoch
        self.method = self.config.method  # Negative sampling method
        self.sample_wgt = self.config.get(
            "sample_wgt", False
        )  # Whether to use weighted sampling
        self.wgt = None  # Will store computed sample weights

        # Validate configuration parameters
        assert 100 >= self.percent > 0, "percent must be between 1 and 100"
        assert isinstance(self.percent, int), "percent must be an integer"
        assert self.method.name in {
            "local",
            "global",
        }, f"method {self.method} is NOT implemented"

        # Set neighbor sampling replacement strategy
        self.replace = self.config.replace

        # Store other configuration parameters
        self.split_edge = split_edge
        self.data_split = data_split
        self.pretrain_mode = pretrain_mode
        self.allow_zero_edges = allow_zero_edges

        # Validate that pretrain mode only works with train split
        assert self.data_split == "train" if self.pretrain_mode else True

        # Set up sparse adjacency tensor for efficient k-hop sampling
        assert hasattr(data, "edge_index"), "data must contain edge_index"
        if adj_t is None:
            # Create sparse adjacency tensor from edge_index
            row, col = data.edge_index.cpu()
            adj_t = SparseTensor(
                row=row,
                col=col,
                value=torch.arange(col.size(0)),  # Store edge indices as values
                sparse_sizes=(data.num_nodes, data.num_nodes),
            ).t()  # Transpose for efficient row-wise operations
        self.adj_t = adj_t

        # Add self-loops to edge_index for negative sampling
        self.new_edge_index, _ = add_self_loops(self.data.edge_index)

        # Set up edge splits for positive/negative sampling
        if self.split_edge is None:
            print(
                "split_edge is None!!!\nSet `data_split` to be 'train', and use data.edge_index as the pos edge!!!"
            )
            self.data_split = "train"
            # Create edge split from existing edge_index (for bidirectional edges)
            tmp_edge_index = self.data.edge_index.T.clone()
            mask = (
                tmp_edge_index[:, 0] < tmp_edge_index[:, 1]
            )  # Keep only upper triangle
            # CAUTION: mask is for bi-directional edge!!!
            self.split_edge = {"train": {"edge": tmp_edge_index[mask]}}
            # Here .clone() must be added, otherwise will be problematic in multiprocessing
            # Check https://github.com/pyg-team/pytorch_geometric/discussions/6919

        # Get the appropriate edge dictionary for the current data split
        self.dict_ = self.split_edge[self.data_split]

        # Initialize training count for weighted sampling (knowledge graph specific)
        self.train_count = None
        if not pretrain_mode and self.data_split == "train" and self.sample_wgt:
            # Compute relation frequency counts for weighted sampling
            # Refer to: https://github.com/snap-stanford/ogb/blob/f631af76359c9687b2fe60905557bbb241916258/examples/linkproppred/wikikg2/run.py#L190
            train_triples = self.dict_
            train_count = {}

            # Count frequency of (head, relation) and (tail, inverse_relation) pairs
            for head, relation, tail in tqdm(
                zip(
                    train_triples["head"].numpy(),
                    train_triples["relation"].numpy(),
                    train_triples["tail"].numpy(),
                )
            ):
                # Initialize count for (head, relation) if not exists
                if (head, relation) not in train_count:
                    train_count[(head, relation)] = 4  # Base count of 4
                # Initialize count for (tail, inverse_relation) if not exists
                if (tail, -relation - 1) not in train_count:
                    train_count[(tail, -relation - 1)] = 4  # Base count of 4

                # Increment counts
                train_count[(head, relation)] += 1
                train_count[(tail, -relation - 1)] += 1

            self.train_count = train_count
            print(f"using sample weight with {len(self.train_count)} dict entries!!!")

        # Initialize sample storage variables (will be set by reset_samples)
        self.all_edges_with_y = None  # All edges with labels (positive=1, negative=0)
        self.all_edge_attr = None  # Edge attributes for all samples
        self.sample_idx = None  # Sample indices
        self.sampler = None  # Shuffled sample list
        self.reset_samples_per_epoch = True  # Flag to reset samples each epoch

        # Initialize samples for the first epoch
        self.reset_samples()

        # Store additional keyword arguments
        self.kwargs = kwargs

    def reset_samples(self, epoch: Optional[int] = 0, seed: Optional[int] = 42):
        """
        Reset and regenerate positive/negative samples for the current epoch.

        This method implements the core sampling logic:
        1. Sample positive edges (potentially using cyclic sampling)
        2. Generate negative edges using configured strategy (global/local)
        3. Combine positive and negative samples with labels
        4. Shuffle the combined samples
        5. Compute sample weights if weighted sampling is enabled

        Args:
            epoch (int, optional): Current training epoch (used for cyclic sampling). Defaults to 0.
            seed (int, optional): Random seed for reproducible sampling. Defaults to 42.
        """
        print(f"RESET samples of {self.__class__.__name__} for epoch {epoch}!")

        # Seed RNG with epoch-specific seed to ensure different randomness per epoch
        epoch_seed = seed * 100 + epoch
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        # Set CUDA seeds for GPU reproducibility
        if torch.cuda.is_available():
            torch.cuda.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)

        # Get positive edges and attributes from the data split
        pos_edges = self.dict_["edge"]  # [N_p, 2]
        pos_edge_attr = self.dict_.get("pos_edge_attr", None)

        # Check if negative edges are precomputed in the split
        if "edge_neg" in self.dict_:
            # Use precomputed negative edges
            print(f"Use precomputed negative edges ...")
            neg_edges = self.dict_["edge_neg"]
            neg_edge_attr = self.dict_.get("neg_edge_attr", None)
        else:
            # Generate negative edges using the configured sampling strategy
            # First, apply cyclic sampling to positive edges if percent < 100
            pos_edges, pos_edge_attr = sample_pos_edges(
                pos_edges,
                pos_edge_attr,
                percent=self.percent,
                seed=seed,
                epoch=epoch,
            )

            # Get candidate edge attributes for negative sampling
            print(f"Generate negative edges using the configured sampling strategy ...")
            neg_edge_attr_candidates = self.dict_.get("neg_edge_attr_candidates", None)

            # Choose the appropriate negative sampling function based on method
            sampling_func = (
                sample_neg_edges_globally
                if self.method.name == "global"
                else sample_neg_edges_locally
            )

            # Generate negative edges using the selected strategy
            neg_edges, neg_edge_attr = sampling_func(
                pos_edges,
                pos_edge_attr,
                # sampling params
                self_looped_edge_index=self.new_edge_index,
                num_nodes=self.data.num_nodes,
                neg_ratio=self.neg_ratio,
                neg_edge_attr_candidates=neg_edge_attr_candidates,
                method=self.method,
            )

        # Validate edge tensor shapes
        assert (
            pos_edges.shape[1] == 2
        ), "positive edges must have 2 columns [head, tail]"
        assert (
            neg_edges.shape[1] == 2
        ), "negative edges must have 2 columns [head, tail]"

        # Attach labels to edges.
        # Priority:
        #   1) If heuristic_labels is provided, use them as continuous regression targets.
        #   2) If labels aren't provided, we attach binary labels (1 for positive, 0 for negative).
        #   3) Else if explicit labels tensor is provided (e.g., efficient heart), defer label
        #      attachment to later.
        if self.heuristic_labels is not None:
            num_nodes = self.data.num_nodes
            pos_idx = _upper_triangular_index(pos_edges, num_nodes)
            neg_idx = _upper_triangular_index(neg_edges, num_nodes)

            y_pos = self.heuristic_labels[pos_idx].view(-1, 1)
            y_neg = self.heuristic_labels[neg_idx].view(-1, 1)

            pos_edges_with_y = torch.cat([pos_edges, y_pos], dim=1)
            neg_edges_with_y = torch.cat([neg_edges, y_neg], dim=1)
        elif self.labels is None:
            # Create labels for positive (1) and negative (0) edges
            y_pos = torch.ones((pos_edges.shape[0], 1), dtype=torch.int64)
            y_neg = torch.zeros((neg_edges.shape[0], 1), dtype=torch.int64)

            # Combine edges with their labels
            pos_edges_with_y = torch.cat([pos_edges, y_pos], dim=1)
            neg_edges_with_y = torch.cat([neg_edges, y_neg], dim=1)
        else:
            # Case where labels are provided.
            # No labels attached because we will attach labels later.
            pos_edges_with_y = pos_edges
            neg_edges_with_y = neg_edges

        # Combine all positive and negative samples
        self.all_edges_with_y = torch.cat(
            [pos_edges_with_y, neg_edges_with_y], dim=0
        )  # [N_p + N_e, 3] where last column is label or [N_p + N_e, 2] if no labels are attached

        # Create sample indices and shuffle them for random ordering
        self.sample_idx = torch.arange(len(self.all_edges_with_y), dtype=torch.int64)
        self.sampler = list(self.sample_idx.tolist())
        random.shuffle(self.sampler)

        print(
            f"FINISH reset of {self.__class__.__name__} with {pos_edges_with_y.shape[0]} pos-samples and {neg_edges_with_y.shape[0]} neg-samples!\n"
        )

        # Handle edge attributes if they exist
        if (pos_edge_attr is not None) or (neg_edge_attr is not None):
            # Combine positive and negative edge attributes
            self.all_edge_attr = torch.cat([pos_edge_attr, neg_edge_attr], dim=0)
            # Verify that edge attributes match the number of samples
            assert self.all_edge_attr.shape[0] == self.all_edges_with_y.shape[0]

        # Compute sample weights for weighted sampling (knowledge graph specific)
        if self.train_count is not None:
            # Create a defaultdict with default value 4 for unseen (head, relation) pairs
            # Note: Can't put `defaultdict(lambda: 4)` in __init__ due to pickle issues
            # Can't pickle local object 'ShaDowKHopSeqFromEdgesMapDataset.__init__.<locals>.<lambda>'
            train_count = defaultdict(lambda: 4)
            train_count.update(self.train_count)

            # Compute weights based on relation frequency counts
            ls_wgt = [
                (train_count[(head, relation)], train_count[(tail, -relation - 1)])
                for head, tail, relation in tqdm(
                    zip(
                        self.all_edges_with_y[:, 0].numpy(),  # head nodes
                        self.all_edges_with_y[:, 1].numpy(),  # tail nodes
                        self.all_edge_attr[:, 1].numpy(),  # relations
                    )
                )
            ]

            # Convert to tensor and compute inverse square root weights
            arr_wgt = torch.tensor(ls_wgt).float()
            self.wgt = torch.sqrt(1 / arr_wgt.sum(dim=-1))

            # Verify that weights match the number of samples
            assert self.wgt.shape[0] == self.all_edges_with_y.shape[0]
            print(f"top 10 ls_wgt: {ls_wgt[:10]}")

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of positive and negative edge samples combined
        """
        return len(self.all_edges_with_y)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        This method implements the core k-hop sampling logic:
        1. Extract the edge pair and label from the sample index
        2. Randomly select k-hop sampling parameters (depth, num_neighbors)
        3. Perform k-hop sampling around the edge endpoints
        4. Extract the subgraph and relabel nodes
        5. Remove the target edge to prevent data leakage (unless in pretrain mode)
        6. Copy relevant node/edge attributes to the subgraph

        Args:
            idx (int): Sample index in the dataset

        Returns:
            tuple: (idx, data)
                - idx (int): The original sample index
                - data (torch_geometric.data.Data): Subgraph data object containing:
                    - edge_index: Subgraph edges with relabeled nodes
                    - root_n_id: Relabeled indices of the original edge endpoints
                    - seed_node: Original global node IDs of the edge endpoints
                    - y: Edge label (1 for positive, 0 for negative) - only in fine-tuning mode
                    - tgt_edge_attr: Target edge attributes (if available)
                    - wgt: Sample weight (if weighted sampling is enabled)
                    - id: ID map for the nodes in the subgraph.
                        e.g [2, 3, 4] means [0, 1, 2] in local id map to [2, 3, 4] in global id map.
                    - Other node/edge attributes from the original graph
        """
        # Validate input index
        assert isinstance(idx, int), "idx must be an integer"

        # Extract edge information and label from the sample
        edge_with_y = self.all_edges_with_y[idx]
        index = edge_with_y[:2].tolist()  # [head, tail] node IDs
        if self.labels is None:
            y = edge_with_y[2].view([1])  # label (1 for positive, 0 for negative)
        else:
            y = None
        assert len(index) == 2, "edge index must have exactly 2 nodes"

        # Convert to tensor for k-hop sampling
        seed_node_ids = torch.tensor(index, dtype=torch.int64)  # 1-D tensor; NOT scalar

        # Randomly select k-hop sampling parameters (depth, num_neighbors)
        depth, num_neighbors = random.choice(self.depth_neighbors)

        # Perform k-hop sampling using torch_sparse operations
        rowptr, col, _ = (
            self.adj_t.csr()
        )  # Convert to CSR format for efficient sampling
        out = torch.ops.torch_sparse.ego_k_hop_sample_adj(
            rowptr,
            col,
            seed_node_ids,
            depth,
            num_neighbors,
            self.replace,  # Whether to sample with replacement
        )
        n_id = out[2]  # Node IDs in the sampled subgraph
        n_id_unique = torch.unique(
            n_id
        )  # Unique nodes (output tensor is always sorted!)

        # Extract subgraph using saint_subgraph (much faster than subgraph function)
        # Note: saint_subgraph is preferred over the slower subgraph function:
        # a). func `subgraph` is too slow, change to saint_subgraph
        # b). below use `saint_subgraph` to extract subgraph, very fast!
        adj, e_id = self.adj_t.saint_subgraph(n_id_unique)
        row, col, _ = adj.t().coo()  # Convert back to COO format
        edge_index = torch.vstack([row, col])  # Create edge_index tensor
        edge_mask = e_id  # Mask indicating which original edges are in subgraph

        # Find relabeled indices of the original edge endpoints in the subgraph
        root_n_id_src = (n_id_unique == index[0]).nonzero(as_tuple=True)[0]
        root_n_id_dst = (n_id_unique == index[1]).nonzero(as_tuple=True)[0]
        root_n_id = torch.tensor([root_n_id_src, root_n_id_dst], dtype=torch.int64)

        # Create the subgraph data object
        data = Data(num_nodes=n_id_unique.numel())
        data.root_n_id = root_n_id  # Relabeled indices of edge endpoints
        data.seed_node = seed_node_ids  # Original global node IDs

        # Handle target edge masking (remove target edge to prevent data leakage)
        if self.pretrain_mode:
            # In pretrain mode, don't mask the target edge
            non_tgt_mask = torch.tensor([True] * edge_index.shape[1], dtype=bool)
        elif self.heuristic_labels is not None:
            # Heuristic-regression mode: Don't mask the target edge because heuristic scores
            # are computed on the full training graph.
            non_tgt_mask = torch.tensor([True] * edge_index.shape[1], dtype=bool)
            # y is the heuristic score
            data.y = y
        else:
            # In fine-tuning mode, remove the target edge
            edge_index, non_tgt_mask = _remove_target_edge(
                edge_index, root_n_id_src, root_n_id_dst
            )
            data.y = y  # Add label only in fine-tuning mode

        data.edge_index = edge_index

        # Copy relevant attributes from the original graph to the subgraph
        for k, v in self.data:
            # Skip certain keys that are not relevant for subgraphs
            if k in ["edge_index", "adj_t", "num_nodes", "batch", "ptr", "y"]:
                continue

            # Handle node-level attributes (copy selected nodes)
            if isinstance(v, Tensor) and v.size(0) == self.data.num_nodes:
                data[k] = v[n_id_unique]  # Copy attributes for nodes in subgraph

            # Handle edge-level attributes (copy selected edges)
            elif isinstance(v, Tensor) and v.size(0) == self.data.num_edges:
                # Handle case where subgraph has no edges
                if self.allow_zero_edges and (
                    edge_mask.numel() == 0 or non_tgt_mask.numel() == 0
                ):
                    print(
                        f"[{datetime.now()}] Subgraph got no edges => [src, tgt]: {index} when assigning {k} !!!\n"
                        f"edge_mask.numel()=={edge_mask.numel()}\n"
                        f"non_tgt_mask.numel()=={non_tgt_mask.numel()}"
                    )
                    # Verify that edge_index is indeed empty
                    assert (
                        edge_index.numel() == 0
                    ), f"edge_index=={edge_index}\nedge_index.numel()=={edge_index.numel()}"
                    # Create empty tensor with same dtype and shape (except first dimension)
                    data[k] = torch.empty((0,) + v.shape[1:], dtype=v.dtype)
                else:
                    # Apply two masks: 1st to get subgraph edges, 2nd to remove target edge
                    data[k] = v[edge_mask][non_tgt_mask]
            else:
                # Copy other attributes as-is (graph-level attributes)
                data[k] = v

        # Add sample metadata
        data.idx = idx  # Original sample index

        # Add target edge attributes if available
        if self.all_edge_attr is not None:
            # [N_p+N_e, edge_attr_dim] -> [edge_attr_dim]
            data.tgt_edge_attr = self.all_edge_attr[idx]

        # Add sample weight if weighted sampling is enabled
        if self.wgt is not None:
            data.wgt = self.wgt[idx]

        return idx, data

    def scatter_preds(self, preds):
        """
        Scatter predictions to the original edges.

        Args:
            preds: Predictions tensor of shape (M, ) where M <= N

        Returns:
            original_preds: Original predictions tensor of shape (N, )
        """
        original_preds = preds[self.orig_to_unique]
        assert original_preds.shape[0] == self.labels.shape[0]
        return original_preds

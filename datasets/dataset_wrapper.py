import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional

from transformers import PreTrainedTokenizer

import random
import threading
from utils import rank_zero_print as print


class DatasetWrapper(Dataset):
    """
    Wrapper dataset that transforms ogbl Data objects into dictionary format
    compatible with PyTorch's default collator.

    The wrapper converts each (idx, data) tuple from the original dataset into
    a dictionary with the following structure:
    - input_embeds: Input embeddings for nodes and edges (excluding task tokens)
    - task_embeds: Task embeddings for link prediction tokens
    - input_attention_mask: Attention mask for input_embeds (all 1s)
    - task_attention_mask: Attention mask for task_embeds (all 1s)
    - labels: Edge labels (1 for positive, 0 for negative)
    - encoding_scheme: Encoding scheme for the input embeddings
        - full: input=[node_embeddings, edge_embeddings], task=[link_embedding] (default)
            - dim: 4 * max_num_nodes + 3 (is_node, is_edge, is_task)
        - adjacency_row: input=[node_embeddings], task=[src_link_token, dst_link_token] (Yehudai et al.)
            - dim: 2 * max_num_nodes + 2 (is_node, is_task)
            - each node is represented by one-hot encoding of its index concatenated with the adjacency row
            - no edges are included in the sequence
            - link prediction uses 2 separate tokens: src token and dst token, each with is_node=0 and is_task=1
        - edge_list: input=[node_embeddings, edge_list_embeddings], task=[link_embedding] (Sanford et al.)
            - dim: 2 * max_num_nodes + 3 (is_node, is_edge, is_task)
            - each node is represented by one-hot encoding of its index
            - no adjacency row is included in the node embeddings, but edges are included in the sequence
    - Additional fields: All original data attributes preserved

    Note: The collator pads input_embeds and task_embeds separately, then concatenates them
    to ensure task tokens are always in the same columns after padding.
    """

    def __init__(
        self,
        original_dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_sequence_length: Optional[int] = 1024,  # GPT-2 max sequence length
        node_remapping: bool = False,
        is_eval: bool = False,
        use_features: bool = False,
        encoding_scheme: str = "full",  # "full", "adjacency_row", "edge_list"
        use_edge_weight: bool = False,
    ):
        """
        Initialize the wrapper dataset.

        Args:
            original_dataset: The original ogbl dataset (ogbl-collab, ogbl-citation2, ogbl-ppa)
            tokenizer: The tokenizer to use
            max_sequence_length: Maximum sequence length for padding/truncation
            node_remapping: Whether to remap node IDs
            is_eval: Evaluation mode, no node remapping
            use_features: Whether to create feature embeddings from data.x
        """
        assert encoding_scheme in ["full", "adjacency_row", "edge_list"], (
            f"Invalid encoding scheme: {encoding_scheme}. "
            f"Supported schemes: full (default), adjacency_row, edge_list"
        )

        self.encoding_scheme = encoding_scheme
        self.original_dataset = original_dataset
        self.tokenizer = tokenizer

        self.max_sequence_length = max_sequence_length
        self.node_remapping = node_remapping
        self.is_eval = is_eval  # evaluation mode, no node remapping
        self.use_features = use_features
        self.use_edge_weight = use_edge_weight

        # Pre-allocate reusable tensors for performance optimization
        self.max_num_nodes = self.tokenizer.max_num_nodes()
        if self.encoding_scheme == "full":
            self.max_num_edges = (
                self.max_sequence_length - 1 - 2
            )  # 1 for task, 2 for nodes
            self.embedding_dim = 4 * self.max_num_nodes + 3
        elif self.encoding_scheme == "adjacency_row":
            self.max_num_edges = 0  # no edges
            self.embedding_dim = 2 * self.max_num_nodes + 2
        elif self.encoding_scheme == "edge_list":
            self.max_num_edges = (
                self.max_sequence_length - 1 - 2
            )  # 1 for task, 2 for nodes
            self.embedding_dim = 2 * self.max_num_nodes + 3

        # Use thread-local storage for thread-safe node embeddings buffer
        self._thread_local = threading.local()

        # Pre-compute all ones and zeros to avoid creating new tensors
        self.all_ones = torch.ones(
            (max(self.embedding_dim, self.max_num_edges), 1), dtype=torch.float
        )
        self.all_zeros = torch.zeros(
            (max(self.embedding_dim, self.max_num_edges), 1), dtype=torch.float
        )

    def __len__(self):
        """Return the length of the original dataset."""
        return len(self.original_dataset)

    def _get_thread_local_node_embeddings_buffer(self):
        """Get thread-local buffers, initializing them if needed."""
        if not hasattr(self._thread_local, "node_embeddings_buffer"):
            # Initialize thread-local buffers for this worker
            self._thread_local.node_embeddings_buffer = torch.zeros(
                (self.max_num_nodes, self.embedding_dim), dtype=torch.float
            )
        return self._thread_local.node_embeddings_buffer

    def _create_node_embeddings(
        self, connected_nodes, adj_indices, max_num_nodes, adj_values
    ):
        """Create node embeddings for connected nodes only (lazy computation).

        Args:
            connected_nodes: Tensor of connected node indices
            adj_indices: Adjacency matrix indices (2, num_edges)
            max_num_nodes: Maximum number of nodes
            adj_values: Edge weights for adjacency rows (num_edges,).
        """
        num_connected_nodes = len(connected_nodes)

        # Get thread-local buffers
        node_embeddings_buffer = self._get_thread_local_node_embeddings_buffer()

        # REUSE pre-allocated node embeddings tensor
        if num_connected_nodes > 0 and node_embeddings_buffer.size(0) > 0:
            node_embeddings = node_embeddings_buffer[
                :num_connected_nodes
            ]  # Slice to actual size
            node_embeddings.zero_()  # Zero out for reuse
        else:
            # Fallback to creating new tensor if buffer is empty or no nodes
            node_embeddings = torch.zeros(
                (num_connected_nodes, self.embedding_dim), dtype=torch.float
            )

        # Set node flags for connected nodes only
        if self.encoding_scheme == "full":
            node_embeddings[:, 4 * max_num_nodes] = 1.0  # is_node = 1
        elif self.encoding_scheme == "adjacency_row":
            node_embeddings[:, 2 * max_num_nodes] = 1.0  # is_node = 1
        elif self.encoding_scheme == "edge_list":
            node_embeddings[:, 2 * max_num_nodes] = 1.0  # is_node = 1
        else:
            raise ValueError(f"Invalid encoding scheme: {self.encoding_scheme}")

        # is_edge = 0 and is_task = 0 are already set by zero_() above, no need to explicitly set

        # Vectorize adjacency row assignment using advanced indexing for maximum efficiency
        # Create mapping from node_id to local_idx for vectorized operations
        node_id_to_local = torch.zeros(max_num_nodes, dtype=torch.long)
        node_id_to_local[connected_nodes] = torch.arange(num_connected_nodes)

        # Set adjacency information for connected nodes using fully vectorized operations
        # Vectorize 1-hot node index assignment
        node_embeddings[node_id_to_local[connected_nodes], connected_nodes] = 1.0

        if self.encoding_scheme in ["full", "adjacency_row"]:
            # Get all source nodes and their corresponding local indices
            src_nodes = adj_indices[0]
            dst_nodes = adj_indices[1]
            src_local_indices = node_id_to_local[src_nodes]

            # Create adjacency positions [max_num_nodes:max_num_nodes*2]
            adj_positions = max_num_nodes + dst_nodes

            # Set adjacency relationships with edge weights
            node_embeddings[src_local_indices, adj_positions] = adj_values

        # IMPORTANT: Return a copy to avoid buffer reuse issues across samples of the same batch
        node_embeddings = node_embeddings.clone()

        return node_embeddings, connected_nodes.tolist(), node_id_to_local

    def _create_edge_embeddings(
        self,
        filtered_src,
        filtered_dst,
        node_embeddings,
        node_id_to_local,
        num_edges,
        max_num_nodes,
    ):
        """Create edge embeddings using vectorized operations."""
        if num_edges > 0:
            # Shuffle edges
            perm = torch.randperm(len(filtered_src))
            shuffled_src = filtered_src[perm]
            shuffled_dst = filtered_dst[perm]

            # Map global node IDs to local indices for connected nodes using tensor indexing
            src_local_indices = node_id_to_local[shuffled_src]
            dst_local_indices = node_id_to_local[shuffled_dst]

            # Retrieve source and destination node embeddings directly
            edge_dim = (
                2 * max_num_nodes
                if self.encoding_scheme in ["full", "adjacency_row"]
                else max_num_nodes
            )
            src_embeddings = node_embeddings[src_local_indices, :edge_dim]
            dst_embeddings = node_embeddings[dst_local_indices, :edge_dim]

            # Concatenate source and destination embeddings
            edge_content = torch.cat([src_embeddings, dst_embeddings], dim=1)

            # Concatenate edge content with flags
            all_ones = self.all_ones[:num_edges]
            all_zeros = self.all_zeros[:num_edges]
            edge_embeddings = torch.cat(
                [edge_content, all_zeros, all_ones, all_zeros], dim=1
            )
        else:
            # Empty edge embeddings
            shuffled_src, shuffled_dst = filtered_src, filtered_dst
            edge_embeddings = torch.empty((0, self.embedding_dim), dtype=torch.float)

        return edge_embeddings, shuffled_src, shuffled_dst

    def _create_link_embedding(
        self, src_mapped, dst_mapped, node_embeddings, node_id_to_local, max_num_nodes
    ):
        """Create link embedding for prediction task."""
        # Map to local indices for lazy computation using tensor indexing
        src_local_idx = node_id_to_local[src_mapped]
        dst_local_idx = node_id_to_local[dst_mapped]

        link_dim = (
            2 * max_num_nodes
            if self.encoding_scheme in ["full", "adjacency_row"]
            else max_num_nodes
        )
        # Retrieve source and destination node embeddings directly
        src_embedding = node_embeddings[[src_local_idx], :link_dim]
        dst_embedding = node_embeddings[[dst_local_idx], :link_dim]

        # Concatenate link content with flags
        all_zeros = self.all_zeros[:1]
        all_ones = self.all_ones[:1]
        if self.encoding_scheme in ["full", "edge_list"]:
            # Concatenate source and destination embeddings
            link_content = torch.cat([src_embedding, dst_embedding], dim=1)
            link_embedding = torch.cat(
                [link_content, all_zeros, all_zeros, all_ones], dim=1
            )
            return link_embedding
        elif self.encoding_scheme == "adjacency_row":
            # For adjacency_row, return 2 separate tokens: src and dst
            # Each token has: [node_embedding, is_node=0, is_task=1]
            src_link_embedding = torch.cat([src_embedding, all_zeros, all_ones], dim=1)
            dst_link_embedding = torch.cat([dst_embedding, all_zeros, all_ones], dim=1)
            # Stack to create (2, embedding_dim) tensor
            link_embeddings = torch.cat([src_link_embedding, dst_link_embedding], dim=0)
            return link_embeddings
        else:
            raise ValueError(f"Invalid encoding scheme: {self.encoding_scheme}")

    def _create_feature_embeddings(
        self,
        data,
        connected_nodes_list,
        num_connected_nodes,
        shuffled_src,
        shuffled_dst,
        num_edges,
        idx_map_tensor,
        src_mapped,
        dst_mapped,
        node_id_to_local,
    ):
        """
        Create feature embeddings from data.x matching the full sequence length.

        Sequence: [node_embeddings, edge_embeddings, link_embedding] for full/edge_list
        Sequence: [node_embeddings, src_link_token, dst_link_token] for adjacency_row

        For full/edge_list:
            - Nodes: copy x to first half of feature_embeddings
            - Edges: copy x of two endpoints to first half (src) and second half (dst)
            - Link: copy x of two endpoints to first half (src) and second half (dst)
        For adjacency_row:
            - Nodes: copy x to feature_embeddings (full dimension, not half)
            - Link tokens: each token has its own node's x features (full dimension)

        Args:
            data: PyG data object with node features in data.x
            connected_nodes_list: List of connected node indices (after remapping)
            num_connected_nodes: Number of connected nodes
            shuffled_src: Source nodes for edges (remapped)
            shuffled_dst: Destination nodes for edges (remapped)
            num_edges: Number of edges
            idx_map_tensor: Mapping from original index to remapped index
            src_mapped: Source node for link (remapped)
            dst_mapped: Destination node for link (remapped)
            node_id_to_local: Mapping from remapped node ID to local index in node_embeddings

        Returns:
            feature_embeddings with shape:
            - (num_connected_nodes + num_edges + 1, 2 * feature_dim) for full/edge_list
            - (num_connected_nodes + 2, feature_dim) for adjacency_row
            or None if data.x doesn't exist
        """
        # Check if data.x exists
        if not hasattr(data, "x") or data.x is None:
            return None

        # Get feature dimension
        feature_dim = data.x.size(1)

        if self.encoding_scheme in ["full", "edge_list"]:
            # Total number of items: nodes + edges + 1 (link)
            total_items = num_connected_nodes + num_edges + 1
        elif self.encoding_scheme == "adjacency_row":
            # Total number of items: nodes + 2 (link: src and dst tokens)
            total_items = num_connected_nodes + 2
        else:
            raise ValueError(f"Invalid encoding scheme: {self.encoding_scheme}")

        # Create feature embeddings tensor
        # For full/edge_list: each item uses 2*feature_dim (src+dst concatenated)
        # For adjacency_row: each item uses feature_dim (nodes and link tokens each have their own features)
        if self.encoding_scheme == "adjacency_row":
            feature_emb_dim = feature_dim
        else:
            feature_emb_dim = 2 * feature_dim

        feature_embeddings = torch.zeros(
            total_items, feature_emb_dim, dtype=torch.float
        )

        # Convert connected_nodes_list to tensor for indexing
        connected_nodes_tensor = torch.tensor(connected_nodes_list, dtype=torch.long)

        # Create inverse mapping from remapped to original
        # idx_map_tensor[original_idx] = remapped_idx
        # We need remapped_to_original[remapped_idx] = original_idx
        # Since idx_map_tensor is a permutation, argsort gives us the inverse
        remapped_to_original = torch.argsort(idx_map_tensor)

        # Step 2: For nodes, copy x to feature embeddings
        # For full/edge_list: copy to first half (positions 0 to feature_dim-1)
        # For adjacency_row: copy to full dimension (positions 0 to feature_dim-1, which is the entire dimension)
        # Use node_id_to_local to ensure features are placed in the same positions as node_embeddings
        if num_connected_nodes > 0:
            # connected_nodes_list contains remapped indices, so we need to map back to original
            # Get original node indices from remapped indices
            original_indices = remapped_to_original[connected_nodes_tensor]
            # Get local indices using node_id_to_local to match node_embeddings ordering
            local_indices = node_id_to_local[connected_nodes_tensor]
            # Copy features - data.x is locally indexed so we can use original_indices directly
            # :feature_dim works for both cases (first half for full/edge_list, full for adjacency_row)
            # Use local_indices to place features in the correct positions matching node_embeddings
            feature_embeddings[local_indices, :feature_dim] = data.x[original_indices]

        # Step 3: For edges, copy x of two endpoints
        # Position: num_connected_nodes to num_connected_nodes + num_edges
        if num_edges > 0 and self.encoding_scheme in ["full", "edge_list"]:
            # Get original node indices for shuffled_src and shuffled_dst
            original_src_indices = remapped_to_original[shuffled_src]
            original_dst_indices = remapped_to_original[shuffled_dst]

            # Copy src features to first half, dst features to second half - data.x is locally indexed
            edge_start_idx = num_connected_nodes
            feature_embeddings[
                edge_start_idx : edge_start_idx + num_edges, :feature_dim
            ] = data.x[original_src_indices]
            feature_embeddings[
                edge_start_idx : edge_start_idx + num_edges, feature_dim:
            ] = data.x[original_dst_indices]
        elif num_edges > 0 and self.encoding_scheme == "adjacency_row":
            # adjacency_row doesn't have edges, but handle edge case
            pass

        # Step 4: For link, copy x of two endpoints
        # Position: num_connected_nodes + num_edges (last position)
        if self.encoding_scheme in ["full", "edge_list"]:
            link_pos = num_connected_nodes + num_edges
            original_src_idx = remapped_to_original[src_mapped]
            original_dst_idx = remapped_to_original[dst_mapped]
            feature_embeddings[link_pos, :feature_dim] = data.x[original_src_idx]
            feature_embeddings[link_pos, feature_dim:] = data.x[original_dst_idx]
        elif self.encoding_scheme == "adjacency_row":
            # For adjacency_row, we have 2 separate link tokens: src and dst
            # Each token has its own node feature (feature_dim, not 2*feature_dim)
            src_link_pos = num_connected_nodes
            dst_link_pos = num_connected_nodes + 1
            original_src_idx = remapped_to_original[src_mapped]
            original_dst_idx = remapped_to_original[dst_mapped]
            # Each token has its own node feature
            feature_embeddings[src_link_pos, :] = data.x[original_src_idx]
            feature_embeddings[dst_link_pos, :] = data.x[original_dst_idx]
        else:
            raise ValueError(f"Invalid encoding scheme: {self.encoding_scheme}")

        return feature_embeddings

    def _filter_and_shuffle_edges(
        self,
        remapped_edge_index,
        max_sequence_length,
        num_connected_nodes,
        edge_weights=None,
    ):
        """Filter and shuffle edges using vectorized operations.

        Args:
            remapped_edge_index: Remapped edge indices (2, num_edges)
            max_sequence_length: Maximum sequence length
            num_connected_nodes: Number of connected nodes
            edge_weights: Optional edge weights (num_edges,).
                If provided and self.encoding_scheme is edge_list, edges will be repeated according to their weights.

        Returns:
            filtered_src, filtered_dst, num_edges
        """
        # Filter remapped_edge_index using torch.where for efficiency
        # Create mask for src < dst (on remapped indices)
        src_mask = remapped_edge_index[0] < remapped_edge_index[1]

        # Use torch.where for more efficient filtering
        valid_indices = torch.where(src_mask)[0]
        filtered_src = remapped_edge_index[0][valid_indices]
        filtered_dst = remapped_edge_index[1][valid_indices]

        # For edge_list scheme, expand edges based on edge_weights if provided
        if self.encoding_scheme == "edge_list" and edge_weights is not None:
            # Get filtered edge weights
            filtered_weights = edge_weights[valid_indices]
            # Convert weights to integers (they should be positive integers)
            filtered_weights_int = filtered_weights.long()

            # Repeat each edge according to its weight using vectorized operations
            # Create repeat indices using torch.repeat_interleave for efficiency
            if len(filtered_weights_int) > 0:
                repeat_indices = torch.repeat_interleave(
                    torch.arange(
                        len(filtered_weights_int),
                        dtype=torch.long,
                        device=filtered_weights_int.device,
                    ),
                    filtered_weights_int,
                )
                filtered_src = filtered_src[repeat_indices]
                filtered_dst = filtered_dst[repeat_indices]

        # Limit number of edges if needed
        num_edges = len(filtered_src)
        if self.encoding_scheme in ["full", "edge_list"]:
            permitted_num_edges = (
                max_sequence_length - num_connected_nodes - 1
            )  # reserve 1 slot for link embedding
        else:
            raise ValueError(f"Invalid encoding scheme: {self.encoding_scheme}")

        assert (
            permitted_num_edges >= 0
        ), f"Permitted number of edges {permitted_num_edges} is less than 0"

        if num_edges > permitted_num_edges:
            # Randomly sample edges if too many
            perm = torch.randperm(num_edges)[:permitted_num_edges]
            filtered_src = filtered_src[perm]
            filtered_dst = filtered_dst[perm]
            num_edges = permitted_num_edges

        return filtered_src, filtered_dst, num_edges

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample and convert it to embedding format.

        Format: [n_emb_0, n_emb_1, ... e_emb_0, e_emb_1, ... link_emb]

        Args:
            idx: Index of the sample

        Returns:
            Dictionary with embedding-based format
        """
        # Get the original sample
        _, data = self.original_dataset[idx]

        # Extract basic information
        num_nodes = data.num_nodes

        # Cache max_num_nodes for performance
        max_num_nodes = self.max_num_nodes
        assert (
            num_nodes <= max_num_nodes
        ), f"Number of nodes {num_nodes} exceeds max number of nodes {max_num_nodes} per sample"

        if self.node_remapping and not self.is_eval:
            assert (
                not self.use_features
            ), "Node remapping and use_features cannot be True at the same time"
            r = random.randint(0, max_num_nodes - 1)
            # Optimize idx_map creation using vectorized operations
            idx_map_tensor = torch.arange(max_num_nodes, dtype=torch.long)
            idx_map_tensor = (idx_map_tensor + r) % max_num_nodes
            idx_map = idx_map_tensor.tolist()
        else:
            # Enforce idx for link must be 0 and 1
            src, dst = data.root_n_id.tolist()

            # Initialize empty tensor and set src→0, dst→1
            # Only create mapping for nodes that actually exist (num_nodes instead of max_num_nodes)
            idx_map_tensor = torch.zeros(num_nodes, dtype=torch.long)
            idx_map_tensor[dst] = 1
            idx_map_tensor[src] = (
                0  # CRITICAL: src must be set after dst for edge case src == dst
            )

            # Assign remaining nodes values from 2 to num_nodes
            # Create mask for nodes that are not src or dst
            all_indices = torch.arange(num_nodes, dtype=torch.long)
            mask = (all_indices != src) & (all_indices != dst)
            remaining_nodes = all_indices[mask]

            if len(remaining_nodes) > 0:
                if src != dst:
                    remaining_values = torch.arange(
                        2, 2 + len(remaining_nodes), dtype=torch.long
                    )
                else:
                    # src == dst, so we start from 1 instead of 2 because src = dst = 0
                    remaining_values = torch.arange(
                        1, 1 + len(remaining_nodes), dtype=torch.long
                    )
                # shuffle remaining values
                if not self.is_eval:
                    remaining_values = remaining_values[
                        torch.randperm(len(remaining_values))
                    ]
                idx_map_tensor[remaining_nodes] = remaining_values

            idx_map = idx_map_tensor.tolist()
            r = -1

        # Build sparse adjacency representation using PyG utilities
        edge_index = data.edge_index

        # Remap edge indices efficiently using vectorized tensor operations
        src_indices = edge_index[0]
        dst_indices = edge_index[1]
        src_mapped = idx_map_tensor[src_indices]
        dst_mapped = idx_map_tensor[dst_indices]

        # Create remapped edge index (already undirected)
        remapped_edge_index = torch.stack([src_mapped, dst_mapped], dim=0)

        # Create sparse adjacency matrix in COO format using direct sparse tensor creation
        # This is more efficient than to_torch_coo_tensor for our use case
        if (
            self.use_edge_weight
            and hasattr(data, "edge_weight")
            and data.edge_weight is not None
        ):
            edge_weights = data.edge_weight.float()
            # Ensure edge_weights is 1D and has the correct length
            if edge_weights.dim() > 1:
                edge_weights = edge_weights.squeeze()
            # Ensure it matches the number of edges
            if edge_weights.size(0) != remapped_edge_index.size(1):
                raise ValueError(
                    f"edge_weight size ({edge_weights.size(0)}) does not match "
                    f"edge_index size ({remapped_edge_index.size(1)})"
                )
        else:
            edge_weights = torch.ones(remapped_edge_index.size(1), dtype=torch.float)
        sparse_adj = torch.sparse_coo_tensor(
            remapped_edge_index, edge_weights, (max_num_nodes, max_num_nodes)
        ).coalesce()

        # Extract adjacency information efficiently using sparse tensor indices
        # Get all non-zero positions (connected nodes)
        adj_indices = sparse_adj.indices()
        adj_values = sparse_adj.values()

        # Get all nodes that have edges (connected nodes only) - vectorized
        # Include both source and destination nodes from the adjacency matrix
        all_edge_nodes = torch.cat([adj_indices[0], adj_indices[1]])
        connected_nodes = torch.unique(all_edge_nodes)

        # Ensure root nodes are included in connected nodes for link prediction
        root_nodes = data.root_n_id.tolist()
        src, dst = root_nodes
        # enforce src < dst for link prediction
        src_mapped, dst_mapped = min(idx_map[src], idx_map[dst]), max(
            idx_map[src], idx_map[dst]
        )

        # Add root nodes to connected nodes if they're not already there
        root_nodes_tensor = torch.tensor([src_mapped, dst_mapped], dtype=torch.long)
        # Only add root nodes that are not already in connected_nodes
        missing_root_nodes = root_nodes_tensor[
            ~torch.isin(root_nodes_tensor, connected_nodes)
        ]
        if len(missing_root_nodes) > 0:
            # add missing root nodes to connected nodes
            all_nodes = torch.cat([connected_nodes, missing_root_nodes])
            connected_nodes = torch.unique(all_nodes)

        # Create node embeddings using helper method
        # connected_node_list is the list of indices after remapping
        node_embeddings, connected_nodes_list, node_id_to_local = (
            self._create_node_embeddings(
                connected_nodes, adj_indices, max_num_nodes, adj_values
            )
        )
        num_connected_nodes = len(connected_nodes_list)

        # Filter and shuffle edges using helper method
        if self.encoding_scheme in ["full", "edge_list"]:
            # tmp_edge_weights is edge_weights if use_edge_weight is enabled, otherwise None
            # for efficiency so that we don't need to repeat edges if edge_weights is not provided
            tmp_edge_weights = (
                edge_weights
                if self.use_edge_weight
                and hasattr(data, "edge_weight")
                and data.edge_weight is not None
                else None
            )
            filtered_src, filtered_dst, num_edges = self._filter_and_shuffle_edges(
                remapped_edge_index,
                self.max_sequence_length,
                num_connected_nodes,
                tmp_edge_weights,
            )

            # Create edge embeddings using helper method
            edge_embeddings, shuffled_src, shuffled_dst = self._create_edge_embeddings(
                filtered_src,
                filtered_dst,
                node_embeddings,
                node_id_to_local,
                num_edges,
                max_num_nodes,
            )
        else:
            num_edges = 0

        # Separate embeddings into input_embeddings (nodes + edges) and task_embeddings (link)
        input_embeddings_to_stack = []

        if num_connected_nodes > 0:
            input_embeddings_to_stack.append(node_embeddings)

        if num_edges > 0 and self.encoding_scheme in ["full", "edge_list"]:
            input_embeddings_to_stack.append(edge_embeddings)

        # Stack input embeddings (nodes + edges)
        if len(input_embeddings_to_stack) > 0:
            input_embeddings = torch.cat(input_embeddings_to_stack, dim=0)
        else:
            # Empty input embeddings (shouldn't happen, but handle gracefully)
            input_embeddings = torch.empty((0, self.embedding_dim), dtype=torch.float)

        # Create link embedding using helper method
        # src_mapped and dst_mapped are already computed above
        task_embeddings = self._create_link_embedding(
            src_mapped, dst_mapped, node_embeddings, node_id_to_local, max_num_nodes
        )

        # Create feature embeddings using helper method only if use_features is True
        if self.use_features:
            feature_embeddings = self._create_feature_embeddings(
                data,
                connected_nodes_list,
                num_connected_nodes,
                shuffled_src if self.encoding_scheme in ["full", "edge_list"] else None,
                shuffled_dst if self.encoding_scheme in ["full", "edge_list"] else None,
                num_edges,
                idx_map_tensor,
                src_mapped,
                dst_mapped,
                node_id_to_local,
            )
            # Separate feature embeddings into input_features and task_features
            # input_features: nodes + edges
            # task_features: link token(s)
            if self.encoding_scheme in ["full", "edge_list"]:
                num_input_items = num_connected_nodes + num_edges
                num_task_items = 1
            elif self.encoding_scheme == "adjacency_row":
                num_input_items = num_connected_nodes
                num_task_items = 2  # src and dst link tokens
            else:
                raise ValueError(f"Invalid encoding scheme: {self.encoding_scheme}")

            input_feature_embeddings = feature_embeddings[:num_input_items]
            task_feature_embeddings = feature_embeddings[num_input_items:]
        else:
            feature_embeddings = None
            input_feature_embeddings = None
            task_feature_embeddings = None

        # Create separate attention masks for input and task embeddings
        input_attention_mask = [1] * input_embeddings.size(0)
        task_attention_mask = [1] * task_embeddings.size(0)

        # Create the dictionary with all original data attributes
        sample_dict = {
            "input_embeds": input_embeddings,
            "task_embeds": task_embeddings,
            "input_attention_mask": input_attention_mask,
            "task_attention_mask": task_attention_mask,
            "num_nodes": num_connected_nodes,  # Only connected nodes in output
            "num_edges": num_edges,
            "idx_offset": r,  # idx offset for cyclical sampling
            # uncomment the below keys for testing
            # "dense_adj": sparse_adj.to_dense(),
            # "edge_index": remapped_edge_index,
            # "shuffled_edge_index": (
            #     torch.stack([shuffled_src, shuffled_dst])
            #     if self.encoding_scheme in ["full", "edge_list"]
            #     else None
            # ),
            # "root_nodes": (src_mapped, dst_mapped),
            # "connected_nodes": connected_nodes_list,  # Use the list version
        }

        # Add feature embeddings if they were created
        if input_feature_embeddings is not None:
            sample_dict["input_feature_embeds"] = input_feature_embeddings
            sample_dict["task_feature_embeds"] = task_feature_embeddings

        # idx in the dataset list
        if hasattr(data, "idx"):
            sample_dict["idx"] = data.idx

        # label
        if hasattr(data, "y") and data.y is not None:
            sample_dict["y"] = data.y.item()

        return sample_dict

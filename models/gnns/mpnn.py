import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

from typing import Optional


CONV_TYPE_REGISTRY = {
    "gcn": GCNConv,
    "gat": GATConv,
    "sage": SAGEConv,
}


class MPNN(torch.nn.Module):
    def __init__(
        self,
        config,
        feature_dim,
        residual: bool = False,
        id_awareness: bool = False,
        ortho_embedding: bool = False,
        is_binary: bool = True,
        max_num_nodes: Optional[int] = None,
    ):
        super(MPNN, self).__init__()
        self.name = config.name
        assert self.name in CONV_TYPE_REGISTRY.keys()
        self.conv = CONV_TYPE_REGISTRY[self.name]
        self.id_awareness = id_awareness
        self.ortho_embedding = ortho_embedding
        # make sure ortho_embedding and id_awareness are not both True
        assert not (
            ortho_embedding and id_awareness
        ), "ortho_embedding and id_awareness cannot both be True"
        # Whether this run is standard binary link prediction or heuristic regression
        self.is_binary = is_binary
        self.hidden_channels = config.hidden_size
        # Adjust in_channels if ortho_embedding is used (will concatenate embedding with x)
        self.in_channels = feature_dim + (
            self.hidden_channels if ortho_embedding else 0
        )
        self.num_layers = config.num_layers
        self.num_layers_predictor = config.num_layers_predictor
        self.dropout = config.dropout
        self.residual = residual
        if self.residual:
            self.res_lins = torch.nn.ModuleList()
        if self.id_awareness:
            # id_embedding uses self.in_channels which is already adjusted for ortho_embedding
            self.id_embedding = torch.nn.Embedding(1, self.in_channels)

        self.max_num_nodes = max_num_nodes
        if self.ortho_embedding:
            assert (
                self.max_num_nodes is not None
            ), "max_num_nodes must be provided when ortho_embedding is True"
            assert (
                self.max_num_nodes < 1e5
            ), "max_num_nodes must be less than 1e5 for stable sorting"

            # similar to BERTLP's input_proj
            self.ortho_x = torch.nn.Parameter(
                torch.empty(self.max_num_nodes, self.hidden_channels)
            )
            torch.nn.init.orthogonal_(self.ortho_x)
            self.ortho_x.requires_grad = False

            # Lazy-initialized repeated ortho_x for batch processing
            self.repeat_ortho_x = None
            self._cached_batch_size = None

        self.convs = torch.nn.ModuleList()
        # Explicitly set root_weight=False to avoid residual connections for SAGE. We'll add
        # residual connection in our forward pass.
        if self.num_layers == 1:
            if self.residual:
                # For num_layers == 1, res_lins[0] is only used at the end where x is already hidden_channels
                self.res_lins.append(
                    torch.nn.Linear(self.hidden_channels, self.hidden_channels)
                )
            if self.name == "sage":
                self.convs.append(
                    self.conv(self.in_channels, self.hidden_channels, root_weight=False)
                )
            else:
                self.convs.append(self.conv(self.in_channels, self.hidden_channels))

        elif self.num_layers > 1:
            if self.residual:
                self.res_lins.append(
                    torch.nn.Linear(self.in_channels, self.hidden_channels)
                )
            if self.name == "sage":
                self.convs.append(
                    self.conv(self.in_channels, self.hidden_channels, root_weight=False)
                )
            else:
                self.convs.append(self.conv(self.in_channels, self.hidden_channels))

            for _ in range(self.num_layers - 1):
                if self.residual:
                    self.res_lins.append(
                        torch.nn.Linear(self.hidden_channels, self.hidden_channels)
                    )
                if self.name == "sage":
                    self.convs.append(
                        self.conv(
                            self.hidden_channels,
                            self.hidden_channels,
                            root_weight=False,
                        )
                    )
                else:
                    self.convs.append(
                        self.conv(self.hidden_channels, self.hidden_channels)
                    )

        self.lins = torch.nn.ModuleList()
        if self.num_layers_predictor == 1:
            self.lins.append(torch.nn.Linear(self.hidden_channels, 1))
        else:
            for _ in range(self.num_layers_predictor - 1):
                self.lins.append(
                    torch.nn.Linear(self.hidden_channels, self.hidden_channels)
                )
            self.lins.append(torch.nn.Linear(self.hidden_channels, 1))

    def _get_ortho_x(self, data, src, dst):
        """
        Draw x from self.ortho_x such that each graph in the batch gets its own slice.
        Ensure src always gets row 0 and dst always gets row 1.
        Vectorized implementation using lazy-initialized repeat_ortho_x.

        Args:
            data: PyG Batch object containing graph data
            src: Source node indices (batch_size,)
            dst: Destination node indices (batch_size,)

        Returns:
            x: Node features tensor (total_nodes, hidden_channels)
        """
        # Get number of nodes per graph using ptr (PyG Batch attribute)
        if hasattr(data, "ptr"):
            # ptr is [0, n0, n0+n1, n0+n1+n2, ...] where ni is num_nodes for graph i
            num_nodes_per_graph = data.ptr[1:] - data.ptr[:-1]
            batch_size = len(num_nodes_per_graph)
        else:
            raise ValueError("ptr attribute not found in data")

        # Lazy initialize repeat_ortho_x if needed
        if self.repeat_ortho_x is None or self._cached_batch_size != batch_size:
            # Repeat ortho_x batch_size times and reshape to (batch_size, max_num_nodes, hidden_channels)
            # This allows direct indexing by [batch_idx, local_idx]
            self.repeat_ortho_x = self.ortho_x.unsqueeze(0).repeat(batch_size, 1, 1)
            # Shape: (batch_size, max_num_nodes, hidden_channels)
            self._cached_batch_size = batch_size

        # Create row assignments for all graphs vectorized
        total_nodes = data.ptr[-1].item()

        # Create graph indices associated with each node [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2, ...]
        graph_indices_expanded = torch.repeat_interleave(
            torch.arange(batch_size, device=self.ortho_x.device), num_nodes_per_graph
        )

        # Create row assignment from scratch (similar to dataset_wrapper.py)
        # Initialize to zeros, then assign src→0, dst→1, and remaining nodes
        row_assignment = torch.zeros(
            total_nodes, dtype=torch.long, device=self.ortho_x.device
        )

        # Handle dst assignment only for graphs with more than 1 node
        valid_dst_mask = num_nodes_per_graph > 1

        # Set dst to 1 first (for graphs with >1 node)
        # CRITICAL: dst must be set before src to handle edge case src == dst
        if valid_dst_mask.any():
            row_assignment[dst[valid_dst_mask]] = 1

        # Set src to 0 (after dst, handles edge case src == dst)
        row_assignment[src] = 0

        # Assign remaining nodes (not src and not dst) with values starting from 2
        # Create mask for nodes that are not src or dst
        all_global_indices = torch.arange(total_nodes, device=self.ortho_x.device)
        remaining_mask = torch.ones(
            total_nodes, dtype=torch.bool, device=self.ortho_x.device
        )
        remaining_mask[src] = False
        if valid_dst_mask.any():
            remaining_mask[dst[valid_dst_mask]] = False

        remaining_global_indices = all_global_indices[remaining_mask]

        if len(remaining_global_indices) > 0:
            # For each graph, assign sequential values starting from 2 to remaining nodes
            remaining_graph_indices = graph_indices_expanded[remaining_mask]

            # Group by graph and assign values 2, 3, 4, ... to remaining nodes in each graph
            unique_graphs, inverse_indices, counts = torch.unique(
                remaining_graph_indices, return_inverse=True, return_counts=True
            )

            # For each graph segment, assign sequential values starting from 2
            segment_starts = torch.cumsum(
                torch.cat([torch.tensor([0], device=self.ortho_x.device), counts[:-1]]),
                dim=0,
            )
            segment_offsets = torch.repeat_interleave(segment_starts, counts)
            # remaining_values = [2, 3, ..., 2, 3, ..., 2, 3, ...]
            remaining_values = (
                torch.arange(len(remaining_global_indices), device=self.ortho_x.device)
                - segment_offsets
            ) + 2

            # Shuffle values within each graph segment
            # Use segment-based permutation: group by graph and permute within each group
            random_keys = torch.rand(
                len(remaining_global_indices), device=self.ortho_x.device
            )
            # Combine graph index and random key for stable sort within segments
            # Use a large multiplier to ensure graph boundaries are respected
            # Safe multiplier: supports up to 10k graphs (max sort_key 10,000 * 1e5 = 1e9 << float32 max .4e38)
            sort_keys = remaining_graph_indices.float() * 1e5 + random_keys
            sort_indices = torch.argsort(sort_keys)

            # Apply permutation: reorder the values using the sorted indices
            permuted_values = remaining_values[sort_indices]
            row_assignment[remaining_global_indices] = permuted_values

        # Use graph_indices_expanded and row_assignment to directly index into reshaped repeat_ortho_x
        # graph_indices_expanded gives us the batch index, row_assignment gives us the local index
        x = self.repeat_ortho_x[graph_indices_expanded, row_assignment]
        return x

    def forward(self, data, labels: Optional[torch.Tensor] = None):
        link_indices = (
            data.root_n_index
        )  # IMPORTANT: use root_n_index for correct offsetting during batching
        src, dst = link_indices[:, 0], link_indices[:, 1]

        x = data.x
        if self.ortho_embedding:
            # Concatenate ortho embedding with x
            ortho_x = self._get_ortho_x(data, src, dst)
            x = torch.cat([x, ortho_x], dim=-1)
        if self.id_awareness:
            # Add id embedding to source nodes
            x[src] += self.id_embedding.weight
        for i, conv in enumerate(self.convs[:-1]):
            if self.residual:
                x = conv(x, data.edge_index) + self.res_lins[i](x)
            else:
                x = conv(x, data.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.edge_index)
        if self.residual:
            x = x + self.res_lins[-1](x)

        if not self.id_awareness:
            src_embeddings, dst_embeddings = x[src], x[dst]
            link_embeddings = src_embeddings * dst_embeddings  # element-wise product
        else:
            link_embeddings = x[dst]  # NBFNet only reads dst nodes
        for lin in self.lins[:-1]:
            link_embeddings = lin(link_embeddings)
            link_embeddings = F.relu(link_embeddings)
            link_embeddings = F.dropout(
                link_embeddings, p=self.dropout, training=self.training
            )
        link_logits = self.lins[-1](link_embeddings).squeeze(-1)  # Shape: (batch_size,)

        loss = None
        if labels is not None:
            target = labels.view_as(link_logits).float()
            if self.is_binary:
                # Standard link prediction with binary labels {0, 1}
                loss_fn = torch.nn.BCEWithLogitsLoss()
            else:
                # Regression loss for heuristic experiments (MSE on raw scores)
                loss_fn = torch.nn.MSELoss()
            loss = loss_fn(link_logits, target)

        return {"loss": loss, "logits": link_logits}

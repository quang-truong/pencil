import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import AutoConfig, PreTrainedTokenizer, AutoModel
from typing import Optional, Tuple, Union
from utils import rank_zero_print as print, get_rank

from rich import print as rich_print


class LinkPredictor(ABC, nn.Module):
    """
    Abstract base class for link prediction models.

    This class defines the common interface and initialization for models that
    perform link prediction tasks, ensuring consistent lp_head initialization
    and forward pass logic for extracting representations at [A] token positions.
    """

    def __init__(
        self,
        config,
        tokenizer,
        encoding_scheme: str = "full",  # "full", "adjacency_row", "edge_list"
        *,
        is_binary: bool = True,
    ):
        """
        Initialize the link prediction head.

        Args:
            config: Model configuration containing n_embd (embedding dimension)
            tokenizer: Tokenizer for accessing IDs
            use_features: Whether to use feature embeddings (default: False)
            feature_dim: Dimension of feature embeddings.
                         - For full/edge_list: 2 * node_feature_dim (src+dst concatenated)
                         - For adjacency_row: node_feature_dim (each token has its own features)
                         Required if use_features=True.
            feature_fusion: Fusion strategy for features - 'early' or 'late' (default: None).
                           Required if use_features=True.
        """
        # Note: nn.Module.__init__ is not called here because LinkPredictor is always used
        # as a mixin with model classes (e.g., BertModel, GPT2Model) that initialize
        # nn.Module first. Calling it here would reset _modules and remove attributes
        # like self.embeddings that were registered by the model class.

        # Task type: binary link prediction (BCE) vs. regression (MSE)
        self.is_binary = is_binary

        # get input length
        self.max_num_nodes = tokenizer.max_num_nodes()
        if encoding_scheme == "full":
            self.input_length = (
                tokenizer.max_num_nodes() * 2 * 2 + 3
            )  # 2 * 2(num_nodes + num_nodes) + 3 (is_node, is_edge, is_task)
        elif encoding_scheme == "adjacency_row":
            self.input_length = (
                tokenizer.max_num_nodes() * 2 + 2
            )  # 2 * num_nodes + 2 (is_node, is_task)
        elif encoding_scheme == "edge_list":
            self.input_length = (
                tokenizer.max_num_nodes() * 2 + 3
            )  # 2 * (num_nodes + num_nodes) + 3 (is_node, is_edge, is_task)
        else:
            raise ValueError(f"Invalid encoding scheme: {encoding_scheme}")

        self.task_idx = self.input_length - 1  # last index is task
        self.encoding_scheme = encoding_scheme

        if hasattr(config, "n_embd"):
            self.hidden_size = config.n_embd
        elif hasattr(config, "hidden_size"):
            self.hidden_size = config.hidden_size
        else:
            raise ValueError("n_embd or hidden_size must be provided")

        # For adjacency_row, we have 2 task tokens (src and dst), so we need to concatenate them
        # For other schemes, we have 1 task token
        if encoding_scheme == "adjacency_row":
            self.lp_head = nn.Linear(2 * self.hidden_size, 1, bias=True)
        else:
            self.lp_head = nn.Linear(self.hidden_size, 1, bias=True)

        # Initialize the new head
        self.lp_head.weight.data.normal_(mean=0.0, std=0.02)
        if self.lp_head.bias is not None:
            self.lp_head.bias.data.zero_()

    @staticmethod
    def build_dense_adjacency_matrix(
        input_embeds: torch.FloatTensor,
        num_nodes: torch.LongTensor,
        max_num_nodes: int,
    ) -> torch.FloatTensor:
        """
        Build a dense adjacency matrix from input embeddings.

        This method extracts adjacency information from input embeddings and constructs
        a normalized dense adjacency matrix for message passing operations.

        Args:
            input_embeds: Input embeddings of shape (batch_size, max_seq_len, input_length)
            num_nodes: A vector of shape (batch_size,) indicating the number of nodes in the
                subgraph for each sample in the batch
            max_num_nodes: Maximum number of nodes in the subgraph, which is bounded by the sampling config

        Returns:
            dense_adj: Normalized dense adjacency matrix of shape (batch_size, max_seq_len, max_seq_len)
        """
        # input_embeds shape: (batch_size, max_seq_len, input_length)
        max_seq_len = input_embeds.shape[1]
        max_adj_size = torch.max(num_nodes)
        max_task_size = max_seq_len - max_adj_size
        dense_adj = (
            input_embeds[:, :, max_num_nodes : max_num_nodes + max_adj_size]
            + input_embeds[:, :, :max_adj_size]
        )  # (batch_size, max_seq_len, max_adj_size)
        dense_adj = torch.nn.functional.pad(
            dense_adj, (0, max_task_size, 0, 0), value=0
        )  # (batch_size, max_seq_len, max_seq_len)
        # normalize by degree
        degree = dense_adj.sum(dim=2, keepdim=True)
        dense_adj = dense_adj / (
            degree + 1e-8
        )  # Add small epsilon to prevent division by zero
        return dense_adj

    @abstractmethod
    def _call_transformer(
        self,
        input_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        """
        Call the transformer backbone. Must be implemented by subclasses.

        Args:
            input_embeds: Input embeddings
            attention_mask: Attention mask
            **kwargs: Additional arguments passed to transformer

        Returns:
            Transformer outputs (format varies by model)
        """
        pass

    def forward(
        self,
        input_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        feature_embeds: Optional[torch.FloatTensor] = None,
        num_nodes: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        """
        Forward pass for link prediction.

        This method implements the common logic for:
        1. Getting hidden states from the transformer
        2. Extracting representations at [A] token positions
        3. Making predictions using the lp_head
        4. Computing loss if labels are provided

        Args:
            input_embeds: Input embeddings
            attention_mask: Attention mask
            labels: Binary labels (0 for negative, 1 for positive links)
            feature_embeds: Optional feature embeddings (batch_size, seq_len, feature_dim)
            **kwargs: Additional arguments passed to transformer

        Returns:
            CausalLMOutputWithCrossAttentions with:
            - logits: Binary classification logits at [A] token positions
            - loss: Binary cross-entropy loss if labels provided
        """
        # Extract task mask: find tokens where task feature (at task_idx) is 1
        # input_embeds shape: (batch_size, seq_len, input_length)
        # task_idx is the last dimension (task feature)
        task_mask = input_embeds[:, :, self.task_idx] == 1.0  # (batch_size, seq_len)

        # Apply attention mask to only consider valid (non-padded) positions
        if attention_mask is not None:
            task_mask = task_mask & (attention_mask == 1)

        hidden_states = self._call_transformer(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            feature_embeds=feature_embeds,
            num_nodes=num_nodes,
            **kwargs,
        )

        # Extract hidden states at task token positions using task_mask (vectorized)
        if task_mask.any():
            batch_size, seq_len, hidden_size = hidden_states.shape

            # Find positions of task tokens for each batch
            # task_mask: (batch_size, seq_len)
            task_positions = torch.nonzero(
                task_mask, as_tuple=False
            )  # (num_task_tokens, 2) [batch_idx, seq_idx]

            if self.encoding_scheme == "adjacency_row":
                # For adjacency_row, we expect 2 task tokens per batch (src and dst)
                # Group by batch_idx and extract both tokens
                num_task_tokens_per_batch = 2
                assert len(task_positions) == batch_size * num_task_tokens_per_batch, (
                    f"Expected {batch_size * num_task_tokens_per_batch} task tokens, "
                    f"got {len(task_positions)}"
                )

                # Extract task token hidden states
                task_hidden_states_all = hidden_states[
                    task_positions[:, 0], task_positions[:, 1]
                ]  # (batch_size * 2, hidden_size)

                # Reshape to (batch_size, 2, hidden_size) and concatenate
                task_hidden_states = task_hidden_states_all.view(
                    batch_size, num_task_tokens_per_batch, hidden_size
                )  # (batch_size, 2, hidden_size)
                task_hidden_states = task_hidden_states.view(
                    batch_size, num_task_tokens_per_batch * hidden_size
                )  # (batch_size, 2 * hidden_size)
            else:
                # For other schemes, we expect 1 task token per batch
                assert all(
                    task_positions[:, 0]
                    == torch.arange(batch_size, device=task_positions.device)
                ), "Expected exactly one task token per batch for non-adjacency_row schemes"

                # Use advanced indexing to extract task token hidden states
                task_hidden_states = hidden_states[
                    task_positions[:, 0], task_positions[:, 1]
                ]  # (batch_size, hidden_size)

            # Compute logits using the lp_head
            task_logits = self.lp_head(task_hidden_states).squeeze(-1)  # (batch_size,)
        else:
            raise ValueError("No task tokens found in the input")

        loss = None
        if labels is not None:
            target = labels.view_as(task_logits).float()
            if self.is_binary:
                # Standard link prediction with binary labels {0, 1}
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(task_logits, target)
            else:
                # Regression loss for heuristic experiments (MSE on raw scores)
                loss_fct = nn.MSELoss()
                loss = loss_fct(task_logits, target)

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=task_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    @classmethod
    def from_default_configs_and_yaml(
        cls,
        tokenizer: PreTrainedTokenizer,
        model_config: dict,
        use_features: bool = False,
        feature_dim: Optional[int] = None,
        feature_fusion: Optional[str] = None,
        encoding_scheme: str = "full",  # "full", "adjacency_row", "edge_list"
        is_binary: bool = True,
    ):
        """
        Create LinkPredictor model with configuration from HuggingFace defaults and YAML overrides.

        This method should be overridden by subclasses to specify the default model_name.

        Args:
            tokenizer: Tokenizer for accessing [A] token ID
            model_config: Model configuration dict from YAML (e.g., configs.model)
            use_features: Whether to use feature embeddings (default: False)
            feature_dim: Dimension of feature embeddings.
                         - For full/edge_list: 2 * node_feature_dim (src+dst concatenated)
                         - For adjacency_row: node_feature_dim (each token has its own features)
                         Required if use_features=True.
            feature_fusion: Fusion strategy for features - 'early' or 'late' (default: None).
                           Required if use_features=True.
            encoding_scheme: Encoding scheme for the input embeddings
                - full: [node_embeddings, edge_embeddings, link_embedding] (default)
                    - dim: 4 * max_num_nodes + 3 (is_node, is_edge, is_task)
                - adjacency_row: [node_embeddings, link_embeddings] (Yehudai et al.)
                    - dim: 2 * max_num_nodes + 2 (is_node, is_task)
                    - link prediction uses 2 separate tokens: src token and dst token, each with is_node=0 and is_task=1
                - edge_list: [node_embeddings, edge_list_embeddings, link_embedding] (Sanford et al.)
                    - dim: 2 * max_num_nodes + 3 (is_node, is_edge, is_task)

        Returns:
            LinkPredictor instance
        """
        model_name = model_config.get("name", None)
        assert model_name is not None, "Model name must be provided"

        # 1. Load default config from HuggingFace
        config = AutoConfig.from_pretrained(model_name)

        # 2. Populate config with tokenizer-specific values
        config.vocab_size = tokenizer.vocab_size
        if hasattr(config, "bos_token_id"):
            config.bos_token_id = tokenizer.bos_token_id
        if hasattr(config, "eos_token_id"):
            config.eos_token_id = tokenizer.eos_token_id
        if hasattr(config, "pad_token_id"):
            config.pad_token_id = tokenizer.pad_token_id

        # Disable relative attention
        if hasattr(config, "relative_attention"):
            config.relative_attention = False

        # 3. Override model-specific values from YAML config
        # This is generic and works for most transformers
        # Only override if not from pretrained
        if not model_config.get("from_pretrained", False):
            # List of config attributes to potentially override
            config_attributes = [
                "n_layer",
                "n_head",
                "n_embd",
                "n_inner",
                "n_positions",
                "num_hidden_layers",
                "num_attention_heads",
                "hidden_size",
                "intermediate_size",
                "max_position_embeddings",
            ]

            for attr in config_attributes:
                if (
                    model_config.get(attr, None)
                    is not None  # check if the attribute exists in the default pretrained config
                    and hasattr(
                        config, attr
                    )  # check if the attribute exists in yaml config
                    and getattr(config, attr, None)
                    is not None  # check if the attribute has a value in yaml config
                ):
                    setattr(
                        config, attr, model_config.get(attr)
                    )  # set the attribute in the yaml config to the default pretrained config

        # 4. Create model with updated config
        model = cls(
            config,
            tokenizer=tokenizer,
            use_features=use_features,
            feature_dim=feature_dim,
            feature_fusion=feature_fusion,
            encoding_scheme=encoding_scheme,
            is_binary=is_binary,
        )
        if get_rank() == 0:
            print("Transformer Configs: ")
            rich_print(config)
        if model_config.get("from_pretrained", False):
            print(f"Loading pretrained model {model_name}")
            # Load the base pretrained model using AutoModel
            # This automatically selects the appropriate model class based on model_name
            pretrained_model = AutoModel.from_pretrained(model_name)

            # Transfer weights from pretrained model to our LinkPredictor instance
            # model.load_state_dict() will automatically handle compatible weight transfer
            print(model.load_state_dict(pretrained_model.state_dict(), strict=False))

        return model

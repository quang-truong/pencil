import torch
import torch.nn as nn
from transformers import BertConfig, PreTrainedTokenizer
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask_for_sdpa,
)
from typing import Optional
from .lp_model import LinkPredictor


class BERTLP(LinkPredictor, BertEncoder):
    """
    BERT model modified for Link Prediction tasks.

    This model replaces the standard language modeling head with a binary
    classification head that outputs a single logit for link prediction.
    The model uses the hidden states at the [A] token position to make predictions.
    """

    def __init__(
        self,
        config: BertConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        use_features: bool = False,
        feature_dim: Optional[int] = None,
        feature_fusion: Optional[str] = None,
        encoding_scheme: str = "full",
        *,
        is_binary: bool = True,
    ):
        # Initialize both parent classes
        config._attn_implementation = (
            "sdpa"  # override the default attention implementation
        )
        BertEncoder.__init__(self, config)
        LinkPredictor.__init__(
            self,
            config,
            tokenizer,
            encoding_scheme,
            is_binary=is_binary,
        )

        self.input_proj = nn.Parameter(torch.empty(self.input_length, self.hidden_size))
        nn.init.orthogonal_(self.input_proj)
        self.input_proj.requires_grad = False

        self.mp_proj = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size), torch.nn.GELU()
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        for i, mp_proj in enumerate(self.mp_proj):
            mp_proj[0].weight.data.normal_(mean=0.0, std=0.02)
            if mp_proj[0].bias is not None:
                mp_proj[0].bias.data.zero_()

        # Feature projection
        if use_features:
            if feature_dim is None:
                raise ValueError("feature_dim must be provided when use_features=True")
            assert feature_fusion in [
                "early",
                "late",
            ], "feature_fusion must be either 'early' or 'late' when use_features=True"

            self.feature_fusion = feature_fusion
            self.feature_proj = torch.nn.Sequential(
                nn.Linear(feature_dim, self.hidden_size, bias=True),
                torch.nn.GELU(),
            )
            # Initialize projection weights
            self.feature_proj[0].weight.data.normal_(mean=0.0, std=0.02)
            if self.feature_proj[0].bias is not None:
                self.feature_proj[0].bias.data.zero_()
        else:
            self.feature_fusion = None
            self.feature_proj = None

    def _call_transformer(
        self,
        input_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        feature_embeds: Optional[torch.FloatTensor] = None,
        num_nodes: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Call the BERT transformer backbone.

        Args:
            input_embeds: Input embeddings
            attention_mask: Attention mask
            **kwargs: Additional arguments

        Returns:
            hidden_states: Hidden states
                (batch_size, seq_len, hidden_size)
        """
        # Convert attention mask to the format expected by BERT
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Use SDPA-compatible mask preparation
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, input_embeds.dtype, tgt_len=input_embeds.shape[1]
                )
            else:
                raise ValueError(
                    f"Only 2D attention masks are supported, got {attention_mask.dim()}D mask"
                )
        else:
            extended_attention_mask = None

        # Build dense adjacency matrix using the shared static method
        dense_adj = BERTLP.build_dense_adjacency_matrix(
            input_embeds, num_nodes, self.max_num_nodes
        )

        # input_embeds projection using orthonormal matrix
        # input_embeds shape: (batch_size, seq_len, input_length)
        # hidden_states shape: (batch_size, seq_len, hidden_size)
        hidden_states = torch.matmul(input_embeds, self.input_proj)

        # Early fusion: project and add features to input embeddings before transformer
        if feature_embeds is not None and self.feature_fusion == "early":
            encoded_features = self.feature_proj(feature_embeds)
            hidden_states = hidden_states + encoded_features

        for i, encoder_layer in enumerate(self.layer):
            # Process through encoder layer
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=extended_attention_mask,
            )
            hidden_states = layer_outputs[0]
            mp_hidden_states = torch.bmm(dense_adj, hidden_states)
            hidden_states = hidden_states + self.mp_proj[i](mp_hidden_states)

        # Late fusion: project and add features to hidden states after transformer
        if feature_embeds is not None and self.feature_fusion == "late":
            encoded_features = self.feature_proj(feature_embeds)
            hidden_states = hidden_states + encoded_features

        return hidden_states

    def get_ignored_modules(self):
        return set()

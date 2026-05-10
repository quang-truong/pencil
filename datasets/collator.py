# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase


@dataclass
class Collator:

    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100
    id_pad_token_id: Optional[int] = -999
    position_pad_value: Optional[int] = 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Pad the batch like this to maximize the reuse of kv cache.
        E.g.,

        xxxxxxxxxx<latent><latent>xxxxx--
        -----xxxxx<latent>xxxxxxxx-------
        ---xxxxxxx<latent><latent>xxxxxxx


        ("x" is word token, "-" is pad token)

        Args:
            features: List of dictionaries, each element is corresponding to a sample in the batch.
                - input_ids: List of token ids
                - attention_mask: List of attention mask
                - labels: List of labels
            device: The device to place tensors on

        Returns:
            batch: Dictionary containing the padded features
                - input_ids: Tensor of shape (batch_size, max_length)
                - attention_mask: Tensor of shape (batch_size, max_length)
                - labels: Tensor of shape (batch_size, max_length) if labels are provided

        Examples:
        >> features = [
            {
            "input_ids": [1, 2, 3, 4],
            "attention_mask": [1, 1, 1, 1],
            "labels": [1, 2, 3, 4]
            },
            {
                "input_ids": [5, 6, 7],
                "attention_mask": [1, 1, 1],
                "labels": [5, 6, 7]
            }
        ]
        >> collator(features)
        {
            'input_ids': tensor([[1, 2, 3, 4],
                                 [5, 6, 7, 38]]),
            'attention_mask': tensor([[1, 1, 1, 1],
                                       [1, 1, 1, 0]]),
            'labels': tensor([[1, 2, 3, 4],
                              [5, 6, 7, -100]])
        }
        >> features = [
            {
                "input_ids": [1, 2, 33, 3],
                "attention_mask": [1, 1, 1, 1],
                "labels": [1, 2, 33, 3],
            },
            {
                "input_ids": [4, 33, 5],
                "attention_mask": [1, 1, 1],
                "labels": [4, 33, 5],
            }
        ]
        >> collator(features)
        {
            'input_ids': tensor([[1, 2, 33, 3],
                                 [38, 4, 33, 5]]),
            'attention_mask': tensor([[1, 1, 1, 1],
                                       [0, 1, 1, 1]]),
            'labels': tensor([[1, 2, 33, 3],
                              [-100, 4, 33, 5]]),
        }
        """
        if not features:
            raise ValueError("Empty features list provided")

        assert (
            self.tokenizer.padding_side == "right"
        ), "This collator requires right-side padding."

        # Handle embedding-based inputs (no latent token support)
        if "input_embeds" in features[0] and features[0]["input_embeds"] is not None:
            # Pad input_embeds and task_embeds separately
            input_embeds = pad_sequence(
                [
                    torch.as_tensor(f["input_embeds"], dtype=torch.float)
                    for f in features
                ],
                batch_first=True,
                padding_value=0.0,
            )  # (B, S_input, D)

            task_embeds = pad_sequence(
                [
                    torch.as_tensor(f["task_embeds"], dtype=torch.float)
                    for f in features
                ],
                batch_first=True,
                padding_value=0.0,
            )  # (B, S_task, D)

            # Concatenate padded sequences along sequence dimension
            # This ensures task tokens are always in the same columns after padding
            input_embeds = torch.cat(
                [input_embeds, task_embeds], dim=1
            )  # (B, S_input + S_task, D)

            # Pad attention masks separately
            if (
                "input_attention_mask" in features[0]
                and features[0].get("input_attention_mask") is not None
                and "task_attention_mask" in features[0]
                and features[0].get("task_attention_mask") is not None
            ):
                input_attention_mask = pad_sequence(
                    [
                        torch.tensor(f["input_attention_mask"], dtype=torch.long)
                        for f in features
                    ],
                    batch_first=True,
                    padding_value=0,
                )  # (B, S_input)

                task_attention_mask = pad_sequence(
                    [
                        torch.tensor(f["task_attention_mask"], dtype=torch.long)
                        for f in features
                    ],
                    batch_first=True,
                    padding_value=0,
                )  # (B, S_task)

                # Concatenate attention masks along sequence dimension
                attention_mask = torch.cat(
                    [input_attention_mask, task_attention_mask], dim=1
                )  # (B, S_input + S_task)
            else:
                raise ValueError(
                    "Both input_attention_mask and task_attention_mask must be provided"
                )

            # Create batch dictionary
            batch = {
                "input_embeds": input_embeds,
                "attention_mask": attention_mask,
            }

            # Labels
            if "labels" in features[0] and features[0].get("labels") is not None:
                batch["labels"] = pad_sequence(
                    [torch.tensor(f["labels"], dtype=torch.long) for f in features],
                    batch_first=True,
                    padding_value=self.label_pad_token_id,
                )

            # Handle feature embeddings separately for input and task
            if (
                "input_feature_embeds" in features[0]
                and features[0].get("input_feature_embeds") is not None
                and "task_feature_embeds" in features[0]
                and features[0].get("task_feature_embeds") is not None
            ):
                input_feature_embeds = pad_sequence(
                    [
                        torch.tensor(f["input_feature_embeds"], dtype=torch.float)
                        for f in features
                    ],
                    batch_first=True,
                    padding_value=0.0,
                )  # (B, S_input, feature_dim)

                task_feature_embeds = pad_sequence(
                    [
                        torch.tensor(f["task_feature_embeds"], dtype=torch.float)
                        for f in features
                    ],
                    batch_first=True,
                    padding_value=0.0,
                )  # (B, S_task, feature_dim)

                # Concatenate feature embeddings along sequence dimension
                batch["feature_embeds"] = torch.cat(
                    [input_feature_embeds, task_feature_embeds], dim=1
                )  # (B, S_input + S_task, feature_dim)

            # 3. Handle dense_adj,edge_index, shuffled_edge_index, root_nodes, and connected_nodes
            # These are optional testing/debugging fields - pad and stack them if present
            if "dense_adj" in features[0] and features[0].get("dense_adj") is not None:
                dense_adj = torch.stack([f["dense_adj"] for f in features])
                batch["dense_adj"] = dense_adj
            if (
                "edge_index" in features[0]
                and features[0].get("edge_index") is not None
            ):
                # edge_index has shape (2, num_edges) - pad along dimension 1
                edge_indices = [f["edge_index"] for f in features]
                max_num_edges = max(ei.size(1) for ei in edge_indices)
                padded_edge_indices = []
                for ei in edge_indices:
                    if ei.size(1) < max_num_edges:
                        # Pad with zeros along dimension 1
                        padding = torch.zeros(
                            (2, max_num_edges - ei.size(1)), dtype=ei.dtype
                        )
                        padded = torch.cat([ei, padding], dim=1)
                    else:
                        padded = ei
                    padded_edge_indices.append(padded)
                batch["edge_index"] = torch.stack(padded_edge_indices)

            if (
                "shuffled_edge_index" in features[0]
                and features[0].get("shuffled_edge_index") is not None
            ):
                # Handle None values (for adjacency_row encoding)
                shuffled_edge_indices = []
                for f in features:
                    if f["shuffled_edge_index"] is not None:
                        shuffled_edge_indices.append(f["shuffled_edge_index"])
                    else:
                        # Create empty tensor with same shape as others
                        shuffled_edge_indices.append(
                            torch.empty((2, 0), dtype=torch.long)
                        )
                # Pad to max length
                max_num_shuffled_edges = max(
                    sei.size(1) for sei in shuffled_edge_indices
                )
                padded_shuffled_edge_indices = []
                for sei in shuffled_edge_indices:
                    if sei.size(1) < max_num_shuffled_edges:
                        padding = torch.zeros(
                            (2, max_num_shuffled_edges - sei.size(1)), dtype=sei.dtype
                        )
                        padded = torch.cat([sei, padding], dim=1)
                    else:
                        padded = sei
                    padded_shuffled_edge_indices.append(padded)
                batch["shuffled_edge_index"] = torch.stack(padded_shuffled_edge_indices)

            if (
                "root_nodes" in features[0]
                and features[0].get("root_nodes") is not None
            ):
                # root_nodes is a tuple, convert to tensor and stack
                root_nodes_list = []
                for f in features:
                    src, dst = f["root_nodes"]
                    root_nodes_list.append(torch.tensor([src, dst], dtype=torch.long))
                batch["root_nodes"] = torch.stack(root_nodes_list)

            if (
                "connected_nodes" in features[0]
                and features[0].get("connected_nodes") is not None
            ):
                # connected_nodes is a list, convert each to tensor and stack
                connected_nodes_list = []
                max_connected_nodes = max(len(f["connected_nodes"]) for f in features)
                for f in features:
                    connected_nodes = f["connected_nodes"]
                    # Pad to max length with -1 (invalid index)
                    padded = connected_nodes + [-1] * (
                        max_connected_nodes - len(connected_nodes)
                    )
                    connected_nodes_list.append(torch.tensor(padded, dtype=torch.long))
                batch["connected_nodes"] = torch.stack(connected_nodes_list)

            # 4. Handle any other fields automatically
            processed_keys = set(batch.keys()) | {
                "input_embeds",
                "task_embeds",
                "input_attention_mask",
                "task_attention_mask",
                "input_feature_embeds",
                "task_feature_embeds",
            }
            other_keys = [k for k in features[0].keys() if k not in processed_keys]
            for key in other_keys:
                values = [feature[key] for feature in features]
                try:
                    batch[key] = torch.tensor(values)
                except Exception as e:
                    print(
                        f"[WARNING] Could not collate field '{key}' into a tensor. Error: {e}"
                    )

            return batch
        else:
            raise ValueError("This collator requires input_embeds to be provided.")

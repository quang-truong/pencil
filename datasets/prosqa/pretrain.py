# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import random

import torch
import torch.distributed as dist

# TODO: the current pre-train is based on the number of samples in the downstream task.
# Need to make the pre-train stage independent of the downstream task.


def expand_data(
    data,
    k,
    max_steps,
    max_num_nodes,
    is_eval=False,
    return_idx_map=False,
    use_latent_labels=False,
):
    """
    Expand a single data instance into training examples with different reasoning depths.

    Args:
        data: Dictionary containing graph data with edges, targets, neighbors, etc.
        k: Current stage for curriculum learning (1 <= k <= max_steps+1)
        max_steps: Maximum number of reasoning steps to reach the target
        max_num_nodes: Maximum number of nodes in the graph
        is_eval: Evaluation mode
        return_idx_map: Whether to return the index map

    Returns:
        tuple: (question, continuation) where question is the input and continuation is the target.
            question format: <eos> edge1|edge2|... [R] root <|latent|> * (k-1) continuation
                where:
                - edge1, edge2, ... are the edges of the graph
                - root is the root of the graph
                - <|latent|> is the latent token
                - <eos> is the end of sequence token
                - [R] is the root token
            continuation format:
                - neighbor_k[k] if k <= max_steps
                - neighbor_k[k-1] if k == max_steps + 1 with probability 0.5, else sample from neighbor_k
    """

    # Ensure k doesn't exceed the maximum allowed steps + 1
    assert 1 <= k <= max_steps + 1
    # k = 1, 2, 3, 4, 5 (different reasoning depths)

    if is_eval:
        idx_map = [i for i in range(max_num_nodes)]
    else:
        r = random.randint(0, max_num_nodes - 1)
        idx_map = [(i + r) % max_num_nodes for i in range(max_num_nodes)]

    def get_prefix(data):
        """
        Generate the question prefix containing graph edges and target information.
        Format:
            <eos> edge1|edge2|... [R] root
                where:
                - edge1, edge2, ... are the edges of the graph
                - root is the root of the graph
                - <eos> is the end of sequence token
                - [R] is the root token

        Args:
            data: Graph data dictionary

        Returns:
            str: Formatted question string with edges, targets, and root information
        """

        # Randomize edge order to increase data diversity
        random.shuffle(data["edges"])

        # Construct question with format: <eos> edge1|edge2|... [Q] target1 target2 [R] root
        question = (
            "<eos> "
            + "|".join(
                [f" {idx_map[e[0]]} {idx_map[e[1]]} " for e in data["edges"]]
            ).strip()
            + " |"
        )

        # Add root information to complete the question
        question += " [R] " + str(idx_map[data["root"]])

        return question

    # If k >= max_steps + 1, choose k = max_steps with prob 0.5, else k sampled from [1, max_steps - 1]
    if k >= max_steps + 1:
        if random.random() < 0.5:
            k = random.randint(1, max_steps - 1)
        else:
            k = max_steps

    # Append latent tokens to the question prefix
    # If the current stage is k, we append k-1 latent tokens
    # because we want to predict the k-th neighbor using (k-1)-th latent token
    question = get_prefix(data)
    labels = []
    for i in range(1, k):
        n = random.choice(data["neighbor_k"][str(i)])
        labels.append(str(idx_map[n]))
        question += " <|latent|>"
    # append the k-th neighbor if k <= max_steps
    # if k == max_steps + 1, we do nothing because the labels are already sufficiently appended
    if k <= max_steps:
        n = random.choice(data["neighbor_k"][str(k)])
        labels.append(str(idx_map[n]))

    # continuation is always the last latent token
    continuation = labels[-1]

    return_data = (question, continuation)
    if use_latent_labels:
        return_data = (*return_data, labels)
    if return_idx_map:
        return_data = (*return_data, idx_map)
    return return_data


# def get_multi_labels(data, k, vocab_size):


def get_pretrain_graph_latent_cot_dataset(
    dataset_path,
    scheduled_stage,
    configs,
    tokenizer,
    use_latent_labels=False,
    is_eval=False,
):
    """
    Creates a dataset for training graph reasoning with latent tokens and chain of thought.
    Each sample will contain the graph edges, question, and the full reasoning path to the answer.

    Args:
        dataset_path: Path to the dataset
        scheduled_stage: The stage in curriculum learning [0, ..., epoch // configs.epochs_per_stage]
        configs: Configuration dictionary
        tokenizer: Tokenizer

    Returns:
        dataset: List of processed samples, each sample is a dictionary with the following keys:
            - input_ids: List of token ids
            - labels: List of labels
            - attention_mask: List of attention mask
            - position_ids: List of position ids
    """
    # base_dataset is a list of samples, each sample is a dictionary with the following keys:
    # - steps: List of steps
    # - edges: List of edges
    # - target: Target value
    # - neg_target: Negative target value
    # - root: Root value
    # - idx_to_symbol: List of indices to symbols
    # - neighbor_k: Dictionary of neighbors for each step
    base_dataset = json.load(open(dataset_path))

    if configs.debug:
        base_dataset = base_dataset[:10000]

    def process_dataset(sample):
        """
        Process a single sample from the base dataset.
        Returns:
            [(question_tokenized, continuation_tokenized)]
        """

        if (
            random.random() < configs.uniform_prob
        ):  # with some prob, randomly sample stage
            scheduled_stage_to_train = random.randint(
                0, min(scheduled_stage, len(sample["steps"]))
            )
        else:
            scheduled_stage_to_train = min(scheduled_stage, len(sample["steps"]))
            # 0, 1, 2, 3, 4

        # this range is [0, ..., len(sample["steps"])]
        # including both ends
        # expanded_data is a tuple of (question, continuation) for each sample
        expanded_data = expand_data(
            sample,
            scheduled_stage_to_train + 1,
            len(sample["steps"]),
            max_num_nodes=tokenizer.max_num_nodes(),
            use_latent_labels=use_latent_labels,
            is_eval=is_eval,
        )

        # Process each question-continuation pair
        processed_samples = []
        if use_latent_labels:
            question, continuation, latent_labels = expanded_data
        else:
            question, continuation = expanded_data

        question_tokenized = tokenizer.encode(question, add_special_tokens=False)
        continuation_tokenized = tokenizer.encode(
            continuation, add_special_tokens=False
        )

        # Create labels for question_tokenized + continuation_tokenized: -100 for all positions initially
        labels = [-100] * (len(question_tokenized) + len(continuation_tokenized))
        if use_latent_labels:
            # Find latent token positions in question_tokenized and set them with latent_labels
            root_token_id = tokenizer.convert_tokens_to_ids("[R]")

            # [R] root_node <|latent|> * (k-1), so the first latent token position is 2 after [R]
            # Find the first latent token position using built-in index method
            first_latent_pos = question_tokenized.index(root_token_id) + 2

            # If latent tokens exist, set them with latent_labels
            # Latent tokens are consecutive, so we can calculate all positions
            labels[first_latent_pos : first_latent_pos + len(latent_labels)] = (
                tokenizer.encode(latent_labels, add_special_tokens=False)
            )

            assert (
                continuation_tokenized[0]
                == labels[first_latent_pos + len(latent_labels) - 1]
            )

        # Replace the last label with the continuation token
        labels[-1] = continuation_tokenized[0]

        tokens = question_tokenized + continuation_tokenized

        processed_sample = {
            "input_ids": tokens,
            "labels": labels,
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }
        processed_samples.append(processed_sample)

        return processed_samples

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            # Process each sample and collect all results
            all_processed_samples = []
            for sample in base_dataset:
                processed_samples = process_dataset(sample)
                all_processed_samples.extend(processed_samples)

            processed_dataset = all_processed_samples

            if not is_eval:
                random.shuffle(processed_dataset)
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        # broadcast the processed dataset to all ranks
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        # single GPU
        # Process each sample and collect all results
        all_processed_samples = []
        for sample in base_dataset:
            processed_samples = process_dataset(sample)
            all_processed_samples.extend(processed_samples)

        processed_dataset = all_processed_samples
        if not is_eval:
            random.shuffle(processed_dataset)
        dataset = processed_dataset
    return dataset

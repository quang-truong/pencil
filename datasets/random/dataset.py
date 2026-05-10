# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import random

import torch
import torch.distributed as dist


def expand_data(
    data,
    target,
    k,
    max_steps,
    max_num_nodes,
    is_eval=False,
    return_idx_map=False,
    task_dependent_latent=False,
    use_latent_labels=False,
):
    """
    Expand a single data instance into training examples with different reasoning depths.

    Args:
        data: Dictionary containing graph data with edges, targets, neighbors, etc.
        target: Target value
        k: Current stage for curriculum learning (1 <= k <= max_steps + 1). k = 1 means no latent token.
        max_steps: Maximum number of reasoning steps to reach the target
        max_num_nodes: Maximum number of nodes in the graph
        is_eval: Evaluation mode, negative target is the first negative target
        return_idx_map: Whether to return the index map
        task_dependent_latent: Whether task-related tokens are before the latent tokens
        use_latent_labels: Whether to return the labels for latent tokens, instead of just using continuation as the only label

    Returns:
        tuple: (question, continuation) where question is the input and continuation is the target.
            question format: <eos> edge1|edge2|... [Q] target neg_target [R] root <|latent|> * (k - 1) + [A]
                where:
                - edge1, edge2, ... are the edges of the graph
                - root is the root of the graph
                - <|latent|> is the latent token
                - <eos> is the end of sequence token
                - [Q] is the question token
                - [R] is the root token
                - [A] is the answer token
            continuation format:
                - target if k == max_steps + 1
                - continuation if k <= max_steps
    """
    assert 1 <= k <= max_steps + 1

    if is_eval:
        idx_map = [i for i in range(max_num_nodes)]
    else:
        r = random.randint(0, max_num_nodes - 1)
        idx_map = [(i + r) % max_num_nodes for i in range(max_num_nodes)]

    # unreacheable node is the negative target
    if not is_eval:
        neg_target = random.choice(data["neg_targets_dict"][str(max_steps)])
    else:
        # in evaluation, we use the first negative target
        neg_target = data["neg_targets_dict"][str(max_steps)][0]

    def get_prefix(data, k, max_steps):
        """
        Generate the question prefix containing graph edges and target information.

        Args:
            data: Graph data dictionary, with the following keys:
                - edges: List of edges
                - root: Root value
                - targets: Target values are list of nodes randomly sampled from the graph.
                - neighbor_k_dict: Dictionary of neighbors for each step of each target node.
                - num_nodes: Number of nodes in the graph
                - num_edges: Number of edges in the graph
                - neg_targets_dict: Dictionary of negative target values for each step,
                    values are nodes in next-hop (k+1) for each key k
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
        if task_dependent_latent:
            # Add question token and target information when task-dependent latent is used.
            question += " [Q] "
            if random.random() < 0.5:
                question += str(idx_map[target]) + " " + str(idx_map[neg_target])
            else:
                question += str(idx_map[neg_target]) + " " + str(idx_map[target])

        # Add root information to complete the question
        question += " [R] " + str(idx_map[data["root"]])

        labels = []
        for i in range(1, k):
            n = random.choice(data["neighbor_k_dict"][str(target)][str(i)])
            labels.append(str(idx_map[n]))
            question += " <|latent|>"
        if k <= max_steps:
            n = random.choice(data["neighbor_k_dict"][str(target)][str(k)])
            labels.append(str(idx_map[n]))

        continuation = labels[-1]
        if not task_dependent_latent:
            # Add question token and target information when task-dependent latent is used.
            question += " [Q] "
            if random.random() < 0.5:
                question += str(idx_map[target]) + " " + str(idx_map[neg_target])
            else:
                question += str(idx_map[neg_target]) + " " + str(idx_map[target])

        question += " [A] " if k == max_steps + 1 else " "
        return_data = (question, continuation)

        if use_latent_labels:
            return_data = (*return_data, labels)

        return return_data

    return_data = get_prefix(data, k, max_steps)
    if return_idx_map:
        return_data = (*return_data, idx_map)
    return return_data


def get_graph_latent_cot_dataset(
    dataset_path,
    scheduled_stage,
    configs,
    tokenizer,
    task_dependent_latent=False,
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
    base_dataset = json.load(open(dataset_path))

    if configs.debug:
        base_dataset = base_dataset[:10000]

    def process_dataset(sample):
        """
        Process a single sample from the base dataset.
        Returns:
            [(question_tokenized, continuation_tokenized)]
        """

        # sample a target from the list of targets
        target = random.choice(sample["targets"])
        # number of hops to the target
        max_steps = len(sample["neighbor_k_dict"][str(target)].keys()) - 1

        if (
            random.random() < configs.uniform_prob
        ):  # with some prob, randomly sample stage
            scheduled_stage_to_train = random.randint(
                0, min(scheduled_stage, max_steps)
            )
        else:
            scheduled_stage_to_train = min(scheduled_stage, max_steps)
            # 0, 1, 2, 3, 4

        # this range is [0, ..., len(sample["steps"])]
        # including both ends
        # expanded_data is a tuple of (question, continuation) for each sample
        expanded_data = expand_data(
            sample,
            target,
            scheduled_stage_to_train + 1,
            max_steps,
            max_num_nodes=tokenizer.max_num_nodes(),
            task_dependent_latent=task_dependent_latent,
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


def get_graph_latent_question_dataset(
    dataset_path,
    scheduled_stage,
    configs,
    tokenizer,
    task_dependent_latent=False,
):
    base_dataset = json.load(open(dataset_path))
    if configs.debug:
        base_dataset = base_dataset[:10000]
    # similar to get_graph_latent_dataset, but we only keep the question
    # without the continuation

    def process_dataset(sample, idx):
        processed_samples = []

        for target in sample["targets"]:
            max_steps = len(sample["neighbor_k_dict"][str(target)].keys()) - 1
            question, continuation = expand_data(
                sample,
                target,
                max_steps + 1,
                max_steps,
                max_num_nodes=tokenizer.max_num_nodes(),
                is_eval=True,
                task_dependent_latent=task_dependent_latent,
                use_latent_labels=False,
            )

            question_tokenized = tokenizer.encode(question, add_special_tokens=False)

            # the last token should be [A]
            assert question_tokenized[-1] == tokenizer.convert_tokens_to_ids("[A]")

            processed_samples.append(
                {
                    "input_ids": question_tokenized,
                    "attention_mask": [1] * len(question_tokenized),
                    "position_ids": list(range(len(question_tokenized))),
                    "answer": target,  # target is integer, not string
                }
            )
        return processed_samples

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            # Process each sample and collect all results
            all_processed_samples = []
            for idx, sample in enumerate(base_dataset):
                processed_samples = process_dataset(sample, idx)
                all_processed_samples.extend(processed_samples)

            processed_dataset = all_processed_samples
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        all_processed_samples = []
        for idx, sample in enumerate(base_dataset):
            processed_samples = process_dataset(sample, idx)
            all_processed_samples.extend(processed_samples)

        processed_dataset = all_processed_samples
        dataset = processed_dataset

    return dataset

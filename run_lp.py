import os
import argparse
import functools
import shutil
import json
import builtins

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import yaml

from transformers.models.bert.modeling_bert import BertLayer

from tqdm import tqdm
import wandb
import rich

from utils import (
    Config,
    set_seed,
    seed_worker,
    compute_params,
    concat_all_gather_1d,
    save_checkpoint,
    load_model_checkpoint,
    load_optimizer_checkpoint,
    suppress_warnings,
    generate_name,
    log_final_table,
    get_feature_dim,
    normalize_score,
    str_to_bool,
)
from utils import rank_zero_print as print
from definitions import ROOT_DIR

from models.transformers.stokenizer import STokenizer
from models.transformers.bert_lp import BERTLP
from models.gnns.mpnn import MPNN
from datasets import Collator, DatasetWrapper, load_dataset
from datasets.gnn_collator import GNNCollator

from evaluator import Evaluator

from datetime import timedelta

# Model registry mapping model names to their classes
MODEL_REGISTRY = {
    "bert-base-uncased": BERTLP,
    "gcn": MPNN,
    "gat": MPNN,
    "sage": MPNN,
}

# FSDP layer registry mapping model names to their transformer layer classes
FSDP_LAYER_REGISTRY = {
    "bert-base-uncased": BertLayer,
}

# Dataset -> metric used for model selection (validation)
SELECTION_METRIC_BY_DATASET = {
    "cora": "mrr",
    "citeseer": "mrr",
    "pubmed": "mrr",
    # Heuristic regression variants (optimize RMSE; lower is better, handled in utils.normalize_score)
    "cn-cora": "rmse",
    "aa-cora": "rmse",
    "ra-cora": "rmse",
    "katz-cora": "rmse",
    "shortest-path-cora": "rmse",
    "pagerank-cora": "rmse",
    "cn-citeseer": "rmse",
    "aa-citeseer": "rmse",
    "ra-citeseer": "rmse",
    "katz-citeseer": "rmse",
    "shortest-path-citeseer": "rmse",
    "pagerank-citeseer": "rmse",
    "ogbl-collab": "hits@50",
    "ogbl-ppa": "hits@100",
    "ogbl-citation2": "mrr",
    "ogbl-ddi": "hits@20",
    "heart-cora": "mrr",
    "heart-citeseer": "mrr",
    "heart-pubmed": "mrr",
    "heart-ogbl-collab": "mrr",
    "heart-ogbl-ppa": "mrr",
    "heart-ogbl-citation2": "mrr",
    "heart-ogbl-ddi": "mrr",
}


def build_loaders(
    epoch: int,
    tokenizer,
    configs,
    collator,
    train_dataset_raw=None,
    valid_dataset_raw=None,
    test_dataset_raw=None,
    use_features=False,
    encoding_scheme="full",
    is_gnn=False,
):
    """
    Build DataLoaders for training, validation, and testing.
    Resets training samples for each epoch to ensure proper data shuffling.

    Args:
        epoch: Current epoch number
        train_dataset_raw: Raw training dataset
        valid_dataset_raw: Raw validation dataset
        test_dataset_raw: Raw test dataset
        tokenizer: Tokenizer for data processing (None for GNN models)
        configs: Configuration object
        collator: Collator for batching
        use_features: Whether to use feature embeddings
        encoding_scheme: Encoding scheme for the input embeddings ('full', 'adjacency_row', or 'edge_list')
        is_gnn: Whether the model is a GNN model (default: False)

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    # For GNN models, use raw dataset directly with GNNCollator
    if is_gnn:
        if train_dataset_raw is not None:
            train_dataset_raw.reset_samples(epoch=epoch, seed=configs.seed)
            train_g = torch.Generator()
            train_g.manual_seed(configs.seed * 100 + epoch)
            train_loader = torch.utils.data.DataLoader(
                train_dataset_raw,
                batch_size=configs.batch_size_training,
                num_workers=configs.num_workers,
                pin_memory=True,
                shuffle=False,  # Shuffling handled by DistributedSampler
                sampler=DistributedSampler(train_dataset_raw, shuffle=True),
                collate_fn=collator,
                generator=train_g,
                worker_init_fn=lambda worker_id: seed_worker(
                    worker_id, configs.seed * 100 + epoch
                ),
            )
            train_loader.sampler.set_epoch(epoch)
        else:
            train_loader = None

        g = torch.Generator()
        g.manual_seed(configs.seed)

        if valid_dataset_raw is not None:
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset_raw,
                batch_size=configs.batch_size_training,
                num_workers=configs.num_workers,
                pin_memory=True,
                shuffle=False,
                sampler=DistributedSampler(valid_dataset_raw, shuffle=False),
                collate_fn=collator,
                generator=g,
                worker_init_fn=lambda worker_id: seed_worker(worker_id, configs.seed),
            )
        else:
            valid_loader = None

        if test_dataset_raw is not None:
            test_loader = torch.utils.data.DataLoader(
                test_dataset_raw,
                batch_size=configs.batch_size_training,
                num_workers=configs.num_workers,
                pin_memory=True,
                shuffle=False,
                sampler=DistributedSampler(test_dataset_raw, shuffle=False),
                collate_fn=collator,
                generator=g,
                worker_init_fn=lambda worker_id: seed_worker(worker_id, configs.seed),
            )
        else:
            test_loader = None

        return train_loader, valid_loader, test_loader

    # Create dataset wrappers that format graph data for transformer input
    if train_dataset_raw is not None:
        train_dataset_raw.reset_samples(epoch=epoch, seed=configs.seed)
        train_g = torch.Generator()
        train_g.manual_seed(configs.seed * 100 + epoch)
        train_ds = DatasetWrapper(
            train_dataset_raw,
            tokenizer,
            max_sequence_length=configs.model.get("max_position_embeddings"),
            node_remapping=configs.node_remap,
            is_eval=False,
            use_features=use_features,
            encoding_scheme=encoding_scheme,
        )
        # Create training DataLoader with shuffling enabled
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=configs.batch_size_training,
            num_workers=configs.num_workers,
            pin_memory=True,
            shuffle=False,  # Shuffling handled by DistributedSampler
            sampler=DistributedSampler(train_ds, shuffle=True),
            collate_fn=collator,
            generator=train_g,
            worker_init_fn=lambda worker_id: seed_worker(
                worker_id, configs.seed * 100 + epoch
            ),
        )
        # necessary according to the docs
        train_loader.sampler.set_epoch(epoch)
    else:
        train_loader = None

    g = torch.Generator()
    g.manual_seed(configs.seed)

    if valid_dataset_raw is not None:
        valid_ds = DatasetWrapper(
            valid_dataset_raw,
            tokenizer,
            max_sequence_length=configs.model.get("max_position_embeddings"),
            node_remapping=False,  # no node remapping for validation
            is_eval=True,
            use_features=use_features,
            encoding_scheme=encoding_scheme,
        )
        # Create validation DataLoader without shuffling
        valid_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=configs.batch_size_training,
            num_workers=configs.num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(valid_ds, shuffle=False),
            collate_fn=collator,
            generator=g,
            worker_init_fn=lambda worker_id: seed_worker(worker_id, configs.seed),
        )
    else:
        valid_loader = None

    if test_dataset_raw is not None:
        test_ds = DatasetWrapper(
            test_dataset_raw,
            tokenizer,
            max_sequence_length=configs.model.get("max_position_embeddings"),
            node_remapping=False,  # no node remapping for test
            is_eval=True,
            use_features=use_features,
            encoding_scheme=encoding_scheme,
        )
        # Create test DataLoader without shuffling
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=configs.batch_size_training,
            num_workers=configs.num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(test_ds, shuffle=False),
            collate_fn=collator,
            generator=g,
            worker_init_fn=lambda worker_id: seed_worker(worker_id, configs.seed),
        )
    else:
        test_loader = None

    return train_loader, valid_loader, test_loader


def get_model(
    configs,
    tokenizer,
    rank,
    world_size,
    local_rank,
    wandb_run,
    save_dir,
    use_features=False,
    feature_dim=None,
    encoding_scheme="full",
    is_gnn=False,
    use_bf16=False,
):
    """
    Create and configure the model with appropriate parallelization strategy.

    Args:
        configs: Configuration object containing model settings
        tokenizer: Tokenizer for the model (just to extract max_num_nodes)
        rank: Current process rank
        world_size: Total number of processes
        local_rank: Local rank for device placement
        wandb_run: Wandb run object for logging
        save_dir: Directory to save model checkpoints
        use_features: Whether to use feature embeddings (default: False)
        feature_dim: Dimension of feature embeddings. For transformers: 2 * node_feature_dim. For GNNs: node_feature_dim.
        encoding_scheme: Encoding scheme for the input embeddings ('full', 'adjacency_row', or 'edge_list')
        is_gnn: Whether the model is a GNN model (default: False)
        use_bf16: Whether to use bfloat16 mixed precision for FSDP (default: False)

    Returns:
        parallel_model: The parallelized model (DDP or FSDP wrapped)
    """
    # eval mode: load checkpoint
    strict_mode = True
    if configs.only_eval:
        if configs.load_model_path != "None":
            strict_mode = True  # strict because we are loading a checkpoint for eval
        else:
            # automatically fetch the best model checkpoint from the train directory for eval mode
            # if load_model_path isn't set
            assert configs.name.startswith("eval-"), "Not in eval mode"
            train_folder_name = configs.name[5:]  # strip "eval-"
            train_dir = os.path.join(configs.save_path, train_folder_name)
            assert os.path.exists(train_dir), "Train directory not found"
            train_ckpts = os.listdir(train_dir)
            assert (
                "best_model_checkpoint.pth" in train_ckpts
            ), "Best model checkpoint not found"
            load_dir = os.path.join(train_dir, "best_model_checkpoint.pth")
            configs.load_model_path = load_dir
            strict_mode = True  # strict because we are resuming from a previous run
            print(f"Resume from the best model checkpoint at {load_dir}")
    # training mode
    else:
        # check if the job is preempted and resumed.
        cur_ckpts = os.listdir(save_dir)
        if len(cur_ckpts) > 1:  # yaml file + checkpoint files
            if configs.resume == 0:
                # if there are previous checkpoints, and only_eval is False
                # and resume is 0
                # it means the previous run was preempted and the program is restarted.
                # need to find the latest checkpoint and resume from that.

                print(
                    f"[WARNING] Found previous run and gonna resume from that. The inputted `resume` argument is ignored!"
                )

                numbered_checkpoints = [
                    f for f in cur_ckpts if f.startswith("checkpoint_")
                ]
                if len(numbered_checkpoints) > 0:
                    # checkpoints for each epoch are saved. we need to find the latest checkpoint.
                    numbered_checkpoints.sort(
                        key=lambda x: int(x.split("_")[1].split(".")[0])
                    )

                    # Get the last item in the sorted list
                    latest_checkpoint = (
                        numbered_checkpoints[-1] if numbered_checkpoints else None
                    )
                    configs.resume = int(latest_checkpoint.split("_")[1].split(".")[0])
                    load_dir = os.path.join(save_dir, latest_checkpoint)

                    configs.load_model_path = load_dir
                    strict_mode = (
                        True  # strict because we are resuming from a previous run
                    )
                    print(
                        f"Resume from {load_dir} and skip the first {configs.resume} epochs"
                    )
                else:
                    # Resume from latest checkpoint saved after every epoch.
                    latest_checkpoint = "latest_checkpoint.pth"
                    assert latest_checkpoint in cur_ckpts, "Latest checkpoint not found"
                    load_dir = os.path.join(save_dir, latest_checkpoint)
                    configs.load_model_path = load_dir
                    strict_mode = (
                        True  # strict because we are resuming from a previous run
                    )
                    print(f"Resume from the latest checkpoint at {load_dir}")

            else:
                # resume is not 0, it means we are resuming from a run in the current directory
                checkpoint = f"checkpoint_{configs.resume}.pt"
                assert checkpoint in cur_ckpts, f"Checkpoint {checkpoint} not found"
                load_dir = os.path.join(save_dir, checkpoint)
                configs.load_model_path = load_dir
                strict_mode = True  # strict because we are resuming from a run in the current directory
                print(
                    f"Resume from {load_dir} and skip the first {configs.resume} epochs"
                )
        # No previous run at the current directory, but resume is set (not 0)
        # it means we are resuming from a run in a different directory
        elif configs.resume != 0:
            # by setting `resume`, we can skip a few epoches at the beginning.
            if configs.load_model_path == "None":
                raise ValueError(
                    f"[ERROR] You want to skip the first {configs.resume} but you are not loading any existing checkpoint!"
                )
            # load_model_path is the path to the checkpoint to load
            strict_mode = True  # strict because we are resuming from a previous run
            print(
                f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
            )
        else:
            # configs.resume is 0 here
            # no previous run, no resume, not eval mode
            # it means we are training from scratch or from a pre-trained model
            if configs.load_model_path != "None":
                strict_mode = (
                    False  # not strict because we are loading a pre-trained model
                )
                print(f"Loading pre-trained model from {configs.load_model_path}")

    # Choose model type based on configuration
    full_model_name = configs.model.get("name")

    # Extract model type from full HuggingFace identifier
    if "/" in full_model_name:
        model_type = full_model_name.split("/")[
            -1
        ].lower()  # e.g., "microsoft/deberta-base" -> "deberta-base"
    else:
        model_type = full_model_name.lower()

    if model_type not in MODEL_REGISTRY:
        supported_types = ", ".join(f"'{t}'" for t in MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unsupported model type: {model_type} (extracted from {full_model_name}). Supported types: {supported_types}"
        )

    model_class = MODEL_REGISTRY[model_type]

    # Determine whether this run is binary link prediction or regression (heuristic prefix).
    dataset_name = str(configs.dataset)
    heuristic_prefixes = ("cn-", "aa-", "ra-", "katz-", "shortest-path-", "pagerank-")
    is_binary = not any(
        dataset_name.startswith(prefix) for prefix in heuristic_prefixes
    )
    print(f"Is binary link prediction: {is_binary} for dataset {dataset_name}")

    # Initialize GNN models differently
    if is_gnn:
        if feature_dim is None:
            raise ValueError("feature_dim must be provided for GNN models")
        # For GNN models, feature_dim is the node feature dimension (not 2 * node_feature_dim)
        model = model_class(
            config=configs.model,
            feature_dim=feature_dim,
            residual=configs.model.get("residual", False),
            id_awareness=configs.model.get("id_awareness", False),
            ortho_embedding=configs.model.get("ortho_embedding", False),
            is_binary=is_binary,
            max_num_nodes=tokenizer.max_num_nodes(),
        )
    else:
        model = model_class.from_default_configs_and_yaml(
            tokenizer=tokenizer,
            model_config=configs.model,
            use_features=use_features,
            feature_dim=feature_dim,
            feature_fusion=configs.get("feature_fusion", None),
            encoding_scheme=encoding_scheme,
            is_binary=is_binary,
        )

    # Load model weights if specified
    if configs.load_model_path != "None":
        # Load model weights from checkpoint
        # resume/eval mode: strict is True
        # pre-trained mode: strict is False
        checkpoint_metadata = load_model_checkpoint(
            model, configs.load_model_path, rank, strict=strict_mode
        )
        if strict_mode:
            # Resume from the latest checkpoint or best model checkpoint
            if configs.resume == 0 and (
                "latest_checkpoint" in configs.load_model_path
                or "best_model_checkpoint" in configs.load_model_path
            ):
                configs.resume = checkpoint_metadata["epoch"]
                print(f"Resume from epoch {configs.resume}")
            # Restore training tracking variables from checkpoint if available
            if "step" in checkpoint_metadata:
                configs.checkpoint_step = checkpoint_metadata["step"]
            if "total_trained_samples" in checkpoint_metadata:
                configs.checkpoint_total_trained_samples = checkpoint_metadata[
                    "total_trained_samples"
                ]
            if "best_score" in checkpoint_metadata:
                configs.checkpoint_best_score = checkpoint_metadata["best_score"]
        else:
            configs.resume = 0
            configs.checkpoint_step = None
            configs.checkpoint_total_trained_samples = None

    # Log model parameters to wandb
    if rank == 0:
        num_trainable_params, num_total_params = compute_params(
            model, print_params=False
        )

        # Log model parameters to wandb
        if wandb_run:
            wandb_run.log(
                {
                    "model/trainable_params": num_trainable_params,
                    "model/total_params": num_total_params,
                }
            )

    # Choose parallelization strategy based on evaluation mode or use_ddp flag
    # Use DDP for evaluation to avoid FSDP bugs, or when use_ddp is explicitly set
    # GNN models always use DDP (FSDP not well-supported for PyG models)
    if configs.use_ddp or is_gnn:
        print(f"Running DDP on rank = {rank}, world size = {world_size}")
        model = model.to(rank)
        parallel_model = DDP(model, device_ids=[rank])
    else:
        print(f"Running FSDP on rank = {rank}, world size = {world_size}")

        # Configure FSDP auto-wrap policy to wrap transformer layers
        # The root module is always wrapped, but we'll exclude embeddings from sharding
        transformer_layer_cls = FSDP_LAYER_REGISTRY.get(model_type)

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_layer_cls},
        )

        # Configure mixed precision for FSDP if bf16 is enabled
        mixed_precision_policy = None
        if use_bf16:
            print("Configuring FSDP with bf16 mixed precision")
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.float32,  # Keep buffers in fp32 for numerical stability
            )

        parallel_model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=rank,
            mixed_precision=mixed_precision_policy,
        )

    del model
    return parallel_model


@torch.no_grad()
def evaluate_loop(
    parallel_model,
    data_loader,
    *,
    rank,
    world_size,
    evaluator=None,
    dataset_len=None,
    split_name="eval",
    show_progress=True,
    compute_loss=False,
    check_sequential_indices=False,
    is_gnn=False,
    use_bf16=False,
):
    """
    Run a full evaluation loop over a loader, gather across ranks, and optionally compute loss/metrics.

    For heart datasets (detected by presence of 'labels' and 'scatter_preds' attributes):
    - batch_t doesn't have 'y', so we retrieve y directly from dataset.labels
    - y_pred needs to be scattered using dataset.orig_to_unique mapping to match the shape of labels

    Returns a dict with tensors gathered across ranks and optionally average loss and metrics (rank 0 only).
    { 'y_pred_all', 'y_true_all', 'idx_all', 'avg_loss' (optional), 'metrics' (rank 0 or None) }
    """
    if data_loader is None:
        raise ValueError("data_loader must not be None for evaluation")

    # Clear CUDA cache before evaluation
    torch.cuda.empty_cache()

    # Detect if this is a heart dataset (datasets whose name starts with "heart-")
    dataset = data_loader.dataset if is_gnn else data_loader.dataset.original_dataset
    is_heart_dataset = (
        hasattr(dataset, "labels")
        and dataset.labels is not None
        and hasattr(dataset, "orig_to_unique")
        and dataset.orig_to_unique is not None
    )

    parallel_model.module.eval()
    autocast_kwargs = {
        "device_type": "cuda",
        "dtype": torch.bfloat16,
        "enabled": use_bf16,
    }
    y_preds = []
    y_trues = (
        [] if not is_heart_dataset else None
    )  # Don't collect y_trues for heart datasets
    idxs = []
    total_loss = 0.0

    pbar = None
    if show_progress:
        pbar = tqdm(
            total=len(data_loader),
            desc=f"Eval {split_name}",
            colour="green",
            dynamic_ncols=True,
        )

    for batch in data_loader:
        if is_gnn:
            # GNN models: batch is (batch_indices, batched_data) from GNNCollator
            batch_indices, batched_data = batch
            batched_data = batched_data.to(rank)
            with torch.autocast(**autocast_kwargs):
                if is_heart_dataset:
                    # For heart datasets, batched_data.y might not exist, so we pass None
                    outputs = parallel_model(batched_data, labels=None)
                    y_pred = outputs["logits"]
                else:
                    outputs = parallel_model(batched_data, labels=batched_data.y)
                    y_pred = outputs["logits"]
                    y_true = batched_data.y
                    if y_true is None:
                        raise ValueError("Batch missing 'y' for evaluation")
                    if compute_loss:
                        loss_val = outputs["loss"]
                        dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                        total_loss += loss_val.item() / world_size
                    y_trues.append(y_true.detach().to(y_pred.dtype))

            idx = batch_indices.to(rank)

        else:
            # Transformer models: batch is a dictionary
            batch_t = {
                k: (v.to(rank) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            # Prepare forward kwargs, including feature_embeds if available
            if is_heart_dataset:
                # For heart datasets, batch_t doesn't have 'y'
                forward_kwargs = {
                    "input_embeds": batch_t["input_embeds"],
                    "attention_mask": batch_t["attention_mask"],
                    "feature_embeds": (
                        batch_t["feature_embeds"]
                        if "feature_embeds" in batch_t
                        else None
                    ),
                    "num_nodes": batch_t["num_nodes"],
                }
                with torch.autocast(**autocast_kwargs):
                    outputs = parallel_model(**forward_kwargs, labels=None)
                y_pred = outputs.logits
            else:
                forward_kwargs = {
                    "input_embeds": batch_t["input_embeds"],
                    "attention_mask": batch_t["attention_mask"],
                    "labels": batch_t["y"],
                    "feature_embeds": (
                        batch_t["feature_embeds"]
                        if "feature_embeds" in batch_t
                        else None
                    ),
                    "num_nodes": batch_t["num_nodes"],
                }
                with torch.autocast(**autocast_kwargs):
                    outputs = parallel_model(**forward_kwargs)
                y_pred = outputs.logits
                y_true = batch_t.get("y")
                if y_true is None:
                    raise ValueError("Batch missing 'y' for evaluation")
                if compute_loss:
                    loss_val = outputs.loss
                    dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                    total_loss += loss_val.item() / world_size
                y_trues.append(y_true.detach().to(y_pred.dtype))

            idx = batch_t.get("idx")

        y_preds.append(y_pred.detach())
        idxs.append(idx)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    y_pred_flat = torch.cat(y_preds, dim=0)
    idx_flat = torch.cat(idxs, dim=0).to(torch.long)

    if dataset_len is None:
        raise ValueError("dataset_len must be provided for distributed gather")

    if is_heart_dataset:
        # For heart datasets: gather predictions for unique edges, then scatter to original edges
        y_pred_all_unique = concat_all_gather_1d(y_pred_flat, dataset_len=dataset_len)
        idx_all = concat_all_gather_1d(idx_flat, dataset_len=dataset_len)

        if check_sequential_indices and rank == 0:
            assert all(idx_all == torch.arange(len(y_pred_all_unique), device=rank))

        # Scatter predictions from unique edges to original edges (only on rank 0)
        if rank == 0:
            y_pred_all = dataset.scatter_preds(y_pred_all_unique)

            # dataset.labels contains labels for all original edges (including duplicates)
            # After scattering, y_pred_all has predictions for all original edges in the same order
            y_true_all = dataset.labels.to(rank)

            # Compute loss if requested (only on rank 0 since data is already gathered)
            if compute_loss:
                loss_fn = torch.nn.BCEWithLogitsLoss()
                loss_val = loss_fn(y_pred_all.float(), y_true_all.float())
                total_loss = loss_val.item()

            # Compute idx_all for heart datasets (only on rank 0)
            idx_all = torch.arange(len(y_pred_all), device=rank)
        else:
            y_pred_all = None
            y_true_all = None
            idx_all = None

        metrics = None
        if evaluator is not None and rank == 0:
            metrics = evaluator.eval(
                {
                    "y_true": y_true_all,
                    "y_pred": y_pred_all,
                    "idx": idx_all,
                }
            )
    else:
        # For regular datasets: gather predictions and labels normally
        y_true_flat = torch.cat(y_trues, dim=0)
        y_pred_all = concat_all_gather_1d(y_pred_flat, dataset_len=dataset_len)
        y_true_all = concat_all_gather_1d(y_true_flat, dataset_len=dataset_len)
        idx_all = concat_all_gather_1d(idx_flat, dataset_len=dataset_len)

        if check_sequential_indices and rank == 0:
            assert all(idx_all == torch.arange(len(y_pred_all), device=rank))

        metrics = None
        if evaluator is not None and rank == 0:
            metrics = evaluator.eval(
                {"y_true": y_true_all, "y_pred": y_pred_all, "idx": idx_all}
            )
    # Synchronize all processes: ensure rank 0 finishes metrics computation before return
    torch.distributed.barrier()
    result = {
        "y_pred_all": y_pred_all,
        "y_true_all": y_true_all,
        "idx_all": idx_all,
        "metrics": metrics,
    }
    if compute_loss:
        # For regular datasets: total_loss accumulates batch losses, divide by num batches
        # For heart datasets: total_loss is already mean per sample (computed on all samples)
        if is_heart_dataset:
            avg_loss = total_loss
        else:
            avg_loss = total_loss / max(1, len(data_loader))
        result["avg_loss"] = avg_loss

    # Clear CUDA cache after evaluation
    torch.cuda.empty_cache()

    return result


def train_loop(
    parallel_model,
    train_loader,
    optimizer,
    epoch,
    num_epochs,
    gradient_accumulation_steps,
    max_num_samples,
    total_train_steps,
    total_trained_samples,
    rank,
    local_rank,
    configs,
    wandb_run,
    world_size,
    is_gnn=False,
    use_bf16=False,
):
    """
    Execute one training epoch.

    Args:
        parallel_model: The parallelized model (DDP or FSDP)
        train_loader: DataLoader for training data
        optimizer: Optimizer for model parameters
        epoch: Current epoch number (0-indexed)
        num_epochs: Total number of epochs
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_num_samples: Total maximum number of samples per epoch across all devices (-1 for no limit)
        total_train_steps: Total number of training batches processed so far (across all ranks)
        total_trained_samples: Total number of training samples processed so far (across all ranks)
        rank: Current process rank
        local_rank: Local rank for device placement
        configs: Configuration object
        wandb_run: Wandb run object for logging (None if not logging)
        world_size: Total number of processes
        is_gnn: Whether the model is a GNN model (default: False)
    Returns:
        tuple: (Updated total_train_steps count, Updated total_trained_samples count)
    """
    # Set model to training mode
    parallel_model.module.train()
    total_length = len(train_loader) // gradient_accumulation_steps

    # Get batch size from configs for calculating sample limits
    batch_size = configs.batch_size_training

    # Adjust total_length to respect max_num_samples (total samples across all devices) per epoch if specified
    if max_num_samples != -1:
        # max_num_samples is total samples across all devices
        # Formula: max_num_samples = batch_size * num_gpus * num_batches_per_rank
        # Solving for num_batches_per_rank: num_batches_per_rank = max_num_samples / (batch_size * world_size)
        max_batches_per_rank = max_num_samples // (batch_size * world_size)
        total_length = min(
            total_length, max_batches_per_rank // gradient_accumulation_steps
        )

    pbar = tqdm(
        total=total_length,
        desc=f"Training Epoch: {epoch+1}",
        colour="blue",
        dynamic_ncols=True,
    )

    # Initialize gradients for the epoch
    optimizer.zero_grad()

    # Track total samples processed per rank (for max_num_samples limit)
    total_samples_per_rank = 0

    autocast_kwargs = {
        "device_type": "cuda",
        "dtype": torch.bfloat16,
        "enabled": use_bf16,
    }

    # Training loop over all batches in the current epoch
    for step, batch in enumerate(train_loader):
        total_train_steps += world_size

        if is_gnn:
            # GNN models: batch is (batch_indices, batched_data) from GNNCollator
            batch_indices, batched_data = batch
            batched_data = batched_data.to(rank)

            # Extract labels
            if hasattr(batched_data, "y") and batched_data.y is not None:
                y = batched_data.y
            else:
                raise ValueError("Batch missing 'y' for training")

            # Get actual batch size
            actual_batch_size = (
                y.shape[0] if isinstance(y, torch.Tensor) else batch_size
            )
            total_samples_per_rank += actual_batch_size
            total_trained_samples += actual_batch_size * world_size

            # Forward pass with labels for loss computation
            with torch.autocast(**autocast_kwargs):
                outputs = parallel_model(batched_data, labels=y)
                loss = outputs["loss"]
            loss = loss.float() / gradient_accumulation_steps
            loss.backward()
        else:
            # Transformer models: batch is a dictionary
            batch_t = {
                k: v.to(rank) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Extract labels for link prediction at [A] token positions
            y = batch_t.get("y")
            if y is None:
                raise ValueError("Batch missing 'y' for training")

            # Get actual batch size from the batch (may vary for last batch)
            actual_batch_size = (
                y.shape[0] if isinstance(y, torch.Tensor) else batch_size
            )
            total_samples_per_rank += actual_batch_size
            total_trained_samples += actual_batch_size * world_size

            # Forward pass with labels for loss computation
            # The model learns to predict link existence at [A] positions
            # Prepare forward kwargs, including feature_embeds if available
            forward_kwargs = {
                "input_embeds": batch_t["input_embeds"],
                "attention_mask": batch_t["attention_mask"],
                "labels": y,
                "feature_embeds": (
                    batch_t["feature_embeds"] if "feature_embeds" in batch_t else None
                ),
                "num_nodes": batch_t["num_nodes"],
            }
            with torch.autocast(**autocast_kwargs):
                outputs = parallel_model(**forward_kwargs)
                loss = outputs.loss
            loss = loss.float() / gradient_accumulation_steps
            loss.backward()

        # Update model weights after accumulating gradients
        if (step + 1) % gradient_accumulation_steps == 0 or step == len(
            train_loader
        ) - 1:
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)

        # Log training metrics to wandb
        if wandb_run and rank == 0:
            log_dict = {
                "train/epoch": epoch + 1,
                "train/step": total_train_steps,
                "train/total_trained_samples": total_trained_samples,
                "train/loss": loss.detach().float() * gradient_accumulation_steps,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            wandb_run.log(log_dict)

        # Update progress bar with current loss
        # Show step limits based on max_num_samples (total samples across all devices) per epoch if specified
        if max_num_samples != -1:
            # max_num_samples is total samples across all devices
            # Calculate max batches per rank: max_num_samples / (batch_size * world_size)
            max_batches_per_rank = max_num_samples // (batch_size * world_size)
            step_limit = min(len(train_loader), max_batches_per_rank)
            step_info = f"batch {step}/{step_limit}"
        else:
            step_info = f"batch {step}/{len(train_loader)}"

        pbar.set_description(
            f"Training Epoch: {epoch+1}/{num_epochs}, {step_info} "
            f"completed (loss: {round(float(loss.detach().float()*gradient_accumulation_steps), 4)})"
        )

        # Break conditions for early stopping
        if configs.get("debug", False) and step == 5:
            print("Debug mode: breaking after 5 steps")
            # Apply any accumulated gradients before breaking
            if (step + 1) % gradient_accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
            break
        # Break condition for max_num_samples (total samples across all devices) per epoch if specified
        elif max_num_samples != -1:
            # Estimate total samples across all devices: total_samples_per_rank * world_size
            total_samples_across_devices = total_samples_per_rank * world_size

            if total_samples_across_devices >= max_num_samples:
                print(
                    f"Reached max_num_samples ({max_num_samples} total samples across all devices, processed {total_samples_across_devices}), breaking epoch"
                )
                # Apply any accumulated gradients before breaking
                if (step + 1) % gradient_accumulation_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad()
                break
    pbar.close()

    # Synchronize all processes after training epoch
    torch.distributed.barrier(device_ids=[local_rank])

    return total_train_steps, total_trained_samples


def main():
    # Suppress warnings for cleaner output
    suppress_warnings()

    # Enable TF32 for faster matmul/convolution on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Transformer4LP")

    parser.add_argument("config_file", help="Path to the YAML config file")
    parser.add_argument(
        "--only_eval",
        action="store_true",
        help="Override YAML config and run in evaluation-only mode",
    )
    parser.add_argument(
        "--only_eval_test",
        action="store_true",
        help="Override YAML config and run in evaluation-only mode for test set",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Note for the run",
    )
    parser.add_argument(
        "--use_features",
        action="store_true",
        help="Use features for the model",
    )
    parser.add_argument(
        "--feature_fusion",
        type=str,
        default="early",
        help="Feature fusion strategy for the model. 'early' or 'late' when use_features=True. Default: 'early'.",
    )

    # GNN model arguments
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=None,
        help="Hidden size for the model",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=None,
        help="Number of hidden layers for the model",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Number of layers for the model",
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=None,
        help="Intermediate size for the model",
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=None,
        help="Number of attention heads for the model",
    )
    parser.add_argument(
        "--num_layers_predictor",
        type=int,
        default=None,
        help="Number of layers for the predictor",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout rate for the model",
    )
    parser.add_argument(
        "--residual",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
        help="Use residual connections in the model",
    )
    parser.add_argument(
        "--id_awareness",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
        help="Use id awareness for the GNN model.",
    )
    parser.add_argument(
        "--ortho",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
        help="Use orthogonal embedding for the GNN model.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate for the model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset from config file",
    )
    parser.add_argument(
        "--heart",
        action="store_true",
        help="Enable heart mode: add 'heart-' prefix to dataset and load checkpoints from original directory",
    )
    parser.add_argument(
        "--heart_resume",
        type=int,
        default=None,
        help="Resume training from a specific epoch for --heart",
    )
    parser.add_argument(
        "--skip_val",
        action="store_true",
        help="Skip validation/test evaluation and go directly to the final test set evaluation after training 1 epoch",
    )
    args = parser.parse_args()

    # Initialize distributed environment for multi-GPU training
    # Extract rank information from environment variables with fallback defaults
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    # Set the device before initializing the process group
    torch.cuda.set_device(torch.device(local_rank))

    # Initialize process group with device specification
    dist.init_process_group(
        "nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=30),
    )

    # Load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    configs = Config(config_dict)

    # Override dataset if dataset argument is provided (do this early so other checks can use it)
    if args.dataset is not None:
        configs.dataset = args.dataset
        config_dict["dataset"] = args.dataset

    # Override dataset with "heart-" prefix if --heart flag is set
    if args.heart:
        if not configs.dataset.startswith("heart-"):
            configs.dataset = f"heart-{configs.dataset}"
            config_dict["dataset"] = configs.dataset
            print(f"Added 'heart-' prefix to dataset: {configs.dataset}")

    # Override use_features for ogbl-ddi as early as possible (ogbl-ddi has no node features)
    if "ogbl-ddi" in configs.dataset:
        print("ogbl-ddi has no node features, setting use_features=False")
        configs.use_features = False
        config_dict["use_features"] = False

    # Calculate max_num_nodes from sampling config (needed for adjacency_row encoding and tokenizer creation)
    depth, num_neighbors = configs.sampling_config.edge_ego.depth_neighbors[-1]
    max_num_nodes = 2 * sum(
        [num_neighbors**d for d in range(depth + 1)]
    )  # 2 * sum([10^0, 10^1, 10^2])

    if (
        hasattr(configs, "encoding_scheme")
        and configs.encoding_scheme == "adjacency_row"
    ):
        print(
            "Detected encoding scheme: adjacency_row, overriding max_sequence_length to max_num_nodes + 2"
        )
        configs.max_sequence_length = max_num_nodes + 2
        config_dict["max_sequence_length"] = configs.max_sequence_length

    # Override only_eval if only_eval argument is provided
    if args.only_eval:
        configs.only_eval = True
        config_dict["only_eval"] = True
    # Override only_eval_test if only_eval_test argument is provided
    if args.only_eval_test:
        configs.only_eval_test = True
        config_dict["only_eval_test"] = True
        configs.only_eval = True
        config_dict["only_eval"] = True
    # Override skip_val if skip_val argument is provided
    if args.skip_val:
        configs.skip_val = True
        config_dict["skip_val"] = True
    # Override seed if seed argument is provided
    if args.seed is not None:
        configs.seed = args.seed
        config_dict["seed"] = args.seed

    # Override debug if debug argument is provided
    if args.debug:
        configs.debug = True
        config_dict["debug"] = True

    # Override num_workers if num_workers argument is provided
    if args.num_workers is not None:
        configs.num_workers = args.num_workers
        config_dict["num_workers"] = args.num_workers

    # Override num_epochs if num_epochs argument is provided
    if args.num_epochs is not None:
        configs.num_epochs = args.num_epochs
        config_dict["num_epochs"] = args.num_epochs

    # Override note if note argument is provided
    if args.note is not None:
        configs.note = args.note
        config_dict["note"] = args.note

    # Override use_features if use_features argument is provided
    if args.use_features:
        # Warn if trying to use features with ogbl-ddi (which has no node features)
        if "ogbl-ddi" in configs.dataset:
            print(
                "[WARNING] --use_features was specified but ogbl-ddi has no node features. "
                "Ignoring --use_features flag and using use_features=False."
            )
        else:
            configs.use_features = True
            config_dict["use_features"] = True
            # automatically set feature_fusion to "early" if not provided and use_features=True
            # args.feature_fusion is 'early' by default
            # else, override using user-provided value
            if args.feature_fusion is not None and args.feature_fusion in [
                "early",
                "late",
            ]:
                configs.feature_fusion = args.feature_fusion
                config_dict["feature_fusion"] = args.feature_fusion
            else:
                raise ValueError(
                    f"Invalid feature fusion strategy: {args.feature_fusion}. Must be 'early' or 'late' when use_features=True."
                )

    # Override hidden_size if hidden_size argument is provided
    if args.hidden_size is not None:
        configs.model.hidden_size = args.hidden_size
        config_dict["model"]["hidden_size"] = args.hidden_size

    # Override num_hidden_layers if num_hidden_layers argument is provided
    if args.num_hidden_layers is not None:
        configs.model.num_hidden_layers = args.num_hidden_layers
        config_dict["model"]["num_hidden_layers"] = args.num_hidden_layers

    # Override num_layers if num_layers argument is provided
    if args.num_layers is not None:
        configs.model.num_layers = args.num_layers
        config_dict["model"]["num_layers"] = args.num_layers

    if args.intermediate_size is not None:
        configs.model.intermediate_size = args.intermediate_size
        config_dict["model"]["intermediate_size"] = args.intermediate_size

    if args.num_attention_heads is not None:
        configs.model.num_attention_heads = args.num_attention_heads
        config_dict["model"]["num_attention_heads"] = args.num_attention_heads

    # Override num_layers_predictor if num_layers_predictor argument is provided
    if args.num_layers_predictor is not None:
        configs.model.num_layers_predictor = args.num_layers_predictor
        config_dict["model"]["num_layers_predictor"] = args.num_layers_predictor

    # Override dropout if dropout argument is provided
    if args.dropout is not None:
        configs.model.dropout = args.dropout
        config_dict["model"]["dropout"] = args.dropout

    # Override residual if residual argument is provided
    if args.residual is not None:
        configs.model.residual = args.residual
        config_dict["model"]["residual"] = args.residual

    # Override id_awareness if id_awareness argument is provided
    if args.id_awareness is not None:
        configs.model.id_awareness = args.id_awareness
        config_dict["model"]["id_awareness"] = args.id_awareness

    # Override ortho_embedding if ortho argument is provided
    if args.ortho is not None:
        configs.model.ortho_embedding = args.ortho
        config_dict["model"]["ortho_embedding"] = args.ortho

    # Override lr if lr argument is provided
    if args.lr is not None:
        configs.lr = args.lr
        config_dict["lr"] = args.lr

    configs.name = generate_name(config_dict)

    # Store original directory name if --heart flag is set
    if args.heart:
        configs.orig_dir = configs.name[6:]  # Strip "heart-" prefix (6 characters)
        print(f"Original directory: {configs.orig_dir}")

    # Initialize wandb
    if not configs.get("debug", False) and rank == 0:
        wandb_run = wandb.init(
            project=configs.get("project", "transformer4lp"),
            name=configs.name,
            notes=configs.note,
        )
        wandb_run.config.update(config_dict, allow_val_change=True)
        wandb_run.log_code(
            root=ROOT_DIR,
            include_fn=lambda path: path.endswith(".py")
            or "configs" in path
            or "args" in path,
        )
    else:
        wandb_run = None

    # Create save directory for model checkpoints
    save_dir = os.path.join(configs.save_path, configs.name)
    if rank == 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # copy config file to save_dir
    if rank == 0:
        shutil.copy(args.config_file, os.path.join(save_dir, "config.yaml"))

    # Synchronize all processes before proceeding
    torch.distributed.barrier(device_ids=[local_rank])

    # Load datasets first (before model creation) so we can get feature_dim if needed
    # These datasets contain graph data formatted for transformer models
    print(f"Max number of nodes per subgraph: {max_num_nodes}")
    print("Loading datasets...")
    ds = load_dataset(configs)
    if isinstance(ds, tuple) and len(ds) == 4:
        train_dataset_raw, valid_dataset_raw, test_dataset_raw, _ = ds
    else:
        raise ValueError(
            "Dataset loader must return train/valid/test datasets for fine-tuning"
        )

    # Synchronize all processes after dataset loading
    torch.distributed.barrier(device_ids=[local_rank])

    # Reset seed after dataset loading to ensure model initialization uses the correct seed
    # (dataset initialization may have called reset_samples() with default seed=42, overriding the global seed)
    set_seed(configs.seed)

    # Get feature dimension and determine if model is GNN
    encoding_scheme = configs.get("encoding_scheme", None)
    feature_dim, is_gnn = get_feature_dim(
        configs, train_dataset_raw, encoding_scheme=encoding_scheme
    )

    use_features = configs.get("use_features", False)
    # Safety check for feature fusion
    if use_features and not is_gnn:
        feature_fusion = configs.get("feature_fusion", None)
        assert feature_fusion in [
            "early",
            "late",
        ], "feature_fusion must be 'early' or 'late' when use_features=True"

    # Create tokenizer to store the max_num_nodes (legacy code)
    tokenizer = STokenizer(num_nodes=max_num_nodes)

    # Create and configure the model with appropriate parallelization strategy
    parallel_model = get_model(
        configs,
        tokenizer,
        rank,
        world_size,
        local_rank,
        wandb_run,
        save_dir,
        use_features=use_features,
        feature_dim=feature_dim,
        encoding_scheme=encoding_scheme,
        is_gnn=is_gnn,
        use_bf16=configs.get("bf16", False),
    )

    # Initialize optimizer for training
    optimizer = optim.AdamW(
        parallel_model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay
    )
    # Load optimizer state from checkpoint if load_model_path is not None
    if configs.load_model_path != "None":
        load_optimizer_checkpoint(
            optimizer, parallel_model, configs.load_model_path, rank
        )

    if rank == 0:
        rich.print("Config:", config_dict)
        rich.print(parallel_model)

    # Create collator for batching - GNNCollator for GNN models, Collator for transformers
    if is_gnn:
        collator = GNNCollator()
    else:
        collator = Collator(tokenizer)

    # Initialize evaluator for link prediction metrics
    evaluator = Evaluator(metric=configs.dataset)

    # Get training configuration
    num_epochs = configs.num_epochs
    # Set num_epochs to 1 if skip_val is True
    if configs.get("skip_val", False):
        num_epochs = 1
        print(
            f"skip_val is enabled: setting num_epochs to 1, and going directly to the final test set evaluation"
        )

    # Adjust gradient_accumulation_steps to maintain constant total samples
    # before first optimizer step regardless of world_size
    # Config value represents total gradient accumulation steps across all devices
    total_gradient_accumulation_steps = configs.gradient_accumulation_steps
    gradient_accumulation_steps = total_gradient_accumulation_steps // world_size
    if gradient_accumulation_steps == 0:
        raise ValueError(
            f"gradient_accumulation_steps ({total_gradient_accumulation_steps}) must be >= world_size ({world_size})"
        )
    if total_gradient_accumulation_steps % world_size != 0:
        print(
            f"[WARNING] gradient_accumulation_steps ({total_gradient_accumulation_steps}) "
            f"is not divisible by world_size ({world_size}). "
            f"Using {gradient_accumulation_steps} per device "
            f"(total effective: {gradient_accumulation_steps * world_size})"
        )
    print(
        f"Gradient accumulation: {gradient_accumulation_steps} steps per device "
        f"(total: {gradient_accumulation_steps * world_size} steps across {world_size} devices)"
    )

    # Handle max_num_samples configuration
    max_num_samples = configs.get("max_num_samples", -1)
    if max_num_samples != -1:
        assert max_num_samples > 0, "max_num_samples must be positive when specified"

    # Handle evaluation-only mode
    if configs.only_eval:
        only_eval_test = configs.get("only_eval_test", False)

        # Build loaders - we always need test_loader, and valid_loader only if not only_eval_test
        _, valid_loader, test_loader = build_loaders(
            epoch=0,
            tokenizer=tokenizer,
            configs=configs,
            collator=collator,
            train_dataset_raw=None,
            valid_dataset_raw=None if only_eval_test else valid_dataset_raw,
            test_dataset_raw=test_dataset_raw,
            use_features=use_features,
            encoding_scheme=encoding_scheme,
            is_gnn=is_gnn,
        )

        # Run evaluation on validation set (if not only_eval_test)
        if only_eval_test:
            valid_out = {"metrics": {}}
            print("Skipping validation evaluation (only_eval_test=True)")
        else:
            valid_out = evaluate_loop(
                parallel_model,
                valid_loader,
                rank=rank,
                world_size=world_size,
                evaluator=evaluator,
                dataset_len=len(valid_dataset_raw),
                split_name="valid",
                show_progress=True,
                compute_loss=False,
                check_sequential_indices=True,
                is_gnn=is_gnn,
                use_bf16=configs.get("bf16", False),
            )
            print(f"valid metrics: {valid_out.get('metrics', {})}")

        # Run evaluation on test set
        test_out = evaluate_loop(
            parallel_model,
            test_loader,
            rank=rank,
            world_size=world_size,
            evaluator=evaluator,
            dataset_len=len(test_dataset_raw),
            split_name="test",
            show_progress=True,
            compute_loss=False,
            check_sequential_indices=True,
            is_gnn=is_gnn,
            use_bf16=configs.get("bf16", False),
        )
        print(f"test metrics: {test_out.get('metrics', {})}")

        if rank == 0:
            if wandb_run:
                log_dict = {}
                if not only_eval_test:
                    for metric_name, metric_value in (
                        valid_out.get("metrics", {})
                    ).items():
                        log_dict[f"valid/{metric_name}"] = metric_value
                for metric_name, metric_value in (test_out.get("metrics", {})).items():
                    log_dict[f"test/{metric_name}"] = metric_value
                wandb_run.log(log_dict)
                # Append log_dict to a single JSON file in save_dir
                best_metrics_file = os.path.join(save_dir, "best_metrics.json")
                with open(best_metrics_file, "a") as f:
                    f.write(json.dumps(log_dict) + "\n")
        log_final_table(
            wandb_run,
            configs,
            valid_out.get("metrics", {}) if not only_eval_test else {},
            test_out.get("metrics", {}),
        )

        # Clean up distributed environment
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    #########################################################################################
    #########################################################################################
    ############################## TRAINING LOOP ############################################
    #########################################################################################
    #########################################################################################

    # Restore from checkpoint if available (for resume mode)
    total_train_steps = getattr(configs, "checkpoint_step", None) or 0
    total_trained_samples = (
        getattr(configs, "checkpoint_total_trained_samples", None) or 0
    )
    # Get best_score from checkpoint and normalize it (ensures consistency)
    # Compute metric_key on all ranks (needed for normalization)
    dataset_key = str(configs.dataset).lower()
    metric_key = SELECTION_METRIC_BY_DATASET.get(dataset_key)
    if metric_key is None:
        raise ValueError(
            f"No selection metric defined for dataset {dataset_key}. "
            f"Available datasets: {list(SELECTION_METRIC_BY_DATASET.keys())}"
        )
    raw_best_score = getattr(configs, "checkpoint_best_score", None)
    if raw_best_score is not None:
        best_score = normalize_score(raw_best_score, metric_key)
        best_raw_score = raw_best_score
    else:
        best_score = float("-inf")
        best_raw_score = None
    best_val_metrics = None  # Store best validation metrics for final table

    # Main training loop over all epochs
    skip_val = configs.get("skip_val", False)
    if args.heart_resume is not None:
        configs.resume = args.heart_resume
    for epoch in range(configs.resume, num_epochs):
        # Build test_loader if eval_test is True, otherwise set to None
        train_loader, valid_loader, test_loader = build_loaders(
            epoch=epoch,
            tokenizer=tokenizer,
            configs=configs,
            collator=collator,
            train_dataset_raw=train_dataset_raw,
            valid_dataset_raw=None if skip_val else valid_dataset_raw,
            test_dataset_raw=(
                test_dataset_raw
                if configs.get("eval_test", False) and not skip_val
                else None
            ),
            use_features=use_features,
            encoding_scheme=encoding_scheme,
            is_gnn=is_gnn,
        )

        # If --heart flag is set, skip training and load checkpoint from original directory
        if args.heart:
            orig_dir_path = os.path.join(configs.save_path, configs.orig_dir)
            checkpoint_path = os.path.join(orig_dir_path, f"checkpoint_{epoch + 1}.pth")

            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}. "
                    f"Expected checkpoint from original training directory: {orig_dir_path}"
                )

            print(f"Loading checkpoint from original directory: {checkpoint_path}")
            load_model_checkpoint(parallel_model, checkpoint_path, rank, strict=True)
        else:
            # Execute training loop for this epoch
            total_train_steps, total_trained_samples = train_loop(
                parallel_model=parallel_model,
                train_loader=train_loader,
                optimizer=optimizer,
                epoch=epoch,
                num_epochs=num_epochs,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_num_samples=max_num_samples,
                total_train_steps=total_train_steps,
                total_trained_samples=total_trained_samples,
                rank=rank,
                local_rank=local_rank,
                configs=configs,
                wandb_run=wandb_run,
                world_size=world_size,
                is_gnn=is_gnn,
                use_bf16=configs.get("bf16", False),
            )

        # Save model checkpoint after each epoch if save_only_improve is disabled
        if (
            not configs.get("save_only_improve", False)
            and not configs.get("debug", False)
            and not args.heart
        ):
            save_checkpoint(
                parallel_model,
                optimizer,
                os.path.join(save_dir, f"checkpoint_{epoch + 1}.pth"),
                rank,
                "checkpoint",
                epoch=epoch + 1,
                step=total_train_steps,
                total_trained_samples=total_trained_samples,
            )

        # Validation evaluation after each training epoch (controlled by eval_every)
        eval_every = configs.get("eval_every", 1)
        skip_val = configs.get("skip_val", False)
        if (epoch + 1) % eval_every == 0 and not skip_val:
            val_out = evaluate_loop(
                parallel_model,
                valid_loader,
                rank=rank,
                world_size=world_size,
                evaluator=evaluator,
                dataset_len=len(valid_dataset_raw),
                split_name="valid",
                show_progress=True,
                compute_loss=True,
                check_sequential_indices=True,
                is_gnn=is_gnn,
                use_bf16=configs.get("bf16", False),
            )

            # Test evaluation during training if eval_test is True
            if configs.get("eval_test", False):
                test_out = evaluate_loop(
                    parallel_model,
                    test_loader,
                    rank=rank,
                    world_size=world_size,
                    evaluator=evaluator,
                    dataset_len=len(test_dataset_raw),
                    split_name="test",
                    show_progress=True,
                    compute_loss=True,
                    check_sequential_indices=True,
                    is_gnn=is_gnn,
                    use_bf16=configs.get("bf16", False),
                )

            if rank == 0:
                # Validation evaluation
                valid_result = val_out["metrics"]
                print(f"valid metrics: {valid_result}")
                if isinstance(valid_result, dict) and metric_key in valid_result:
                    raw_score = float(valid_result[metric_key])
                else:
                    raise ValueError(
                        f"Metric {metric_key} not found in validation results for dataset {dataset_key}"
                    )
                print(f"Validation loss: {val_out.get('avg_loss', 0.0)}")

                # Test evaluation
                if configs.get("eval_test", False):
                    test_result = test_out["metrics"]
                    print(f"test metrics: {test_result}")
                    print(f"Test loss: {test_out.get('avg_loss', 0.0)}")

                # Log metrics to wandb
                if wandb_run:
                    log_dict = {
                        "valid/epoch": epoch + 1,
                        "valid/loss": val_out.get("avg_loss", 0.0),
                    }
                    for metric_name, metric_value in valid_result.items():
                        log_dict[f"valid/{metric_name}"] = metric_value

                    # Log test metrics if eval_test is True
                    if configs.get("eval_test", False):
                        log_dict["test/epoch"] = epoch + 1
                        log_dict["test/loss"] = test_out.get("avg_loss", 0.0)
                        for metric_name, metric_value in test_result.items():
                            log_dict[f"test/{metric_name}"] = metric_value

                    wandb_run.log(log_dict)

                    # Append log_dict to a single JSONL file in save_dir
                    metrics_file = os.path.join(save_dir, "metrics.jsonl")
                    with open(metrics_file, "a") as f:
                        f.write(json.dumps(log_dict) + "\n")
            else:
                valid_result = None
                raw_score = 0.0

            # Broadcast raw_score from rank 0 to all ranks
            raw_score_tensor = torch.tensor(raw_score, device=rank)
            dist.broadcast(raw_score_tensor, src=0)
            raw_score = raw_score_tensor.item()

            # Normalize score so we always maximize (negate if lower is better)
            score = normalize_score(raw_score, metric_key)

            # Check if this is the best score (all ranks can evaluate this)
            if score > best_score:
                # Store best validation metrics for final table (rank 0 only)
                if rank == 0:
                    best_val_metrics = valid_result.copy()

                # Update best score (normalized for comparison) and best raw score (for saving)
                best_score = score
                best_raw_score = raw_score

                # Save the best model checkpoint (all ranks participate in FSDP/DDP checkpoint saving)
                # Save raw_score to checkpoint (not normalized) for consistency
                if not configs.get("debug", False):
                    save_checkpoint(
                        parallel_model,
                        optimizer,
                        os.path.join(save_dir, f"best_model_checkpoint.pth"),
                        rank,
                        "best checkpoint",
                        epoch=epoch + 1,
                        step=total_train_steps,
                        total_trained_samples=total_trained_samples,
                        best_score=best_raw_score,
                    )

        # Save latest checkpoint after every epoch (for resuming from previous run)
        # Save after validation (if it happened) so it includes the updated best_score
        # Use best_raw_score if available (after validation), otherwise use None (will default appropriately)
        if not configs.get("debug", False) and not args.heart:
            save_checkpoint(
                parallel_model,
                optimizer,
                os.path.join(save_dir, "latest_checkpoint.pth"),
                rank,
                "latest checkpoint",
                epoch=epoch + 1,
                step=total_train_steps,
                total_trained_samples=total_trained_samples,
                best_score=best_raw_score,
            )

    # Synchronize all processes after training completion
    torch.distributed.barrier(device_ids=[local_rank])

    #########################################################################################
    #########################################################################################
    ############################## FINAL EVALUATION #########################################
    #########################################################################################
    #########################################################################################

    # Test set evaluation after training completion
    if configs.resume != 0:
        print(
            "The model was resumed from a previous run, skipping test set evaluation. \
              Please run the script again with only_eval set to True to evaluate the test set."
        )
        best_val_metrics = None
        best_test_metrics = None
    else:
        print("Starting test set evaluation...")

        # Load the best model checkpoint if available
        # If skip_val is True, use latest_checkpoint instead (no best checkpoint exists)
        skip_val = configs.get("skip_val", False)
        if not configs.get("debug", False):
            if skip_val:
                # Use latest checkpoint when validation was skipped
                latest_checkpoint_path = os.path.join(save_dir, "latest_checkpoint.pth")
                if not os.path.exists(latest_checkpoint_path):
                    raise ValueError(
                        f"No latest checkpoint found at {latest_checkpoint_path}. "
                        "Cannot evaluate test set when skip_val=True without a checkpoint."
                    )
                print(
                    f"Loading latest model from {latest_checkpoint_path} (skip_val=True)"
                )
                load_model_checkpoint(
                    parallel_model, latest_checkpoint_path, rank, strict=True
                )
            else:
                # Find the best model checkpoint
                checkpoint_files = [
                    f
                    for f in os.listdir(save_dir)
                    if f.startswith("best_model_checkpoint")
                ]
                if len(checkpoint_files) > 1:
                    raise ValueError("Multiple best model checkpoints found")
                elif len(checkpoint_files) == 0:
                    raise ValueError("No best model checkpoint found")

                best_checkpoint_path = os.path.join(save_dir, checkpoint_files[0])
                print(f"Loading best model from {best_checkpoint_path}")

                # Load the best model checkpoint
                load_model_checkpoint(
                    parallel_model, best_checkpoint_path, rank, strict=True
                )
        else:
            print(
                "[WARNING] The best model checkpoint is not found, using final model for test evaluation"
            )
            print(
                "[WARNING] The valid metrics are not corresponding to the test metrics"
            )

        _, _, test_loader = build_loaders(
            epoch=0,
            tokenizer=tokenizer,
            configs=configs,
            collator=collator,
            test_dataset_raw=test_dataset_raw,
            use_features=use_features,
            encoding_scheme=encoding_scheme,
            is_gnn=is_gnn,
        )

        test_out = evaluate_loop(
            parallel_model,
            test_loader,
            rank=rank,
            world_size=world_size,
            evaluator=evaluator,
            dataset_len=len(test_dataset_raw),
            split_name="test",
            show_progress=True,
            compute_loss=False,
            check_sequential_indices=True,
            is_gnn=is_gnn,
            use_bf16=configs.get("bf16", False),
        )
        best_test_metrics = test_out["metrics"] if rank == 0 else None

        print(f"Best validation metrics: {best_val_metrics}")
        print(f"Best test metrics: {best_test_metrics}")

    # Finish wandb run
    if wandb_run and rank == 0:
        # Save best metrics to JSON file
        best_metrics_dict = {}
        if best_val_metrics is not None:
            for metric_name, metric_value in best_val_metrics.items():
                best_metrics_dict[f"valid/{metric_name}"] = metric_value
        if best_test_metrics is not None:
            for metric_name, metric_value in best_test_metrics.items():
                best_metrics_dict[f"test/{metric_name}"] = metric_value
        if best_metrics_dict:
            best_metrics_file = os.path.join(save_dir, "best_metrics.json")
            with open(best_metrics_file, "a") as f:
                f.write(json.dumps(best_metrics_dict) + "\n")

        # Create and log final table with all metrics in separate columns
        if best_test_metrics is not None:
            # Pass empty dict for validation metrics if skip_val was used
            val_metrics_for_table = (
                best_val_metrics if best_val_metrics is not None else {}
            )
            log_final_table(
                wandb_run, configs, val_metrics_for_table, best_test_metrics
            )
        else:
            print("No test metrics found")
        wandb_run.finish()

    # Clean up distributed environment after training completion
    if dist.is_initialized():
        dist.destroy_process_group()

    return configs.name


if __name__ == "__main__":
    name = main()
    builtins.print(name)

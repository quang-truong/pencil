# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import gc
import random, torch, os
import numpy as np
import torch.distributed as dist
import inspect
import warnings
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
import wandb

# Metric -> whether higher is better (True) or lower is better (False)
# This allows the same comparison logic to work for both cases
METRIC_HIGHER_IS_BETTER = {
    "mrr": True,
    "hits@1": True,
    "hits@3": True,
    "hits@10": True,
    "hits@20": True,
    "hits@50": True,
    "hits@100": True,
    # Add metrics where lower is better as False, e.g.:
    "rmse": False,
    "mae": False,
    "mse": False,
}


def str_to_bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def normalize_score(score: float, metric_key: str) -> float:
    """
    Normalize a score so that we always maximize it.

    For metrics where higher is better, returns the score as-is.
    For metrics where lower is better, returns the negated score.

    Args:
        score: Raw metric score
        metric_key: Name of the metric (e.g., "mrr", "loss", "hits@50")

    Returns:
        Normalized score (always maximize)
    """
    higher_is_better = METRIC_HIGHER_IS_BETTER.get(metric_key, True)
    if higher_is_better:
        return score
    else:
        return -score


def suppress_warnings():
    """Suppress common warnings for cleaner output during training."""
    # Suppress Python warnings
    # warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Suppress PyTorch warnings
    # warnings.filterwarnings("ignore", message=".*FSDP.*")
    # warnings.filterwarnings("ignore", message=".*DDP.*")
    # warnings.filterwarnings("ignore", message=".*distributed.*")
    # warnings.filterwarnings("ignore", message=".*CUDA.*")

    # Suppress transformers warnings
    try:
        from transformers import logging

        logging.set_verbosity_error()
    except ImportError:
        pass

    # Set PyTorch backend settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # to access a dict with object.key
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Recursively convert nested dictionaries to Config objects
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key, default=None):
        """Get a value from the config, similar to dict.get()"""
        return getattr(self, key, default)


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set CUDA seeds for GPU reproducibility (important for multi-GPU setups)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


def seed_worker(worker_id: int, base_seed: int):
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Set CUDA seeds for GPU reproducibility in data loading workers
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def compute_params(model, print_params=False):
    num_trainable_params = -1
    num_total_params = -1
    if get_rank() == 0:
        for name, child in model.named_children():
            tmp_trainable_params = 0
            tmp_total_params = 0
            for param in child.parameters():
                if param.requires_grad:
                    tmp_trainable_params += param.numel()
                tmp_total_params += param.numel()
            print(f"=================== {name} ========================")
            print(f"Trainable params: {tmp_trainable_params}")
            print(f"Total params    : {tmp_total_params}")

    if get_rank() == 0:
        if print_params:
            print("==================== Model Parameters =======================")
        trainable_params = 0
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if print_params:
                    print(name, param.size())
                trainable_params += param.numel()
            total_params += param.numel()
        print("=================== Params stats ========================")
        print(f"Trainable params: {trainable_params}")
        print(f"Total params    : {total_params}")
        print("=========================================================")
        num_trainable_params = trainable_params
        num_total_params = total_params
    return num_trainable_params, num_total_params


def rank_zero_print(*args, print_details=True, **kwargs):
    """Custom print function that only prints if the current process is rank zero."""
    if get_rank() == 0:
        if print_details:
            # Get the name of the file where rank_zero_print was called
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            full_file_path = caller_frame.f_code.co_filename
            file_name = os.path.relpath(
                full_file_path
            )  # Make file path relative to current directory
            line_number = caller_frame.f_lineno  # Get line number
            function_name = caller_frame.f_code.co_name  # Get function name

            # Format the message with datetime, filename, line number, and function name
            prefix = f"[{file_name}:{line_number}][{function_name}] -"
        else:
            prefix = ""
        print(prefix, *args, **kwargs)


def concat_all_gather_1d(
    local_tensor: torch.Tensor, dataset_len: int = -1
) -> torch.Tensor:
    """
    Distributed function to gather and concatenate 1D tensors from all processes.

    This function performs an all-gather operation where each process contributes
    a 1D tensor of potentially different lengths, and all tensors are gathered,
    transposed, and returned in sequential order to each process.

    Args:
        local_tensor: 1D tensor from the current process to be gathered
        dataset_len: Length of the dataset, used to trim the gathered tensor
            if -1, no trimming is performed.

    Returns:
        Concatenated tensor containing data from all processes in sequential order
    """
    # Ensure input is 1-dimensional
    assert local_tensor.dim() == 1

    # Get distributed training configuration
    world_size = dist.get_world_size()  # Number of processes in the distributed group
    device = local_tensor.device  # Device where the tensor is located
    dtype = local_tensor.dtype  # Data type of the tensor

    # Step 1: Gather the size of each process's tensor
    # Create a tensor containing the length of the local tensor
    local_len = torch.tensor([local_tensor.numel()], device=device, dtype=torch.int64)

    # Create a list to hold the lengths from all processes
    sizes = [torch.zeros_like(local_len) for _ in range(world_size)]

    # Gather all tensor lengths across processes
    dist.all_gather(sizes, local_len)

    # Convert to Python integers for easier manipulation
    sizes_list = [int(s.item()) for s in sizes]

    # Find the maximum length across all processes
    max_len = max(sizes_list)

    # Step 2: Pad the local tensor to match the maximum length
    # This is necessary because all_gather requires all tensors to have the same size
    if local_tensor.numel() < max_len:
        # Create padding tensor to fill the difference
        pad = torch.zeros(max_len - local_tensor.numel(), device=device, dtype=dtype)
        # Concatenate the local tensor with padding
        padded = torch.cat([local_tensor, pad], dim=0)
    else:
        # No padding needed if local tensor is already at maximum length
        padded = local_tensor

    # Step 3: Gather all padded tensors from all processes to rank 0 only
    rank = get_rank()
    if rank == 0:
        # Create a list to hold gathered tensors from all processes (only on rank 0)
        gathered = [
            torch.zeros(max_len, device=device, dtype=dtype) for _ in range(world_size)
        ]
    else:
        gathered = None

    # Perform the gather operation (only rank 0 receives the data)
    dist.gather(padded, gathered, dst=0)

    # Step 4-5: Post-processing operations only on rank 0
    if rank == 0:
        # Step 4: Stack and transpose to get sequential order (following inference_entity pattern)
        # Stack the gathered predictions: Shape (num_processes, max_len)
        stacked_preds = torch.stack(gathered)

        # Transpose to get sequential order: Shape (max_len, num_processes)
        stacked_preds = stacked_preds.transpose(0, 1)

        # Flatten to get the final sequential tensor
        flattened_preds = stacked_preds.flatten(0, 1)

        # Step 5: Remove padding by calculating total valid elements
        # Calculate total valid elements across all processes
        if dataset_len == -1:
            total_valid = sum(sizes_list)
        else:
            assert dataset_len <= sum(
                sizes_list
            ), "Dataset length is greater than the total length of the gathered tensors"
            total_valid = dataset_len

        # Return only the valid predictions (remove padding)
        return flattened_preds[:total_valid]
    else:
        # Return None on non-rank-0 processes
        return None


def save_checkpoint(
    parallel_model,
    optimizer,
    save_path,
    rank,
    model_type="checkpoint",
    scheduler=None,
    epoch=None,
    step=None,
    total_trained_samples=None,
    best_score=None,
):
    """
    Save FSDP model, optimizer, and optionally scheduler state.

    Args:
        parallel_model: FSDP-wrapped model
        optimizer: Optimizer instance
        save_path: Path to save the checkpoint
        rank: Current process rank
        model_type: Type of model being saved (for logging)
        scheduler: Optional learning rate scheduler
        epoch: Current epoch number
        step: Current step number
        total_trained_samples: Total number of training samples processed so far
    """
    checkpoint = {}

    # Save model state
    if isinstance(parallel_model, FSDP):
        # Use the deprecated but working FSDP state dict API
        rank_zero_print(f"Saving FSDP model state dict to {save_path}")
        full_state_dict_config = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )

        with FSDP.state_dict_type(
            parallel_model, StateDictType.FULL_STATE_DICT, full_state_dict_config
        ):
            # Gather the full state dictionary from all ranks
            checkpoint["model_state_dict"] = parallel_model.state_dict()
        # Get the full optimizer state dictionary from all ranks
        checkpoint["optimizer_state_dict"] = FSDP.full_optim_state_dict(
            parallel_model, optimizer
        )

    else:
        # For DDP models, use regular state_dict
        rank_zero_print(f"Saving DDP model state dict to {save_path}")
        checkpoint["model_state_dict"] = parallel_model.state_dict()
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    # Save scheduler state if provided
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Save training metadata
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if step is not None:
        checkpoint["step"] = step
    if total_trained_samples is not None:
        checkpoint["total_trained_samples"] = total_trained_samples
    if best_score is not None:
        checkpoint["best_score"] = best_score

    # Save only on rank 0
    if rank == 0:
        try:
            torch.save(checkpoint, save_path)
            rank_zero_print(f"Successfully saved {model_type} to {save_path}")
            if epoch is not None:
                rank_zero_print(f"  - Epoch: {epoch}")
            if step is not None:
                rank_zero_print(f"  - Step: {step}")
            if total_trained_samples is not None:
                rank_zero_print(f"  - Total trained samples: {total_trained_samples}")
        except Exception as e:
            rank_zero_print(f"Error saving checkpoint: {e}")

    # Synchronize all processes after saving checkpoint
    torch.distributed.barrier(device_ids=[rank])
    gc.collect()
    torch.cuda.empty_cache()


def load_model_checkpoint(model, checkpoint_path, rank, strict=True):
    """
    Load weights from a checkpoint to a model.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to the checkpoint
        rank: Current process rank
        strict: Whether to use strict mode for loading model weights
    Returns:
        metadata: Dictionary containing metadata from the checkpoint
            - epoch: Epoch number
            - step: Step number
            - total_trained_samples: Total number of training samples processed so far
    """
    # Load just model weights (for pre-trained models)
    rank_zero_print(f"Loading model weights from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")

    # Check if this is a full checkpoint or just model weights
    if "model_state_dict" in state:
        # This is a full checkpoint, extract just the model weights
        model_state = state["model_state_dict"]
        rank_zero_print("Detected full checkpoint, extracting model_state_dict")
    else:
        # This is just model weights
        model_state = state
        rank_zero_print("Detected model weights only")

    # Detect if model is FSDP or DDP wrapped
    is_fsdp = isinstance(model, FSDP)
    is_ddp = hasattr(model, "module") and not is_fsdp

    # Handle different model types
    if is_fsdp:
        # For FSDP models, use FULL_STATE_DICT context to load the full state dict
        rank_zero_print("Loading into FSDP model using FULL_STATE_DICT context")
        full_state_dict_config = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=False
        )
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, full_state_dict_config
        ):
            rank_zero_print(model.load_state_dict(model_state, strict=strict))
    elif is_ddp:
        # For DDP models, load into the underlying module
        rank_zero_print("Loading into DDP model")
        rank_zero_print(model.load_state_dict(model_state, strict=strict))
    else:
        # Regular model (not wrapped)
        # Strip 'module.' prefix if present (from DDP/FSDP wrapping)
        if any(key.startswith("module.") for key in model_state.keys()):
            rank_zero_print("Stripping 'module.' prefix from checkpoint keys")
            model_state = {
                key.replace("module.", ""): value for key, value in model_state.items()
            }
        rank_zero_print("Loading model weights into model")
        rank_zero_print(model.load_state_dict(model_state, strict=strict))

    rank_zero_print(f"Successfully loaded model weights from {checkpoint_path}")

    # Synchronize all processes after loading model weights
    torch.distributed.barrier(device_ids=[rank])
    gc.collect()
    torch.cuda.empty_cache()

    # Return metadata
    metadata = {}
    if "epoch" in state:
        metadata["epoch"] = state["epoch"]
    if "step" in state:
        metadata["step"] = state["step"]
    if "total_trained_samples" in state:
        metadata["total_trained_samples"] = state["total_trained_samples"]
    if "best_score" in state:
        metadata["best_score"] = state["best_score"]

    if rank == 0:
        if "epoch" in metadata:
            rank_zero_print(f"  - Epoch: {metadata['epoch']}")
        if "step" in metadata:
            rank_zero_print(f"  - Step: {metadata['step']}")
        if "total_trained_samples" in metadata:
            rank_zero_print(
                f"  - Total trained samples: {metadata['total_trained_samples']}"
            )
        if "best_score" in metadata:
            rank_zero_print(f"  - Best score: {metadata['best_score']}")
    return metadata


def load_optimizer_checkpoint(optimizer, model, checkpoint_path, rank):
    """
    Load optimizer state from a checkpoint.

    Need to load the checkpoint first, then wrap the model with FSDP or DDP,
    then initialize the optimizer with the wrapped model parameters, and finally
    load the optimizer state from the checkpoint.

    Args:
        optimizer: Optimizer to load state into
        model: a parallel model wrapped with FSDP or DDP
        checkpoint_path: Path to the checkpoint
        rank: Current process rank

    Returns:
        None
    """
    # Load optimizer state directly
    rank_zero_print(f"Loading optimizer state dict from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    if "optimizer_state_dict" in state:
        rank_zero_print("Detected optimizer state dict")
        if isinstance(model, FSDP):
            full_optimizer_state_dict = (
                None  # will be scattered from rank 0 to all ranks
            )
            if rank == 0:
                full_optimizer_state_dict = state["optimizer_state_dict"]
        else:
            # every rank has the full optimizer state dict for DDP
            full_optimizer_state_dict = state["optimizer_state_dict"]
    else:
        raise ValueError("No optimizer state dict found")

    if isinstance(model, FSDP):
        sharded_optimizer_state_dict = FSDP.scatter_full_optim_state_dict(
            full_optimizer_state_dict, model
        )
        rank_zero_print(optimizer.load_state_dict(sharded_optimizer_state_dict))
    else:
        rank_zero_print(optimizer.load_state_dict(full_optimizer_state_dict))
    rank_zero_print(f"Successfully loaded optimizer state dict from {checkpoint_path}")

    # Synchronize all processes after loading optimizer state
    torch.distributed.barrier(device_ids=[rank])


def log_final_table(wandb_run, configs, val_metrics, test_metrics):
    """
    Create and log a final evaluation table with all metrics in separate columns.

    Args:
        wandb_run: Wandb run object for logging
        configs: Configuration object containing run name
        val_metrics: Dictionary of validation metrics
        test_metrics: Dictionary of test metrics
        rank: Current process rank
    """
    if wandb_run and val_metrics is not None and test_metrics is not None:
        columns = ["run_name"]
        row_data = [configs.name]

        # Add validation metrics columns
        for metric_name in sorted(val_metrics.keys()):
            columns.append(f"val/{metric_name}")
            row_data.append(val_metrics[metric_name])

        # Add test metrics columns
        for metric_name in sorted(test_metrics.keys()):
            columns.append(f"test/{metric_name}")
            row_data.append(test_metrics[metric_name])

        table = wandb.Table(data=[row_data], columns=columns)
        wandb_run.log({"Final Evaluation": table})


def is_gnn_model(model_type: str) -> bool:
    """
    Check if the model type is a GNN model.

    Args:
        model_type: Model type string (e.g., "gcn", "gat", "sage", "gpt2")

    Returns:
        bool: True if the model is a GNN model, False otherwise
    """
    return model_type in ["gcn", "gat", "sage"]


def get_feature_dim(configs, train_dataset_raw, encoding_scheme=None):
    """
    Determine the feature dimension based on model type, dataset, and encoding scheme.

    Args:
        configs: Configuration object containing model and dataset settings
        train_dataset_raw: Training dataset to extract feature dimensions from
        encoding_scheme: Encoding scheme for transformer models ('full', 'adjacency_row', 'edge_list')

    Returns:
        tuple: (feature_dim, is_gnn) where feature_dim is the feature dimension
               (or None if not applicable) and is_gnn indicates if the model is a GNN
    """
    # Detect model type to determine if it's a GNN model
    full_model_name = configs.model.get("name")
    if "/" in full_model_name:
        model_type = full_model_name.split("/")[-1].lower()
    else:
        model_type = full_model_name.lower()
    is_gnn = is_gnn_model(model_type)

    # Extract feature_dim - GNN models always need features (node features)
    use_features = configs.get("use_features", False)
    feature_dim = None

    # Get feature dimension from dataset
    if train_dataset_raw is not None and len(train_dataset_raw) > 0:
        _, sample_data = train_dataset_raw[0]
        if hasattr(sample_data, "x") and sample_data.x is not None:
            node_feature_dim = sample_data.x.size(1)
            if is_gnn:
                # For GNN models, feature_dim is the node feature dimension
                feature_dim = node_feature_dim
                rank_zero_print(f"GNN model - Feature dimension: {feature_dim}")
            else:
                # For transformer models, feature_dim depends on encoding scheme
                if use_features:
                    if encoding_scheme == "adjacency_row":
                        # For adjacency_row, each token uses node_feature_dim (not 2*node_feature_dim)
                        feature_dim = node_feature_dim
                        rank_zero_print(
                            f"Transformer model (adjacency_row) - Feature dimension: {feature_dim}"
                        )
                    else:
                        # For full/edge_list, feature_dim is 2 * node_feature_dim (src+dst concatenated)
                        feature_dim = 2 * node_feature_dim
                        rank_zero_print(
                            f"Transformer model ({encoding_scheme}) - Feature dimension: {feature_dim} (2 * {node_feature_dim})"
                        )
        else:
            if is_gnn:
                raise ValueError(
                    f"GNN models require node features. Dataset {configs.dataset} does not have node features (data.x)"
                )
            elif use_features:
                raise ValueError(
                    f"Dataset {configs.dataset} does not have node features (data.x)"
                )
    else:
        if is_gnn:
            raise ValueError(
                "Cannot determine feature dimension: train dataset is empty or not provided"
            )
        elif use_features:
            raise ValueError(
                "Cannot determine feature dimension: train dataset is empty or not provided"
            )

    return feature_dim, is_gnn


def generate_name(config) -> str:
    """Generate a run name from a YAML config file.

    Rules (applied in order):
    1) base = configs.name
    2) prepend dataset: "{dataset}-{base}"
    3) if configs.only_eval: prepend "eval-"
    4) if sampling_config.pretrain_mode: append "-pretrain"
    4.5) if use_features: append "-use_features"
    5) append depth_neighbors (e.g., [[2, 8]] -> "-2,8")
    6) append sampling_config.method.name
    7) append model.hidden_size as "-{hidden_size}d"
    7.5) append model.intermediate_size as "-{intermediate_size}i"
    8) append model.num_hidden_layers as "-{num_hidden_layers}l"
    9) append model.num_attention_heads as "-{num_attention_heads}h"
    10) append batch_size_training as "-bs{batch_size_training}"
    11) append gradient_accumulation_steps as "-ga{gradient_accumulation_steps}"
    12) append lr as "-lr{lr}"
    13) append note as "-{note}"
    14) append seed as "-seed{seed}"
    """

    base_name = config["name"]
    dataset = config["dataset"]
    if "encoding_scheme" in config:
        encoding_scheme = config["encoding_scheme"]
        name = f"{dataset}-{encoding_scheme}-{base_name}"
    else:
        name = f"{dataset}-{base_name}"

    # Step 3: only_eval prefix
    if config.get("only_eval", False):
        name = f"eval-{name}"

    # Step 4: pretrain_mode suffix
    pretrain_mode = config.get("sampling_config", {}).get("pretrain_mode", False)
    if pretrain_mode:
        name = f"{name}-pretrain"

    # Step 4.5: use_features suffix
    use_features = config.get("use_features", False)
    if use_features:
        name = f"{name}-use_features"
        feature_fusion = config.get("feature_fusion", None)
        if feature_fusion is not None:
            name = f"{name}-{feature_fusion}"

    # Step 5: depth_neighbors (list like [[2, 8]])
    depth_neighbors = config["sampling_config"]["edge_ego"]["depth_neighbors"][0]
    depth_neighbors_str = f"{depth_neighbors[0]},{depth_neighbors[1]}"
    name = f"{name}-{depth_neighbors_str}"

    # Step 6: sampling method name
    method_name = config["sampling_config"]["edge_ego"]["method"]["name"]
    name = f"{name}-{method_name}"

    # Model fields
    model_cfg = config["model"]

    # Step 7: hidden_size
    hidden_size = model_cfg["hidden_size"]
    name = f"{name}-{hidden_size}d"

    # Step 7.5: intermediate_size
    if "intermediate_size" in model_cfg:
        intermediate_size = model_cfg["intermediate_size"]
        name = f"{name}-{intermediate_size}i"

    # Step 8: num_hidden_layers
    if "num_hidden_layers" in model_cfg:
        num_hidden_layers = model_cfg["num_hidden_layers"]
        name = f"{name}-{num_hidden_layers}l"

    # Step 9: num_attention_heads
    if "num_attention_heads" in model_cfg:
        num_attention_heads = model_cfg["num_attention_heads"]
        name = f"{name}-{num_attention_heads}h"

    # GNN's fields
    if "num_layers" in model_cfg:
        num_layers = model_cfg["num_layers"]
        name = f"{name}-{num_layers}l"
    if "num_layers_predictor" in model_cfg:
        num_layers_predictor = model_cfg["num_layers_predictor"]
        name = f"{name}-{num_layers_predictor}p"
    if "dropout" in model_cfg:
        dropout = model_cfg["dropout"]
        name = f"{name}-drop{dropout}"
    if "residual" in model_cfg:
        residual = model_cfg["residual"]
        if residual:
            name = f"{name}-residual"
    if "id_awareness" in model_cfg:
        id_awareness = model_cfg["id_awareness"]
        if id_awareness:
            name = f"{name}-nbfnet"
    if "ortho_embedding" in model_cfg:
        ortho_embedding = model_cfg["ortho_embedding"]
        if ortho_embedding:
            name = f"{name}-ortho"

    # Step 10: batch_size_training
    batch_size_training = config["batch_size_training"]
    name = f"{name}-bs{batch_size_training}"

    # Step 11: gradient_accumulation_steps
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    name = f"{name}-ga{gradient_accumulation_steps}"

    # Step 12: lr
    lr = config["lr"]
    name = f"{name}-lr{lr}"

    # Step 13: note
    note = config["note"]
    if note:
        name = f"{name}-{note}"

    # Step 14: seed
    seed = config["seed"]
    name = f"{name}-seed{seed}"

    return name

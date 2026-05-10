import numpy as np
import torch
from typing import Any, Dict

from sklearn import metrics as met

from ogb.linkproppred import Evaluator as OGBLinkEvaluator
from typing import List


class CustomOGBLinkEvaluator(OGBLinkEvaluator):
    def __init__(self, name: str, k_list: List[int] = [20, 50, 100]):
        super().__init__(name)
        self.k_list = k_list

    def _eval_mrr(self, y_pred_pos, y_pred_neg, type_info):
        """
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
        """

        if type_info == "torch":
            # calculate ranks
            y_pred_pos = y_pred_pos.view(-1, 1)
            # optimistic rank: "how many negatives have a larger score than the positive?"
            # ~> the positive is ranked first among those with equal score
            optimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
            # pessimistic rank: "how many negatives have at least the positive score?"
            # ~> the positive is ranked last among those with equal score
            pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
            ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
            hits1_list = (ranking_list <= 1).to(torch.float)
            hits3_list = (ranking_list <= 3).to(torch.float)
            hits10_list = (ranking_list <= 10).to(torch.float)
            hits20_list = (ranking_list <= 20).to(torch.float)
            hits50_list = (ranking_list <= 50).to(torch.float)
            hits100_list = (ranking_list <= 100).to(torch.float)
            mrr_list = 1.0 / ranking_list.to(torch.float)

            return {
                "hits@1_list": hits1_list,
                "hits@3_list": hits3_list,
                "hits@10_list": hits10_list,
                "hits@20_list": hits20_list,
                "hits@50_list": hits50_list,
                "hits@100_list": hits100_list,
                "mrr_list": mrr_list,
            }

        else:
            y_pred_pos = y_pred_pos.reshape(-1, 1)
            optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
            pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
            ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
            hits1_list = (ranking_list <= 1).astype(np.float32)
            hits3_list = (ranking_list <= 3).astype(np.float32)
            hits10_list = (ranking_list <= 10).astype(np.float32)
            hits20_list = (ranking_list <= 20).astype(np.float32)
            hits50_list = (ranking_list <= 50).astype(np.float32)
            hits100_list = (ranking_list <= 100).astype(np.float32)
            mrr_list = 1.0 / ranking_list.astype(np.float32)

            return {
                "hits@1_list": hits1_list,
                "hits@3_list": hits3_list,
                "hits@10_list": hits10_list,
                "hits@20_list": hits20_list,
                "hits@50_list": hits50_list,
                "hits@100_list": hits100_list,
                "mrr_list": mrr_list,
            }

    def _eval_hits(self, y_pred_pos, y_pred_neg, type_info):
        """
        compute Hits@K for multiple K values
        For each positive target node, the negative target nodes are the same.

        y_pred_neg is an array.
        rank y_pred_pos[i] against y_pred_neg for each i
        """

        # If k_list is not provided, use the default self.K
        if self.k_list is None:
            k_list = [self.K]

        k_list = self.k_list
        result = {}

        for K in k_list:
            if len(y_pred_neg) < K:
                result["hits@{}".format(K)] = 1.0
                continue

            if type_info == "torch":
                kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
                hitsK = float(
                    torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()
                ) / len(y_pred_pos)

            # type_info is numpy
            else:
                kth_score_in_negative_edges = np.sort(y_pred_neg)[-K]
                hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(
                    y_pred_pos
                )

            result["hits@{}".format(K)] = hitsK

        return result


class Evaluator(object):
    """Evaluate predictions for various tasks with a consistent API.

    Args:
        metric: Name of the metric or OGB benchmark (e.g., "accuracy", "ogbl-ppa").
        **kwargs: Optional overrides or dataset-specific parameters (e.g., cnt_neg).
    """

    def __init__(self, metric: str, **kwargs: Any) -> None:
        self.metric = metric

        if metric.startswith("ogbl-"):
            self._ogb_link_evaluator = CustomOGBLinkEvaluator(name=metric)
            self.eval_fn = self._ogb_link
            # Optional specialization parameters (used by citation2-style MRR formatting)
            self._cnt_neg = int(kwargs.get("cnt_neg", 1000))
        elif metric in ("cora", "citeseer", "pubmed"):
            self._ogb_link_evaluator = CustomOGBLinkEvaluator(
                name="ogbl-citation2"
            )  # MRR
            self.eval_fn = self._ogb_link
        # Heuristics datasets
        elif metric.startswith(
            ("cn-", "aa-", "ra-", "shortest-path-", "katz-", "pagerank-")
        ):
            self.eval_fn = self._regression
        # Heart splits
        elif metric.startswith("heart-"):
            self._ogb_link_evaluator = CustomOGBLinkEvaluator(name="ogbl-citation2")
            self.eval_fn = self._ogb_link
            self._cnt_neg = int(kwargs.get("cnt_neg", 500))
        else:
            raise NotImplementedError(f"Metric {metric} is not yet supported.")

    def eval(self, input_dict: Dict[str, Any]) -> Any:
        """
        Evaluate the predictions for the given input dictionary.

        Args:
            input_dict: A dictionary containing the predictions and ground truth labels.
                For heuristics datasets, must contain:
                    - y_pred: The predictions (continuous values).
                    - y_true: The ground truth labels (continuous values).
        Returns:
            The evaluation result.
        """
        return self.eval_fn(input_dict)

    def _ogb_link(self, input_dict: Dict[str, Any], **kwargs: Any) -> Dict[str, float]:
        """Evaluate OGB link prediction metrics.

        Accepts flexible inputs and normalizes them to the
        `{ "y_pred_pos": <Tensor>, "y_pred_neg": <Tensor> }` structure.
        For datasets like `ogbl-citation2`, average list/tensor outputs to scalars.
        """
        # Accept a flexible input format and convert to what OGB expects.
        # Preferred: input contains tensors for 'y_pred_pos' and 'y_pred_neg'.
        ogb_input = self._prepare_input(input_dict)
        result = self._ogb_link_evaluator.eval(ogb_input)

        # Some datasets (e.g., citation2) return lists/tensors of per-head values.
        # Average them and strip the trailing "_list" from the metric key.
        summarized: Dict[str, float] = {}
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                mean_value = torch.mean(v.float()).item()
                base_key = k[:-5] if k.endswith("_list") else k
                summarized[base_key] = float(mean_value)
            elif (
                isinstance(v, (list, tuple))
                and len(v) > 0
                and isinstance(v[0], (float, int))
            ):
                mean_value = float(np.mean(v))
                base_key = k[:-5] if k.endswith("_list") else k
                summarized[base_key] = mean_value
            else:
                try:
                    summarized[k] = float(v)  # type: ignore[arg-type]
                except Exception:
                    raise ValueError(f"Could not convert {k} to float")
        return summarized

    def _prepare_input(self, input_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Normalize multiple possible input formats to OGB's expected dict.

        Supported input formats:
          1) Direct: `{"y_pred_pos": Tensor, "y_pred_neg": Tensor}`
          2) Tuple/list: `{"y_pred": (pos, neg)}`
          3) Dict: `{"y_pred": {"pos": pos, "neg": neg}}`
          4) Flat scores + labels: `{"y_true": ..., "y_pred": ...}` with
             dataset-specific reformat (HR for ppa/collab, MRR for citation2).
        """
        # # Already in desired format
        # if "y_pred_pos" in input_dict and "y_pred_neg" in input_dict:
        #     y_pred_pos = self._to_tensor(input_dict["y_pred_pos"])  # type: ignore[index]
        #     y_pred_neg = self._to_tensor(input_dict["y_pred_neg"])  # type: ignore[index]
        #     return {"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg}

        # # Common alternates: 'y_pred' contains (pos, neg) or {'pos': ..., 'neg': ...}
        # if "y_pred" in input_dict and input_dict["y_pred"] is not None:
        #     y_pred = input_dict["y_pred"]
        #     if isinstance(y_pred, (tuple, list)) and len(y_pred) == 2:
        #         y_pred_pos, y_pred_neg = y_pred
        #         return {
        #             "y_pred_pos": self._to_tensor(y_pred_pos),
        #             "y_pred_neg": self._to_tensor(y_pred_neg),
        #         }
        #     if isinstance(y_pred, dict) and "pos" in y_pred and "neg" in y_pred:
        #         return {
        #             "y_pred_pos": self._to_tensor(y_pred["pos"]),  # type: ignore[index]
        #             "y_pred_neg": self._to_tensor(y_pred["neg"]),  # type: ignore[index]
        #         }

        # If y_true/y_pred given in flat form, try dataset-specific reformats
        if "y_true" in input_dict and "y_pred" in input_dict:
            if self.metric in ("ogbl-ppa", "ogbl-collab", "ogbl-ddi"):
                y_pred_pos, y_pred_neg = self._reformat_pred(input_dict)
                return {"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg}
            elif self.metric in ("cora", "citeseer", "pubmed"):
                y_pred_pos, y_pred_neg = self._reformat_pred_for_planetoid(input_dict)
                return {"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg}
            elif self.metric in ("ogbl-citation2") or self.metric.startswith("heart-"):
                y_pred_pos, y_pred_neg = self._reformat_pred_for_citation2(
                    input_dict, cnt_neg=self._cnt_neg
                )
                return {"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg}

        raise ValueError(
            f"For {self.metric} metrics, provide y_pred and y_true in the input dictionary."
        )

    @staticmethod
    def _reformat_pred(input_dict: Dict[str, Any]) -> (torch.Tensor, torch.Tensor):
        """Split flat scores into positive and negative sets using boolean labels.

        Used by datasets that report hit-rate style metrics (e.g., ogbl-ppa, ogbl-collab).

        Args:
            input_dict: A dictionary containing the predictions and ground truth labels.
            It must contain:
                - y_pred: The predictions.
                - y_true: The ground truth labels.
        """
        y_true = input_dict["y_true"]
        y_pred = input_dict["y_pred"]
        # Convert inputs to tensors if needed to enable boolean masking.
        y_true_t = (
            y_true if isinstance(y_true, torch.Tensor) else torch.as_tensor(y_true)
        )
        y_pred_t = (
            y_pred if isinstance(y_pred, torch.Tensor) else torch.as_tensor(y_pred)
        )

        # Positive mask: True where label indicates a positive edge
        pos_mask = y_true_t.bool()
        y_pred_pos = y_pred_t[pos_mask]

        # Negative mask: all non-positive entries
        # If labels are bool, use logical not; if 0/1 ints, convert to bool negation.
        neg_mask = (
            (~pos_mask) if y_true_t.dtype == torch.bool else (1 - y_true_t).bool()
        )
        y_pred_neg = y_pred_t[neg_mask]

        return y_pred_pos, y_pred_neg

    @staticmethod
    def _reformat_pred_for_planetoid(
        input_dict: Dict[str, Any],
    ) -> (torch.Tensor, torch.Tensor):
        """Split flat scores into positive and negative sets using boolean labels.

        Used by datasets that report MRR style metrics (e.g., cora, citeseer, pubmed).

        Args:
            input_dict: A dictionary containing the predictions and ground truth labels.
            It must contain:
                - y_pred: The predictions.
                - y_true: The ground truth labels.
            metric: The metric name to determine if special handling is needed.
        """
        y_true = input_dict["y_true"]
        y_pred = input_dict["y_pred"]
        # Convert inputs to tensors if needed to enable boolean masking.
        y_true_t = (
            y_true if isinstance(y_true, torch.Tensor) else torch.as_tensor(y_true)
        )
        y_pred_t = (
            y_pred if isinstance(y_pred, torch.Tensor) else torch.as_tensor(y_pred)
        )

        # Positive mask: True where label indicates a positive edge
        pos_mask = y_true_t.bool()
        y_pred_pos = y_pred_t[pos_mask]

        # Negative mask: all non-positive entries
        # If labels are bool, use logical not; if 0/1 ints, convert to bool negation.
        neg_mask = (
            (~pos_mask) if y_true_t.dtype == torch.bool else (1 - y_true_t).bool()
        )

        # Need to repeat negative predictions: (num_negatives,) -> (num_positives, num_negatives)
        y_pred_neg = y_pred_t[neg_mask].repeat(y_pred_pos.size(0), 1)

        return y_pred_pos, y_pred_neg

    @staticmethod
    def _reformat_pred_for_citation2(
        input_dict: Dict[str, Any], cnt_neg: int
    ) -> (torch.Tensor, torch.Tensor):
        """Build (pos, neg) scores in the citation2-style MRR layout.

        Validation/Test set is structured as
            [pos1, pos2, ..., posN, neg1_1, neg1_2, ..., neg1_cnt_neg, neg2_1, neg2_2, ..., neg2_cnt_neg, ..., negN_1, negN_2, ..., negN_cnt_neg]
        where N is the number of queries.
        Returns tensors shaped for OGB evaluator:
          - y_pred_pos: (N,)
          - y_pred_neg: (N, cnt_neg)

        Args:
            input_dict: A dictionary containing the predictions and ground truth labels.
            It must contain:
                - y_pred: The predictions.
                - y_true: The ground truth labels.
                - idx: The index of the sample.
        """
        y_true = input_dict["y_true"]
        y_pred = input_dict["y_pred"]
        idx = input_dict["idx"]
        # Sanity check: idx should range from 0 to num_queries-1 without gaps.
        assert idx.max().item() + 1 == len(idx), f"{idx.max().item()}+1 != {len(idx)}"

        pos_mask = y_true.bool()
        y_pred_pos = y_pred[pos_mask]  # (num_queries,)
        y_pred_neg = y_pred[~pos_mask]  # (num_queries * cnt_neg,)
        # Negatives are flattened; reshape to (num_queries, cnt_neg) as required by OGB.
        y_pred_neg = y_pred_neg.reshape((-1, cnt_neg))

        # Ensure each positive score has exactly one row of negatives.
        assert (
            y_pred_pos.shape[0] == y_pred_neg.shape[0]
        ), f"{y_pred_pos.shape[0]} != {y_pred_neg.shape[0]}"
        return y_pred_pos, y_pred_neg

    def _regression(self, input_dict: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate regression metrics (RMSE, MAE, MSE).

        Args:
            input_dict: A dictionary containing:
                - y_pred: The predictions (continuous values).
                - y_true: The ground truth labels (continuous values).

        Returns:
            A dictionary containing the computed metric value.
        """
        if "y_true" not in input_dict or "y_pred" not in input_dict:
            raise ValueError(
                f"For {self.metric} metric, input_dict must contain 'y_true' and 'y_pred'."
            )

        y_true = input_dict["y_true"]
        y_pred = input_dict["y_pred"]

        # Convert to numpy arrays for sklearn metrics
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        elif isinstance(y_true, np.ndarray):
            pass
        else:
            y_true = np.asarray(y_true)

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        elif isinstance(y_pred, np.ndarray):
            pass
        else:
            y_pred = np.asarray(y_pred)

        # Flatten arrays if needed
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Compute all regression metrics
        mse = met.mean_squared_error(y_true, y_pred)
        mae = met.mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        result = {"mse": float(mse), "mae": float(mae), "rmse": float(rmse)}

        return result

    @staticmethod
    def _to_tensor(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        if isinstance(x, (list, tuple)):
            return torch.tensor(x)
        # Attempt best-effort conversion
        return torch.as_tensor(x)

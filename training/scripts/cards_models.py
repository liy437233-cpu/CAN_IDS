"""Probe models and normal-traffic threshold calibration for CarDS."""
from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


FEATURE_DIM = 21


class CausalEncoder(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(features, 16, 3)
        self.c2 = nn.Conv1d(16, 16, 3)
        self.n1 = nn.GroupNorm(4, 16)
        self.n2 = nn.GroupNorm(4, 16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.transpose(1, 2)
        value = F.gelu(self.n1(self.c1(F.pad(value, (2, 0)))))
        return F.gelu(self.n2(self.c2(F.pad(value, (2, 0)))))[:, :, -1]


class EarlyFusionTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = CausalEncoder(FEATURE_DIM)
        self.head = nn.Linear(16, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(value)).squeeze(1)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_torch(
    x: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
    config: dict[str, Any],
    kind: str,
    seed: int,
    device: torch.device,
) -> nn.Module:
    del group
    if kind != "early_fusion_tcn":
        raise ValueError(f"unsupported torch probe: {kind}")
    seed_all(seed)
    model = EarlyFusionTCN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    order = np.arange(len(x))
    batch = int(config["training"]["batch_size"])
    for epoch in range(int(config["training"]["epochs"])):
        np.random.default_rng(seed + epoch).shuffle(order)
        model.train()
        for start in range(0, len(order), batch):
            index = order[start : start + batch]
            xb = torch.from_numpy(x[index]).to(device)
            yb = torch.from_numpy(y[index].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            score = model(xb)
            loss = F.binary_cross_entropy_with_logits(score, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
            optimizer.step()
    return model.eval()


def fit_sklearn(x: np.ndarray, y: np.ndarray, config: dict[str, Any], kind: str) -> Any:
    current = x[:, -1, :]
    if kind == "logistic_regression":
        return LogisticRegression(**config["comparators"]["logistic_regression"]).fit(current, y)
    if kind == "hgb":
        return HistGradientBoostingClassifier(**config["comparators"]["hgb"]).fit(current, y)
    raise ValueError(f"unsupported sklearn probe: {kind}")


@torch.inference_mode()
def torch_scores(model: nn.Module, x: np.ndarray, device: torch.device, batch: int = 8192) -> np.ndarray:
    values: list[np.ndarray] = []
    for start in range(0, len(x), batch):
        output = model(torch.from_numpy(x[start : start + batch]).to(device))
        values.append(torch.sigmoid(output).cpu().numpy())
    return np.concatenate(values)


def residual_max_scores(x: np.ndarray) -> np.ndarray:
    return np.max(np.abs(x[:, -1, :]), axis=1)


def alarm_count(scores: np.ndarray, timestamps: np.ndarray, threshold: float, cooldown: float) -> int:
    selected = timestamps[np.flatnonzero(scores >= threshold)]
    alarms = 0
    index = 0
    while index < len(selected):
        alarms += 1
        next_index = int(np.searchsorted(selected, selected[index] + cooldown, side="left"))
        index = max(index + 1, next_index)
    return alarms


def calibrate_threshold(
    normal_streams: list[tuple[np.ndarray, np.ndarray]],
    budget_fae_h: float,
    cooldown: float,
) -> dict[str, float]:
    duration_h = sum(max(0.0, float(ts[-1] - ts[0])) for _, ts in normal_streams if len(ts) > 1) / 3600
    if duration_h <= 0:
        raise RuntimeError("normal calibration duration is zero")
    candidates = np.unique(np.concatenate([scores for scores, _ in normal_streams]))
    left, right, selected = 0, len(candidates) - 1, None
    evaluations = 0
    while left <= right:
        middle = (left + right) // 2
        threshold = float(candidates[middle])
        events = sum(alarm_count(scores, ts, threshold, cooldown) for scores, ts in normal_streams)
        evaluations += 1
        if events / duration_h <= budget_fae_h:
            selected = (middle, threshold, events)
            right = middle - 1
        else:
            left = middle + 1
    if selected is None:
        threshold = float(np.nextafter(candidates.max(), math.inf))
        return {
            "threshold": threshold,
            "calibration_events": 0.0,
            "calibration_hours": duration_h,
            "calibration_fae_h": 0.0,
            "candidate_count": float(len(candidates)),
            "search_evaluations": float(evaluations),
            "selection_rule": "above_maximum_zero_alarm_fallback",
        }
    index, threshold, events = selected
    return {
        "threshold": threshold,
        "calibration_events": float(events),
        "calibration_hours": duration_h,
        "calibration_fae_h": events / duration_h,
        "candidate_count": float(len(candidates)),
        "search_evaluations": float(evaluations),
        "selection_rule": "lowest_empirical_threshold_with_fae_h_at_or_below_budget",
        "selected_candidate_index": float(index),
    }

"""CTAT training-row extraction with deterministic normal-frame sampling."""
from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
import pandas as pd

import ctat_features as features


class NormalReservoir:
    def __init__(self, cap: int, seed: int) -> None:
        self.cap = cap
        self.rng = np.random.default_rng(seed)
        self.priority = np.empty(0, dtype=np.float64)
        self.x = np.empty((0, len(features.RICH_NAMES)), dtype=np.float32)
        self.position = np.empty(0, dtype=np.int64)

    def update(self, x: np.ndarray, position: np.ndarray) -> None:
        if not len(x):
            return
        priority = np.concatenate([self.priority, self.rng.random(len(x))])
        values = np.concatenate([self.x, x])
        positions = np.concatenate([self.position, position])
        take = np.argpartition(priority, self.cap - 1)[: self.cap] if len(priority) > self.cap else np.arange(len(priority))
        self.priority = priority[take]
        self.x = values[take]
        self.position = positions[take]

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(self.position)
        return self.x[order], self.position[order]


def extract(zf: Any, row: dict[str, str], normal_cap: int = 20_000, chunksize: int = 100_000):
    member = row["zip_path"]
    if row["official_split"] != "train_01" or "/train_01/" not in member:
        raise AssertionError("training extraction accepts train_01 files only")
    info = zf.getinfo(member)
    if info.file_size != int(row["uncompressed_bytes"]) or f"{info.CRC:08X}" != row["crc32_hex"].upper():
        raise ValueError("archive and manifest do not match")
    seed = int(hashlib.sha256((member + ":P1:42").encode()).hexdigest()[:16], 16)
    normal = NormalReservoir(normal_cap, seed)
    attack_x: list[np.ndarray] = []
    attack_position: list[np.ndarray] = []
    state = features.CausalFeatures()
    counts = np.zeros(2, dtype=np.int64)
    offset = 0
    first_timestamp = None
    last_timestamp = None
    started = time.perf_counter()
    with zf.open(member) as source:
        reader = pd.read_csv(
            source,
            chunksize=chunksize,
            keep_default_na=False,
            dtype={"timestamp": "float64", "arbitration_id": "str", "data_field": "str", "attack": "int8"},
        )
        for frame in reader:
            y = frame["attack"].to_numpy(dtype=np.int8)
            if not np.isin(y, [0, 1]).all():
                raise ValueError("unexpected CTAT label")
            x = state.transform(frame.drop(columns="attack"))
            position = np.arange(offset, offset + len(y), dtype=np.int64)
            normal.update(x[y == 0], position[y == 0])
            if (y == 1).any():
                attack_x.append(x[y == 1])
                attack_position.append(position[y == 1])
            counts += np.bincount(y, minlength=2)
            if first_timestamp is None:
                first_timestamp = float(frame["timestamp"].iloc[0])
            last_timestamp = float(frame["timestamp"].iloc[-1])
            offset += len(y)
    normal_x, normal_position = normal.arrays()
    positives = np.concatenate(attack_x) if attack_x else np.empty((0, len(features.RICH_NAMES)), dtype=np.float32)
    positive_positions = np.concatenate(attack_position) if attack_position else np.empty(0, dtype=np.int64)
    x = np.concatenate([normal_x, positives])
    y = np.concatenate([np.zeros(len(normal_x), dtype=np.int8), np.ones(len(positives), dtype=np.int8)])
    position = np.concatenate([normal_position, positive_positions])
    order = np.argsort(position)
    x, y, position = x[order], y[order], position[order]
    if len(positives) != counts[1] or len(normal_x) != min(counts[0], normal_cap):
        raise RuntimeError("CTAT sampling contract failed")
    weights = np.where(y == 0, counts[0] / len(normal_x), 1.0).astype(np.float64)
    metadata = {
        **row,
        "total_rows": int(counts.sum()),
        "full_label_counts": counts.tolist(),
        "sample_rows": len(y),
        "sample_label_counts": np.bincount(y, minlength=2).tolist(),
        "normal_sample_probability": float(len(normal_x) / counts[0]),
        "attack_sample_probability": 1.0,
        "normal_frame_weight": float(counts[0] / len(normal_x)),
        "attack_frame_weight": 1.0,
        "duration_seconds": last_timestamp - first_timestamp,
        "backwards_timestamp_count": state.backwards_count,
        "maximum_backwards_seconds": state.maximum_backwards_seconds,
        "extraction_seconds": time.perf_counter() - started,
    }
    return x, y, weights, position, metadata

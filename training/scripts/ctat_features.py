"""Causal frame features used by the CTAT probes."""
from __future__ import annotations

import numpy as np
import pandas as pd


EXPECTED_SHA = "a9c607b38bd28f1768021ad01c29ffbfe4e82bb0ae5815ac3ce7ad74751ae061"
BYTE_NAMES = [f"byte_{index}" for index in range(8)]
CHEAP_NAMES = ["can_id", "dlc", *BYTE_NAMES]
RICH_NAMES = [
    *CHEAP_NAMES,
    "has_same_id_history",
    "log_global_iat_us",
    "log_same_id_iat_us",
    *[f"abs_delta_{index}" for index in range(8)],
    "hamming",
]
NO_CLOCK_COLUMNS = [
    index for index, name in enumerate(RICH_NAMES)
    if name not in ["log_global_iat_us", "log_same_id_iat_us"]
]
BIT_COUNT = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)
PARAMS = {
    "max_iter": 60,
    "max_leaf_nodes": 15,
    "learning_rate": 0.1,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 42,
}


class CausalFeatures:
    """Construct features from the current frame and prior frames in the same file."""

    def __init__(self) -> None:
        self.previous = None
        self.last_timestamp = None
        self.last_raw_timestamp = None
        self.backwards_count = 0
        self.maximum_backwards_seconds = 0.0

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
        if not len(timestamps) or not np.isfinite(timestamps).all():
            raise ValueError("empty or non-finite timestamp chunk")
        identifiers = np.array([int(value, 16) for value in frame["arbitration_id"]], dtype=np.uint32)
        payload = frame["data_field"].str.strip()
        lengths = payload.str.len().to_numpy()
        if ((lengths > 16) | (lengths % 2 != 0)).any() or not payload.str.fullmatch(r"[0-9a-fA-F]*").all():
            raise ValueError("invalid classic-CAN payload")
        data = np.frombuffer(
            bytes.fromhex("".join(payload.str.pad(16, side="right", fillchar="0"))),
            dtype=np.uint8,
        ).reshape(-1, 8)
        raw_step = np.diff(
            timestamps,
            prepend=timestamps[0] if self.last_raw_timestamp is None else self.last_raw_timestamp,
        )
        self.backwards_count += int((raw_step < 0).sum())
        self.maximum_backwards_seconds = max(self.maximum_backwards_seconds, float(max(0, -raw_step.min())))
        self.last_raw_timestamp = float(timestamps[-1])
        prior_timestamp = timestamps[0] if self.last_timestamp is None else self.last_timestamp
        timestamps = np.maximum.accumulate(np.concatenate([[prior_timestamp], timestamps]))[1:]
        global_iat = np.diff(timestamps, prepend=prior_timestamp)
        current = pd.DataFrame(data, columns=BYTE_NAMES)
        current.insert(0, "timestamp", timestamps)
        current.insert(0, "id", identifiers)
        joined = current if self.previous is None else pd.concat([self.previous, current], ignore_index=True)
        shifted = joined.groupby("id", sort=False)[["timestamp", *BYTE_NAMES]].shift().tail(len(current))
        history = shifted["timestamp"].notna().to_numpy()
        same_iat = timestamps - shifted["timestamp"].fillna(pd.Series(timestamps, index=shifted.index)).to_numpy()
        previous_bytes = shifted[BYTE_NAMES].to_numpy(copy=True)
        previous_bytes[~history] = data[~history]
        previous_bytes = previous_bytes.astype(np.uint8)
        delta = np.abs(data.astype(np.int16) - previous_bytes.astype(np.int16)) / 255.0
        hamming = BIT_COUNT[np.bitwise_xor(data, previous_bytes)].sum(axis=1) / 64.0
        cheap = np.column_stack([identifiers / 2047.0, lengths / 16.0, data / 255.0])
        rich = np.column_stack([
            cheap,
            history,
            np.log1p(global_iat * 1e6),
            np.log1p(same_iat * 1e6),
            delta,
            hamming,
        ]).astype(np.float32)
        if rich.shape[1] != len(RICH_NAMES) or not np.isfinite(rich).all():
            raise ValueError("invalid CTAT features")
        self.previous = joined.drop_duplicates("id", keep="last").reset_index(drop=True)
        self.last_timestamp = float(timestamps[-1])
        return rich

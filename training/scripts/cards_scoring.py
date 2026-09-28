"""CarDS full-stream scoring and event-level result aggregation."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

import cards_cache as cache_io
import cards_features as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cards_training.json"
CACHE = ROOT / "work" / "cards_cache"


def dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stream_windows(row: dict[str, str], profile: dict[str, Any], center: np.ndarray, scale: np.ndarray, sequence_length: int = 16) -> Iterator[tuple[float, int, np.ndarray]]:
    """Causal full stream with no sampling or future-frame access."""
    state = base.StreamState(); sequences: dict[int, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=sequence_length))
    for timestamp, bus, can_id, frame_type, payload, direction in cache_io.cached_records(CACHE, row):
        raw = base.feature_raw(state.step(timestamp, bus, can_id, payload), bus, can_id, frame_type, payload, profile)
        window = sequences[bus]; window.append(base.norm_feature(raw, center, scale))
        if len(window) == sequence_length:
            yield float(timestamp), int(direction == "T"), np.asarray(tuple(window), dtype=np.float32)


def emit_alarm_times(scores: np.ndarray, timestamps: np.ndarray, threshold: float, cooldown_seconds: float) -> np.ndarray:
    emitted, last = [], -math.inf
    for score, timestamp in zip(scores, timestamps):
        if score >= threshold and timestamp - last >= cooldown_seconds:
            emitted.append(float(timestamp)); last = float(timestamp)
    return np.asarray(emitted, dtype=float)


def trace_receipt(trace_name: str, mechanism: str | None, timestamps: np.ndarray, labels: np.ndarray, scores: np.ndarray, threshold: float, cooldown_seconds: float) -> dict[str, Any]:
    if not (len(timestamps) == len(labels) == len(scores)) or not len(timestamps): raise RuntimeError("trace score/label/timestamp shape contract failed")
    pred = scores >= threshold; alarms = emit_alarm_times(scores, timestamps, threshold, cooldown_seconds)
    tp = int(np.logical_and(pred, labels == 1).sum()); fp = int(np.logical_and(pred, labels == 0).sum()); tn = int(np.logical_and(~pred, labels == 0).sum()); fn = int(np.logical_and(~pred, labels == 1).sum())
    onset = float(timestamps[np.flatnonzero(labels == 1)[0]]) if (labels == 1).any() else None
    deadline = onset + 0.100 if onset is not None else None
    first_alarm = float(alarms[alarms >= onset][0]) if onset is not None and (alarms >= onset).any() else None
    duration = max(0.0, float(timestamps[-1] - timestamps[0]))
    delay = None if first_alarm is None or onset is None else max(0.0, first_alarm - onset)
    return {"trace_name": trace_name, "mechanism": mechanism, "frames": int(len(labels)), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "frame_macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)), "auprc": float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else None, "attack_onset": onset, "first_alarm_after_onset": first_alarm, "der_100ms": None if deadline is None else bool(first_alarm is not None and first_alarm <= deadline), "detection_delay_seconds": delay, "alarm_event_count": int(len(alarms)), "duration_seconds": duration, "fae_h": None if onset is not None or duration <= 0 else len(alarms) / (duration / 3600)}


def aggregate_receipts(receipts: list[dict[str, Any]], rmttd_horizon_seconds: float = 5.0) -> dict[str, Any]:
    attacks = [value for value in receipts if value["attack_onset"] is not None]; normals = [value for value in receipts if value["attack_onset"] is None]
    if not attacks or not normals: raise RuntimeError("aggregate requires both attack and normal trace receipts")
    by_mechanism: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in attacks: by_mechanism[str(value["mechanism"])].append(value)
    mechanism_tpr = {name: float(np.mean([item["first_alarm_after_onset"] is not None for item in values])) for name, values in by_mechanism.items()}
    der = [float(item["der_100ms"]) for item in attacks if item["der_100ms"] is not None]
    macro_auprc = [item["auprc"] for values in by_mechanism.values() for item in values if item["auprc"] is not None]
    frame_f1 = [item["frame_macro_f1"] for item in receipts]
    trace_f1 = [item["frame_macro_f1"] for item in receipts]
    normal_hours = sum(item["duration_seconds"] for item in normals) / 3600
    normal_events = sum(item["alarm_event_count"] for item in normals)
    delays = [min(rmttd_horizon_seconds, item["detection_delay_seconds"] if item["detection_delay_seconds"] is not None else rmttd_horizon_seconds) for item in attacks]
    return {"mechanism_macro_tpr": float(np.mean(list(mechanism_tpr.values()))), "mechanism_count": len(mechanism_tpr), "der_100ms": float(np.mean(der)) if der else None, "outer_normal_fae_h": normal_events / normal_hours if normal_hours else None, "mechanism_macro_auprc": float(np.mean(macro_auprc)) if macro_auprc else None, "trace_macro_f1": float(np.mean(trace_f1)), "frame_macro_f1_trace_mean": float(np.mean(frame_f1)), "rmttd_5s": float(np.mean(delays)), "normal_alarm_events": int(normal_events), "normal_hours": normal_hours, "per_mechanism_tpr": mechanism_tpr}


def bootstrap_trace_cluster(receipts: list[dict[str, Any]], draws: int = 2000, seed: int = 20260908) -> dict[str, list[float]]:
    """Cluster bootstrap over whole trace receipts, never over individual frames."""
    rng = np.random.default_rng(seed); values: dict[str, list[float]] = defaultdict(list)
    for _ in range(draws):
        sampled = [receipts[index] for index in rng.integers(0, len(receipts), size=len(receipts))]
        try: result = aggregate_receipts(sampled)
        except RuntimeError: continue
        for key in ("mechanism_macro_tpr", "der_100ms", "outer_normal_fae_h", "mechanism_macro_auprc", "trace_macro_f1", "frame_macro_f1_trace_mean", "rmttd_5s"):
            if result.get(key) is not None: values[key].append(float(result[key]))
    return {key: [float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975))] for key, value in values.items() if value}


def static_result(config: dict[str, Any]) -> dict[str, Any]:
    ts = np.asarray([0.0, 0.03, 0.06, 0.12, 0.15]); labels = np.asarray([0, 0, 1, 1, 0]); scores = np.asarray([0.01, 0.1, 0.95, 0.7, 0.8])
    attack = trace_receipt("synthetic_attack", "synthetic::mechanism", ts, labels, scores, 0.8, 1.0)
    normal = trace_receipt("synthetic_normal", None, ts, np.zeros(5, dtype=int), scores, 0.8, 1.0)
    aggregate = aggregate_receipts([attack, normal])
    checks = {"synthetic_alarm_cooldown": attack["alarm_event_count"] == 1 and normal["alarm_event_count"] == 1, "synthetic_der": attack["der_100ms"] is True, "receipt_confusion_nonnegative": all(attack[key] >= 0 for key in ("tp", "fp", "tn", "fn")), "aggregate_primary_finite": math.isfinite(aggregate["mechanism_macro_tpr"]), "protocol_primary_metric_exact": config["metrics"]["primary"] == "mechanism_macro_tpr_at_calibrated_30_fae_per_hour", "cooldown_exact": float(config["alarm"]["cooldown_seconds"]) == 1.0, "no_real_data_read": True, "no_model_fit": True}
    return {"status": "SCORING_STATIC_PASS" if all(checks.values()) else "SCORING_STATIC_FAIL", "checks": checks, "scope": {"synthetic_scores_only": True, "raw_archive_content_read": False, "model_training": False, "outer_test_labels_read": False}}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--check", action="store_true"); args = parser.parse_args()
    if not args.check: parser.error("only --check is allowed before the future authorized formal executor")
    output, config_path = args.out.resolve(), args.config.resolve()
    if output.exists(): raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True); shutil.copy2(Path(__file__), output / "script_snapshot.py"); shutil.copy2(config_path, output / "protocol_snapshot.json")
    result = static_result(json.loads(config_path.read_text(encoding="utf-8"))); dump(output / "result.json", result); marker = "COMPLETED.json" if "PASS" in result["status"] else "FAILED.json"; dump(output / marker, {"status": result["status"]}); print(json.dumps(result, ensure_ascii=False, indent=2)); return 0 if marker == "COMPLETED.json" else 2


if __name__ == "__main__": raise SystemExit(main())

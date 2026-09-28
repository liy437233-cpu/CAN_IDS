"""ROAD group-disjoint probe training and operational evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import road_training as base


ROOT = Path(__file__).resolve().parents[1]
NORMAL_CAP = 8000
SEED = 20260907


def dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def reservoir(values: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if len(values) <= cap:
        return values
    rng = np.random.default_rng(seed)
    selected = np.arange(cap)
    for index in range(cap, len(values)):
        candidate = rng.integers(index + 1)
        if candidate < cap:
            selected[candidate] = index
    return values[np.sort(selected)]


def macro_f1(labels: np.ndarray, probabilities: np.ndarray) -> float:
    predicted = (probabilities >= 0.5).astype(np.int8)
    values = []
    for target in (0, 1):
        true_positive = ((predicted == target) & (labels == target)).sum()
        denominator = (
            2 * true_positive
            + ((predicted == target) & (labels != target)).sum()
            + ((predicted != target) & (labels == target)).sum()
        )
        values.append(0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(values))


def confusion(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predicted = (probabilities >= 0.5).astype(np.int8)
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": None if tp + fn == 0 else tp / (tp + fn),
        "candidate_frames": int(labels.sum()),
        "frames": int(len(labels)),
    }


def event_row(
    capture: str,
    group: str,
    elapsed: np.ndarray,
    probabilities: np.ndarray,
    metadata: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    start, end = metadata["injection_interval"]
    in_interval = (elapsed >= start) & (elapsed <= end)
    alarm_times = elapsed[in_interval & (probabilities >= 0.5)]
    delay = None if len(alarm_times) == 0 else float(alarm_times[0] - start)
    duration = float(end - start)
    detected = delay is not None
    return {
        "capture": capture,
        "group_id": group,
        "model": model,
        "event_duration_sec": duration,
        "detected": int(detected),
        "ttd_sec": delay,
        "censored": int(not detected),
        "der_10ms": int(detected and delay <= min(0.01, duration)),
        "der_50ms": int(detected and delay <= min(0.05, duration)),
        "der_100ms": int(detected and delay <= min(0.1, duration)),
        "der_500ms": int(detected and delay <= min(0.5, duration)),
    }


def false_alarm_row(
    capture: str,
    group: str,
    elapsed: np.ndarray,
    probabilities: np.ndarray,
    model: str,
) -> dict[str, Any]:
    alarm_times = elapsed[probabilities >= 0.5]

    def episode_count(cooldown_seconds: float) -> int:
        count = 0
        last = None
        for timestamp in alarm_times:
            if last is None or timestamp - last > cooldown_seconds:
                count += 1
            last = timestamp
        return count

    count_01 = episode_count(0.1)
    count_1 = episode_count(1.0)
    count_5 = episode_count(5.0)
    hours = (elapsed[-1] - elapsed[0]) / 3600 if len(elapsed) > 1 else 0
    return {
        "capture": capture,
        "group_id": group,
        "model": model,
        "false_alarm_episodes_0_1s": count_01,
        "false_alarm_episodes": count_1,
        "false_alarm_episodes_1s": count_1,
        "false_alarm_episodes_5s": count_5,
        "normal_hours": hours,
        "fae_per_hour": None if hours == 0 else count_1 / hours,
        "fae_per_hour_0_1s": None if hours == 0 else count_01 / hours,
        "fae_per_hour_1s": None if hours == 0 else count_1 / hours,
        "fae_per_hour_5s": None if hours == 0 else count_5 / hours,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check(zip_path: Path, manifest: Path) -> dict[str, Any]:
    if base.sha256(zip_path) != base.ROAD_SHA256:
        raise ValueError("ROAD hash mismatch")
    rows = base.activity_manifest(manifest)
    primary = [row for row in rows if row["task_scope"] == "primary_injection"]
    normal = [row for row in rows if row["task_scope"] == "primary_normal"]
    return {
        "status": "CHECK_PASS_NO_MODEL_FIT",
        "road_sha256": base.ROAD_SHA256,
        "manifest_sha256": base.sha256(manifest),
        "primary_records": len(primary),
        "attack_groups": len({row["group_id"] for row in primary}),
        "normal_groups": len(normal),
        "fits_per_model": 28,
        "total_fits": 84,
        "normal_cap": NORMAL_CAP,
    }


def formal(out: Path, zip_path: Path, manifest: Path) -> None:
    contract = check(zip_path, manifest)
    rows = base.activity_manifest(manifest)
    attacks = [row for row in rows if row["task_scope"] == "primary_injection"]
    normals = [row for row in rows if row["task_scope"] == "primary_normal"]
    dump(out / "RUNNING.json", {"pid": os.getpid(), "started": time.strftime("%F %T"), **contract})
    dump(
        out / "config.json",
        {
            **contract,
            "hgb": base.HGB,
            "features": base.FEATURE_SETS,
            "threshold": 0.5,
            "cooldown_sec": 1,
            "cooldown_sensitivity_sec": [0.1, 1.0, 5.0],
        },
    )

    with zipfile.ZipFile(zip_path) as archive:
        metadata = json.loads(archive.read("road/attacks/capture_metadata.json").decode("utf-8-sig"))
        attack_data = {
            row["capture"]: base.parse_capture(archive, row["log_path"], metadata[row["capture"]])[:3]
            for row in attacks
        }
        normal_train = {}
        for row in normals:
            features, _, _, _ = base.parse_capture(archive, row["log_path"], None)
            seed = int(hashlib.sha256((row["capture"] + str(SEED)).encode()).hexdigest()[:16], 16)
            normal_train[row["capture"]] = reservoir(features, NORMAL_CAP, seed)

        events: list[dict[str, Any]] = []
        frames: list[dict[str, Any]] = []
        normal_rows: list[dict[str, Any]] = []
        inventory: list[dict[str, Any]] = []
        completed = 0
        attack_groups = sorted({row["group_id"] for row in attacks})

        for model, columns in base.FEATURE_SETS.items():
            for group in attack_groups:
                held_out = [row for row in attacks if row["group_id"] == group]
                positives = np.concatenate(
                    [
                        attack_data[row["capture"]][0][attack_data[row["capture"]][1] == 1]
                        for row in attacks
                        if row["group_id"] != group
                    ]
                )
                negatives = np.concatenate(list(normal_train.values()))
                features = np.concatenate([positives, negatives])
                labels = np.r_[np.ones(len(positives)), np.zeros(len(negatives))]
                weights = np.where(
                    labels == 1,
                    len(labels) / (2 * len(positives)),
                    len(labels) / (2 * len(negatives)),
                )
                classifier = HistGradientBoostingClassifier(**base.HGB).fit(
                    features[:, columns], labels, sample_weight=weights
                )
                for row in held_out:
                    capture_features, capture_labels, timestamps = attack_data[row["capture"]]
                    probabilities = classifier.predict_proba(capture_features[:, columns])[:, 1]
                    events.append(
                        event_row(
                            row["capture"], group, timestamps, probabilities, metadata[row["capture"]], model
                        )
                    )
                    frames.append(
                        {
                            "capture": row["capture"],
                            "group_id": group,
                            "model": model,
                            "frame_macro_f1": macro_f1(capture_labels, probabilities),
                            **confusion(capture_labels, probabilities),
                        }
                    )
                inventory.append(
                    {
                        "model": model,
                        "heldout_group": group,
                        "kind": "attack",
                        "positive_frames": len(positives),
                        "negative_frames": len(negatives),
                    }
                )
                completed += 1
                dump(
                    out / "progress.json",
                    {
                        "completed_fits": completed,
                        "total_fits": 84,
                        "model": model,
                        "heldout_group": group,
                        "kind": "attack",
                    },
                )

            for row in normals:
                positives = np.concatenate(
                    [attack_data[item["capture"]][0][attack_data[item["capture"]][1] == 1] for item in attacks]
                )
                negatives = np.concatenate(
                    [normal_train[item["capture"]] for item in normals if item["capture"] != row["capture"]]
                )
                features = np.concatenate([positives, negatives])
                labels = np.r_[np.ones(len(positives)), np.zeros(len(negatives))]
                weights = np.where(
                    labels == 1,
                    len(labels) / (2 * len(positives)),
                    len(labels) / (2 * len(negatives)),
                )
                classifier = HistGradientBoostingClassifier(**base.HGB).fit(
                    features[:, columns], labels, sample_weight=weights
                )
                capture_features, _, timestamps, _ = base.parse_capture(archive, row["log_path"], None)
                probabilities = classifier.predict_proba(capture_features[:, columns])[:, 1]
                normal_rows.append(
                    false_alarm_row(row["capture"], row["group_id"], timestamps, probabilities, model)
                )
                inventory.append(
                    {
                        "model": model,
                        "heldout_group": row["group_id"],
                        "kind": "normal",
                        "positive_frames": len(positives),
                        "negative_frames": len(negatives),
                    }
                )
                completed += 1
                dump(
                    out / "progress.json",
                    {
                        "completed_fits": completed,
                        "total_fits": 84,
                        "model": model,
                        "heldout_group": row["group_id"],
                        "kind": "normal",
                    },
                )

    write_csv(out / "attack_event_oof.csv", events)
    write_csv(out / "frame_capture_oof.csv", frames)
    write_csv(out / "normal_capture_oof.csv", normal_rows)
    write_csv(out / "training_inventory.csv", inventory)
    (out / "RUNNING.json").unlink()
    dump(out / "COMPLETED.json", {"status": "COMPLETED", "completed_fits": completed})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["check", "formal"], default="check")
    parser.add_argument("--zip", type=Path, default=ROOT / "data/raw/road.zip")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/protocol/road_activity_groups.csv")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.mode == "check":
        print(json.dumps(check(args.zip.resolve(), args.manifest.resolve()), ensure_ascii=False))
        return
    if args.out is None:
        raise ValueError("--out required for formal")
    output = args.out.resolve()
    allowed = {"stdout.log", "stderr.log", "launch.json"}
    if output.exists() and any(path.name not in allowed for path in output.iterdir()):
        raise RuntimeError("Refusing to overwrite existing run artifacts")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, output / "script_snapshot.py")
    try:
        formal(output, args.zip.resolve(), args.manifest.resolve())
    except Exception as exc:
        dump(output / "FAILED.json", {"status": "FAILED", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()

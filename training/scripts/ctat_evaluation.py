"""CTAT preprocessing, probe fitting, and evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

import ctat_features as feature_io
import ctat_models as model_io


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/cantrainandtest.zip"
MANIFEST = ROOT / "data/protocol/ctat_manifest.csv"
ARCHIVE_SHA = "A9C607B38BD28F1768021AD01C29FFBFE4E82BB0AE5815AC3CE7AD74751AE061"
SEED = 42
NORMAL_CAP = 20_000
THRESHOLD = 0.5
FEATURES = {
    "frame": list(range(len(feature_io.CHEAP_NAMES))),
    "timing": [0, 10, 11, 12],
    "context": list(range(len(feature_io.RICH_NAMES))),
}
NORMAL_POOL = {
    "set_01": ("set_03", ["attack-free-3.csv", "attack-free-4.csv"]),
    "set_02": ("set_04", ["attack-free-3.csv", "attack-free-4.csv"]),
    "set_03": ("set_04", ["attack-free-3.csv", "attack-free-4.csv"]),
    "set_04": ("set_02", ["attack-free-1.csv", "attack-free-2.csv"]),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def dump(path: Path, value: Any) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def macro_f1(tp: int, fp: int, tn: int, fn: int) -> float:
    positive = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    negative = 0.0 if 2 * tn + fp + fn == 0 else 2 * tn / (2 * tn + fp + fn)
    return (positive + negative) / 2.0


def episode_count(timestamps: np.ndarray, probabilities: np.ndarray, cooldown: float) -> int:
    selected = timestamps[probabilities >= THRESHOLD]
    count = 0
    last = None
    for timestamp in selected:
        if last is None or timestamp - last > cooldown:
            count += 1
        last = timestamp
    return count


def microsegment_rows(
    task: str,
    filename: str,
    timestamps: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    model: str,
) -> list[dict[str, Any]]:
    positive = labels == 1
    starts = np.flatnonzero(positive & ~np.r_[False, positive[:-1]])
    ends = np.flatnonzero(positive & ~np.r_[positive[1:], False])
    rows = []
    for index, (start, end) in enumerate(zip(starts, ends), start=1):
        hits = np.flatnonzero(probabilities[start : end + 1] >= THRESHOLD)
        detected = len(hits) > 0
        first = start + int(hits[0]) if detected else None
        duration = max(0.0, float(timestamps[end] - timestamps[start]))
        delay = duration if first is None else max(0.0, float(timestamps[first] - timestamps[start]))
        rows.append(
            {
                "task_set": task,
                "filename": filename,
                "model": model,
                "event_index": index,
                "event_frames": int(end - start + 1),
                "event_duration_sec": duration,
                "detected": int(detected),
                "censored": int(not detected),
                "ttd_sec": delay,
                "frames_to_first_alarm": int(end - start + 1 if first is None else first - start),
                "der_10ms": int(detected and delay <= min(0.01, duration)),
                "der_50ms": int(detected and delay <= min(0.05, duration)),
                "der_100ms": int(detected and delay <= min(0.1, duration)),
                "der_500ms": int(detected and delay <= min(0.5, duration)),
            }
        )
    return rows


def test_scan(
    archive: zipfile.ZipFile,
    row: dict[str, str],
    classifier: HistGradientBoostingClassifier,
    columns: list[int],
    model: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = feature_io.CausalFeatures()
    timestamp_chunks = []
    label_chunks = []
    probability_chunks = []
    last = None
    with archive.open(row["zip_path"]) as source:
        reader = pd.read_csv(
            source,
            chunksize=100_000,
            keep_default_na=False,
            dtype={"timestamp": "float64", "arbitration_id": "str", "data_field": "str", "attack": "int8"},
        )
        for frame in reader:
            labels = frame["attack"].to_numpy(np.int8)
            if not np.isin(labels, [0, 1]).all():
                raise ValueError("unexpected CTAT label")
            raw_timestamps = frame["timestamp"].to_numpy(float)
            timestamps = np.maximum.accumulate(
                np.r_[raw_timestamps[0] if last is None else last, raw_timestamps]
            )[1:]
            last = float(timestamps[-1])
            features = state.transform(frame.drop(columns="attack"))
            timestamp_chunks.append(timestamps)
            label_chunks.append(labels)
            probability_chunks.append(classifier.predict_proba(features[:, columns])[:, 1])

    timestamps = np.concatenate(timestamp_chunks)
    labels = np.concatenate(label_chunks)
    probabilities = np.concatenate(probability_chunks)
    predicted = probabilities >= THRESHOLD
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    frame_result = {
        "task_set": row["set"],
        "filename": row["filename"],
        "model": model,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": None if tp + fn == 0 else tp / (tp + fn),
        "fpr": None if tn + fp == 0 else fp / (tn + fp),
        "frame_macro_f1": macro_f1(tp, fp, tn, fn),
        "attack_frames": int(labels.sum()),
        "frames": int(len(labels)),
    }
    return frame_result, microsegment_rows(
        row["set"], row["filename"], timestamps, labels, probabilities, model
    )


def normal_scan(
    archive: zipfile.ZipFile,
    row: dict[str, str],
    classifier: HistGradientBoostingClassifier,
    columns: list[int],
    model: str,
    task: str,
) -> dict[str, Any]:
    state = feature_io.CausalFeatures()
    timestamp_chunks = []
    probability_chunks = []
    label_chunks = []
    last = None
    with archive.open(row["zip_path"]) as source:
        reader = pd.read_csv(
            source,
            chunksize=100_000,
            keep_default_na=False,
            dtype={"timestamp": "float64", "arbitration_id": "str", "data_field": "str", "attack": "int8"},
        )
        for frame in reader:
            labels = frame["attack"].to_numpy(np.int8)
            label_chunks.append(labels)
            raw_timestamps = frame["timestamp"].to_numpy(float)
            timestamps = np.maximum.accumulate(
                np.r_[raw_timestamps[0] if last is None else last, raw_timestamps]
            )[1:]
            last = float(timestamps[-1])
            features = state.transform(frame.drop(columns="attack"))
            timestamp_chunks.append(timestamps)
            probability_chunks.append(classifier.predict_proba(features[:, columns])[:, 1])

    timestamps = np.concatenate(timestamp_chunks)
    probabilities = np.concatenate(probability_chunks)
    if np.concatenate(label_chunks).any():
        raise ValueError("normal evaluation pool is not attack-free")
    hours = max(0.0, float(timestamps[-1] - timestamps[0])) / 3600
    count_01 = episode_count(timestamps, probabilities, 0.1)
    count_1 = episode_count(timestamps, probabilities, 1.0)
    count_5 = episode_count(timestamps, probabilities, 5.0)
    return {
        "task_set": task,
        "normal_pool_set": row["set"],
        "filename": row["filename"],
        "model": model,
        "normal_hours": hours,
        "false_alarm_episodes_0_1s": count_01,
        "false_alarm_episodes_1s": count_1,
        "false_alarm_episodes_5s": count_5,
        "fae_per_hour_0_1s": None if hours == 0 else count_01 / hours,
        "fae_per_hour": None if hours == 0 else count_1 / hours,
        "fae_per_hour_5s": None if hours == 0 else count_5 / hours,
    }


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def check() -> dict[str, Any]:
    if sha256(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError("CTAT archive SHA-256 mismatch")
    rows = read_manifest()
    training = [row for row in rows if row["official_split"] == "train_01"]
    tests = [row for row in rows if row["official_split"] == "test_04_unknown_vehicle_unknown_attack"]
    if len(rows) != 236 or len(training) != 60 or len(tests) != 44:
        raise ValueError("CTAT manifest count contract failed")
    if {row["set"] for row in tests} != {"set_01", "set_02", "set_03", "set_04"}:
        raise ValueError("missing UV-UA task")
    if not all("/test_04_unknown_vehicle_unknown_attack/" in row["zip_path"] for row in tests):
        raise ValueError("unexpected CTAT test path")
    normal_rows = []
    for task, (pool, names) in NORMAL_POOL.items():
        found = [row for row in training if row["set"] == pool and row["filename"] in names]
        if len(found) != 2:
            raise ValueError(f"normal-pool contract failed: {task}")
        normal_rows.extend(found)
    return {
        "status": "CHECK_PASS",
        "archive_sha256": ARCHIVE_SHA,
        "manifest_sha256": sha256(MANIFEST),
        "train_files": 60,
        "allowed_test_files": 44,
        "allowed_test_split": "test_04_unknown_vehicle_unknown_attack",
        "normal_evaluation_files": len(normal_rows),
        "models": list(FEATURES),
        "fits": 12,
        "threshold": THRESHOLD,
    }


def self_test() -> None:
    timestamps = np.asarray([0.0, 0.05, 0.2, 1.1, 1.15, 7.0])
    probabilities = np.ones(6)
    assert [episode_count(timestamps, probabilities, value) for value in (0.1, 1.0, 5.0)] == [4, 2, 2]
    labels = np.asarray([0, 1, 1, 0, 1])
    scores = np.asarray([0.1, 0.7, 0.2, 0.1, 0.8])
    rows = microsegment_rows("set", "file", np.arange(5, dtype=float), labels, scores, "model")
    assert len(rows) == 2 and rows[0]["detected"] == 1 and rows[1]["frames_to_first_alarm"] == 0
    print("SELF_TEST_PASS: label segmentation, causal alarm timing, and cooldown aggregation", flush=True)


def formal(output: Path) -> None:
    contract = check()
    rows = read_manifest()
    training = [row for row in rows if row["official_split"] == "train_01"]
    tests = [row for row in rows if row["official_split"] == "test_04_unknown_vehicle_unknown_attack"]
    normal = {
        task: [row for row in training if row["set"] == pool and row["filename"] in names]
        for task, (pool, names) in NORMAL_POOL.items()
    }
    dump(output / "RUNNING.json", {"pid": os.getpid(), "started": time.strftime("%F %T"), **contract})
    dump(
        output / "config.json",
        {
            **contract,
            "hgb": feature_io.PARAMS,
            "normal_cap_per_train_file": NORMAL_CAP,
            "seed": SEED,
            "source_boundary": "Each set trains only on its own train_01; normal pools are evaluation-only.",
        },
    )

    events: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    normals: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    fits = 0
    with zipfile.ZipFile(ARCHIVE) as archive, threadpool_limits(limits=6):
        for task in sorted({row["set"] for row in tests}):
            task_training = [row for row in training if row["set"] == task]
            arrays = []
            for row in task_training:
                features, labels, weights, _, metadata = model_io.extract(
                    archive, row, normal_cap=NORMAL_CAP
                )
                arrays.append((features, labels, weights))
                inventory.append(
                    {
                        "task_set": task,
                        "kind": "train_file",
                        "filename": row["filename"],
                        "sample_frames": len(labels),
                        "full_attack_frames": metadata["full_label_counts"][1],
                    }
                )
            features, labels, weights = [np.concatenate(values) for values in zip(*arrays)]
            mass = np.bincount(labels, weights=weights, minlength=2)
            balanced = weights * mass.sum() / (2 * mass[labels])
            balanced /= balanced.mean()

            for model, columns in FEATURES.items():
                classifier = HistGradientBoostingClassifier(**feature_io.PARAMS).fit(
                    features[:, columns], labels, sample_weight=balanced
                )
                fits += 1
                dump(
                    output / "progress.json",
                    {
                        "stage": "fit_complete",
                        "completed_fits": fits,
                        "total_fits": 12,
                        "task_set": task,
                        "model": model,
                    },
                )
                for row in [item for item in tests if item["set"] == task]:
                    frame_result, microsegments = test_scan(archive, row, classifier, columns, model)
                    frames.append(frame_result)
                    events.extend(microsegments)
                    dump(
                        output / "progress.json",
                        {
                            "stage": "test_scanned",
                            "completed_fits": fits,
                            "total_fits": 12,
                            "task_set": task,
                            "model": model,
                            "filename": row["filename"],
                        },
                    )
                for row in normal[task]:
                    normals.append(normal_scan(archive, row, classifier, columns, model, task))

    write_csv(output / "ctat_microsegment_test.csv", events)
    write_csv(output / "ctat_frame_file_test.csv", frames)
    write_csv(output / "ctat_normal_fae_test.csv", normals)
    write_csv(output / "training_inventory.csv", inventory)
    (output / "RUNNING.json").unlink()
    dump(output / "COMPLETED.json", {"status": "COMPLETED", "completed_fits": fits, "official_test_files_read": 44})


def main() -> None:
    global ARCHIVE, MANIFEST
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["check", "formal"], default="check")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    ARCHIVE = args.archive.resolve()
    MANIFEST = args.manifest.resolve()
    if args.mode == "check":
        self_test()
        print(json.dumps(check(), ensure_ascii=False))
        return
    if args.out is None:
        raise ValueError("--out required for formal")
    output = args.out.resolve()
    allowed = {"stdout.log", "stderr.log", "launch.json"}
    if output.exists() and any(path.name not in allowed for path in output.iterdir()):
        raise RuntimeError("Refusing to overwrite existing artifacts")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, output / "script_snapshot.py")
    try:
        formal(output)
    except BaseException as exc:
        dump(output / "FAILED.json", {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()

"""CarDS probe training and operational-metric evaluation."""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import platform
import shutil
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import torch

import cards_models as models
import cards_training as assembly


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cards_operational.json"
TRAINING_CONFIG = ROOT / "configs" / "cards_training.json"
SOURCE_FILES = {
    "cards_training": ROOT / "scripts" / "cards_training.py",
    "cards_models": ROOT / "scripts" / "cards_models.py",
    "cards_scoring": ROOT / "scripts" / "cards_scoring.py",
}


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def progress(output: Path, fold: int | None, stage: str, **detail: Any) -> None:
    dump(
        output / "PROGRESS.json",
        {
            "protocol_id": "CARDS_OPERATIONAL_EVALUATION_V1",
            "fold": fold,
            "stage": stage,
            **detail,
        },
    )


def load_training_config(config: dict[str, Any]) -> dict[str, Any]:
    base = json.loads(TRAINING_CONFIG.read_text(encoding="utf-8"))
    for section in ("normal_reference_split", "inputs", "sampling", "training", "comparators"):
        if section == "inputs":
            continue
        if section in config and section in base:
            base[section] = config[section]
    base["data"]["archive"] = config["data"]["archive"]
    base["data"]["expected_bytes"] = config["data"]["expected_bytes"]
    base["data"]["expected_md5"] = config["data"]["expected_md5"]
    base["data"]["role_csv"] = config["data"]["role_csv"]
    base["data"]["role_csv_sha256"] = config["data"]["role_csv_sha256"]
    base["data"]["outer_folds"] = config["data"]["outer_folds"]
    return base


def probe_manifest(config: dict[str, Any]) -> list[dict[str, Any]]:
    result = [
        {"name": "logistic_regression", "kind": "logistic_regression", "seed": None},
        {"name": "hgb", "kind": "hgb", "seed": None},
    ]
    result.extend(
        {"name": "early_fusion_tcn", "kind": "early_fusion_tcn", "seed": int(seed)}
        for seed in config["training"]["stochastic_seeds"]
    )
    result.append({"name": "residual_max", "kind": "residual_max", "seed": None})
    return result


def probe_key(item: dict[str, Any]) -> str:
    suffix = "fixed" if item["seed"] is None else str(item["seed"])
    return f"{item['name']}__{suffix}"


def fit_records(data: dict[str, Any], config: dict[str, Any], training_config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in probe_manifest(config):
        kind = item["kind"]
        if kind in {"logistic_regression", "hgb"}:
            model = models.fit_sklearn(data["x"], data["y"], training_config, kind)
        elif kind == "early_fusion_tcn":
            model = models.fit_torch(data["x"], data["y"], data["group"], training_config, kind, int(item["seed"]), device)
        elif kind == "residual_max":
            model = None
        else:
            raise RuntimeError(f"unsupported probe kind: {kind}")
        records.append({**item, "key": probe_key(item), "model": model})
    return records


def score_batch(record: dict[str, Any], windows: np.ndarray, device: torch.device) -> np.ndarray:
    kind = record["kind"]
    if kind == "residual_max":
        value = models.residual_max_scores(windows)
    elif kind in {"logistic_regression", "hgb"}:
        value = record["model"].predict_proba(windows[:, -1, :])[:, 1]
    else:
        value = models.torch_scores(record["model"], windows, device)
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise RuntimeError(f"nonfinite scores from {record['key']}")
    return value


def trace_scores(
    row: dict[str, str],
    profile: dict[str, Any],
    center: np.ndarray,
    scale: np.ndarray,
    records: list[dict[str, Any]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    import cards_scoring as scoring

    timestamps: list[float] = []
    labels: list[int] = []
    values: dict[str, list[np.ndarray]] = {record["key"]: [] for record in records}
    batch: list[np.ndarray] = []
    batch_timestamps: list[float] = []
    batch_labels: list[int] = []

    def flush() -> None:
        if not batch:
            return
        windows = np.stack(batch).astype(np.float32)
        timestamps.extend(batch_timestamps)
        labels.extend(batch_labels)
        for record in records:
            values[record["key"]].append(score_batch(record, windows, device))
        batch.clear()
        batch_timestamps.clear()
        batch_labels.clear()

    for timestamp, label, window in scoring.stream_windows(row, profile, center, scale):
        batch.append(window)
        batch_timestamps.append(float(timestamp))
        batch_labels.append(int(label))
        if len(batch) >= 8192:
            flush()
    flush()
    if not timestamps:
        raise RuntimeError(f"trace produced no causal windows: {row['trace_name']}")
    return (
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(labels, dtype=np.uint8),
        {key: np.concatenate(parts) for key, parts in values.items()},
    )


def alarm_times(scores: np.ndarray, timestamps: np.ndarray, threshold: float, cooldown: float) -> np.ndarray:
    selected = timestamps[np.flatnonzero(scores >= threshold)]
    if not len(selected):
        return np.empty(0, dtype=np.float64)
    accepted: list[float] = []
    index = 0
    while index < len(selected):
        current = float(selected[index])
        accepted.append(current)
        index = int(np.searchsorted(selected, current + cooldown, side="left"))
    return np.asarray(accepted, dtype=np.float64)


def trace_grid_receipt(
    row: dict[str, str],
    timestamps: np.ndarray,
    labels: np.ndarray,
    emitted: np.ndarray,
    deadlines: list[float],
) -> dict[str, Any]:
    onset = float(timestamps[np.flatnonzero(labels == 1)[0]]) if bool((labels == 1).any()) else None
    after = emitted[emitted >= onset] if onset is not None else np.empty(0, dtype=np.float64)
    first_alarm = float(after[0]) if len(after) else None
    delay = None if onset is None or first_alarm is None else max(0.0, first_alarm - onset)
    duration = max(0.0, float(timestamps[-1] - timestamps[0]))
    mechanism = row.get("attack_mechanism_group") if onset is not None else None
    return {
        "trace_name": row["trace_name"],
        "mechanism": mechanism,
        "frames": int(len(labels)),
        "attack_onset": onset,
        "first_alarm_after_onset": first_alarm,
        "detection_delay_seconds": delay,
        "detected_any": bool(first_alarm is not None),
        "der_by_deadline": {
            f"{deadline:g}": bool(first_alarm is not None and first_alarm <= float(onset) + deadline)
            if onset is not None
            else None
            for deadline in deadlines
        },
        "alarm_event_count": int(len(emitted)),
        "duration_seconds": duration,
        "fae_h": None if onset is not None or duration <= 0 else float(len(emitted) / (duration / 3600.0)),
    }


def aggregate_receipts(receipts: list[dict[str, Any]], deadlines: list[float], horizons: list[float]) -> dict[str, Any]:
    attacks = [item for item in receipts if item["attack_onset"] is not None]
    normals = [item for item in receipts if item["attack_onset"] is None]
    if not attacks or not normals:
        raise RuntimeError("aggregate requires attack and normal traces")
    by_mechanism: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in attacks:
        by_mechanism[str(item["mechanism"])].append(item)
    per_mechanism_tpr = {
        mechanism: float(np.mean([float(item["detected_any"]) for item in values]))
        for mechanism, values in by_mechanism.items()
    }
    der_trace: dict[str, float] = {}
    der_mechanism: dict[str, float] = {}
    for deadline in deadlines:
        key = f"{deadline:g}"
        der_trace[key] = float(np.mean([float(item["der_by_deadline"][key]) for item in attacks]))
        der_mechanism[key] = float(
            np.mean(
                [
                    np.mean([float(item["der_by_deadline"][key]) for item in values])
                    for values in by_mechanism.values()
                ]
            )
        )
    rmttd_trace: dict[str, float] = {}
    rmttd_mechanism: dict[str, float] = {}
    for horizon in horizons:
        key = f"{horizon:g}"

        def restricted(item: dict[str, Any]) -> float:
            delay = item["detection_delay_seconds"]
            return float(min(horizon, horizon if delay is None else delay))

        rmttd_trace[key] = float(np.mean([restricted(item) for item in attacks]))
        rmttd_mechanism[key] = float(
            np.mean([np.mean([restricted(item) for item in values]) for values in by_mechanism.values()])
        )
    normal_hours = float(sum(item["duration_seconds"] for item in normals) / 3600.0)
    normal_events = int(sum(item["alarm_event_count"] for item in normals))
    return {
        "mechanism_macro_tpr": float(np.mean(list(per_mechanism_tpr.values()))),
        "worst_mechanism_tpr": float(min(per_mechanism_tpr.values())),
        "mechanism_count": int(len(per_mechanism_tpr)),
        "per_mechanism_tpr": per_mechanism_tpr,
        "der_trace_mean_by_deadline": der_trace,
        "der_mechanism_macro_by_deadline": der_mechanism,
        "outer_normal_fae_per_hour": float(normal_events / normal_hours) if normal_hours else None,
        "normal_alarm_events": normal_events,
        "normal_hours": normal_hours,
        "rmttd_trace_mean_by_horizon": rmttd_trace,
        "rmttd_mechanism_macro_by_horizon": rmttd_mechanism,
    }


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in ("mechanism_macro_tpr", "worst_mechanism_tpr", "outer_normal_fae_per_hour"):
        if metrics.get(key) is not None:
            result[key] = float(metrics[key])
    for family in (
        "der_trace_mean_by_deadline",
        "der_mechanism_macro_by_deadline",
        "rmttd_trace_mean_by_horizon",
        "rmttd_mechanism_macro_by_horizon",
    ):
        for parameter, value in metrics[family].items():
            result[f"{family}__{parameter}"] = float(value)
    return result


def nested_bootstrap_ci(
    receipts: list[dict[str, Any]],
    deadlines: list[float],
    horizons: list[float],
    draws: int,
    seed: int,
) -> dict[str, list[float]]:
    attacks = [item for item in receipts if item["attack_onset"] is not None]
    normals = [item for item in receipts if item["attack_onset"] is None]
    by_mechanism: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in attacks:
        by_mechanism[str(item["mechanism"])].append(item)
    names = sorted(by_mechanism)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(draws):
        sampled_attacks: list[dict[str, Any]] = []
        for position, mechanism in enumerate(rng.choice(names, size=len(names), replace=True)):
            source = by_mechanism[str(mechanism)]
            indexes = rng.integers(0, len(source), size=len(source))
            for index in indexes:
                cloned = dict(source[int(index)])
                cloned["mechanism"] = f"bootstrap_mechanism_{position}"
                sampled_attacks.append(cloned)
        sampled_normals = [normals[int(index)] for index in rng.integers(0, len(normals), size=len(normals))]
        flat = flatten_metrics(aggregate_receipts(sampled_attacks + sampled_normals, deadlines, horizons))
        for key, value in flat.items():
            values[key].append(value)
    return {
        key: [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))]
        for key, sample in values.items()
        if sample
    }


def verify_cache_payloads(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Formal-only byte-integrity gate; compressed CAN records are not parsed."""
    cache = Path(config["data"]["cache"])
    manifest = json.loads((cache / "cache_manifest.json").read_text(encoding="utf-8"))
    receipts: dict[str, Any] = {}
    for index, family in enumerate(manifest["families"], start=1):
        progress(output, None, "verify_cache_payload_sha256", completed=index - 1, total=len(manifest["families"]), family=family)
        path = cache / f"{family}.zip"
        observed = sha256_file(path)
        expected = str(manifest["sha256"][family]).lower()
        if observed.lower() != expected:
            raise RuntimeError(f"cache SHA-256 mismatch: {family}")
        receipts[family] = {
            "bytes": path.stat().st_size,
            "sha256": observed,
        }
    result = {
        "status": "CACHE_INTEGRITY_PASS",
        "cache_manifest_sha256": sha256_file(cache / "cache_manifest.json"),
        "payloads": receipts,
        "scope": "compressed_bytes_hashed_without_parsing_CAN_records",
    }
    dump(output / "cache_integrity_receipt.json", result)
    return result


def setting_id(budget: float, cooldown: float) -> str:
    return f"budget_{budget:g}__cooldown_{cooldown:g}"


def save_checkpoints(output: Path, fold: int, records: list[dict[str, Any]], data: dict[str, Any]) -> dict[str, str]:
    folder = output / "checkpoints" / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for record in records:
        key = record["key"]
        if record["kind"] == "residual_max":
            continue
        path = folder / (f"{key}.pt" if record["kind"] == "early_fusion_tcn" else f"{key}.joblib")
        if record["kind"] == "early_fusion_tcn":
            torch.save(record["model"].state_dict(), path)
        else:
            joblib.dump(record["model"], path)
        hashes[str(path.relative_to(output))] = sha256_file(path)
    normalizer = folder / "normalizer.npz"
    np.savez_compressed(normalizer, center=data["center"], scale=data["scale"])
    hashes[str(normalizer.relative_to(output))] = sha256_file(normalizer)
    profile = folder / "profile.joblib"
    joblib.dump(data["profile"], profile)
    hashes[str(profile.relative_to(output))] = sha256_file(profile)
    return hashes


def validate_threshold_monotonicity(thresholds: dict[str, Any], budgets: list[float], cooldowns: list[float]) -> None:
    for model_key, values in thresholds.items():
        for cooldown in cooldowns:
            ordered = [float(values[setting_id(budget, cooldown)]["threshold"]) for budget in budgets]
            if any(left < right for left, right in zip(ordered, ordered[1:])):
                raise RuntimeError(f"threshold monotonicity failed for {model_key}, cooldown={cooldown}: {ordered}")


def run_fold(config: dict[str, Any], training_config: dict[str, Any], fold: int, output: Path) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress(output, fold, "assemble_training_side", device=str(device))
    data = assembly.assemble_formal_fold(training_config, fold)
    progress(output, fold, "fit_probe_families", sample_count=int(len(data["x"])))
    records = fit_records(data, config, training_config, device)
    checkpoint_hashes = save_checkpoints(output, fold, records, data)

    budgets = [float(value) for value in config["alarm_sensitivity_grid"]["calibration_budgets_fae_per_hour"]]
    cooldowns = [float(value) for value in config["alarm_sensitivity_grid"]["cooldown_seconds"]]
    deadlines = [float(value) for value in config["alarm_sensitivity_grid"]["event_deadlines_seconds"]]
    horizons = [float(value) for value in config["alarm_sensitivity_grid"]["rmttd_horizons_seconds"]]
    calibration: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {record["key"]: [] for record in records}

    for index, row in enumerate(data["roles"]["normal_calibration"], start=1):
        progress(output, fold, "score_normal_calibration", completed=index - 1, total=len(data["roles"]["normal_calibration"]), trace_name=row["trace_name"])
        timestamps, labels, scores = trace_scores(row, data["profile"], data["center"], data["scale"], records, device)
        if bool(labels.any()):
            raise RuntimeError("normal calibration unexpectedly contains T labels")
        for key, value in scores.items():
            calibration[key].append((value, timestamps))

    thresholds: dict[str, dict[str, Any]] = defaultdict(dict)
    for record in records:
        key = record["key"]
        for cooldown in cooldowns:
            for budget in budgets:
                thresholds[key][setting_id(budget, cooldown)] = models.calibrate_threshold(
                    calibration[key], budget, cooldown
                )
    validate_threshold_monotonicity(thresholds, budgets, cooldowns)

    receipts: dict[str, dict[str, list[dict[str, Any]]]] = {
        record["key"]: {setting_id(budget, cooldown): [] for cooldown in cooldowns for budget in budgets}
        for record in records
    }
    event_path = output / f"fold_{fold}_alarm_events.jsonl.gz"
    with gzip.open(event_path, "wt", encoding="utf-8", compresslevel=6) as event_file:
        for record in records:
            key = record["key"]
            for trace_index, row in enumerate(data["roles"]["normal_calibration"]):
                scores, timestamps = calibration[key][trace_index]
                for cooldown in cooldowns:
                    for budget in budgets:
                        sid = setting_id(budget, cooldown)
                        threshold = float(thresholds[key][sid]["threshold"])
                        emitted = alarm_times(scores, timestamps, threshold, cooldown)
                        event_file.write(
                            json.dumps(
                                {
                                    "fold": fold,
                                    "role": "normal_calibration",
                                    "trace_name": row["trace_name"],
                                    "mechanism": None,
                                    "probe": key,
                                    "budget_fae_h": budget,
                                    "cooldown_seconds": cooldown,
                                    "threshold": threshold,
                                    "alarm_event_count": int(len(emitted)),
                                    "alarm_times": emitted.tolist(),
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
        for role in ("outer_attack", "outer_normal"):
            rows = data["roles"][role]
            for index, row in enumerate(rows, start=1):
                progress(output, fold, "score_outer_stream", role=role, completed=index - 1, total=len(rows), trace_name=row["trace_name"], probe_count=len(records))
                timestamps, labels, scores = trace_scores(row, data["profile"], data["center"], data["scale"], records, device)
                if role == "outer_normal" and bool(labels.any()):
                    raise RuntimeError("outer normal unexpectedly contains T labels")
                for record in records:
                    key = record["key"]
                    for cooldown in cooldowns:
                        for budget in budgets:
                            sid = setting_id(budget, cooldown)
                            threshold = float(thresholds[key][sid]["threshold"])
                            emitted = alarm_times(scores[key], timestamps, threshold, cooldown)
                            receipt = trace_grid_receipt(row, timestamps, labels, emitted, deadlines)
                            receipts[key][sid].append(receipt)
                            event_file.write(
                                json.dumps(
                                    {
                                        "fold": fold,
                                        "role": role,
                                        "trace_name": row["trace_name"],
                                        "mechanism": receipt["mechanism"],
                                        "probe": key,
                                        "budget_fae_h": budget,
                                        "cooldown_seconds": cooldown,
                                        "threshold": threshold,
                                        "alarm_event_count": int(len(emitted)),
                                        "alarm_times": emitted.tolist(),
                                    },
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                del scores, timestamps, labels
                gc.collect()

    draws = int(config["metrics"]["bootstrap_draws"])
    base_seed = int(config["metrics"]["bootstrap_seed"])
    metrics: dict[str, dict[str, Any]] = defaultdict(dict)
    bootstrap: dict[str, dict[str, Any]] = defaultdict(dict)
    for model_index, record in enumerate(records):
        key = record["key"]
        for setting_index, sid in enumerate(sorted(receipts[key])):
            metrics[key][sid] = aggregate_receipts(receipts[key][sid], deadlines, horizons)
            bootstrap[key][sid] = nested_bootstrap_ci(
                receipts[key][sid], deadlines, horizons, draws, base_seed + fold * 1000 + model_index * 100 + setting_index
            )
    return {
        "fold": fold,
        "device": str(device),
        "input_receipt": data["receipt"],
        "probe_manifest": [{key: value for key, value in record.items() if key != "model"} for record in records],
        "checkpoint_sha256": checkpoint_hashes,
        "thresholds": thresholds,
        "metrics": metrics,
        "bootstrap_95ci": bootstrap,
        "per_trace_grid_receipts": receipts,
        "alarm_event_file": event_path.name,
        "alarm_event_file_sha256": sha256_file(event_path),
    }


def synthetic_checks(config: dict[str, Any]) -> dict[str, bool]:
    timestamps = np.asarray([0.00, 0.03, 0.06, 0.12, 0.18, 0.55, 1.20], dtype=float)
    scores = np.asarray([0.1, 0.9, 0.95, 0.85, 0.2, 0.91, 0.92], dtype=float)
    labels = np.asarray([0, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
    deadlines = [float(value) for value in config["alarm_sensitivity_grid"]["event_deadlines_seconds"]]
    horizons = [float(value) for value in config["alarm_sensitivity_grid"]["rmttd_horizons_seconds"]]
    counts = {cooldown: len(alarm_times(scores, timestamps, 0.8, cooldown)) for cooldown in (0.1, 1.0, 5.0)}
    attack = trace_grid_receipt({"trace_name": "attack", "attack_mechanism_group": "m"}, timestamps, labels, alarm_times(scores, timestamps, 0.8, 1.0), deadlines)
    normal = trace_grid_receipt({"trace_name": "normal", "attack_mechanism_group": ""}, timestamps, np.zeros_like(labels), alarm_times(scores, timestamps, 0.8, 1.0), deadlines)
    aggregate = aggregate_receipts([attack, normal], deadlines, horizons)
    bootstrap = nested_bootstrap_ci([attack, normal], deadlines, horizons, 20, 20260923)
    synthetic_stream = [(np.asarray([0.1, 0.2, 0.3, 0.4]), np.asarray([0.0, 0.5, 1.0, 1.5]))]
    synthetic_thresholds = [models.calibrate_threshold(synthetic_stream, budget, 0.5)["threshold"] for budget in (600.0, 1200.0, 2400.0)]
    return {
        "alarm_counts_nonincreasing_with_cooldown": counts[0.1] >= counts[1.0] >= counts[5.0],
        "deadline_der_nondecreasing": list(attack["der_by_deadline"].values()) == sorted(attack["der_by_deadline"].values()),
        "rmttd_finite": all(math.isfinite(value) for value in aggregate["rmttd_trace_mean_by_horizon"].values()),
        "aggregate_fae_finite": bool(math.isfinite(float(aggregate["outer_normal_fae_per_hour"]))),
        "nested_bootstrap_executes": bool(bootstrap) and all(len(interval) == 2 for interval in bootstrap.values()),
        "thresholds_nonincreasing_with_budget": all(left >= right for left, right in zip(synthetic_thresholds, synthetic_thresholds[1:])),
    }


def static_check(config: dict[str, Any]) -> dict[str, Any]:
    role_csv = Path(config["data"]["role_csv"])
    archive = Path(config["data"]["archive"])
    cache = Path(config["data"]["cache"])
    cache_manifest_path = cache / "cache_manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8")) if cache_manifest_path.is_file() else {}
    checks: dict[str, bool] = {
        "archive_exists_and_size_matches": archive.is_file() and archive.stat().st_size == int(config["data"]["expected_bytes"]),
        "role_csv_sha256_matches": role_csv.is_file() and sha256_file(role_csv).lower() == config["data"]["role_csv_sha256"].lower(),
        "cache_manifest_present": cache_manifest_path.is_file(),
        "cache_families_exact": cache_manifest.get("families") == ["advanced", "benign", "dos", "fuzzing", "replay", "spoofing", "v_mode"],
        "cache_payload_sizes_match_manifest": bool(cache_manifest.get("families")) and all((cache / f"{family}.zip").is_file() and (cache / f"{family}.zip").stat().st_size == int(cache_manifest["inner_zip_bytes"][family]) for family in cache_manifest.get("families", [])),
        "probe_families_exact": config["probe_families"]["main"] == ["logistic_regression", "hgb", "early_fusion_tcn"] and config["probe_families"]["supplement_sanity_only"] == ["residual_max"],
        "probe_manifest_exact_six": len(probe_manifest(config)) == 6,
        "budget_grid_exact": config["alarm_sensitivity_grid"]["calibration_budgets_fae_per_hour"] == [10.0, 30.0, 60.0],
        "cooldown_grid_exact": config["alarm_sensitivity_grid"]["cooldown_seconds"] == [0.1, 1.0, 5.0],
        "deadline_grid_exact": config["alarm_sensitivity_grid"]["event_deadlines_seconds"] == [0.01, 0.05, 0.1, 0.5],
        "rmttd_grid_exact": config["alarm_sensitivity_grid"]["rmttd_horizons_seconds"] == [0.1, 0.5, 5.0],
        "primary_setting_exact": config["primary_setting"] == {"calibration_budget_fae_per_hour": 30.0, "cooldown_seconds": 1.0, "event_deadline_seconds": 0.1, "rmttd_horizon_seconds": 5.0},
        "normal_calibration_only": config["alarm_sensitivity_grid"]["threshold_source"] == "normal_calibration_only" and not config["alarm_sensitivity_grid"]["outer_test_threshold_change"],
        "posthoc_grid_selection_forbidden": not config["stopping_rules"]["posthoc_grid_selection"] and config["publication_boundary"]["primary_setting_cannot_be_replaced_by_best_sensitivity_cell"],
        "automatic_retry_disabled": not config["stopping_rules"]["automatic_retry"],
        "full_scores_not_required_by_contract": not config["output_contract"]["save_full_per_frame_scores"],
        "no_raw_stream_read": True,
        "no_model_fit": True,
        "no_outer_test_label_read": True,
    }
    for name, path in SOURCE_FILES.items():
        checks[f"source_present__{name}"] = path.is_file()
    training_config = load_training_config(config)
    for fold in config["data"]["outer_folds"]:
        roles = assembly.roles_for_fold(training_config, int(fold))
        checks[f"fold_{fold}_role_contract"] = (
            len(roles["references"]) == 10
            and len(roles["normal_train"]) == 9
            and len(roles["normal_calibration"]) == 7
            and len(roles["outer_normal"]) == 13
            and bool(roles["attack_train"])
            and bool(roles["outer_attack"])
        )
    checks.update({f"synthetic__{key}": value for key, value in synthetic_checks(config).items()})
    disk = shutil.disk_usage(ROOT)
    checks["free_disk_at_least_20_gib"] = disk.free >= 20 * 1024**3
    return {
        "status": "STATIC_PASS" if all(checks.values()) else "STATIC_FAIL",
        "checks": checks,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "free_disk_gib": disk.free / 1024**3,
        },
        "scope": {
            "raw_archive_content_read": False,
            "cached_can_stream_read": False,
            "model_training": False,
            "outer_test_labels_read": False,
        },
    }


def snapshot_inputs(output: Path, config_path: Path) -> None:
    shutil.copy2(Path(__file__), output / "script_snapshot.py")
    shutil.copy2(config_path, output / "protocol_snapshot.json")
    shutil.copy2(TRAINING_CONFIG, output / "training_config_snapshot.json")
    for name, path in SOURCE_FILES.items():
        shutil.copy2(path, output / f"{name}_snapshot.py")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    if args.check == args.formal:
        parser.error("pass exactly one of --check or --formal")
    output = args.out.resolve()
    config_path = args.config.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True)
    snapshot_inputs(output, config_path)
    dump(
        output / "RUNNING.json",
        {
            "protocol_id": "CARDS_OPERATIONAL_EVALUATION_V1",
            "mode": "check" if args.check else "formal",
        },
    )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if args.check:
            result = static_check(config)
        else:
            preflight = static_check(config)
            if preflight["status"] != "STATIC_PASS":
                raise RuntimeError("formal preflight failed")
            cache_integrity = verify_cache_payloads(config, output)
            training_config = load_training_config(config)
            fold_results: list[dict[str, Any]] = []
            for fold in config["data"]["outer_folds"]:
                fold_result = run_fold(config, training_config, int(fold), output)
                dump(output / f"fold_{fold}_result.json", fold_result)
                dump(output / f"fold_{fold}_COMPLETED.json", {"fold": fold, "status": "FOLD_COMPLETED"})
                fold_results.append(fold_result)
            result = {
                "status": "FORMAL_COMPLETED",
                "protocol_id": config["protocol_id"],
                "preflight_environment": preflight["environment"],
                "cache_integrity_receipt": "cache_integrity_receipt.json",
                "cache_integrity_status": cache_integrity["status"],
                "folds": [
                    {
                        "fold": item["fold"],
                        "result_file": f"fold_{item['fold']}_result.json",
                        "alarm_event_file": item["alarm_event_file"],
                    }
                    for item in fold_results
                ],
                "interpretation_boundary": "sensitivity_grid_is_not_authorized_for_posthoc_primary_setting_selection",
            }
        dump(output / "result.json", result)
        passed = result["status"] in {"STATIC_PASS", "FORMAL_COMPLETED"}
        marker = "COMPLETED.json" if passed else "FAILED.json"
        dump(output / marker, {"status": result["status"]})
        (output / "RUNNING.json").unlink(missing_ok=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if passed else 2
    except Exception as exc:
        dump(output / "FAILED.json", {"status": "FAILED", "error": str(exc), "traceback": traceback.format_exc()})
        (output / "RUNNING.json").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

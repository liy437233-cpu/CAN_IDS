"""CarDS cached-profile construction and deterministic feature checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

import cards_features as base
import cards_cache as cache_io


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cards_training.json"
WORKERS = 6


def dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["family"], row["trace_name"], row["role"], row.get("attack_mechanism_group", ""))


def build_cached_profile(cache: Path, references: list[dict[str, str]]) -> dict[str, Any]:
    """Byte-for-byte feature logic of base.build_profile, using only cache."""
    profile = base.blank_profile()
    for row in sorted(references, key=canonical_row_key):
        state = base.StreamState()
        for timestamp, bus, can_id, frame_type, payload, direction in cache_io.cached_records(cache, row):
            if direction != "R":
                raise RuntimeError(f"normal reference unexpectedly contains T: {row['trace_name']}")
            event = state.step(timestamp, bus, can_id, payload)
            key = event["key"]
            profile["bus_total"][bus] += 1; profile["route"][key] += 1
            profile["format"][(bus, can_id, frame_type, len(payload))] += 1
            delta_bin = min(16, int(round(event["hamming"] * 16)))
            profile["delta"][(bus, can_id, delta_bin)] += 1; profile["delta_total"][key] += 1
            predecessor_key = (bus, event["predecessor"], can_id)
            profile["predecessor"][predecessor_key] += 1; profile["predecessor_total"][(bus, event["predecessor"])] += 1
            for position, byte in enumerate(payload):
                profile["byte"][(bus, can_id, position, byte)] += 1; profile["byte_total"][(bus, can_id, position)] += 1
            profile["iat"][key].add(event["log_iat"]); profile["iat_global"].add(event["log_iat"])
            profile["hamming"][key].add(event["hamming"]); profile["hamming_global"].add(event["hamming"])
            profile["density10"][bus].add(event["density10"]); profile["density10_global"].add(event["density10"])
            profile["density100"][bus].add(event["density100"]); profile["density100_global"].add(event["density100"])
            profile["entropy"][bus].add(event["entropy"]); profile["entropy_global"].add(event["entropy"])
            values = profile["iat_samples"][key]
            if len(values) < 256:
                values.append(event["log_iat"])
    for bus in profile["bus_total"]:
        total = max(1, profile["bus_total"][bus])
        profile["route_prob"][bus] = {can_id: count / total for (route_bus, can_id), count in profile["route"].items() if route_bus == bus}
    for values in profile["iat_samples"].values():
        values.sort()
    return profile


def fit_cached_normalizer(cache: Path, references: list[dict[str, str]], profile: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    rng = random.Random(20260908)
    seen, cap = 0, 50_000
    for row in sorted(references, key=canonical_row_key):
        state = base.StreamState()
        for timestamp, bus, can_id, frame_type, payload, direction in cache_io.cached_records(cache, row):
            if direction != "R":
                raise RuntimeError("normal reference includes T while fitting normalizer")
            raw = base.feature_raw(state.step(timestamp, bus, can_id, payload), bus, can_id, frame_type, payload, profile)
            if len(values) < cap:
                values.append(raw)
            else:
                slot = rng.randrange(seen + 1)
                if slot < cap:
                    values[slot] = raw
            seen += 1
    matrix = np.stack(values)
    center = np.median(matrix, axis=0)
    scale = np.maximum(np.median(np.abs(matrix - center), axis=0) * 1.4826, 1e-4)
    return center.astype(np.float32), scale.astype(np.float32)


def build_tasks(config: dict[str, Any], selection: dict[str, Any]) -> list[tuple[dict[str, str], int, int, int]]:
    tasks: list[tuple[dict[str, str], int, int, int]] = []
    for row in selection["negatives"]:
        tasks.append((row, -1, base.SMOKE_PER_NORMAL_CAP, base.SMOKE_PER_ATTACK_LABEL_CAP))
    for group, row in enumerate(selection["attacks"]):
        tasks.append((row, group, base.SMOKE_PER_ATTACK_LABEL_CAP, base.SMOKE_PER_ATTACK_LABEL_CAP))
    if len(tasks) != 8 or any(not row["role"].startswith("model_train") for row, *_ in tasks):
        raise RuntimeError("training task contract failed")
    return sorted(tasks, key=lambda item: canonical_row_key(item[0]))


def run_tasks(cache: Path, tasks: list[tuple[dict[str, str], int, int, int]], profile: dict[str, Any], center: np.ndarray, scale: np.ndarray, sequence_length: int, parallel: bool) -> list[dict[str, Any]]:
    invoke = lambda task: cache_io.worker_sample(str(cache), task[0], profile, center, scale, sequence_length, task[1], task[2], task[3])
    if not parallel:
        items = [invoke(task) for task in tasks]
    else:
        items = []
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = [executor.submit(cache_io.worker_sample, str(cache), row, profile, center, scale, sequence_length, group, normal_cap, attack_cap) for row, group, normal_cap, attack_cap in tasks]
            for future in as_completed(futures):
                items.append(future.result())
    return sorted(items, key=lambda item: canonical_row_key(item["row"]))


def canonical_arrays(items: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    samples = [sample for item in items for sample in item["samples"]]
    if not samples:
        raise RuntimeError("equivalence selection retained no windows")
    x = np.stack([sample[0] for sample in samples]).astype("<f4", copy=False)
    y = np.asarray([sample[1] for sample in samples], dtype="<i8")
    group = np.asarray([sample[2] for sample in samples], dtype="<i8")
    receipts = [{"trace_name": item["row"]["trace_name"], "counts": item["counts"]} for item in items]
    return x, y, group, receipts


def array_hash(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes(order="C"))


def static_result(config_path: Path) -> dict[str, Any]:
    config = base.read_config(config_path)
    selection = base.select_rows(config)
    cache_manifest = cache_io.CACHE_DIR / "cache_manifest.json"
    checks = {
        "torch_import": torch.__version__ != "",
        "workers_fixed_at_6": WORKERS == 6,
        "selection_train_side_only": all(row["role"].startswith("model_train") for key in ("references", "negatives", "attacks") for row in selection[key]),
        "forbidden_contract_complete": selection["forbidden_read_contract"] == ["outer_test_attack", "outer_test_normal", "uds", "ethernet"],
        "cache_manifest_exists": cache_manifest.is_file(),
        "no_outer_test_read": True,
        "no_model_training": True,
        "no_protocol_mutation": True,
    }
    return {"status": "CHECK_PASS_NO_RAW_READ" if all(checks.values()) else "CHECK_FAIL", "checks": checks,
            "scope": {"raw_archive_content_read": False, "outer_test_labels_read": False, "model_training": False}}


def equivalence(config_path: Path) -> dict[str, Any]:
    config = base.read_config(config_path)
    selection = base.select_rows(config)
    if any(not row["role"].startswith("model_train") for key in ("references", "negatives", "attacks") for row in selection[key]):
        raise RuntimeError("non-training role passed into the profile check")
    cache = cache_io.CACHE_DIR
    manifest = json.loads((cache / "cache_manifest.json").read_text(encoding="utf-8"))
    required = sorted({row["family"] for key in ("references", "negatives", "attacks") for row in selection[key]})
    if manifest.get("families") != required:
        raise RuntimeError("cache families differ from the training-side selection")
    for family in required:
        if sha256_file(cache / f"{family}.zip") != manifest["sha256"][family]:
            raise RuntimeError(f"cache hash mismatch: {family}")
    profile = build_cached_profile(cache, selection["references"])
    center, scale = fit_cached_normalizer(cache, selection["references"], profile)
    tasks = build_tasks(config, selection)
    sequence_length = int(config["inputs"]["sequence_length_same_bus"])
    sequential = run_tasks(cache, tasks, profile, center, scale, sequence_length, parallel=False)
    parallel = run_tasks(cache, tasks, profile, center, scale, sequence_length, parallel=True)
    sx, sy, sg, sr = canonical_arrays(sequential)
    px, py, pg, pr = canonical_arrays(parallel)
    hashes = {
        "sequential": {"x": array_hash(sx), "y": array_hash(sy), "group": array_hash(sg), "center": array_hash(center), "scale": array_hash(scale)},
        "parallel": {"x": array_hash(px), "y": array_hash(py), "group": array_hash(pg), "center": array_hash(center), "scale": array_hash(scale)},
    }
    receipt_match = sr == pr
    exact_match = hashes["sequential"] == hashes["parallel"] and receipt_match and sx.shape == px.shape and np.array_equal(sx, px) and np.array_equal(sy, py) and np.array_equal(sg, pg)
    return {"status": "DETERMINISM_EQUIVALENCE_PASS_TRAIN_SIDE_ONLY" if exact_match else "DETERMINISM_EQUIVALENCE_FAIL",
            "selection": {"reference_trace_count": len(selection["references"]), "task_trace_count": len(tasks), "task_trace_names": [task[0]["trace_name"] for task in tasks]},
            "cache": {"path": str(cache), "families": required, "hashes_verified": True},
            "canonicalization": {"trace_key": ["family", "trace_name", "role", "attack_mechanism_group"], "sample_order": "per_trace_reservoir_slot_order_after_canonical_trace_sort", "reservoir_seed": "per_trace_fixed"},
            "arrays": {"shape": list(sx.shape), "sample_count": int(len(sx)), "positive_samples": int(sy.sum()), "hashes": hashes, "receipts_match": receipt_match},
            "scope": {"outer_test_read": False, "uds_read": False, "ethernet_read": False, "model_training": False, "raw_archive_mutated": False}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--equivalence", action="store_true")
    args = parser.parse_args()
    if args.check == args.equivalence:
        parser.error("pass exactly one of --check or --equivalence")
    output, config_path = args.out.resolve(), args.config.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output / "script_snapshot.py")
    shutil.copy2(config_path, output / "protocol_snapshot.json")
    dump(output / "RUNNING.json", {"pid": os.getpid(), "mode": "check" if args.check else "equivalence", "outer_test_read": False, "model_training": False})
    try:
        result = static_result(config_path) if args.check else equivalence(config_path)
        dump(output / "result.json", result)
        marker = "COMPLETED.json" if result["status"].endswith("PASS_NO_RAW_READ") or result["status"].endswith("PASS_TRAIN_SIDE_ONLY") else "FAILED.json"
        dump(output / marker, {"status": result["status"]})
        (output / "RUNNING.json").unlink(missing_ok=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if marker == "COMPLETED.json" else 2
    except Exception as exc:
        dump(output / "FAILED.json", {"status": "FAILED", "error": str(exc), "traceback": traceback.format_exc()})
        (output / "RUNNING.json").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

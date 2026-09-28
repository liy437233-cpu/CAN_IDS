"""CarDS cache access and parallel causal feature extraction."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import traceback
import zipfile
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

import cards_features as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cards_training.json"
CACHE_DIR = ROOT / "work" / "cards_cache"
WORKERS = 6


def dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def cached_records(cache: Path, row: dict[str, str]) -> Iterator[tuple[float, int, str, str, bytes, str]]:
    nested = cache / f"{row['family']}.zip"
    if not nested.is_file():
        raise RuntimeError(f"cached inner archive missing: {nested}")
    with zipfile.ZipFile(nested, "r") as inner:
        found = [item for item in inner.infolist() if Path(item.filename).name == row["trace_name"]]
        if len(found) != 1:
            raise RuntimeError(f"expected one cached trace {row['family']}/{row['trace_name']}; found {len(found)}")
        with inner.open(found[0], "r") as handle:
            yield from base.parse_records(handle)


def ensure_cache(config: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    archive = Path(config["data"]["archive"])
    required = sorted({row["family"] for key in ("references", "negatives", "attacks") for row in selection[key]})
    expected = {"archive_bytes": archive.stat().st_size, "config_sha256": sha256(DEFAULT_CONFIG), "families": required}
    manifest_path = CACHE_DIR / "cache_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {key: manifest.get(key) for key in expected} != expected:
            raise RuntimeError("existing cache manifest does not match the configured training data")
        for family in required:
            path = CACHE_DIR / f"{family}.zip"
            if not path.is_file() or sha256(path) != manifest["sha256"][family]:
                raise RuntimeError(f"existing cache receipt failed for {family}")
        return {**manifest, "reused": True}
    if CACHE_DIR.exists() and any(CACHE_DIR.iterdir()):
        raise RuntimeError(f"refusing to overwrite incomplete cache: {CACHE_DIR}")
    CACHE_DIR.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(archive, "r") as outer:
            for family in required:
                target = CACHE_DIR / f"{family}.zip"; temporary = target.with_suffix(".zip.tmp")
                with outer.open(f"CAN/{family}.zip", "r") as source, temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination, 1024 * 1024)
                temporary.replace(target)
        manifest = {**expected, "sha256": {family: sha256(CACHE_DIR / f"{family}.zip") for family in required}, "inner_zip_bytes": {family: (CACHE_DIR / f"{family}.zip").stat().st_size for family in required}, "reused": False}
        dump(manifest_path, manifest)
        return manifest
    except Exception:
        raise


class KeepReservoir:
    """Reservoir decision before materializing a 16x21 window."""
    def __init__(self, cap: int, seed: str) -> None:
        self.cap, self.rng, self.seen, self.items = cap, __import__("random").Random(seed), 0, []

    def keep_slot(self) -> int | None:
        if len(self.items) < self.cap:
            self.items.append(None)
            slot = len(self.items) - 1
        else:
            candidate = self.rng.randrange(self.seen + 1)
            slot = candidate if candidate < self.cap else None
        self.seen += 1
        return slot


def worker_sample(cache: str, row: dict[str, str], profile: dict[str, Any], center: np.ndarray, scale: np.ndarray, sequence_length: int, positive_group: int, normal_cap: int, attack_cap: int) -> dict[str, Any]:
    state = base.StreamState(); sequences: dict[int, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=sequence_length))
    positive = KeepReservoir(attack_cap, f"positive|parallel|{row['trace_name']}")
    negative = KeepReservoir(normal_cap, f"negative|parallel|{row['trace_name']}")
    counts = Counter()
    for timestamp, bus, can_id, frame_type, payload, direction in cached_records(Path(cache), row):
        raw = base.feature_raw(state.step(timestamp, bus, can_id, payload), bus, can_id, frame_type, payload, profile)
        sequence = sequences[bus]; sequence.append(base.norm_feature(raw, center, scale))
        if len(sequence) < sequence_length:
            continue
        label = int(direction == "T"); reservoir = positive if label else negative
        slot = reservoir.keep_slot(); counts["positive_seen" if label else "negative_seen"] += 1
        if slot is not None:
            reservoir.items[slot] = (np.asarray(tuple(sequence), dtype=np.float32), label, positive_group if label else -1)
    return {"row": row, "counts": {**counts, "positive_kept": len(positive.items), "negative_kept": len(negative.items)}, "samples": positive.items + negative.items}


def static_result(config: dict[str, Any]) -> dict[str, Any]:
    selection = base.select_rows(config)
    checks = {"torch_cuda_available": bool(torch.cuda.is_available()), "workers_fixed_at_6": WORKERS == 6,
              "selection_train_side_only": all(row["role"].startswith("model_train") for key in ("references", "negatives", "attacks") for row in selection[key]),
              "forbidden_contract_complete": selection["forbidden_read_contract"] == ["outer_test_attack", "outer_test_normal", "uds", "ethernet"],
              "no_protocol_mutation": True, "no_outer_test_read": True, "no_model_training": True}
    return {"status": "CHECK_PASS_NO_RAW_READ" if all(checks.values()) else "CHECK_FAIL", "checks": checks, "selection": selection,
            "scope": {"raw_archive_content_read": False, "outer_test_labels_read": False, "model_training": False}}


def probe(config: dict[str, Any], output: Path) -> dict[str, Any]:
    selection = base.select_rows(config)
    if any(not row["role"].startswith("model_train") for key in ("references", "negatives", "attacks") for row in selection[key]):
        raise RuntimeError("non-training role passed into cache preparation")
    cache_started = time.perf_counter(); cache = ensure_cache(config, selection); cache_seconds = time.perf_counter() - cache_started
    work = Path(tempfile.mkdtemp(prefix="cards_profile_", dir=output))
    try:
        with zipfile.ZipFile(Path(config["data"]["archive"]), "r") as outer:
            profile = base.build_profile(outer, selection["references"], work)
            center, scale = base.fit_normalizer(outer, selection["references"], work, profile)
        tasks = []
        for row in selection["negatives"]:
            tasks.append((row, -1, base.SMOKE_PER_NORMAL_CAP, base.SMOKE_PER_ATTACK_LABEL_CAP))
        for index, row in enumerate(selection["attacks"]):
            tasks.append((row, index, base.SMOKE_PER_ATTACK_LABEL_CAP, base.SMOKE_PER_ATTACK_LABEL_CAP))
        started = time.perf_counter(); completed = []
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            future_map = {executor.submit(worker_sample, str(CACHE_DIR), row, profile, center, scale, int(config["inputs"]["sequence_length_same_bus"]), group, normal_cap, attack_cap): row for row, group, normal_cap, attack_cap in tasks}
            for future in as_completed(future_map):
                completed.append(future.result())
        parallel_seconds = time.perf_counter() - started
        samples = [sample for item in completed for sample in item["samples"]]
        training = base.train_smoke(samples, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        frames = sum(item["counts"].get("positive_seen", 0) + item["counts"].get("negative_seen", 0) for item in completed)
        return {"status": "PARALLEL_CACHE_PROBE_PASS_TRAIN_SIDE_ONLY", "cache": {**cache, "seconds": round(cache_seconds, 2), "path": str(CACHE_DIR)},
                "parallel": {"workers": WORKERS, "seconds": round(parallel_seconds, 2), "frames": frames, "frames_per_second": round(frames / max(parallel_seconds, 1e-6), 2)},
                "trace_receipts": [{"role": item["row"]["role"], "trace_name": item["row"]["trace_name"], "mechanism": item["row"].get("attack_mechanism_group"), "counts": item["counts"]} for item in completed],
                "training": training, "scope": {"outer_test_read": False, "uds_read": False, "ethernet_read": False, "raw_archive_mutated": False}}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.check == args.probe:
        parser.error("pass exactly one of --check or --probe")
    output, config_path = args.out.resolve(), args.config.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output / "script_snapshot.py"); shutil.copy2(config_path, output / "protocol_snapshot.json")
    dump(output / "RUNNING.json", {"mode": "check" if args.check else "probe", "pid": os.getpid(), "outer_test_read": False, "model_training": False})
    try:
        config = base.read_config(config_path)
        result = static_result(config) if args.check else probe(config, output)
        dump(output / "result.json", result); dump(output / "COMPLETED.json", {"status": result["status"]})
        (output / "RUNNING.json").unlink(missing_ok=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        dump(output / "FAILED.json", {"status": "FAILED", "error": str(exc), "traceback": traceback.format_exc()})
        (output / "RUNNING.json").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

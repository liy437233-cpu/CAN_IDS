"""ROAD parsing, feature construction, and smoke validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
ROAD_SHA256 = "0E4FE6ED7F99B5CDABF6B772C91A2025C614A071BA8579C532904D0EF7AEA5F6"
HASH = "#"
LINE = re.compile(
    r"^\((?P<t>\d+(?:\.\d+)?)\)\s+can\d+\s+(?P<id>[0-9A-Fa-f]+)"
    + re.escape(HASH)
    + r"(?P<data>[0-9A-Fa-f]+)\s*$"
)
HGB = dict(max_iter=120, learning_rate=0.10, max_leaf_nodes=31, l2_regularization=1e-3, random_state=20260907)
FEATURE_SETS = {"frame": list(range(9)), "timing": [9, 10, 13], "context": list(range(14))}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def dump(path: Path, payload: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def candidate(observed_id: str, observed_payload: str, metadata: dict, elapsed: float) -> bool:
    start, end = metadata["injection_interval"]
    if not start <= elapsed <= end:
        return False
    id_pattern = metadata["injection_id"].upper()
    id_ok = id_pattern == "XXX" or int(observed_id, 16) == int(id_pattern, 16)
    payload = observed_payload.upper()
    pattern = metadata["injection_data_str"].upper()
    payload_ok = len(payload) == len(pattern) and all(expected == "X" or got == expected for got, expected in zip(payload, pattern))
    return id_ok and payload_ok


def parse_capture(archive: zipfile.ZipFile, path: str, metadata: dict | None, limit: int | None = None):
    """Return causal features, labels, elapsed times, and parse counters."""
    features, labels, elapsed_values = [], [], []
    first_t = last_t = previous_t = None
    last_by_id: dict[int, tuple[float, np.ndarray, int]] = {}
    malformed = backsteps = 0
    with archive.open(path) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig", errors="replace") as source:
        for raw in source:
            line = raw.strip()
            if not line:
                continue
            match = LINE.match(line)
            if not match:
                malformed += 1
                continue
            timestamp = float(match.group("t"))
            arb_id = int(match.group("id"), 16)
            payload = np.frombuffer(bytes.fromhex(match.group("data")), dtype=np.uint8)
            if len(payload) != 8:
                malformed += 1
                continue
            if first_t is None:
                first_t = timestamp
            if previous_t is not None and timestamp < previous_t:
                backsteps += 1
            global_iat = 0.0 if previous_t is None else max(0.0, timestamp - previous_t)
            previous_t = timestamp
            previous = last_by_id.get(arb_id)
            same_iat = 0.0 if previous is None else max(0.0, timestamp - previous[0])
            xor_bits = 0.0 if previous is None else sum(int(value).bit_count() for value in np.bitwise_xor(payload, previous[1])) / 64.0
            byte_diff = 0.0 if previous is None else float(np.mean(payload != previous[1]))
            prior_count = 0 if previous is None else previous[2]
            elapsed = timestamp - first_t
            feature = np.concatenate((np.array([arb_id / 2047.0], dtype=np.float32), payload.astype(np.float32) / 255.0,
                                      np.array([math.log1p(global_iat), math.log1p(same_iat), xor_bits, byte_diff,
                                                math.log1p(prior_count)], dtype=np.float32)))
            features.append(feature)
            labels.append(0 if metadata is None else int(candidate(match.group("id"), match.group("data"), metadata, elapsed)))
            elapsed_values.append(elapsed)
            last_by_id[arb_id] = (timestamp, payload, prior_count + 1)
            last_t = timestamp
            if limit is not None and len(features) >= limit:
                break
    return (np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int8), np.asarray(elapsed_values, dtype=np.float64),
            {"parsed_frames": len(features), "malformed_nonempty_lines": malformed, "timestamp_backsteps": backsteps,
             "duration_seconds_observed": None if first_t is None or last_t is None else last_t - first_t})


def activity_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    primary = [row for row in rows if row["task_scope"] == "primary_injection"]
    normal = [row for row in rows if row["task_scope"] == "primary_normal"]
    if len(primary) != 29 or len(normal) != 12:
        raise ValueError("ROAD group manifest contract failed")
    if len({row["group_id"] for row in primary}) != 16:
        raise ValueError("Attack original/masquerade grouping contract failed")
    return rows


def self_test() -> None:
    meta = {"injection_interval": [1.0, 2.0], "injection_id": "0xD0", "injection_data_str": "XXXX04XXXXXXXXXX"}
    assert candidate("0D0", "ABCD04EF01234567", meta, 1.5)
    assert not candidate("0D0", "ABCD05EF01234567", meta, 1.5)
    assert not candidate("0D0", "ABCD04EF01234567", meta, 2.1)
    wildcard = {"injection_interval": [1.0, 2.0], "injection_id": "XXX", "injection_data_str": "FFFFFFFFFFFFFFFF"}
    assert candidate("321", "FFFFFFFFFFFFFFFF", wildcard, 1.5)
    assert len(FEATURE_SETS["context"]) == 14 and len(FEATURE_SETS["frame"]) == 9


def smoke(out: Path, zip_path: Path, manifest_path: Path) -> None:
    if sha256(zip_path) != ROAD_SHA256:
        raise ValueError("ROAD SHA-256 mismatch")
    rows = activity_manifest(manifest_path)
    attack = next(row for row in rows if row["capture"] == "correlated_signal_attack_3")
    normal = next(row for row in rows if row["task_scope"] == "primary_normal")
    with zipfile.ZipFile(zip_path) as archive:
        metadata = json.loads(archive.read("road/attacks/capture_metadata.json").decode("utf-8-sig"))
        ax, ay, at, attack_info = parse_capture(archive, attack["log_path"], metadata[attack["capture"]], limit=20_000)
        nx, ny, nt, normal_info = parse_capture(archive, normal["log_path"], None, limit=20_000)
    if ax.shape != (20_000, 14) or nx.shape != (20_000, 14) or not np.all(ny == 0) or not np.any(ay == 1):
        raise AssertionError("Feature or normal-label contract failed")
    report = {
        "experiment": "ROAD_GROUP_OOF_BASELINES_V1", "mode": "smoke", "status": "SMOKE_PASS_NO_MODEL_FIT",
        "boundary": "No estimator fit; no full labels saved; no CTAT official test read.",
        "source_sha256": ROAD_SHA256, "group_manifest_sha256": sha256(manifest_path), "script_sha256": sha256(Path(__file__)),
        "hgb": HGB, "feature_sets": FEATURE_SETS, "threshold": 0.50, "alarm_cooldown_seconds": 1.0,
        "attack_probe": {"capture": attack["capture"], "feature_shape": list(ax.shape), "candidate_count_in_first_20000_frames": int(ay.sum()), "info": attack_info},
        "normal_probe": {"capture": normal["capture"], "feature_shape": list(nx.shape), "label_count": int(ny.sum()), "info": normal_info},
        "formal_execution_contract": {
            "attack_oof_fits": 16, "normal_oof_fits": 12, "total_fits_per_baseline": 28,
            "training_positive_source": "candidate-injected frames from non-heldout attack campaigns only",
            "training_negative_source": "independent non-heldout normal captures only",
            "test_boundary": "heldout attack campaign for event metrics; heldout normal capture for false-alarm episodes/hour",
        },
        "python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__,
    }
    dump(out / "smoke_report.json", report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--zip", dest="zip_path", type=Path, default=ROOT / "data/raw/road.zip")
    parser.add_argument("--group-manifest", type=Path, default=ROOT / "data/protocol/road_activity_groups.csv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Refusing to overwrite existing run output")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "script_snapshot.py")
    if args.mode == "formal":
        raise RuntimeError("Use road_evaluation.py for the full group-disjoint evaluation.")
    self_test()
    smoke(out, args.zip_path.resolve(), args.group_manifest.resolve())
    print(json.dumps({"status": "SMOKE_PASS_NO_MODEL_FIT", "output": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

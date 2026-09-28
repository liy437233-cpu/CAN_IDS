"""CarDS causal feature extraction and training-side smoke test."""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import traceback
import zipfile
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cards_training.json"
EXPECTED_FEATURE_DIMS = (4, 6, 6, 5)
SMOKE_FOLD = 0
SMOKE_REFERENCE_TRACES = 2
SMOKE_NEGATIVE_TRACES = 2
SMOKE_ATTACK_MECHANISMS = 6
SMOKE_PER_ATTACK_LABEL_CAP = 512
SMOKE_PER_NORMAL_CAP = 768
HASH = "#"
DOUBLE_HASH = HASH * 2
FRAME_PATTERN = (
    r"^\((?P<timestamp>[0-9]+(?:\.[0-9]+)?)\)\s+"
    r"can(?P<bus>[0-9]+)\s+(?P<frame>[0-9A-Fa-f]+(?:"
    + re.escape(DOUBLE_HASH)
    + "|"
    + re.escape(HASH)
    + r")[0-9A-Fa-f]*)\s+"
    r"(?P<direction>[RT])\s*$"
)
FRAME_RE = re.compile(FRAME_PATTERN)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def stable_key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_records(handle: Any) -> Iterator[tuple[float, int, str, str, bytes, str]]:
    for raw in handle:
        line = raw.decode("ascii", errors="replace").strip()
        if not line:
            continue
        match = FRAME_RE.match(line)
        if match is None:
            raise ValueError(f"unparseable CAN line: {line[:200]}")
        token = match.group("frame")
        delimiter = DOUBLE_HASH if DOUBLE_HASH in token else HASH
        can_id, payload_hex = token.split(delimiter, 1)
        if len(payload_hex) % 2:
            payload_hex = "0" + payload_hex
        yield (float(match.group("timestamp")), int(match.group("bus")), can_id.upper(), delimiter,
               bytes.fromhex(payload_hex), match.group("direction"))


def extract_inner(outer: zipfile.ZipFile, family: str, work: Path) -> Path:
    target = work / f"{family}.zip"
    if not target.exists():
        with outer.open(f"CAN/{family}.zip", "r") as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination, 1024 * 1024)
    return target


def read_trace(outer: zipfile.ZipFile, row: dict[str, str], work: Path) -> Iterator[tuple[float, int, str, str, bytes, str]]:
    nested = extract_inner(outer, row["family"], work)
    with zipfile.ZipFile(nested, "r") as inner:
        matches = [item for item in inner.infolist() if Path(item.filename).name == row["trace_name"]]
        if len(matches) != 1:
            raise RuntimeError(f"expected one trace {row['family']}/{row['trace_name']}; found {len(matches)}")
        with inner.open(matches[0], "r") as handle:
            yield from parse_records(handle)


class Moments:
    def __init__(self) -> None:
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def add(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    def z(self, value: float) -> float:
        if self.n < 20:
            return 0.0
        sd = math.sqrt(max(self.m2 / max(1, self.n - 1), 1e-8))
        return (value - self.mean) / sd


class StreamState:
    """Causal state, reset per trace.  It never sees a future frame."""
    def __init__(self) -> None:
        self.last_id: dict[tuple[int, str], tuple[float, bytes]] = {}
        self.last_bus_id: dict[int, str] = {}
        self.bus_q10: dict[int, deque[float]] = defaultdict(deque)
        self.bus_q100: dict[int, deque[float]] = defaultdict(deque)
        self.bus_recent_ids: dict[int, deque[str]] = defaultdict(lambda: deque(maxlen=16))
        self.bus_recent_counts: dict[int, Counter[str]] = defaultdict(Counter)

    def step(self, timestamp: float, bus: int, can_id: str, payload: bytes) -> dict[str, Any]:
        key = (bus, can_id)
        previous = self.last_id.get(key)
        log_iat = math.log1p(max(0.0, timestamp - previous[0])) if previous else math.log1p(0.2)
        if previous is None:
            hamming = 1.0
            changed_byte_fraction = 1.0
        else:
            old = previous[1]
            hamming = (sum((a ^ b).bit_count() for a, b in zip(payload, old)) + 8 * abs(len(payload) - len(old))) / max(1, 8 * max(len(payload), len(old)))
            changed_byte_fraction = sum(a != b for a, b in zip(payload, old)) / max(1, max(len(payload), len(old)))
        q10, q100 = self.bus_q10[bus], self.bus_q100[bus]
        while q10 and timestamp - q10[0] > 0.010:
            q10.popleft()
        while q100 and timestamp - q100[0] > 0.100:
            q100.popleft()
        density10, density100 = float(len(q10)), float(len(q100))
        predecessor = self.last_bus_id.get(bus, "<START>")
        recent = self.bus_recent_ids[bus]
        counts = self.bus_recent_counts[bus]
        if len(recent) == recent.maxlen:
            counts[recent[0]] -= 1
            if counts[recent[0]] == 0:
                del counts[recent[0]]
        recent.append(can_id); counts[can_id] += 1
        n = len(recent)
        entropy = -sum((count / n) * math.log(max(count / n, 1e-12)) for count in counts.values()) if n else 0.0
        self.last_id[key] = (timestamp, payload)
        self.last_bus_id[bus] = can_id
        q10.append(timestamp); q100.append(timestamp)
        return {"key": key, "log_iat": log_iat, "hamming": hamming, "changed_byte_fraction": changed_byte_fraction,
                "density10": density10, "density100": density100, "predecessor": predecessor, "entropy": entropy,
                "recent_counts": dict(counts), "recent_n": n, "first_seen": previous is None}


def blank_profile() -> dict[str, Any]:
    return {"bus_total": Counter(), "route": Counter(), "format": Counter(), "delta": Counter(), "delta_total": Counter(),
            "predecessor": Counter(), "predecessor_total": Counter(), "byte": Counter(), "byte_total": Counter(),
            "iat": defaultdict(Moments), "iat_global": Moments(), "hamming": defaultdict(Moments), "hamming_global": Moments(), "density10": defaultdict(Moments), "density10_global": Moments(),
            "density100": defaultdict(Moments), "density100_global": Moments(), "entropy": defaultdict(Moments), "entropy_global": Moments(),
            "route_prob": {}, "iat_samples": defaultdict(list)}


def build_profile(outer: zipfile.ZipFile, references: list[dict[str, str]], work: Path) -> dict[str, Any]:
    profile = blank_profile()
    for row in references:
        state = StreamState()
        for timestamp, bus, can_id, frame_type, payload, direction in read_trace(outer, row, work):
            if direction != "R":
                raise RuntimeError(f"normal reference unexpectedly contains T: {row['trace_name']}")
            event = state.step(timestamp, bus, can_id, payload)
            key = event["key"]
            profile["bus_total"][bus] += 1; profile["route"][key] += 1
            profile["format"][(bus, can_id, frame_type, len(payload))] += 1
            delta_bin = min(16, int(round(event["hamming"] * 16)))
            profile["delta"][(bus, can_id, delta_bin)] += 1; profile["delta_total"][key] += 1
            pred_key = (bus, event["predecessor"], can_id)
            profile["predecessor"][pred_key] += 1; profile["predecessor_total"][(bus, event["predecessor"])] += 1
            for pos, byte in enumerate(payload):
                profile["byte"][(bus, can_id, pos, byte)] += 1; profile["byte_total"][(bus, can_id, pos)] += 1
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


def feature_raw(event: dict[str, Any], bus: int, can_id: str, frame_type: str, payload: bytes, profile: dict[str, Any]) -> np.ndarray:
    key = (bus, can_id)
    route_count = profile["route"][key]
    route_prob = (route_count + 1.0) / (profile["bus_total"][bus] + 10.0)
    fmt_count = profile["format"][(bus, can_id, frame_type, len(payload))]
    fmt_prob = (fmt_count + 1.0) / (route_count + 10.0)
    iat_m = profile["iat"].get(key, profile["iat_global"])
    hamming_m = profile["hamming"].get(key, profile["hamming_global"])
    d10_m = profile["density10"].get(bus, profile["density10_global"])
    d100_m = profile["density100"].get(bus, profile["density100_global"])
    entropy_m = profile["entropy"].get(bus, profile["entropy_global"])
    delta_bin = min(16, int(round(event["hamming"] * 16)))
    delta_prob = (profile["delta"][(bus, can_id, delta_bin)] + 1.0) / (profile["delta_total"][key] + 18.0)
    pred_key = (bus, event["predecessor"], can_id)
    pred_prob = (profile["predecessor"][pred_key] + 1.0) / (profile["predecessor_total"][(bus, event["predecessor"])] + 10.0)
    byte_surprisal = [-math.log((profile["byte"][(bus, can_id, pos, byte)] + 1.0) / (profile["byte_total"][(bus, can_id, pos)] + 256.0)) for pos, byte in enumerate(payload)] or [0.0]
    reference_probs = profile["route_prob"].get(bus, {})
    recent_n = max(1, int(event["recent_n"]))
    overlap = sum(min(count / recent_n, reference_probs.get(obs_id, 0.0)) for obs_id, count in event["recent_counts"].items())
    mix_divergence = max(0.0, 1.0 - overlap)
    samples = profile["iat_samples"].get(key, [])
    rank_z = 0.0 if len(samples) < 20 else 2.0 * (bisect.bisect_right(samples, event["log_iat"]) / len(samples) - 0.5)
    signed_iat = iat_m.z(event["log_iat"])
    signed_d10 = d10_m.z(event["density10"])
    return np.asarray([
        -math.log(route_prob), -math.log(fmt_prob), float(route_count == 0), float(fmt_count == 0),
        signed_iat, abs(signed_iat), signed_d10, abs(signed_d10), abs(d100_m.z(event["density100"])), float(event["first_seen"]),
        event["hamming"], abs(hamming_m.z(event["hamming"])), -math.log(delta_prob), event["changed_byte_fraction"], float(np.mean(byte_surprisal)), float(np.max(byte_surprisal)),
        -math.log(pred_prob), float(profile["predecessor"][pred_key] == 0), abs(entropy_m.z(event["entropy"])), mix_divergence, rank_z
    ], dtype=np.float32)


def fit_normalizer(outer: zipfile.ZipFile, references: list[dict[str, str]], work: Path, profile: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    rng = random.Random(20260908)
    seen, cap = 0, 50_000
    for row in references:
        state = StreamState()
        for timestamp, bus, can_id, frame_type, payload, direction in read_trace(outer, row, work):
            if direction != "R":
                raise RuntimeError("normal reference includes T while fitting normalizer")
            raw = feature_raw(state.step(timestamp, bus, can_id, payload), bus, can_id, frame_type, payload, profile)
            if len(values) < cap:
                values.append(raw)
            else:
                slot = rng.randrange(seen + 1)
                if slot < cap:
                    values[slot] = raw
            seen += 1
    array = np.stack(values)
    center = np.median(array, axis=0)
    scale = np.maximum(np.median(np.abs(array - center), axis=0) * 1.4826, 1e-4)
    return center.astype(np.float32), scale.astype(np.float32)


def norm_feature(raw: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((raw - center) / scale, -10.0, 10.0).astype(np.float32)


class Reservoir:
    def __init__(self, cap: int, seed: str) -> None:
        self.cap, self.rng, self.seen, self.items = cap, random.Random(seed), 0, []

    def add(self, value: tuple[np.ndarray, int, int]) -> None:
        if len(self.items) < self.cap:
            self.items.append(value)
        else:
            slot = self.rng.randrange(self.seen + 1)
            if slot < self.cap:
                self.items[slot] = value
        self.seen += 1


def sample_trace(outer: zipfile.ZipFile, row: dict[str, str], work: Path, profile: dict[str, Any], center: np.ndarray, scale: np.ndarray, seq_len: int, positive_group: int, normal_cap: int, attack_cap: int) -> tuple[list[tuple[np.ndarray, int, int]], dict[str, int]]:
    state = StreamState(); sequences: dict[int, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=seq_len))
    positive = Reservoir(attack_cap, f"positive|{row['trace_name']}")
    negative = Reservoir(normal_cap, f"negative|{row['trace_name']}")
    counts = Counter()
    for timestamp, bus, can_id, frame_type, payload, direction in read_trace(outer, row, work):
        raw = feature_raw(state.step(timestamp, bus, can_id, payload), bus, can_id, frame_type, payload, profile)
        sequence = sequences[bus]; sequence.append(norm_feature(raw, center, scale))
        if len(sequence) < seq_len:
            continue
        label = int(direction == "T")
        sample = (np.stack(tuple(sequence)), label, positive_group if label else -1)
        (positive if label else negative).add(sample); counts["positive_seen" if label else "negative_seen"] += 1
    samples = positive.items + negative.items
    counts["positive_kept"], counts["negative_kept"] = len(positive.items), len(negative.items)
    return samples, dict(counts)


def train_smoke(samples: list[tuple[np.ndarray, int, int]]) -> dict[str, Any]:
    if not samples:
        raise RuntimeError("smoke sampling retained no sequence")
    x = np.stack([sample[0] for sample in samples]).astype(np.float32)
    y = np.asarray([sample[1] for sample in samples], dtype=np.uint8)
    if not (np.isfinite(x).all() and x.shape[2] == sum(EXPECTED_FEATURE_DIMS)):
        raise RuntimeError("non-finite or unexpected feature matrix")
    if len(np.unique(y)) != 2:
        raise RuntimeError("smoke sample must contain normal and attack windows")
    model = HistGradientBoostingClassifier(max_iter=10, max_leaf_nodes=15, random_state=20260908)
    model.fit(x[:, -1, :], y)
    probability = model.predict_proba(x[:, -1, :])[:, 1]
    return {"sample_count": int(len(samples)), "positive_samples": int(y.sum()), "negative_samples": int((1 - y).sum()),
            "feature_shape": list(x.shape), "finite_features": True,
            "prediction_finite": bool(np.isfinite(probability).all()), "model": "hist_gradient_boosting"}


def select_rows(config: dict[str, Any]) -> dict[str, Any]:
    role_csv = Path(config["data"]["role_csv"])
    with role_csv.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["outer_fold"]) == SMOKE_FOLD]
    forbidden = [row for row in rows if row["role"].startswith("outer_test") or row["family"] in {"uds", "ethernet"}]
    if not forbidden:
        raise RuntimeError("expected held-out roles were not found")
    train_attack = [row for row in rows if row["role"] == "model_train_attack"]
    train_normal = [row for row in rows if row["role"] == "model_train_normal"]
    if len(train_attack) != 114 or len(train_normal) != 19:
        raise RuntimeError("fold-0 role counts differ")
    assignment = config["normal_reference_split"]["trace_assignment"][str(SMOKE_FOLD)]
    by_name = {row["trace_name"]: row for row in train_normal}
    references = [by_name[name] for name in assignment["reference_traces"][:SMOKE_REFERENCE_TRACES]]
    negatives = [by_name[name] for name in assignment["normal_training_traces"][:SMOKE_NEGATIVE_TRACES]]
    by_mechanism: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train_attack:
        by_mechanism[row["attack_mechanism_group"]].append(row)
    chosen_attack = []
    for mechanism, group in by_mechanism.items():
        chosen_attack.append(min(group, key=lambda row: stable_key("cards_smoke", mechanism, row["trace_name"])))
    attacks = sorted(chosen_attack, key=lambda row: stable_key("cards_smoke", "mechanism", row["attack_mechanism_group"]))[:SMOKE_ATTACK_MECHANISMS]
    if len(references) != 2 or len(negatives) != 2 or len(attacks) != 6:
        raise RuntimeError("smoke selection contract failed")
    selected = references + negatives + attacks
    if any(row["role"].startswith("outer_test") or row["family"] in {"uds", "ethernet"} for row in selected):
        raise RuntimeError("forbidden trace selected")
    return {"references": references, "negatives": negatives, "attacks": attacks,
            "forbidden_read_contract": ["outer_test_attack", "outer_test_normal", "uds", "ethernet"], "raw_model_inputs": False}


def static_result(config_path: Path) -> dict[str, Any]:
    config = read_config(config_path)
    selected = select_rows(config)
    archive = Path(config["data"]["archive"])
    checks = {"torch_import": True, "cuda_available": bool(torch.cuda.is_available()), "archive_size_matches": archive.is_file() and archive.stat().st_size == config["data"]["expected_bytes"],
              "feature_dim_matches_protocol": sum(EXPECTED_FEATURE_DIMS) == sum(len(value) for value in config["inputs"]["groups"].values()),
              "selected_roles_train_side_only": True, "no_outer_test_read": True, "no_model_training": True}
    return {"status": "CHECK_PASS_NO_RAW_READ" if all(checks.values()) else "CHECK_FAIL", "checks": checks, "selection": selected,
            "scope": {"raw_archive_content_read": False, "outer_test_labels_read": False, "model_training": False}}


def smoke(config_path: Path, output: Path) -> dict[str, Any]:
    config = read_config(config_path); selected = select_rows(config); archive = Path(config["data"]["archive"])
    if archive.stat().st_size != config["data"]["expected_bytes"]:
        raise RuntimeError("archive size receipt mismatch")
    output_tmp = Path(tempfile.mkdtemp(prefix="cards_nested_", dir=output))
    try:
        with zipfile.ZipFile(archive, "r") as outer:
            profile = build_profile(outer, selected["references"], output_tmp)
            center, scale = fit_normalizer(outer, selected["references"], output_tmp, profile)
            all_samples: list[tuple[np.ndarray, int, int]] = []; receipts = []
            for row in selected["negatives"]:
                values, receipt = sample_trace(outer, row, output_tmp, profile, center, scale, int(config["inputs"]["sequence_length_same_bus"]), -1, SMOKE_PER_NORMAL_CAP, SMOKE_PER_ATTACK_LABEL_CAP)
                all_samples.extend(values); receipts.append({"role": row["role"], "trace_name": row["trace_name"], "counts": receipt})
            for index, row in enumerate(selected["attacks"]):
                values, receipt = sample_trace(outer, row, output_tmp, profile, center, scale, int(config["inputs"]["sequence_length_same_bus"]), index, SMOKE_PER_ATTACK_LABEL_CAP, SMOKE_PER_ATTACK_LABEL_CAP)
                all_samples.extend(values); receipts.append({"role": row["role"], "trace_name": row["trace_name"], "mechanism": row["attack_mechanism_group"], "counts": receipt})
        training = train_smoke(all_samples)
        return {"status": "SMOKE_PASS_TRAIN_SIDE_ONLY", "completed_at_utc": now(), "selection": selected, "trace_receipts": receipts, "training": training,
                "scope": {"outer_test_read": False, "uds_read": False, "ethernet_read": False, "raw_archive_mutated": False}}
    finally:
        shutil.rmtree(output_tmp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.check == args.smoke:
        parser.error("pass exactly one of --check or --smoke")
    output, config_path = args.out.resolve(), args.config.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output / "script_snapshot.py"); shutil.copy2(config_path, output / "protocol_snapshot.json")
    dump(output / "RUNNING.json", {"started_at_utc": now(), "pid": os.getpid(), "mode": "check" if args.check else "smoke", "outer_test_read": False})
    try:
        result = static_result(config_path) if args.check else smoke(config_path, output)
        dump(output / "result.json", result)
        marker = "COMPLETED.json" if result["status"].endswith("PASS_NO_RAW_READ") or result["status"].endswith("PASS_TRAIN_SIDE_ONLY") else "FAILED.json"
        dump(output / marker, {"status": result["status"], "completed_at_utc": now()})
        (output / "RUNNING.json").unlink(missing_ok=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if marker == "COMPLETED.json" else 2
    except Exception as exc:
        dump(output / "FAILED.json", {"status": "FAILED", "error": str(exc), "traceback": traceback.format_exc()})
        (output / "RUNNING.json").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

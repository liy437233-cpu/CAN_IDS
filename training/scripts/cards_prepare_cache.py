"""Create a receipt-bearing opaque-byte cache for all CarDS CAN-primary families.

The build mode transfers nested ZIP bytes only.  It does not open nested logs,
parse frames, inspect R/T labels, fit models, or compute a metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import traceback
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cards_training.json"
ROLE_CSV = ROOT / "data" / "protocol" / "cards_trace_roles.csv"
CACHE = ROOT / "work" / "cards_cache"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def families() -> list[str]:
    import csv
    with ROLE_CSV.open(encoding="utf-8", newline="") as handle:
        return sorted({row["family"] for row in csv.DictReader(handle) if row["family"] not in {"uds", "ethernet"}})


def check(config: dict[str, Any]) -> dict[str, Any]:
    expected = families()
    checks = {"role_csv_exists": ROLE_CSV.is_file(), "archive_size_matches": Path(config["data"]["archive"]).is_file() and Path(config["data"]["archive"]).stat().st_size == config["data"]["expected_bytes"], "families_exact": expected == ["advanced", "benign", "dos", "fuzzing", "replay", "spoofing", "v_mode"], "cache_target_absent": not CACHE.exists()}
    return {"status": "FULL_CAN_CACHE_STATIC_PASS_NO_RAW_READ" if all(checks.values()) else "FULL_CAN_CACHE_STATIC_FAIL", "checks": checks, "families": expected, "scope": {"raw_archive_content_read": False, "nested_log_parsed": False, "outer_test_labels_read": False, "model_training": False}}


def build(config: dict[str, Any]) -> dict[str, Any]:
    expected = families()
    archive = Path(config["data"]["archive"])
    CACHE.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(archive, "r") as outer:
            names = set(outer.namelist())
            for family in expected:
                member = f"CAN/{family}.zip"
                if member not in names:
                    raise RuntimeError(f"missing expected opaque nested archive: {member}")
                target = CACHE / f"{family}.zip"
                temporary = target.with_suffix(".zip.tmp")
                with outer.open(member, "r") as source, temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination, 1024 * 1024)
                temporary.replace(target)
        manifest = {"archive_bytes": archive.stat().st_size, "archive_md5_expected": config["data"]["expected_md5"], "config_sha256": sha256(DEFAULT_CONFIG), "families": expected, "sha256": {family: sha256(CACHE / f"{family}.zip") for family in expected}, "inner_zip_bytes": {family: (CACHE / f"{family}.zip").stat().st_size for family in expected}, "transfer_mode": "opaque_nested_zip_bytes_only"}
        dump(CACHE / "cache_manifest.json", manifest)
        return {"status": "FULL_CAN_CACHE_BUILD_PASS", "cache": {"path": str(CACHE), **manifest}, "scope": {"nested_log_parsed": False, "outer_test_labels_read": False, "model_training": False, "raw_archive_mutated": False}}
    except Exception:
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.check == args.build:
        parser.error("pass exactly one of --check or --build")
    out, config_path = args.out.resolve(), args.config.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite: {out}")
    out.mkdir(parents=True)
    shutil.copy2(Path(__file__), out / "script_snapshot.py")
    shutil.copy2(config_path, out / "protocol_snapshot.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result = check(config) if args.check else build(config)
        dump(out / "result.json", result)
        marker = "COMPLETED.json" if "PASS" in result["status"] else "FAILED.json"
        dump(out / marker, {"status": result["status"]})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if marker == "COMPLETED.json" else 2
    except Exception as exc:
        dump(out / "FAILED.json", {"status": "FAILED", "error": str(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())

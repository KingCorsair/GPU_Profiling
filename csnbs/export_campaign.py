"""Copy a stopped campaign into a portable, evidence-preserving artifact tree.

Examples:
  python csnbs/export_campaign.py /tmp/campaign /path/to/new-export
  python csnbs/export_campaign.py /tmp/copied-pod-campaign /path/to/new-export \
      --source-root /workspace/rithvik-results/original-campaign

The second form maps only that exact original root to the supplied local tree.
It never searches for similarly named runs. The original campaign bytes are saved
as campaign.source.json; only trial/attempt artifact paths change in campaign.json.
Run/request bytes, specification, specification hash, schedule and statuses stay
unchanged. Re-run the standard report verifier on the resulting campaign.json.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import tempfile
from typing import Any


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _relative_path(value: str, original_root: PurePosixPath, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError(f"{label} must be a nonempty POSIX artifact path")
    path = PurePosixPath(posixpath.normpath(value))
    if path.is_absolute():
        try:
            path = path.relative_to(original_root)
        except ValueError as error:
            raise ValueError(f"{label} is outside the declared campaign root: {value}") from error
    if ".." in path.parts or path.is_absolute():
        raise ValueError(f"{label} escapes the campaign tree: {value}")
    return path.as_posix()


def _local_path(root: Path, relative: str, label: str) -> Path:
    path = root / relative
    if not path.exists():
        raise ValueError(f"Missing referenced {label}: {relative}")
    try:
        path.resolve(strict=True).relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} resolves outside the campaign tree: {relative}") from error
    if not path.is_dir():
        raise ValueError(f"{label} must reference a directory: {relative}")
    return path


def _validate_final_run(directory: Path, required: bool) -> None:
    manifest_path = directory / "run.json"
    if not manifest_path.exists():
        if required:
            raise ValueError(f"Completed trial is missing finalized run.json: {directory}")
        return  # Failed/interrupted attempts may legitimately contain only a partial journal.
    manifest = _object(json.loads(manifest_path.read_text()), "run.json")
    if manifest.get("schema") != "loadgen-run" or manifest.get("schemaVersion") not in (1, 2):
        raise ValueError(f"Unsupported run manifest: {manifest_path}")
    if required and manifest.get("schemaVersion") == 2 and manifest.get("status") != "complete":
        raise ValueError(f"Completed trial points to a non-complete run: {manifest_path}")
    for key in ("requests", "resourceSamples"):
        descriptor = manifest.get(key)
        if descriptor is None and key == "resourceSamples":
            continue
        descriptor = _object(descriptor, f"{key} descriptor")
        filename = descriptor.get("file")
        if not isinstance(filename, str) or not filename or PurePosixPath(filename).is_absolute() or ".." in PurePosixPath(filename).parts or "\\" in filename:
            raise ValueError(f"Invalid {key} artifact path in {manifest_path}")
        artifact = directory / filename
        if not artifact.is_file():
            raise ValueError(f"Missing finalized {key} artifact: {artifact}")
        expected_hash = descriptor.get("sha256")
        if expected_hash is not None:
            if not isinstance(expected_hash, str) or expected_hash.removeprefix("sha256:") != digest(artifact):
                raise ValueError(f"{key} SHA-256 mismatch in {manifest_path}")


def export_campaign(source: Path, destination: Path, source_root: str | None = None) -> dict[str, Any]:
    source = source.expanduser().resolve(strict=True)
    if source.is_dir():
        local_root = source
    elif source.name == "campaign.json" and source.is_file():
        local_root = source.parent
    else:
        raise ValueError("Source must be a campaign directory or its campaign.json")
    campaign_path = local_root / "campaign.json"
    original_bytes = campaign_path.read_bytes()
    manifest = _object(json.loads(original_bytes), "campaign.json")
    if manifest.get("schema") != "loadgen-campaign" or manifest.get("schemaVersion") != 2:
        raise ValueError("Portable export requires a V2 loadgen campaign")
    if not isinstance(manifest.get("trials"), list):
        raise ValueError("Campaign trials must be an array")
    if (local_root / ".campaign.lock").exists():
        raise ValueError("Campaign lock exists; export only after the runner has stopped and cleanup is verified")
    for name in ("campaign.source.json", "export-provenance.json"):
        if (local_root / name).exists():
            raise ValueError(f"Source already contains reserved export artifact {name}; use the original campaign tree")
    original_root = PurePosixPath(posixpath.normpath(source_root if source_root is not None else str(local_root)))
    if not original_root.is_absolute() or original_root == PurePosixPath("/"):
        raise ValueError("--source-root must be the exact absolute original campaign directory")
    destination = destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite export destination: {destination}")
    destination = destination.resolve()
    if destination == local_root or local_root in destination.parents:
        raise ValueError("Export destination cannot be inside the source campaign")

    rewritten = deepcopy(manifest)
    mappings: list[dict[str, str]] = []
    validated_runs: set[tuple[Path, bool]] = set()

    def rewrite_path(record: dict[str, Any], key: str, field: str, required: bool = False, run: bool = False) -> None:
        value = record.get(key)
        if value is None:
            if required:
                raise ValueError(f"Completed artifact is missing {field}")
            return
        # macOS aliases /tmp and /var to /private/...; resolve local absolute
        # references, but never resolve or guess an explicitly supplied remote root.
        path_value = str(Path(value).resolve()) if source_root is None and isinstance(value, str) and Path(value).is_absolute() else value
        relative = _relative_path(path_value, original_root, field)
        path = _local_path(local_root, relative, field)
        if run and (path, required) not in validated_runs:
            _validate_final_run(path, required)
            validated_runs.add((path, required))
        record[key] = relative
        mappings.append({"field": field, "sourcePath": value, "exportPath": relative})

    for index, raw_trial in enumerate(rewritten["trials"]):
        trial = _object(raw_trial, f"trials[{index}]")
        status = trial.get("status")
        if status == "running":
            raise ValueError("Campaign has a running trial; export a stopped snapshot")
        if status not in ("pending", "complete", "failed", "interrupted"):
            raise ValueError(f"Invalid trial status: {status}")
        rewrite_path(trial, "runDirectory", f"trials[{index}].runDirectory", required=status == "complete", run=True)
        attempts = trial.get("attempts", [])
        if not isinstance(attempts, list):
            raise ValueError("Trial attempts must be an array")
        for attempt_index, raw_attempt in enumerate(attempts):
            attempt = _object(raw_attempt, "attempt")
            prefix = f"trials[{index}].attempts[{attempt_index}]"
            if attempt.get("status") == "running":
                raise ValueError("Campaign has a running attempt; export a stopped snapshot")
            rewrite_path(attempt, "directory", prefix + ".directory", required=True)
            rewrite_path(attempt, "runDirectory", prefix + ".runDirectory", required=attempt.get("status") == "complete", run=True)

    inventory = []
    for path in sorted(local_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Artifact symlinks are not portable; resolve them explicitly before export: {path}")
        if path.is_file():
            inventory.append({"path": path.relative_to(local_root).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size})
        elif not path.is_dir():
            raise ValueError(f"Unsupported special file in campaign tree: {path}")
    canonical_bytes = (json.dumps(rewritten, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    provenance = {
        "schema": "loadgen-campaign-export", "schemaVersion": 1,
        "exportedAtUtc": datetime.now(timezone.utc).isoformat(),
        "campaignId": manifest.get("campaignId"),
        "campaignComplete": bool(rewritten["trials"]) and all(trial.get("status") == "complete" for trial in rewritten["trials"]),
        "sourceCampaignSha256": hashlib.sha256(original_bytes).hexdigest(),
        "exportedCampaignSha256": hashlib.sha256(canonical_bytes).hexdigest(),
        "originalSourceRoot": str(original_root), "copiedLocalSourceRoot": str(local_root),
        "pathMappings": mappings, "sourceFiles": inventory,
        "note": "Only trial/attempt artifact location fields changed. Specification, hash, schedule, statuses and raw run/request bytes are preserved. Export does not certify measurement validity or campaign completion.",
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.export-", dir=destination.parent))
    reserved = False
    try:
        shutil.copytree(local_root, staging, dirs_exist_ok=True)
        copied_paths = {path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file()}
        if copied_paths != {entry["path"] for entry in inventory}:
            raise ValueError("Campaign file inventory changed while exporting")
        for entry in inventory:
            copied = staging / entry["path"]
            if copied.stat().st_size != entry["bytes"] or digest(copied) != entry["sha256"]:
                raise ValueError(f"Source changed while exporting: {entry['path']}")
        if (staging / "campaign.json").read_bytes() != original_bytes:
            raise ValueError("Campaign changed while exporting")
        (staging / "campaign.source.json").write_bytes(original_bytes)
        (staging / "campaign.json").write_bytes(canonical_bytes)
        (staging / "export-provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
        # Reserve the previously nonexistent name exclusively, then replace only
        # our own empty directory with the fully verified staged tree.
        destination.mkdir()
        reserved = True
        staging.rename(destination)
        reserved = False
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if reserved:
            destination.rmdir()  # Only our empty reservation; never delete an existing destination tree.
    return {"destination": str(destination), "campaignId": manifest.get("campaignId"),
            "copiedFiles": len(inventory), "rewrittenPathFields": len(mappings),
            "sourceCampaignSha256": provenance["sourceCampaignSha256"], "campaignComplete": provenance["campaignComplete"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="Local campaign tree or campaign.json")
    parser.add_argument("destination", type=Path, help="New portable artifact directory; must not exist")
    parser.add_argument("--source-root", help="Exact original absolute campaign directory before copying from RunPod")
    args = parser.parse_args()
    print(json.dumps(export_campaign(args.source, args.destination, args.source_root), indent=2))


if __name__ == "__main__":
    main()

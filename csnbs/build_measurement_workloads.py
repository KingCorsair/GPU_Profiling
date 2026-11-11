"""Derive serving prompt policies from existing dev questions; never touch eval splits.

These are performance workloads, not new labeled accuracy datasets. Images,
question IDs, categories and source labels are preserved. Accuracy answers are
deliberately omitted because the response policy has changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

LONG_SUFFIX = (
    "\nAnswer in three complete sentences. First answer the question directly, "
    "then describe the visible evidence and relevant image details supporting "
    "your answer. State any uncertainty instead of inventing details."
)


def build(source: Path, destination: Path) -> dict:
    source = source.resolve(strict=True)
    raw = source.read_bytes()
    records = json.loads(raw)
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty dev workload list")
    if destination.exists():
        raise FileExistsError("Refusing to replace an existing workload bundle")
    destination = destination.absolute()
    outputs = {name: [] for name in ("short", "long", "mixed")}
    category_seen: dict[str, int] = {}
    category_index: dict[str, int] = {}
    assignments = []
    for record in records:
        category = record["category"]
        category_index.setdefault(category, len(category_index))
        index = category_seen.get(category, 0)
        category_seen[category] = index + 1
        # Alternate within category; stagger which policy gets the odd item.
        use_long = (index + category_index[category]) % 2 == 0
        image = (source.parent / record["image"]).resolve(strict=True)
        base = {key: record[key] for key in ("question_id", "category", "source_dataset")}
        base["image"] = os.path.relpath(image, destination)
        for name in outputs:
            long_policy = name == "long" or (name == "mixed" and use_long)
            outputs[name].append({**base, "question": record["question"] + (LONG_SUFFIX if long_policy else ""),
                                  "measurement_response_policy": "three-sentences" if long_policy else "original"})
        assignments.append({"question_id": record["question_id"], "category": category,
                            "mixed_policy": "three-sentences" if use_long else "original"})
    destination.mkdir(parents=True)
    hashes = {}
    for name, rows in outputs.items():
        data = (json.dumps(rows, indent=2, ensure_ascii=False) + "\n").encode()
        (destination / f"{name}.json").write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = {"schema": "measurement-workload-policies", "schemaVersion": 1,
                "sourceSha256": hashlib.sha256(raw).hexdigest(), "sourceRecords": len(records),
                "sourcePathRelative": os.path.relpath(source, destination),
                "categoryCounts": category_seen, "longSuffix": LONG_SUFFIX,
                "workloadSha256": hashes, "mixedAssignments": assignments,
                "limitations": ["Performance workloads only; no new accuracy labels or evaluation splits.",
                                "Natural EOS is retained; a longer requested answer does not guarantee a longer generated answer.",
                                "Verify observed output tokens, EOS and cap-hit counts before interpreting length sensitivity.",
                                "Mixed policy is fixed per question and reused in every paired trial."]}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = build(args.source, args.destination)
    print(json.dumps({"destination": str(args.destination), "sourceRecords": result["sourceRecords"],
                      "workloadSha256": result["workloadSha256"]}))

"""Join verified serving trials to matching owner-validated aggregate accuracy.

This module never scores answers. Generate the canonical campaign report first.
Omitting --accuracy writes a pending handoff, with no accuracy points or chart.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def text(value: Any, label: str) -> str:
    require(isinstance(value, str) and value.strip(), f"Missing {label}")
    return value


def sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", value), f"Invalid {label} SHA-256")
    return value.removeprefix("sha256:")


def commit(value: Any, label: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value), f"Missing immutable {label} commit")
    return value


def number(value: Any, label: str, minimum: float = 0, maximum: float = math.inf) -> float:
    require(type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum, f"Invalid {label}")
    return value


def execution_identity(run: dict[str, Any]) -> dict[str, Any]:
    """Exclude only machine-local checkpoint location and download envelope."""
    server = run["server"]
    require(server.get("mode") == "model" and server.get("modelLoaded") is True and server.get("error") is None,
            "Run lacks a confirmed loaded model")
    require(server["source"].get("gitDirty") is False, "Server revision is dirty or unknown")
    config = server["configuration"]
    require(isinstance(config, dict), "Missing effective configuration")
    controls = ("implementation", "visual_token_num", "important_ratio", "prompt_template", "max_new_tokens",
                "do_sample", "use_cache", "eos_policy", "batch_size", "dtype")
    require(all(config.get(key) is not None for key in controls), "Missing effective model/prompt/decode controls")
    files = config.get("download_provenance", {}).get("files")
    require(isinstance(files, list) and files, "Missing verified checkpoint file hashes")
    checkpoint_files = []
    for entry in files:
        require(type(entry.get("bytes")) is int and entry["bytes"] >= 0, "Invalid checkpoint byte length")
        checkpoint_files.append({"file": text(entry.get("file"), "checkpoint filename"), "bytes": entry["bytes"],
                                 "sha256": sha(entry.get("sha256"), "checkpoint file")})
    require(len({entry["file"] for entry in checkpoint_files}) == len(checkpoint_files), "Duplicate checkpoint file")
    return {
        "modelId": text(server.get("modelId"), "model identity"),
        "checkpointRevision": text(server.get("checkpointRevision"), "checkpoint revision"),
        "checkpointFiles": sorted(checkpoint_files, key=lambda entry: entry["file"]),
        "serverGitCommit": commit(server["source"].get("gitCommit"), "server"),
        "runtime": {key: text(server.get("runtime", {}).get(key), f"runtime {key}") for key in ("torch", "transformers")},
        "configuration": {key: value for key, value in config.items() if key not in ("checkpoint", "download_provenance")},
    }


def validate_evaluation(evaluation: dict[str, Any], scope: str) -> None:
    text(evaluation.get("evaluationId"), "evaluation ID")
    require(evaluation.get("scope") == scope, "Accuracy scope differs from campaign; synthetic fixtures cannot join research evidence")
    commit(evaluation.get("evaluationGitCommit"), "evaluation")
    split = evaluation["split"]
    text(split.get("datasetId"), "evaluation dataset ID")
    text(split.get("name"), "evaluation split name")
    sha(split.get("sha256"), "evaluation split")
    require(split.get("locked") is True, "Evaluation split must be locked")
    scorer = evaluation["scorer"]
    text(scorer.get("id"), "scorer identity")
    commit(scorer.get("gitCommit"), "scorer")
    require(scorer.get("validated") is True, "Scorer is not validated")
    text(scorer.get("validationMethod"), "scorer validation method")
    evidence = scorer.get("validationEvidence")
    require(isinstance(evidence, list) and evidence, "Missing scorer validation evidence")
    for item in evidence:
        text(item.get("reference"), "scorer validation reference")
        sha(item.get("sha256"), "scorer validation evidence")
    control = evaluation["randomControl"]
    text(control.get("evaluationId"), "random-control evaluation ID")
    text(control.get("reference"), "random-control evidence reference")
    sha(control.get("sha256"), "random-control evidence")
    require(control.get("validated") is True, "Random-control provenance is not validated")
    def score(row: dict[str, Any], label: str) -> None:
        require(type(row.get("sampleCount")) is int and row["sampleCount"] > 0, f"Invalid {label} sample count")
        number(row.get("accuracy"), f"{label} accuracy fraction", maximum=1)
    score(evaluation["overall"], "overall")
    categories = evaluation.get("perCategory")
    require(isinstance(categories, list) and categories, "Missing per-category accuracy")
    names = []
    for category in categories:
        names.append(text(category.get("category"), "category name"))
        score(category, "category")
        require(category["sampleCount"] <= evaluation["overall"]["sampleCount"], "Category count exceeds evaluation count")
    require(len(set(names)) == len(names), "Duplicate category")
    text(evaluation.get("categorySemantics"), "category overlap/partition semantics")


def join_evidence(report: dict[str, Any], accuracy: dict[str, Any] | None = None) -> dict[str, Any]:
    require(report.get("schema") == "measurement-report" and report.get("schemaVersion") == 1, "Expected canonical measurement-report v1")
    text(report.get("campaignId"), "campaign ID")
    runs = report.get("runs")
    require(isinstance(runs, list) and runs, "Report contains no verified trials")
    require(isinstance(report.get("trials"), list) and len(report["trials"]) == len(runs)
            and all(trial.get("status") == "complete" for trial in report["trials"]), "Report has incomplete prescribed trials")
    scope = "integration-only" if report.get("purpose") == "integration" else "research"
    require(report.get("purpose") in ("integration", "baseline", "aa", "ab"), "Unknown report purpose")
    identities: dict[str, dict[str, Any]] = {}
    seen = set()
    for run in runs:
        run_id = text(run.get("runId"), "run ID")
        require(run_id not in seen, "Run identity reused in report")
        seen.add(run_id)
        require(run.get("quality", {}).get("reportable") is True, "Run failed canonical reportability checks")
        require(run.get("quality", {}).get("reasons") == [], "Run has unresolved quality reasons")
        sha(run.get("requestsSha256"), "request artifact")
        number(run.get("rate"), "offered rate", minimum=1e-12)
        summary = run["summary"]
        number(summary.get("successfulThroughputWithinWindowRps"), "within-window throughput")
        require(type(summary.get("totalRequests")) is int and summary["totalRequests"] > 0, "Missing measured budget")
        require(summary.get("successfulRequests") == summary["totalRequests"] and summary.get("failedRequests") == 0,
                "Failed/incomplete speed trial cannot become an accuracy-throughput point")
        identity = execution_identity(run)
        key = canonical(identity)
        entry = identities.setdefault(key, {"identity": identity, "runIds": [], "variants": []})
        entry["runIds"].append(run_id)
        if run["variant"] not in entry["variants"]:
            entry["variants"].append(run["variant"])
    result: dict[str, Any] = {
        "schema": "accuracy-throughput-join", "schemaVersion": 1, "status": "pending", "scope": scope,
        "campaignId": report["campaignId"], "points": [], "evaluations": [],
        "requiredExecutionIdentities": list(identities.values()),
        "limitations": [
            "Requires a canonical report generated by the raw-artifact verifier; this join does not rerun timing/statistical verification.",
            "Accuracy is supplied by its owner; validation references are retained, not independently rescored or adjudicated.",
            "Each point is one serving trial. Reusing one aggregate accuracy estimate across speed trials is not independent accuracy replication.",
            "Within-window throughput at an offered load is not maximum sustainable capacity. These points establish no optimization significance.",
            "Evaluation split and serving workload are separate populations. Category scores are retained without pooling overlapping categories.",
        ],
    }
    if accuracy is None:
        result["reason"] = "Matching owner-validated aggregate accuracy has not been supplied."
        return result
    require(accuracy.get("schema") == "owner-aggregate-accuracy" and accuracy.get("schemaVersion") == 1, "Expected owner-aggregate-accuracy v1")
    if accuracy.get("status") == "pending":
        require(accuracy.get("evaluations") == [], "Pending handoff must contain no evaluations or numeric scores")
        result["reason"] = text(accuracy.get("reason"), "pending reason")
        return result
    require(accuracy.get("status") == "validated", "Accuracy input is not validated")
    evaluations = accuracy.get("evaluations")
    require(isinstance(evaluations, list) and evaluations, "Missing validated evaluations")
    by_identity = {}
    eval_ids = set()
    evaluation_context = None
    for evaluation in evaluations:
        validate_evaluation(evaluation, scope)
        context = canonical({"split": evaluation["split"], "evaluationGitCommit": evaluation["evaluationGitCommit"],
                             "scorerId": evaluation["scorer"]["id"], "scorerGitCommit": evaluation["scorer"]["gitCommit"],
                             "sampleCount": evaluation["overall"]["sampleCount"], "categorySemantics": evaluation["categorySemantics"],
                             "categoryCounts": sorted((row["category"], row["sampleCount"]) for row in evaluation["perCategory"])})
        require(evaluation_context is None or context == evaluation_context,
                "Evaluations use different splits, scorers, code, or category/sample populations")
        evaluation_context = context
        require(evaluation["evaluationId"] not in eval_ids, "Duplicate evaluation ID")
        eval_ids.add(evaluation["evaluationId"])
        key = canonical(evaluation["executionIdentity"])
        require(key in identities, "Accuracy execution identity does not exactly match any serving trial")
        require(key not in by_identity, "Ambiguous multiple evaluations for one execution identity; select one locked evaluation artifact")
        by_identity[key] = evaluation
    require(set(by_identity) == set(identities), "Missing accuracy for one or more serving configurations")
    for run in runs:
        evaluation = by_identity[canonical(execution_identity(run))]
        result["points"].append({
            "campaignId": report["campaignId"], "runId": run["runId"], "variant": run["variant"],
            "offeredRps": run["rate"], "successfulThroughputWithinWindowRps": run["summary"]["successfulThroughputWithinWindowRps"],
            "p50Ms": run["summary"].get("successfulRequestLatencyMs", {}).get("p50"),
            "accuracy": evaluation["overall"]["accuracy"], "accuracySampleCount": evaluation["overall"]["sampleCount"],
            "evaluationId": evaluation["evaluationId"], "evaluationSplitSha256": evaluation["split"]["sha256"],
            "requestsSha256": run["requestsSha256"], "scope": scope,
        })
    result.update(status="ready", evaluations=evaluations)
    return result


def read_json(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    data = path.read_bytes()
    value = json.loads(data, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant {item}")))
    require(isinstance(value, dict), "Expected JSON object")
    return value, {"filename": path.name, "sha256": hashlib.sha256(data).hexdigest()}


def export_join(report_path: Path, accuracy_path: Path | None, output: Path) -> dict[str, Any]:
    report, report_source = read_json(report_path)
    accuracy, accuracy_source = read_json(accuracy_path) if accuracy_path else (None, None)
    result = join_evidence(report, accuracy)
    result.update(generatedAtUtc=datetime.now(timezone.utc).isoformat(), inputs={"report": report_source, "accuracy": accuracy_source})
    output.mkdir(parents=True, exist_ok=False)
    (output / "accuracy-throughput.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fields = ["campaignId", "runId", "variant", "offeredRps", "successfulThroughputWithinWindowRps", "p50Ms",
              "accuracy", "accuracySampleCount", "evaluationId", "evaluationSplitSha256", "requestsSha256", "scope"]
    with (output / "accuracy-throughput.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["points"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--accuracy", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory; never overwritten")
    args = parser.parse_args()
    try:
        result = export_join(args.report, args.accuracy, args.output)
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
        parser.exit(2, f"Cannot join accuracy and throughput: {error}\n")
    print(json.dumps({"status": result["status"], "points": len(result["points"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()

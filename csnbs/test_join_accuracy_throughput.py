"""Synthetic integration fixtures only; no fixture score is research evidence."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from csnbs.join_accuracy_throughput import execution_identity, export_join, join_evidence


def fixture():
    run = {
        "runId": "synthetic-trial-1", "variant": "synthetic-baseline", "rate": 2, "runKind": "open-loop",
        "requestsSha256": "sha256:" + "a" * 64, "quality": {"reportable": True, "reasons": []},
        "server": {"mode": "model", "modelLoaded": True, "error": None, "modelId": "synthetic-model",
                   "checkpointRevision": "synthetic-checkpoint", "source": {"gitCommit": "a" * 40, "gitDirty": False},
                   "runtime": {"torch": "synthetic-torch", "transformers": "synthetic-transformers"},
                   "configuration": {"implementation": "synthetic-fixture", "visual_token_num": 576, "important_ratio": 0.5,
                                     "prompt_template": "fixture", "max_new_tokens": 8, "do_sample": False,
                                     "use_cache": True, "eos_policy": "natural", "batch_size": 1, "dtype": "float16",
                                     "checkpoint": "/synthetic/local/path", "download_provenance": {
                                         "verified_at_utc": "synthetic-date", "files": [{"file": "fake-weights", "bytes": 1, "sha256": "b" * 64}]} }},
        "summary": {"successfulThroughputWithinWindowRps": 2, "totalRequests": 10, "successfulRequests": 10,
                    "failedRequests": 0, "successfulRequestLatencyMs": {"p50": 10}},
    }
    report = {"schema": "measurement-report", "schemaVersion": 1, "campaignId": "synthetic-integration-only",
              "purpose": "integration", "runs": [run], "trials": [{"status": "complete"}]}
    evaluation = {
        "evaluationId": "synthetic-eval", "scope": "integration-only", "executionIdentity": execution_identity(run),
        "evaluationGitCommit": "c" * 40, "split": {"datasetId": "synthetic-dataset", "name": "synthetic-test", "sha256": "d" * 64, "locked": True},
        "scorer": {"id": "synthetic-scorer", "gitCommit": "e" * 40, "validated": True, "validationMethod": "synthetic fixture, not real validation",
                   "validationEvidence": [{"reference": "synthetic://validation", "sha256": "f" * 64}]},
        "randomControl": {"evaluationId": "synthetic-random", "reference": "synthetic://random", "sha256": "a" * 64, "validated": True},
        "overall": {"sampleCount": 10, "accuracy": 0.7}, "perCategory": [{"category": "synthetic-category", "sampleCount": 10, "accuracy": 0.7}],
        "categorySemantics": "synthetic single category, exhaustive",
    }
    return report, {"schema": "owner-aggregate-accuracy", "schemaVersion": 1, "status": "validated", "evaluations": [evaluation]}


class JoinTests(unittest.TestCase):
    def test_pending_contains_no_accuracy_points_or_scores(self):
        report, _ = fixture()
        pending = join_evidence(report)
        self.assertEqual(pending["status"], "pending")
        self.assertEqual(pending["points"], [])
        self.assertEqual(pending["evaluations"], [])
        self.assertEqual(len(pending["requiredExecutionIdentities"]), 1)
        example = json.loads((Path(__file__).parent / "measure/campaigns/accuracy-handoff.example.json").read_text())
        self.assertEqual(join_evidence(report, example)["status"], "pending")
        example["evaluations"] = [{"overall": {"accuracy": 0}}]
        with self.assertRaisesRegex(ValueError, "Pending handoff"):
            join_evidence(report, example)

    def test_matching_aggregate_is_retained_without_replication_or_rescoring(self):
        report, accuracy = fixture()
        second = deepcopy(report["runs"][0]); second["runId"] = "synthetic-trial-2"
        report["runs"].append(second); report["trials"].append({"status": "complete"})
        original = deepcopy(accuracy)
        result = join_evidence(report, accuracy)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(len(result["points"]), 2)
        self.assertEqual(len(result["evaluations"]), 1)
        self.assertEqual([row["accuracy"] for row in result["points"]], [0.7, 0.7])
        self.assertEqual(result["evaluations"], original["evaluations"])
        self.assertEqual(accuracy, original)
        self.assertEqual(result["scope"], "integration-only")

    def test_every_scientific_identity_mismatch_is_rejected(self):
        report, source = fixture()
        changes = [("modelId", "other"), ("checkpointRevision", "other"), ("serverGitCommit", "b" * 40),
                   ("checkpointFiles", []), ("runtime", {"torch": "other", "transformers": "other"})]
        for key, value in changes:
            with self.subTest(key=key):
                accuracy = deepcopy(source); accuracy["evaluations"][0]["executionIdentity"][key] = value
                with self.assertRaisesRegex(ValueError, "identity"):
                    join_evidence(report, accuracy)
        for key in ("visual_token_num", "important_ratio", "implementation", "prompt_template", "max_new_tokens", "do_sample", "eos_policy"):
            with self.subTest(config=key):
                accuracy = deepcopy(source); accuracy["evaluations"][0]["executionIdentity"]["configuration"][key] = "mismatched"
                with self.assertRaisesRegex(ValueError, "identity"):
                    join_evidence(report, accuracy)

    def test_local_checkpoint_location_does_not_change_scientific_identity(self):
        report, accuracy = fixture()
        report["runs"][0]["server"]["configuration"]["checkpoint"] = "/another/machine/path"
        report["runs"][0]["server"]["configuration"]["download_provenance"]["verified_at_utc"] = "another-time"
        self.assertEqual(join_evidence(report, accuracy)["status"], "ready")
        report["runs"][0]["server"]["configuration"]["download_provenance"]["files"][0]["sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "identity"):
            join_evidence(report, accuracy)

    def test_isolated_and_unknown_run_kinds_have_no_offered_load(self):
        report, accuracy = fixture()
        self.assertEqual(join_evidence(report, accuracy)["points"][0]["offeredRps"], 2)
        report["runs"][0]["runKind"] = "isolated"
        point = join_evidence(report, accuracy)["points"][0]
        self.assertEqual(point["runKind"], "isolated")
        self.assertIsNone(point["offeredRps"])
        del report["runs"][0]["runKind"]
        self.assertIsNone(join_evidence(report, accuracy)["points"][0]["offeredRps"])

    def test_invalid_or_missing_owner_evidence_is_rejected(self):
        report, source = fixture()
        edits = [("scorer", "validated", False), ("scorer", "validationEvidence", []), ("split", "locked", False),
                 ("randomControl", "validated", False), ("randomControl", "sha256", None),
                 ("overall", "accuracy", float("nan")), ("overall", "accuracy", 1.1), ("overall", "sampleCount", 0)]
        for block, key, value in edits:
            with self.subTest(block=block, key=key):
                accuracy = deepcopy(source); accuracy["evaluations"][0][block][key] = value
                with self.assertRaises(ValueError):
                    join_evidence(report, accuracy)
        for key in ("scorer", "split", "randomControl", "perCategory"):
            accuracy = deepcopy(source); del accuracy["evaluations"][0][key]
            with self.assertRaises((ValueError, KeyError)):
                join_evidence(report, accuracy)
        accuracy = deepcopy(source); accuracy["evaluations"][0]["scope"] = "research"
        with self.assertRaisesRegex(ValueError, "scope"):
            join_evidence(report, accuracy)

    def test_missing_ambiguous_or_unmatched_evaluation_populations_are_rejected(self):
        report, accuracy = fixture()
        second = deepcopy(report["runs"][0]); second["runId"] = "synthetic-trial-2"
        second["variant"] = "synthetic-candidate"; second["server"]["configuration"]["visual_token_num"] = 128
        report["runs"].append(second); report["trials"].append({"status": "complete"})
        with self.assertRaisesRegex(ValueError, "Missing accuracy"):
            join_evidence(report, accuracy)
        candidate = deepcopy(accuracy["evaluations"][0]); candidate["evaluationId"] = "synthetic-candidate-eval"
        candidate["executionIdentity"] = execution_identity(second)
        accuracy["evaluations"].append(candidate)
        self.assertEqual(join_evidence(report, accuracy)["status"], "ready")
        candidate["split"]["sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "different splits"):
            join_evidence(report, accuracy)
        report, accuracy = fixture()
        accuracy["evaluations"].append(deepcopy(accuracy["evaluations"][0]))
        accuracy["evaluations"][1]["evaluationId"] = "another-synthetic-eval"
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            join_evidence(report, accuracy)

    def test_failed_incomplete_or_duplicate_speed_evidence_is_rejected(self):
        report, accuracy = fixture()
        for mutate in (lambda r: r["trials"][0].update(status="failed"),
                       lambda r: r["runs"][0]["quality"].update(reportable=False),
                       lambda r: r["runs"][0]["summary"].update(failedRequests=1),
                       lambda r: r["runs"][0]["server"]["source"].update(gitDirty=True)):
            broken = deepcopy(report); mutate(broken)
            with self.assertRaises(ValueError):
                join_evidence(broken, accuracy)
        report["runs"].append(deepcopy(report["runs"][0])); report["trials"].append({"status": "complete"})
        with self.assertRaisesRegex(ValueError, "reused"):
            join_evidence(report, accuracy)

    def test_export_binds_input_hashes_and_refuses_overwrite(self):
        report, accuracy = fixture()
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); report_path = root / "report.json"; accuracy_path = root / "accuracy.json"
            report_path.write_text(json.dumps(report)); accuracy_path.write_text(json.dumps(accuracy))
            pending = export_join(report_path, None, root / "pending")
            self.assertEqual(pending["status"], "pending")
            with (root / "pending/accuracy-throughput.csv").open() as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
            ready = export_join(report_path, accuracy_path, root / "ready")
            self.assertEqual(len(ready["inputs"]["report"]["sha256"]), 64)
            self.assertEqual(len(ready["inputs"]["accuracy"]["sha256"]), 64)
            with (root / "ready/accuracy-throughput.csv").open() as handle:
                self.assertEqual(list(csv.DictReader(handle))[0]["accuracy"], "0.7")
            with self.assertRaises(FileExistsError):
                export_join(report_path, accuracy_path, root / "ready")


if __name__ == "__main__":
    unittest.main()

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from csnbs.export_campaign import export_campaign


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "copied-source"
        self.source.mkdir()
        self.destination = self.root / "portable"
        self.run = self.source / "attempt-1" / "runs" / "run-1"
        self.run.mkdir(parents=True)
        self.raw = b'{"phase":"measurement","sequence":0,"latencyMs":1.125}\n'
        (self.run / "requests.jsonl").write_bytes(self.raw)
        self.run_bytes = json.dumps({"schema": "loadgen-run", "schemaVersion": 2, "runId": "run-1", "status": "complete",
            "requests": {"file": "requests.jsonl", "schemaVersion": 3, "count": 1,
                         "sha256": "sha256:" + hashlib.sha256(self.raw).hexdigest()}}).encode()
        (self.run / "run.json").write_bytes(self.run_bytes)
        (self.source / "attempt-1" / "server.log").write_text("exact log bytes\n")
        self.remote = "/workspace/rithvik-results/campaign-1"
        self.campaign = {"schema": "loadgen-campaign", "schemaVersion": 2, "campaignId": "campaign-1",
            "spec": {"datasetPath": "/workspace/repo/dev.json", "unrelatedPath": "/keep-this/exactly"},
            "specHash": "do-not-recalculate", "schedule": [{"trialId": "t1"}],
            "trials": [{"trialId": "t1", "status": "complete", "runDirectory": self.remote + "/attempt-1/runs/run-1",
                "error": None, "attempts": [{"attemptId": "attempt-1", "status": "complete", "directory": self.remote + "/attempt-1",
                                             "runDirectory": self.remote + "/attempt-1/runs/run-1"}]}]}
        self.save()

    def save(self):
        self.original = (json.dumps(self.campaign, separators=(",", ":")) + "\n").encode()
        (self.source / "campaign.json").write_bytes(self.original)

    def test_remap_only_location_fields_and_preserve_all_evidence_bytes(self):
        result = export_campaign(self.source, self.destination, self.remote)
        rewritten = json.loads((self.destination / "campaign.json").read_text())
        self.assertEqual(rewritten["spec"], self.campaign["spec"])
        self.assertEqual(rewritten["specHash"], self.campaign["specHash"])
        self.assertEqual(rewritten["schedule"], self.campaign["schedule"])
        self.assertEqual(rewritten["trials"][0]["runDirectory"], "attempt-1/runs/run-1")
        self.assertEqual(rewritten["trials"][0]["attempts"][0]["directory"], "attempt-1")
        self.assertEqual((self.destination / "campaign.source.json").read_bytes(), self.original)
        self.assertEqual((self.source / "campaign.json").read_bytes(), self.original)
        self.assertEqual((self.destination / "attempt-1/runs/run-1/run.json").read_bytes(), self.run_bytes)
        self.assertEqual((self.destination / "attempt-1/runs/run-1/requests.jsonl").read_bytes(), self.raw)
        provenance = json.loads((self.destination / "export-provenance.json").read_text())
        self.assertEqual(provenance["sourceCampaignSha256"], hashlib.sha256(self.original).hexdigest())
        self.assertEqual(provenance["exportedCampaignSha256"], hashlib.sha256((self.destination / "campaign.json").read_bytes()).hexdigest())
        self.assertEqual(len(provenance["pathMappings"]), 3)
        self.assertEqual(result["rewrittenPathFields"], 3)
        self.assertTrue(result["campaignComplete"])

    def test_local_absolute_and_relative_references(self):
        trial = self.campaign["trials"][0]
        trial["runDirectory"] = str(self.run)
        trial["attempts"][0]["directory"] = "attempt-1"
        trial["attempts"][0]["runDirectory"] = "attempt-1/runs/run-1"
        self.save()
        export_campaign(self.source / "campaign.json", self.destination)
        self.assertTrue((self.destination / "campaign.json").is_file())

    def test_refuses_wrong_original_root_without_searching(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            export_campaign(self.source, self.destination, "/workspace/different-campaign")
        self.assertFalse(self.destination.exists())

    def test_refuses_missing_finalized_run_and_changed_raw_hash(self):
        (self.run / "run.json").unlink()
        with self.assertRaisesRegex(ValueError, "finalized run.json"):
            export_campaign(self.source, self.destination, self.remote)
        (self.run / "run.json").write_bytes(self.run_bytes)
        (self.run / "requests.jsonl").write_bytes(b"tampered\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            export_campaign(self.source, self.destination, self.remote)
        self.assertFalse(self.destination.exists())

    def test_refuses_overwrite_and_descendant_destination(self):
        self.destination.mkdir()
        sentinel = self.destination / "keep"
        sentinel.write_text("existing content")
        with self.assertRaises(FileExistsError):
            export_campaign(self.source, self.destination, self.remote)
        self.assertEqual(sentinel.read_text(), "existing content")
        with self.assertRaisesRegex(ValueError, "inside the source"):
            export_campaign(self.source, self.source / "export", self.remote)

    def test_rejects_escaping_paths_symlinks_and_active_runs(self):
        trial = self.campaign["trials"][0]
        original_path = trial["runDirectory"]
        for path in ("../escape", self.remote + "-different/attempt-1/runs/run-1"):
            trial["runDirectory"] = path
            self.save()
            with self.assertRaisesRegex(ValueError, "escapes|outside"):
                export_campaign(self.source, self.destination, self.remote)
        trial["runDirectory"] = original_path
        trial["status"] = "running"
        self.save()
        with self.assertRaisesRegex(ValueError, "running trial"):
            export_campaign(self.source, self.destination, self.remote)
        trial["status"] = "complete"
        self.save()
        (self.source / ".campaign.lock").write_text("{}")
        with self.assertRaisesRegex(ValueError, "lock exists"):
            export_campaign(self.source, self.destination, self.remote)
        (self.source / ".campaign.lock").unlink()
        (self.source / "escape").symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            export_campaign(self.source, self.destination, self.remote)

    def test_preserves_failed_partial_attempts_and_pending_statuses(self):
        partial = self.source / "attempt-failed" / "partial-run"
        partial.mkdir(parents=True)
        (partial / "run.partial.json").write_text('{"status":"partial"}')
        (partial / "requests.jsonl").write_text("{}\n")
        trial = self.campaign["trials"][0]
        trial["attempts"].insert(0, {"attemptId": "failed", "status": "failed", "directory": self.remote + "/attempt-failed",
                                       "runDirectory": self.remote + "/attempt-failed/partial-run", "error": "process failed"})
        self.campaign["trials"].append({"trialId": "not-run", "status": "pending", "runDirectory": None, "error": None})
        self.save()
        result = export_campaign(self.source, self.destination, self.remote)
        rewritten = json.loads((self.destination / "campaign.json").read_text())
        self.assertEqual(rewritten["trials"][1]["status"], "pending")
        self.assertFalse(result["campaignComplete"])
        self.assertTrue((self.destination / "attempt-failed/partial-run/run.partial.json").exists())


if __name__ == "__main__":
    unittest.main()

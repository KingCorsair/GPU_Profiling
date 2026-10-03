"""CPU-only plot contract checks; fixtures are explicitly synthetic."""
import copy
import tempfile
import unittest
from pathlib import Path

from csnbs.plot_campaign import render_report, select_representatives, validate_report


def fixture():
    return {
        "schema": "measurement-report", "schemaVersion": 1,
        "campaignId": "synthetic-test", "purpose": "integration", "runKind": "open-loop",
        "comparisons": [], "limitations": ["Synthetic test fixture only"],
        "runs": [{
            "trialId": f"trial-{index}", "variant": variant, "rate": 1,
            "runId": f"run-{index}", "summary": {
                "successfulThroughputWithinWindowRps": throughput,
                "successfulThroughputIncludingDrainRps": throughput,
                "successfulRequests": 3, "failedRequests": 0,
                "successfulRequestLatencyMs": {"p50": 20, "p95": None, "p99": None},
            },
            "capacity": {"classification": "integration-only"},
            "latenciesMs": [10, 20, 30],
            "timeline": [{"scheduledSeconds": sequence, "latencyMs": latency, "outstanding": 1}
                         for sequence, latency in enumerate([10, 20, 30])],
            "quality": {"reportable": False, "reasons": ["Fake server"]},
        } for index, (variant, throughput) in enumerate([("baseline", 0.8), ("candidate", 0.9)])],
    }


class PlotCampaignTests(unittest.TestCase):
    def test_two_runs_with_null_tails_export_png_and_svg_and_keep_integration_label(self):
        report = fixture()
        before = copy.deepcopy(report)
        with tempfile.TemporaryDirectory() as directory:
            paths = render_report(report, Path(directory))
            self.assertTrue(Path(paths["png"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            svg = Path(paths["svg"]).read_text()
            self.assertIn("<svg", svg)
            self.assertIn("INTEGRATION ONLY", svg)
            self.assertIn("p95, p99 unavailable", svg)
            self.assertIn("run-0", svg)
            self.assertIn("run-1", svg)
            self.assertIn("Offered requests / second", svg)
            self.assertIn("Offered rate (reference)", svg)
        self.assertEqual(report, before)

    def test_isolated_trials_ignore_placeholder_rates_and_label_serial_completion(self):
        report = fixture()
        report["runKind"] = "isolated"
        report["runs"][1]["rate"] = 99  # Placeholder must not affect selection or x-axis.
        selected, runs = select_representatives(report)
        self.assertIsNone(selected)
        self.assertEqual(len(runs), 2)
        with self.assertRaisesRegex(ValueError, "explicitly open-loop"):
            select_representatives(report, 1)
        with tempfile.TemporaryDirectory() as directory:
            paths = render_report(report, Path(directory))
            self.assertTrue(Path(paths["png"]).read_bytes().startswith(b"\x89PNG"))
            svg = Path(paths["svg"]).read_text()
            self.assertIn("Isolated serial requests: completion-paced, no offered arrival rate.", svg)
            self.assertIn("Recorded trial sequence", svg)
            self.assertIn("Serial successful completion rate", svg)
            self.assertNotIn("Offered requests / second", svg)
            self.assertNotIn("Offered rate (reference)", svg)
            self.assertNotIn("99 requests/s", svg)

    def test_missing_or_conflicting_mode_cannot_imply_open_loop(self):
        report = fixture()
        del report["runKind"]
        self.assertIsNone(select_representatives(report)[0])
        report["runKind"] = "isolated"
        report["runs"][0]["runKind"] = "open-loop"
        with self.assertRaisesRegex(ValueError, "consistently"):
            validate_report(report)

    def test_representatives_remain_individual_trials_and_unmeasured_rate_rejected(self):
        report = fixture()
        repeated = copy.deepcopy(report["runs"][0])
        repeated.update(runId="repeat", trialId="repeat-trial", latenciesMs=[9999])
        report["runs"].append(repeated)
        rate, runs = select_representatives(report)
        self.assertEqual(rate, 1)
        self.assertEqual(len(runs), 2)
        self.assertNotIn(9999, runs[0]["latenciesMs"])
        with self.assertRaisesRegex(ValueError, "not recorded"):
            select_representatives(report, 3)

    def test_invalid_metrics_or_schema_cannot_silently_enter_plots(self):
        report = fixture()
        report["runs"][0]["latenciesMs"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_report(report)
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_report({})
        duplicate = fixture()
        duplicate["runs"][1]["runId"] = duplicate["runs"][0]["runId"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_report(duplicate)


if __name__ == "__main__":
    unittest.main()

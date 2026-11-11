"""CPU-only checks of synchronization boundaries and immutable artifact integrity."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from csnbs.gpu_calibration import distribution, save_result, time_gpu, time_host_enqueue


class FakeCuda:
    def __init__(self):
        self.events = []
        self.log = []
        self.drained = True

    def Event(self, enable_timing):
        self.assert_timing = enable_timing
        cuda = self

        class Event:
            def record(self):
                cuda.log.append("event")
                cuda.drained = False

            def elapsed_time(self, other):
                if not cuda.drained:
                    raise AssertionError("Read elapsed time before the final drain")
                return 1.25

        event = Event()
        self.events.append(event)
        return event

    def synchronize(self):
        self.log.append("sync")
        self.drained = True


class CalibrationTests(unittest.TestCase):
    def test_event_measurement_drains_only_at_phase_boundaries(self):
        cuda = FakeCuda()
        result = time_gpu(lambda: cuda.log.append("operation"), cuda=cuda, warmup=2, iters=3)
        self.assertEqual(result, {"warmupGpuMs": [1.25] * 2, "measuredGpuMs": [1.25] * 3})
        segments = []
        operations = 0
        for entry in cuda.log:
            if entry == "operation":
                operations += 1
            elif entry == "sync":
                segments.append(operations)
                operations = 0
        self.assertEqual(segments, [0, 2, 3], "no synchronization may occur between operations in one phase")

    def test_eviction_work_precedes_start_event_without_inner_sync(self):
        cuda = FakeCuda()
        time_gpu(lambda: cuda.log.append("operation"), cuda=cuda, warmup=1, iters=2,
                 before_each=lambda: cuda.log.append("evict"))
        for index, entry in enumerate(cuda.log):
            if entry == "evict":
                self.assertEqual(cuda.log[index:index + 4], ["evict", "event", "operation", "event"])
        self.assertEqual(cuda.log.count("sync"), 3)

    def test_host_enqueue_demonstration_has_one_external_drain_per_boundary(self):
        cuda = FakeCuda()
        values = time_host_enqueue(lambda: cuda.log.append("operation"), cuda=cuda, iters=3)
        self.assertEqual(cuda.log, ["sync", "operation", "operation", "operation", "sync"])
        self.assertEqual(len(values), 3)
        self.assertTrue(all(value >= 0 for value in values))

    def test_nearest_rank_and_invalid_observations(self):
        summary = distribution(list(range(1, 101)))
        self.assertEqual((summary["p50"], summary["p95"], summary["p99"]), (50, 95, 99))
        for values in ([], [-1], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                distribution(values)
        for iterations in (0, -1, 10001, True):
            with self.assertRaises(ValueError):
                time_gpu(lambda: None, cuda=FakeCuda(), iters=iterations)

    def test_saved_raw_file_hash_count_and_no_overwrite(self):
        manifest = {"runId": "test", "trials": [{"trialId": "r1", "condition": "test",
                    "warmupGpuMs": [2.0], "measuredGpuMs": [1.0, 1.5], "hostEnqueueMs": [.01, .02]}]}
        with tempfile.TemporaryDirectory() as directory:
            output = save_result(Path(directory), manifest)
            raw = (output / "samples.jsonl").read_bytes()
            stored = json.loads((output / "calibration.json").read_text())
            self.assertEqual(stored["samples"]["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(stored["samples"]["count"], 5)
            self.assertEqual(json.loads(raw.splitlines()[1])["phase"], "measurement")
            with self.assertRaises(FileExistsError):
                save_result(Path(directory), manifest)


if __name__ == "__main__":
    unittest.main()

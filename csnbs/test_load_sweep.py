"""Check server selection and ownership without running benchmark requests."""
import unittest
from unittest.mock import Mock, patch

from csnbs import run_load_sweep as sweep


class SweepServerTests(unittest.TestCase):
    def test_reuses_gemma_and_does_not_start_or_stop_an_owned_process(self):
        health = {"service": "csnbs-gemma-server", "pid": 42, "mode": "model", "model_loaded": True}
        with patch("sys.argv", ["sweep", "--model", "gemma", "--reuse-server"]), \
             patch.object(sweep, "port_is_open", return_value=True), \
             patch.object(sweep, "read_server_health", return_value=health), \
             patch.object(sweep.subprocess, "Popen") as start, \
             patch.object(sweep.subprocess, "run") as run:
            sweep.main()
        start.assert_not_called()
        self.assertEqual(run.call_count, 5)
        for call in run.call_args_list:
            args = call.args[0]
            self.assertEqual(args[args.index("--endpoint") + 1], "http://127.0.0.1:8002/infer")

    def test_refuses_a_different_service_before_generating_load(self):
        health = {"service": "csnbs-llava-server", "pid": 42, "mode": "model", "model_loaded": True}
        with patch("sys.argv", ["sweep", "--model", "gemma", "--reuse-server"]), \
             patch.object(sweep, "port_is_open", return_value=True), \
             patch.object(sweep, "read_server_health", return_value=health), \
             patch.object(sweep.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "not the expected model server"):
                sweep.main()
        run.assert_not_called()

    def test_managed_gemma_uses_its_launcher_and_stops_its_own_process(self):
        process = Mock(pid=42)
        with patch("sys.argv", ["sweep", "--model", "gemma"]), \
             patch.object(sweep, "port_is_open", return_value=False), \
             patch.object(sweep, "wait_for_server"), \
             patch.object(sweep, "check_model_server"), \
             patch.object(sweep.subprocess, "Popen", return_value=process) as start, \
             patch.object(sweep.subprocess, "run"):
            sweep.main()
        self.assertTrue(start.call_args.args[0][1].endswith("csnbs/start_gemma_server.sh"))
        process.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()

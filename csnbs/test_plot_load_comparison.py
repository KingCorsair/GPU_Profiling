"""GPU-free reporting checks. All generated runs here are synthetic test fixtures."""
import json
import tempfile
import unittest
from pathlib import Path

from csnbs.plot_load_comparison import hydrate_and_verify_raw_runs, story_rows, verify_pair


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = {
            'schema': 'vispruner-load-campaign-v1',
            'campaign': {
                'gpu': 'TEST GPU', 'gitCommit': 'test-stock-commit', 'workloadId': 'test-workload',
                'batchSize': 1, 'durationSecondsPerLevel': 30, 'maxNewTokens': 64,
                'importantRatio': 0.5, 'warmupRequestsExcludedPerLevel': 10,
            }, 'series': [],
        }
        self.add_series('stock', 576, latency=1000)
        self.add_series('stock', 128, latency=1100)

    def add_series(self, path, tokens, *, latency, condition='under-load'):
        run_id = f'{path}-{tokens}-{condition}'
        commit = f'test-{path}-commit'
        raw = {
            'schema': 'loadgen-run', 'schemaVersion': 1, 'runId': run_id,
            'runKind': 'isolated' if condition == 'isolated' else 'open-loop',
            'config': {'requestsPerSecond': 3, 'durationSeconds': 30, 'timeoutMs': 120000},
            'source': {'gitCommit': commit, 'gitDirty': False},
            'hardware': {'gpuModels': ['TEST GPU']}, 'workload': {'id': 'test-workload'},
            'benchmark': {'batchSize': 1, 'modelId': 'test-model', 'checkpoint': 'test-checkpoint',
                          'maxOutputTokens': 64, 'visualTokenNum': tokens, 'importantRatio': 0.5},
            'requests': {'count': 90},
            'summary': {'totalRequests': 80, 'successfulRequests': 80, 'failedRequests': 0,
                        'successfulThroughputRps': 2,
                        'successfulRequestLatencyMs': {'p50': latency, 'p95': latency * 2, 'p99': latency * 3}},
        }
        filename = self.root / f'{run_id}.json'
        filename.write_text(json.dumps(raw))
        self.data['series'].append({
            'id': run_id, 'label': run_id, 'servingPath': path, 'visualTokenNum': tokens,
            'loadCondition': condition, 'gitCommit': commit,
            'runs': [{'runId': run_id, 'runPath': filename.name}],
        })

    def change_raw(self, index, change):
        filename = self.root / self.data['series'][index]['runs'][0]['runPath']
        raw = json.loads(filename.read_text())
        change(raw)
        filename.write_text(json.dumps(raw))

    def rows(self):
        hydrate_and_verify_raw_runs(self.data, self.root)
        return story_rows(self.data, tokens=128, rps=3, percentile='p50')

    def test_missing_conditions_stay_missing_and_slowdown_is_negative(self):
        rows = self.rows()
        self.assertEqual(rows[0]['improvementPercent'], 0)
        self.assertIsNone(rows[1]['improvementPercent'])
        self.assertEqual(rows[2]['improvementPercent'], -10)
        self.assertIsNone(rows[3]['improvementPercent'])
        self.assertEqual(rows[2]['controlRunId'], 'stock-576-under-load')

    def test_cache_tampering_is_rejected(self):
        self.data['series'][0]['runs'][0]['p50LatencyMs'] = 999
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.rows()

    def test_hardware_mismatch_is_rejected(self):
        self.change_raw(1, lambda raw: raw['hardware'].update(gpuModels=['OTHER GPU']))
        with self.assertRaisesRegex(ValueError, 'gpuModels'):
            self.rows()

    def test_nonfinite_latency_is_rejected(self):
        self.change_raw(1, lambda raw: raw['summary']['successfulRequestLatencyMs'].update(p95=float('nan')))
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.rows()

    def test_nonfinite_cached_value_is_rejected(self):
        self.data['series'][0]['runs'][0]['p50LatencyMs'] = float('inf')
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.rows()

    def test_wrong_output_token_setting_is_rejected(self):
        self.change_raw(1, lambda raw: raw['benchmark'].update(maxOutputTokens=128))
        with self.assertRaisesRegex(ValueError, 'maxOutputTokens'):
            self.rows()

    def test_different_checkpoints_cannot_be_normalized(self):
        self.change_raw(1, lambda raw: raw['benchmark'].update(checkpoint='other-model'))
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            self.rows()

    def test_wrong_warmup_count_is_rejected(self):
        self.change_raw(1, lambda raw: raw['requests'].update(count=80))
        with self.assertRaisesRegex(ValueError, 'warmups'):
            self.rows()

    def test_load_run_cannot_be_relabelled_as_isolated(self):
        self.data['series'][0]['loadCondition'] = 'isolated'
        with self.assertRaisesRegex(ValueError, 'runKind=isolated'):
            self.rows()

    def test_engineered_requires_its_own_control(self):
        self.add_series('engineered', 128, latency=250)
        self.assertIsNone(self.rows()[3]['improvementPercent'])
        self.add_series('engineered', 576, latency=500)
        # Correct attribution is 50%, not the 75% obtained using stock's 1,000 ms control.
        row = self.rows()[3]
        self.assertEqual(row['improvementPercent'], 50)
        self.assertEqual(row['controlRunId'], 'engineered-576-under-load')

    def test_isolated_measurements_fill_only_the_no_load_slot(self):
        self.add_series('stock', 576, latency=400, condition='isolated')
        self.add_series('stock', 128, latency=300, condition='isolated')
        rows = self.rows()
        self.assertEqual(rows[1]['improvementPercent'], 25)
        self.assertEqual(rows[2]['improvementPercent'], -10)

    def test_unmeasured_load_is_not_interpolated(self):
        self.rows()
        rows = story_rows(self.data, tokens=128, rps=2, percentile='p50')
        self.assertIsNone(rows[2]['improvementPercent'])

    def test_duplicate_run_ids_are_rejected(self):
        self.data['series'][0]['runs'].append(dict(self.data['series'][0]['runs'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate runId'):
            self.rows()

    def test_mismatched_duration_is_rejected(self):
        self.change_raw(1, lambda raw: raw['config'].update(durationSeconds=60))
        with self.assertRaisesRegex(ValueError, 'durationSeconds'):
            self.rows()

    def test_failures_are_not_counted_as_latency_samples(self):
        self.change_raw(1, lambda raw: raw['summary'].update(successfulRequests=60, failedRequests=20))
        row = self.rows()[2]
        self.assertIn('80/60', row['note'])

    def test_pair_validation_checks_workload_metadata(self):
        self.rows()
        a, b = [item['runs'][0] for item in self.data['series']]
        b['_raw']['workload']['recordCount'] = 123
        with self.assertRaisesRegex(ValueError, 'workload'):
            verify_pair(a, b)


if __name__ == '__main__':
    unittest.main()

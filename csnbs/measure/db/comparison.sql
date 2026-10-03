-- psql "$DATABASE_URL" -v left_run='run-A-id' -v right_run='run-B-id' -f db/comparison.sql
-- Retrieves each original summary. Eligibility and uncertainty belong to the validated
-- report/analyzer, not to a second SQL percentile implementation.
SELECT run_id, recorded_at, workload_id, offered_rps, git_commit, git_dirty,
       manifest->'server' AS server_provenance,
       manifest->'summary' AS original_summary,
       bundle_sha256
FROM benchmark.runs
WHERE run_id IN (:'left_run', :'right_run')
ORDER BY run_id;

-- Coverage includes all saved records. A NULL phase means the legacy file did not
-- identify phases; do not infer that these are all measured observations.
SELECT run_id, phase, count(*) AS saved_outcomes,
       count(*) FILTER (WHERE successful IS TRUE) AS successes,
       count(*) FILTER (WHERE successful IS FALSE) AS failures,
       count(*) FILTER (WHERE successful IS NULL) AS unknown_outcomes
FROM benchmark.request_outcomes
WHERE run_id IN (:'left_run', :'right_run')
GROUP BY run_id, phase
ORDER BY run_id, phase;

-- Every row can be traced back to its source bytes.
SELECT run_id, relative_path, artifact_kind, sha256, byte_length
FROM benchmark.artifacts
WHERE run_id IN (:'left_run', :'right_run')
ORDER BY run_id, relative_path;

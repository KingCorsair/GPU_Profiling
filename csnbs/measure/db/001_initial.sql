-- Raw files remain the evidence source. Import only after measurement finishes.
CREATE SCHEMA IF NOT EXISTS benchmark;

CREATE TABLE IF NOT EXISTS benchmark.campaigns (
    campaign_id text PRIMARY KEY,
    schema_version integer NOT NULL CHECK (schema_version IN (1, 2)),
    manifest jsonb NOT NULL,
    manifest_sha256 char(64) NOT NULL,
    imported_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS benchmark.runs (
    run_id text PRIMARY KEY,
    schema_version integer NOT NULL CHECK (schema_version IN (1, 2)),
    run_kind text,
    recorded_at timestamptz,
    workload_id text,
    git_commit text,
    git_dirty boolean,
    offered_rps double precision,
    request_count integer NOT NULL CHECK (request_count >= 0),
    manifest jsonb NOT NULL,
    bundle_sha256 char(64) NOT NULL,
    imported_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS benchmark.trials (
    campaign_id text NOT NULL REFERENCES benchmark.campaigns(campaign_id),
    trial_id text NOT NULL,
    run_id text REFERENCES benchmark.runs(run_id),
    manifest jsonb NOT NULL,
    PRIMARY KEY (campaign_id, trial_id)
);

CREATE TABLE IF NOT EXISTS benchmark.artifacts (
    run_id text NOT NULL REFERENCES benchmark.runs(run_id),
    relative_path text NOT NULL,
    artifact_kind text NOT NULL CHECK (artifact_kind IN ('manifest', 'requests', 'resources')),
    sha256 char(64) NOT NULL,
    byte_length bigint NOT NULL CHECK (byte_length >= 0),
    PRIMARY KEY (run_id, relative_path)
);

CREATE TABLE IF NOT EXISTS benchmark.request_outcomes (
    run_id text NOT NULL REFERENCES benchmark.runs(run_id),
    record_key text NOT NULL,
    sequence integer NOT NULL CHECK (sequence >= 0),
    request_id text,
    phase text,
    question_id text,
    category text,
    scheduled_at_ms double precision,
    sent_at_ms double precision,
    completed_at_ms double precision,
    latency_ms double precision,
    dispatch_lateness_ms double precision,
    planned_to_complete_ms double precision,
    http_status integer CHECK (http_status BETWEEN 100 AND 599),
    error text,
    successful boolean,
    raw jsonb NOT NULL,
    PRIMARY KEY (run_id, record_key)
);

CREATE TABLE IF NOT EXISTS benchmark.resource_samples (
    run_id text NOT NULL REFERENCES benchmark.runs(run_id),
    sample_index integer NOT NULL CHECK (sample_index >= 0),
    recorded_at timestamptz,
    raw jsonb NOT NULL,
    PRIMARY KEY (run_id, sample_index)
);

CREATE INDEX IF NOT EXISTS runs_recorded_at_idx ON benchmark.runs(recorded_at);
CREATE INDEX IF NOT EXISTS runs_workload_idx ON benchmark.runs(workload_id);
CREATE INDEX IF NOT EXISTS trials_run_id_idx ON benchmark.trials(run_id);
CREATE INDEX IF NOT EXISTS requests_phase_time_idx
    ON benchmark.request_outcomes(run_id, phase, scheduled_at_ms);
CREATE UNIQUE INDEX IF NOT EXISTS requests_request_id_idx
    ON benchmark.request_outcomes(run_id, request_id) WHERE request_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS resources_recorded_at_idx
    ON benchmark.resource_samples(run_id, recorded_at);

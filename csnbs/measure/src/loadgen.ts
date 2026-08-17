import { execFileSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

/* Expected types for server.py. */
type LoadConfig = {
  endpoint: string;
  requestsPerSecond: number;
  durationSeconds: number;
  timeoutMs: number; // Maximum client wait time for each request.
};

type InferRequest = { image_b64: string; question: string };

type InferResponse = { answer: string };

type RequestResult = {
  sequence: number;
  scheduledAtMs: number;
  sentAtMs: number;
  completedAtMs: number;
  status: number | null;
  error: string | null;
};

type LoadRun = {
  plannedStartMs: number;
  plannedEndMs: number;
  results: RequestResult[];
};

/* Null if no requests completed, so no latency can be computed. */
type PercentileSummary = {
  p50: number | null;
  p95: number | null;
  p99: number | null;
};

type LoadSummary = {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  achievedArrivalRateRps: number | null; // actual arrival rate; null if <2 requests sent
  completionRateRps: number; // includes failures
  successfulThroughputRps: number; // excludes failures
  successfulRequestLatencyMs: PercentileSummary; // successful requests only
  dispatchLatenessMs: PercentileSummary; // scheduled dispatch time delta
  plannedToCompleteMs: PercentileSummary; // time from scheduled dispatch to completion; includes failures
};

type WorkloadMetadata = {
  id: string;
  question: string;
  imageSha256: string;
  imageByteLength: number;
};

type SourceMetadata = {
  gitCommit: string | null;
  gitDirty: boolean | null;
};

type HardwareMetadata = {
  gpuModels: string[] | null;
  source: "local-nvidia-smi";
};

type BenchmarkMetadata = {
  modelId: string | null;
  checkpoint: string | null;
  pruningMethod: string | null;
  pruningRatio: number | null;
  tokenRemovalMode: string | null;
  batchSize: number;
  maxOutputTokens: number | null;
};

type RunManifest = {
  schema: "loadgen-run";
  schemaVersion: 1;
  runId: string;
  runKind: "open-loop";
  recordedAtUtc: string;
  timing: {
    clock: "performance.now";
    performanceTimeOriginUnixMs: number;
    plannedStartMs: number;
    plannedEndMs: number;
    finishedMs: number;
    plannedStartAtUtc: string;
    plannedEndAtUtc: string;
    finishedAtUtc: string;
  };
  config: LoadConfig;
  workload: WorkloadMetadata;
  source: SourceMetadata;
  hardware: HardwareMetadata;
  benchmark: BenchmarkMetadata;
  summary: LoadSummary;
  requests: {
    file: "requests.jsonl";
    schemaVersion: 1;
    count: number;
  };
};

const repoRoot = fileURLToPath(new URL("../../../", import.meta.url));
const outputRoot = join(repoRoot, "results", "loadgen");

const payload: InferRequest = {
  image_b64: Buffer.from("fake image bytes").toString("base64"),
  question: "What is in this image?",
};

/* Checks whether the requested load can be scheduled. */
function validateConfig(config: LoadConfig): void {
  if (!Number.isFinite(config.requestsPerSecond) || config.requestsPerSecond <= 0) {
    throw new Error("requestsPerSecond must be a finite number greater than 0");
  } else if (!Number.isFinite(config.durationSeconds) || config.durationSeconds <= 0) {
    throw new Error("durationSeconds must be a finite number greater than 0");
  } else if (config.endpoint.trim() === "") {
    throw new Error("endpoint must be nonempty");
  } else if (config.requestsPerSecond * config.durationSeconds < 1) {
    throw new Error("Configuration must schedule at least one request");
  } else if (!Number.isFinite(config.timeoutMs) || config.timeoutMs <= 0) {
    throw new Error("timeoutMs must be a finite number greater than 0");
  }
}

/* Sends one request and turns both successes and failures into a result record. */
async function sendOne(
  sequence: number,
  scheduledAtMs: number,
  config: LoadConfig,
  payload: InferRequest,
): Promise<RequestResult> {
  const sentAtMs = performance.now();
  let status: number | null = null;
  try {
    const response = await fetch(config.endpoint, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(config.timeoutMs),
    });
    status = response.status;
    if (!response.ok) {
      await response.text();
      return {
        sequence,
        scheduledAtMs,
        sentAtMs,
        completedAtMs: performance.now(),
        status,
        error: "Request failed with status " + status,
      };
    }
    const responseBody = (await response.json()) as InferResponse;
    if (typeof responseBody.answer !== "string") {
      throw new Error("Response body is missing a string answer");
    }
    return {
      sequence,
      scheduledAtMs,
      sentAtMs,
      completedAtMs: performance.now(),
      status,
      error: null,
    };
  } catch (caught: unknown) {
    return {
      sequence,
      scheduledAtMs,
      sentAtMs,
      completedAtMs: performance.now(),
      status,
      error: caught instanceof Error ? caught.message : String(caught),
    };
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/* Launches requests according to the clock, not according to prior completions. */
async function runLoad(
  config: LoadConfig,
  payload: InferRequest,
): Promise<LoadRun> {
  validateConfig(config);
  const intervalMs = 1000 / config.requestsPerSecond;
  const totalRequests = Math.floor(
    config.requestsPerSecond * config.durationSeconds,
  );
  const plannedStartMs = performance.now();
  const plannedEndMs = plannedStartMs + config.durationSeconds * 1000;
  const inFlight: Promise<RequestResult>[] = [];
  for (let sequence = 0; sequence < totalRequests; sequence++) {
    const scheduledAtMs = plannedStartMs + sequence * intervalMs;
    const delayMs = scheduledAtMs - performance.now();
    if (delayMs > 0) {
      await sleep(delayMs);
    }
    inFlight.push(sendOne(sequence, scheduledAtMs, config, payload));
  }
  const results = await Promise.all(inFlight);
  return { plannedStartMs, plannedEndMs, results };
}

/* Uses the nearest-rank definition, such as ceil(0.95 * sampleCount) - 1. */
function percentile(
  values: readonly number[],
  fraction: number,
): number | null {
  if (!Number.isFinite(fraction) || fraction <= 0 || fraction > 1) {
    throw new Error(
      "fraction must be a finite number greater than 0 and at most 1",
    );
  }
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.ceil(fraction * sorted.length) - 1;
  return sorted[index] ?? null;
}

function summarizePercentiles(values: readonly number[]): PercentileSummary {
  return {
    p50: percentile(values, 0.5),
    p95: percentile(values, 0.95),
    p99: percentile(values, 0.99),
  };
}

function summarizeRun(run: LoadRun): LoadSummary {
  const successfulResults = run.results.filter(
    (result) => result.error === null,
  );
  const failedResults = run.results.filter((result) => result.error !== null);
  const successfulRequestLatencies = successfulResults.map(
    (result) => result.completedAtMs - result.sentAtMs,
  );
  const dispatchLateness = run.results.map(
    (result) => result.sentAtMs - result.scheduledAtMs,
  );
  const plannedToComplete = run.results.map(
    (result) => result.completedAtMs - result.scheduledAtMs,
  );
  const firstSentAtMs = run.results.reduce(
    (earliest, result) => Math.min(earliest, result.sentAtMs),
    Number.POSITIVE_INFINITY,
  );
  const lastSentAtMs = run.results.reduce(
    (latest, result) => Math.max(latest, result.sentAtMs),
    Number.NEGATIVE_INFINITY,
  );
  const lastCompletedAtMs = run.results.reduce(
    (latest, result) => Math.max(latest, result.completedAtMs),
    run.plannedStartMs,
  );
  const arrivalSpanMs = lastSentAtMs - firstSentAtMs;
  const achievedArrivalRateRps =
    run.results.length >= 2 && arrivalSpanMs > 0
      ? ((run.results.length - 1) * 1000) / arrivalSpanMs
      : null;
  /* Include the configured window and any extra time spent draining requests. */
  const measurementEndMs = Math.max(run.plannedEndMs, lastCompletedAtMs);
  const elapsedSeconds = (measurementEndMs - run.plannedStartMs) / 1000;
  return {
    totalRequests: run.results.length,
    successfulRequests: successfulResults.length,
    failedRequests: failedResults.length,
    achievedArrivalRateRps,
    completionRateRps: run.results.length / elapsedSeconds,
    successfulThroughputRps: successfulResults.length / elapsedSeconds,
    successfulRequestLatencyMs: summarizePercentiles(successfulRequestLatencies),
    dispatchLatenessMs: summarizePercentiles(dispatchLateness),
    plannedToCompleteMs: summarizePercentiles(plannedToComplete),
  };
}

function runCommand(command: string, args: readonly string[]): string | null {
  try {
    return execFileSync(command, args, {
      cwd: repoRoot,
      encoding: "utf8",
    }).trim();
  } catch {
    return null;
  }
}

function collectSourceMetadata(): SourceMetadata {
  const gitCommit = runCommand("git", ["rev-parse", "HEAD"]);
  const gitStatus = runCommand("git", [
    "status",
    "--porcelain=v1",
    "--untracked-files=normal",
  ]);
  return {
    gitCommit: gitCommit === "" ? null : gitCommit,
    gitDirty: gitStatus === null ? null : gitStatus !== "",
  };
}

function collectHardwareMetadata(): HardwareMetadata {
  const gpuOutput = runCommand("nvidia-smi", [
    "--query-gpu=name",
    "--format=csv,noheader",
  ]);
  return {
    gpuModels: gpuOutput === null
      ? null
      : gpuOutput.split("\n").filter((name) => name !== ""),
    source: "local-nvidia-smi",
  };
}

function describeWorkload(payload: InferRequest): WorkloadMetadata {
  const image = Buffer.from(payload.image_b64, "base64");
  const imageHash = createHash("sha256").update(image).digest("hex");
  const workloadHash = createHash("sha256")
    .update(image)
    .update("\0")
    .update(payload.question, "utf8")
    .digest("hex");
  return {
    id: "sha256:" + workloadHash,
    question: payload.question,
    imageSha256: "sha256:" + imageHash,
    imageByteLength: image.length,
  };
}

function makeRunId(startedAtUtc: string): string {
  return (
    startedAtUtc.replace(/[:.]/g, "-") +
    "-" +
    randomUUID().slice(0, 8)
  );
}

function toUtc(performanceTimeOriginUnixMs: number, offsetMs: number): string {
  return new Date(performanceTimeOriginUnixMs + offsetMs).toISOString();
}

async function persistRun(
  manifest: RunManifest,
  results: readonly RequestResult[],
): Promise<string> {
  await mkdir(outputRoot, { recursive: true });
  const runDirectory = join(outputRoot, manifest.runId);
  await mkdir(runDirectory);

  const requestsJsonl = results.length === 0
    ? ""
    : results.map((result) => JSON.stringify(result)).join("\n") + "\n";

  await writeFile(join(runDirectory, "requests.jsonl"), requestsJsonl, {
    encoding: "utf8",
    flag: "wx",
  });
  await writeFile(
    join(runDirectory, "run.json"),
    JSON.stringify(manifest, null, 2) + "\n",
    { encoding: "utf8", flag: "wx" },
  );

  return runDirectory;
}

async function main(): Promise<void> {
  const config: LoadConfig = {
    endpoint: "http://127.0.0.1:8000/infer",
    requestsPerSecond: 5,
    durationSeconds: 10,
    timeoutMs: 5000,
  };

  /* Collect metadata before timing so these commands cannot disturb arrivals. */
  const source = collectSourceMetadata();
  const hardware = collectHardwareMetadata();
  const workload = describeWorkload(payload);
  const benchmark: BenchmarkMetadata = {
    modelId: null,
    checkpoint: null,
    pruningMethod: null,
    pruningRatio: null,
    tokenRemovalMode: null,
    batchSize: 1,
    maxOutputTokens: null,
  };
  const runId = makeRunId(new Date().toISOString());

  const run = await runLoad(config, payload);
  const finishedMs = performance.now();
  const summary = summarizeRun(run);
  const performanceTimeOriginUnixMs = performance.timeOrigin;

  const manifest: RunManifest = {
    schema: "loadgen-run",
    schemaVersion: 1,
    runId,
    runKind: "open-loop",
    recordedAtUtc: new Date().toISOString(),
    timing: {
      clock: "performance.now",
      performanceTimeOriginUnixMs,
      plannedStartMs: run.plannedStartMs,
      plannedEndMs: run.plannedEndMs,
      finishedMs,
      plannedStartAtUtc: toUtc(
        performanceTimeOriginUnixMs,
        run.plannedStartMs,
      ),
      plannedEndAtUtc: toUtc(
        performanceTimeOriginUnixMs,
        run.plannedEndMs,
      ),
      finishedAtUtc: toUtc(performanceTimeOriginUnixMs, finishedMs),
    },
    config,
    workload,
    source,
    hardware,
    benchmark,
    summary,
    requests: {
      file: "requests.jsonl",
      schemaVersion: 1,
      count: run.results.length,
    },
  };

  const outputDirectory = await persistRun(manifest, run.results);
  console.log(JSON.stringify({ runId, outputDirectory, summary }, null, 2));
}

await main();

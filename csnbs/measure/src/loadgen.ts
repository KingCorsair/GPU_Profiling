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
const payload: InferRequest = {
  image_b64: Buffer.from("fake image bytes").toString("base64"),
  question: "What is in this image?",
};
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
async function main(): Promise<void> {
  const config: LoadConfig = {
    endpoint: "http://127.0.0.1:8000/infer",
    requestsPerSecond: 5,
    durationSeconds: 10,
    timeoutMs: 5000,
  };
  const run = await runLoad(config, payload);
  const summary = summarizeRun(run);
  console.log(JSON.stringify(summary, null, 2));
}

await main();

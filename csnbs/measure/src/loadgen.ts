import { execFileSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { mkdir, open, writeFile, readFile, rename } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { collectServerMetadata, type ServerMetadata } from "./server_metadata.js";

export type RunKind = "open-loop" | "isolated" | "smoke";
export type Phase = "warmup" | "measurement";
export type Outcome = "success" | "http-error" | "invalid-response" | "timeout" | "network-error" | "cancelled";
export type DatasetRecord = {
  image: string; question: string; question_id?: string | number;
  category?: string; source_dataset?: string;
};
export type InferRequest = { image_b64: string; question: string };
export type DatasetPayload = InferRequest & {
  workloadIndex: number; questionId: string; category: string; sourceDataset: string | null;
};
export type LoadConfig = {
  endpoint: string; requestsPerSecond: number; durationSeconds: number;
  measuredRequests: number; timeoutMs: number; datasetPath: string;
  warmupRequests: number; seed: number; settleMs: number; runKind: RunKind;
  outputDir: string; runDeadlineMs: number | null;
};
export type ServerMetrics = {
  schema_version?: number; service_ms?: number; queue_ms?: number | null;
  generation_wall_ms?: number | null; preprocess_ms?: number | null;
  postprocess_ms?: number | null; generated_text_tokens?: number | null;
  prompt_text_tokens?: number | null; visual_tokens?: number | null;
  queue_unavailable_reason?: string; [key: string]: unknown;
};
export type RequestResult = {
  requestId: string; phase: Phase; sequence: number; workloadIndex: number;
  questionId: string; category: string; sourceDataset: string | null;
  scheduledAtMs: number; sentAtMs: number | null; completedAtMs: number;
  latencyMs: number | null; dispatchLatenessMs: number | null;
  plannedToCompleteMs: number | null; status: number | null; error: string | null;
  outcome: Outcome; serverRequestId: string | null; serverMetrics: ServerMetrics | null;
  unattributedClientMs: number | null;
};
export type LoadRun = {
  plannedStartMs: number; plannedEndMs: number; finishedMs: number;
  results: RequestResult[]; warmupResults: RequestResult[];
  status: "complete" | "aborted"; stopReason: string | null;
};
export type PercentileSummary = {
  sampleCount: number; p50: number | null; p95: number | null; p99: number | null;
  p95Reason: string | null; p99Reason: string | null;
};
export type LoadSummary = {
  totalRequests: number; dispatchedRequests: number; successfulRequests: number;
  failedRequests: number; timedOutRequests: number; cancelledRequests: number;
  achievedArrivalRateRps: number | null; completionRateRps: number;
  successfulThroughputRps: number; successfulThroughputIncludingDrainRps: number;
  measurementWindowSeconds: number; completedWithinWindow: number;
  successfulWithinWindow: number; successfulThroughputWithinWindowRps: number;
  outstandingAtWindowEnd: number; drainMs: number;
  successfulRequestLatencyMs: PercentileSummary; dispatchLatenessMs: PercentileSummary;
  plannedToCompleteMs: PercentileSummary; serverServiceMs: PercentileSummary;
  serverQueueMs: PercentileSummary; unattributedClientMs: PercentileSummary;
};
export type SourceMetadata = { gitCommit: string | null; gitDirty: boolean | null };
export type WorkloadMetadata = {
  id: string; recordCount: number; totalImageByteLength: number; orderSeed: number;
  measuredOrderHash: string; measuredCategoryCounts: Record<string, number>;
  measuredQuestionIds: string[]; completeDatasetCycles: number;
};
export type RunManifest = {
  schema: "loadgen-run"; schemaVersion: 2; runId: string; runKind: RunKind;
  recordedAtUtc: string; status: "complete" | "aborted"; stopReason: string | null;
  timing: { clock: "performance.now"; performanceTimeOriginUnixMs: number;
    plannedStartMs: number; plannedEndMs: number; finishedMs: number;
    plannedStartAtUtc: string; plannedEndAtUtc: string; finishedAtUtc: string };
  config: LoadConfig; workload: WorkloadMetadata; source: SourceMetadata;
  hardware: { gpuModels: string[] | null; source: string };
  server: ServerMetadata; summary: LoadSummary;
  warmup: { requested: number; completed: number; successful: number; isolated: true; settleMs: number };
  requests: { file: "requests.jsonl"; schemaVersion: 3; count: number; sha256: string };
  quality: { reportable: boolean; reasons: string[]; percentileConvention: "nearest-rank" };
};

const repoRoot = fileURLToPath(new URL("../../../", import.meta.url));
const defaultOutputRoot = join(repoRoot, "results", "loadgen");

function positive(value: number, name: string, integer = false): void {
  if (!Number.isFinite(value) || value <= 0 || (integer && !Number.isSafeInteger(value))) {
    throw new Error(`${name} must be a positive finite${integer ? " integer" : " number"}`);
  }
}
export function validateConfig(config: LoadConfig): void {
  positive(config.requestsPerSecond, "requestsPerSecond");
  positive(config.durationSeconds, "durationSeconds");
  positive(config.measuredRequests, "measuredRequests", true);
  positive(config.timeoutMs, "timeoutMs", true);
  if (config.timeoutMs > 4294967295) throw new Error("timeoutMs exceeds timer range");
  if (!Number.isSafeInteger(config.warmupRequests) || config.warmupRequests < 0) throw new Error("warmupRequests must be a nonnegative integer");
  if (!Number.isSafeInteger(config.seed) || config.seed < 0 || config.seed > 0xffffffff) throw new Error("seed must be an integer from 0 through 4294967295");
  if (!Number.isFinite(config.settleMs) || config.settleMs < 0) throw new Error("settleMs must be nonnegative and finite");
  if (config.runDeadlineMs !== null) positive(config.runDeadlineMs, "runDeadlineMs");
  if (!["open-loop", "isolated", "smoke"].includes(config.runKind)) throw new Error("Unknown run kind");
  const url = new URL(config.endpoint);
  if (!["http:", "https:"].includes(url.protocol)) throw new Error("endpoint must use HTTP or HTTPS");
  if (!config.datasetPath || !config.outputDir) throw new Error("datasetPath and outputDir are required");
}

export function parseLoadConfig(args = process.argv.slice(2)): LoadConfig {
  const allowed = new Set(["endpoint", "rps", "duration", "requests", "timeout", "dataset", "warmup", "seed", "settle-ms", "run-kind", "output-dir", "run-deadline-ms"]);
  const flags = new Map<string, string>();
  for (let i = 0; i < args.length; i += 2) {
    const flag = args[i]!;
    if (!flag.startsWith("--") || !allowed.has(flag.slice(2))) throw new Error(`Unknown argument ${flag}`);
    const value = args[i + 1];
    if (value === undefined || value.startsWith("--")) throw new Error(`Missing value for ${flag}`);
    if (flags.has(flag.slice(2))) throw new Error(`Duplicate argument ${flag}`);
    flags.set(flag.slice(2), value);
  }
  const required = (key: string): string => {
    const value = flags.get(key);
    if (value === undefined || !value.trim()) throw new Error(`Missing --${key} argument`);
    return value;
  };
  const kind = (flags.get("run-kind") ?? "open-loop") as RunKind;
  const rps = Number(flags.get("rps") ?? (kind === "isolated" ? "1" : required("rps")));
  const requested = flags.has("requests") ? Number(flags.get("requests")) : null;
  const duration = flags.has("duration") ? Number(flags.get("duration")) : null;
  if (requested === null && duration === null) throw new Error("Supply --requests or --duration");
  if (duration !== null) positive(duration, "duration");
  if (requested !== null) positive(requested, "requests", true);
  positive(rps, "rps");
  const count = requested ?? Math.floor(rps * duration! + 1e-9);
  if (requested !== null && duration !== null && Math.floor(rps * duration + 1e-9) !== requested) throw new Error("--requests and --duration imply different request counts");
  const config: LoadConfig = {
    endpoint: required("endpoint"), requestsPerSecond: rps, durationSeconds: duration ?? count / rps,
    measuredRequests: count, timeoutMs: Number(flags.get("timeout") ?? "120000"), datasetPath: required("dataset"),
    warmupRequests: Number(flags.get("warmup") ?? (kind === "smoke" ? "0" : "10")),
    seed: Number(flags.get("seed") ?? "90"), settleMs: Number(flags.get("settle-ms") ?? "0"), runKind: kind,
    outputDir: resolve(flags.get("output-dir") ?? defaultOutputRoot),
    runDeadlineMs: flags.has("run-deadline-ms") ? Number(flags.get("run-deadline-ms")) : null,
  };
  validateConfig(config);
  return config;
}

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let value = state;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}
function shuffle<T>(items: readonly T[], random: () => number): T[] {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [copy[i], copy[j]] = [copy[j]!, copy[i]!];
  }
  return copy;
}
/** Every full cycle visits each record once; category round-robin balances each prefix. */
export function buildWorkloadOrder(payloads: readonly DatasetPayload[], count: number, seed: number): number[] {
  if (payloads.length === 0) throw new Error("Dataset is empty");
  if (!Number.isSafeInteger(count) || count < 0) throw new Error("count must be a nonnegative integer");
  const random = seededRandom(seed);
  const groups = new Map<string, number[]>();
  payloads.forEach((payload, index) => {
    const group = groups.get(payload.category) ?? [];
    group.push(index); groups.set(payload.category, group);
  });
  const order: number[] = [];
  while (order.length < count) {
    const categories = shuffle([...groups.keys()].sort(), random);
    const shuffled = categories.map((category) => shuffle(groups.get(category)!, random));
    for (let offset = 0; shuffled.some((group) => offset < group.length); offset++) {
      for (const group of shuffled) {
        if (offset < group.length) order.push(group[offset]!);
        if (order.length === count) return order;
      }
    }
  }
  return order;
}

export async function loadDataset(path: string): Promise<DatasetPayload[]> {
  const data: unknown = JSON.parse(await readFile(path, "utf8"));
  if (!Array.isArray(data) || data.length === 0) throw new Error("Dataset must be a nonempty array");
  const seen = new Set<string>();
  return Promise.all(data.map(async (unknownRecord: unknown, workloadIndex: number) => {
    if (unknownRecord === null || typeof unknownRecord !== "object") throw new Error(`Invalid dataset record ${workloadIndex}`);
    const record = unknownRecord as DatasetRecord;
    if (typeof record.image !== "string" || !record.image || typeof record.question !== "string") throw new Error(`Record ${workloadIndex} needs image and question strings`);
    const questionId = String(record.question_id ?? `record-${workloadIndex}`);
    if (seen.has(questionId)) throw new Error(`Duplicate question_id ${questionId}`);
    seen.add(questionId);
    const image = await readFile(join(dirname(path), record.image));
    return {
      image_b64: image.toString("base64"), question: record.question, workloadIndex, questionId,
      category: typeof record.category === "string" && record.category.trim() ? record.category.trim() : "unknown",
      sourceDataset: typeof record.source_dataset === "string" ? record.source_dataset : null,
    };
  }));
}

export function describeWorkload(payloads: readonly DatasetPayload[], count: number, seed: number): WorkloadMetadata {
  const hash = createHash("sha256");
  let totalImageByteLength = 0;
  for (const payload of payloads) {
    const image = Buffer.from(payload.image_b64, "base64");
    totalImageByteLength += image.length;
    hash.update(JSON.stringify([image.length, payload.questionId, payload.question, payload.category, payload.sourceDataset]));
    hash.update(image);
  }
  const order = buildWorkloadOrder(payloads, count, seed);
  const measuredQuestionIds = order.map((index) => payloads[index]!.questionId);
  const measuredCategoryCounts: Record<string, number> = {};
  for (const index of order) {
    const category = payloads[index]!.category;
    measuredCategoryCounts[category] = (measuredCategoryCounts[category] ?? 0) + 1;
  }
  return { id: `sha256:${hash.digest("hex")}`, recordCount: payloads.length, totalImageByteLength,
    orderSeed: seed, measuredOrderHash: `sha256:${createHash("sha256").update(JSON.stringify(measuredQuestionIds)).digest("hex")}`,
    measuredCategoryCounts, measuredQuestionIds, completeDatasetCycles: Math.floor(count / payloads.length) };
}

function delay(ms: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted || ms <= 0) return Promise.resolve();
  return new Promise((done) => {
    const finish = () => { clearTimeout(timer); signal.removeEventListener("abort", finish); done(); };
    const timer = setTimeout(finish, Math.min(ms, 2147483647));
    signal.addEventListener("abort", finish, { once: true });
  });
}
async function waitUntil(target: number, signal: AbortSignal): Promise<void> {
  while (!signal.aborted && performance.now() < target) await delay(target - performance.now(), signal);
}
function baseResult(runId: string, phase: Phase, sequence: number, scheduledAtMs: number, payload: DatasetPayload): RequestResult {
  return {
    requestId: `${runId}:${phase}:${sequence}`, phase, sequence,
    workloadIndex: payload.workloadIndex, questionId: payload.questionId, category: payload.category,
    sourceDataset: payload.sourceDataset, scheduledAtMs, sentAtMs: null, completedAtMs: performance.now(),
    latencyMs: null, dispatchLatenessMs: null, plannedToCompleteMs: null,
    status: null, error: "Not dispatched", outcome: "cancelled", serverRequestId: null,
    serverMetrics: null, unattributedClientMs: null,
  };
}
function object(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
}
function validMetrics(value: unknown): ServerMetrics | null {
  const metrics = object(value);
  if (metrics === null) return null;
  for (const key of ["service_ms", "queue_ms", "generation_wall_ms", "preprocess_ms", "postprocess_ms", "generated_text_tokens", "prompt_text_tokens", "visual_tokens"]) {
    const value = metrics[key];
    if (value !== undefined && value !== null && (typeof value !== "number" || !Number.isFinite(value) || value < 0)) throw new Error(`Invalid server metric ${key}`);
  }
  return metrics as ServerMetrics;
}

export async function sendOne(result: RequestResult, config: LoadConfig, payload: DatasetPayload, signal: AbortSignal): Promise<RequestResult> {
  if (signal.aborted) return { ...result, completedAtMs: performance.now(), error: String(signal.reason ?? "Run interrupted") };
  const timeout = AbortSignal.timeout(config.timeoutMs);
  result.sentAtMs = performance.now();
  try {
    const response = await fetch(config.endpoint, {
      method: "POST", headers: { "content-type": "application/json", "x-request-id": result.requestId },
      body: JSON.stringify({ image_b64: payload.image_b64, question: payload.question, request_id: result.requestId }),
      signal: AbortSignal.any([signal, timeout]),
    });
    result.status = response.status;
    if (!response.ok) {
      await response.text(); result.outcome = "http-error"; result.error = `HTTP ${response.status}`;
    } else {
      result.outcome = "invalid-response";
      const body = object(await response.json());
      if (typeof body?.answer !== "string") throw new Error("Response body is missing a string answer");
      result.serverRequestId = typeof body.request_id === "string" ? body.request_id : null;
      if (result.serverRequestId !== null && result.serverRequestId !== result.requestId) throw new Error("Server request_id does not match client requestId");
      result.serverMetrics = validMetrics(body.metrics);
      result.outcome = "success"; result.error = null;
    }
  } catch (error: unknown) {
    result.error = error instanceof Error ? error.message : String(error);
    if (signal.aborted) result.outcome = "cancelled";
    else if (timeout.aborted) result.outcome = "timeout";
    else if (result.outcome !== "invalid-response") result.outcome = "network-error";
  }
  result.completedAtMs = performance.now();
  result.latencyMs = result.completedAtMs - result.sentAtMs;
  result.dispatchLatenessMs = result.sentAtMs - result.scheduledAtMs;
  result.plannedToCompleteMs = result.completedAtMs - result.scheduledAtMs;
  const service = result.serverMetrics?.service_ms;
  result.unattributedClientMs = typeof service === "number" ? result.latencyMs - service : null;
  return result;
}

export type RunOptions = { runId?: string; signal?: AbortSignal; onResult?: (result: RequestResult) => void };
/** Warmup drains before a fresh clock. Only isolated mode waits for each measured response. */
export async function runLoad(config: LoadConfig, payloads: readonly DatasetPayload[], options: RunOptions = {}): Promise<LoadRun> {
  validateConfig(config);
  const runId = options.runId ?? randomUUID();
  const controller = new AbortController();
  const signal = options.signal ? AbortSignal.any([controller.signal, options.signal]) : controller.signal;
  const deadline = config.runDeadlineMs === null ? null : setTimeout(() => controller.abort("Run deadline exceeded"), config.runDeadlineMs);
  const order = buildWorkloadOrder(payloads, config.measuredRequests, config.seed);
  const warmupOrder = buildWorkloadOrder(payloads, config.warmupRequests, config.seed ^ 0x51ed);
  const warmupResults: RequestResult[] = [];
  const measured: Promise<RequestResult>[] = [];
  let plannedStartMs = performance.now();
  let plannedEndMs = plannedStartMs + config.durationSeconds * 1000;
  const emit = (result: RequestResult): RequestResult => { options.onResult?.(result); return result; };
  try {
    for (let sequence = 0; sequence < config.warmupRequests; sequence++) {
      const payload = payloads[warmupOrder[sequence]!]!;
      const result = emit(await sendOne(baseResult(runId, "warmup", sequence, performance.now(), payload), config, payload, signal));
      warmupResults.push(result);
      if (result.outcome !== "success") controller.abort(`Warmup failed: ${result.outcome}; server drain is unverified`);
    }
    await delay(config.settleMs, signal);
    plannedStartMs = performance.now();
    plannedEndMs = plannedStartMs + config.durationSeconds * 1000;
    for (let sequence = 0; sequence < config.measuredRequests; sequence++) {
      const scheduledAtMs = config.runKind === "isolated" ? performance.now() : plannedStartMs + sequence * 1000 / config.requestsPerSecond;
      await waitUntil(scheduledAtMs, signal);
      const payload = payloads[order[sequence]!]!;
      const pending = sendOne(baseResult(runId, "measurement", sequence, scheduledAtMs, payload), config, payload, signal).then(emit);
      measured.push(pending);
      if (config.runKind === "isolated") await pending;
    }
    if (config.runKind !== "isolated") await waitUntil(plannedEndMs, signal);
    const results = await Promise.all(measured);
    const finishedMs = performance.now();
    if (config.runKind === "isolated") plannedEndMs = finishedMs;
    return { plannedStartMs, plannedEndMs, finishedMs, results, warmupResults,
      status: signal.aborted ? "aborted" : "complete", stopReason: signal.aborted ? String(signal.reason ?? "Interrupted") : null };
  } finally {
    if (deadline !== null) clearTimeout(deadline);
  }
}

/** Nearest rank: sorted[ceil(fraction*n)-1], with explicit finite input validation. */
export function percentile(values: readonly number[], fraction: number): number | null {
  if (!Number.isFinite(fraction) || fraction <= 0 || fraction > 1) throw new Error("fraction must be >0 and <=1");
  if (values.some((value) => !Number.isFinite(value))) throw new Error("Percentile observations must be finite");
  if (values.length === 0) return null;
  return [...values].sort((a, b) => a - b)[Math.ceil(fraction * values.length) - 1]!;
}
export function summarizePercentiles(values: readonly number[]): PercentileSummary {
  return {
    sampleCount: values.length, p50: percentile(values, 0.5),
    p95: values.length >= 200 ? percentile(values, 0.95) : null,
    p99: values.length >= 1000 ? percentile(values, 0.99) : null,
    p95Reason: values.length >= 200 ? null : "Fewer than 200 observations; exploratory tail withheld",
    p99Reason: values.length >= 1000 ? null : "Fewer than 1000 observations; exploratory tail withheld",
  };
}
function finiteValues(values: readonly (number | null | undefined)[]): number[] {
  return values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
}
export function summarizeRun(run: LoadRun): LoadSummary {
  const results = run.results;
  const dispatched = results.filter((row) => row.sentAtMs !== null);
  const success = results.filter((row) => row.outcome === "success");
  const sent = dispatched.map((row) => row.sentAtMs!);
  const span = sent.length >= 2 ? sent.reduce((a, b) => Math.max(a, b), -Infinity) - sent.reduce((a, b) => Math.min(a, b), Infinity) : 0;
  const windowSeconds = Math.max(0.000001, (run.plannedEndMs - run.plannedStartMs) / 1000);
  const end = dispatched.reduce((latest, row) => Math.max(latest, row.completedAtMs), run.plannedEndMs);
  const elapsedSeconds = Math.max(0.000001, (end - run.plannedStartMs) / 1000);
  const inWindow = dispatched.filter((row) => row.completedAtMs <= run.plannedEndMs);
  const successfulWithinWindow = inWindow.filter((row) => row.outcome === "success").length;
  return {
    totalRequests: results.length, dispatchedRequests: dispatched.length, successfulRequests: success.length,
    failedRequests: results.length - success.length, timedOutRequests: results.filter((row) => row.outcome === "timeout").length,
    cancelledRequests: results.filter((row) => row.outcome === "cancelled").length,
    achievedArrivalRateRps: span > 0 ? (sent.length - 1) * 1000 / span : null,
    completionRateRps: dispatched.length / elapsedSeconds,
    successfulThroughputRps: success.length / elapsedSeconds,
    successfulThroughputIncludingDrainRps: success.length / elapsedSeconds,
    measurementWindowSeconds: windowSeconds, completedWithinWindow: inWindow.length,
    successfulWithinWindow, successfulThroughputWithinWindowRps: successfulWithinWindow / windowSeconds,
    outstandingAtWindowEnd: dispatched.filter((row) => row.sentAtMs! <= run.plannedEndMs && row.completedAtMs > run.plannedEndMs).length,
    drainMs: Math.max(0, end - run.plannedEndMs),
    successfulRequestLatencyMs: summarizePercentiles(finiteValues(success.map((row) => row.latencyMs))),
    dispatchLatenessMs: summarizePercentiles(finiteValues(dispatched.map((row) => row.dispatchLatenessMs))),
    plannedToCompleteMs: summarizePercentiles(finiteValues(dispatched.map((row) => row.plannedToCompleteMs))),
    serverServiceMs: summarizePercentiles(finiteValues(success.map((row) => row.serverMetrics?.service_ms))),
    serverQueueMs: summarizePercentiles(finiteValues(success.map((row) => row.serverMetrics?.queue_ms))),
    unattributedClientMs: summarizePercentiles(finiteValues(success.map((row) => row.unattributedClientMs))),
  };
}

function runCommand(command: string, args: readonly string[]): string | null {
  try { return execFileSync(command, args, { cwd: repoRoot, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim(); }
  catch { return null; }
}
export function collectSourceMetadata(): SourceMetadata {
  const commit = runCommand("git", ["rev-parse", "HEAD"]);
  const status = runCommand("git", ["status", "--porcelain=v1", "--untracked-files=normal"]);
  return { gitCommit: commit || null, gitDirty: status === null ? null : status !== "" };
}
const utc = (offsetMs: number): string => new Date(performance.timeOrigin + offsetMs).toISOString();

/** Serialized asynchronous appends preserve completed outcomes without blocking dispatch. */
export async function createJournal(directory: string): Promise<{
  append: (row: RequestResult) => void; finish: () => Promise<{ count: number; sha256: string }>;
}> {
  const file = await open(join(directory, "requests.jsonl"), "wx");
  const hash = createHash("sha256");
  let count = 0;
  let pending = Promise.resolve();
  let failure: unknown = null;
  return {
    append(row) {
      const line = JSON.stringify(row) + "\n";
      count++; hash.update(line);
      pending = pending.then(async () => { if (failure === null) await file.writeFile(line); }).catch((error: unknown) => { failure = error; });
    },
    async finish() {
      await pending;
      try { if (failure !== null) throw failure; await file.sync(); }
      finally { await file.close(); }
      return { count, sha256: `sha256:${hash.digest("hex")}` };
    },
  };
}

export function runQualityReasons(run: Pick<RunManifest, "config" | "source" | "workload" | "server" | "summary" | "hardware" | "status" | "runKind">): string[] {
  const { config, source, workload, server, summary, hardware } = run;
  const reasons: string[] = [];
  if (run.runKind === "smoke") reasons.push("Smoke/integration run");
  if (run.status !== "complete") reasons.push("Run aborted");
  if (source.gitDirty !== false) reasons.push("Load-generator source is dirty or unknown");
  if (source.gitCommit === null) reasons.push("Load-generator commit unknown");
  const extendedServer = server as unknown as Record<string, unknown>;
  const serverSource = object(extendedServer.source);
  if (serverSource?.gitDirty !== false) reasons.push("Server source is dirty or unknown");
  if (typeof serverSource?.gitCommit !== "string") reasons.push("Server commit unknown");
  if (extendedServer.mode !== "model" || extendedServer.modelLoaded !== true) reasons.push("Server is not a confirmed loaded model");
  if (server.error !== null) reasons.push("Server metadata unavailable");
  if (server.modelId === null || server.checkpointRevision === null) reasons.push("Model identity/checkpoint revision incomplete");
  if (server.configuration === null) reasons.push("Effective server configuration unavailable");
  else {
    const controls = ["implementation", "max_new_tokens", "do_sample", "use_cache", "eos_policy", "prompt_template", "batch_size", "dtype"];
    if (server.modelId?.includes("llava")) controls.push("visual_token_num", "important_ratio");
    const missing = controls.filter((key) => server.configuration![key] === null || server.configuration![key] === undefined);
    if (missing.length) reasons.push(`Effective server controls unavailable: ${missing.join(", ")}`);
  }
  if (workload.completeDatasetCycles < 1 || config.measuredRequests % workload.recordCount !== 0) reasons.push("Measured workload is not complete dataset cycles");
  if (config.warmupRequests < 10) reasons.push("Fewer than ten warmup requests");
  if (summary.failedRequests > 0) reasons.push("Measured failures; successful-only tails are conditional on success");
  if (hardware.gpuModels === null || hardware.gpuModels.length === 0) reasons.push("Server GPU identity unavailable");
  return reasons;
}

export async function main(args = process.argv.slice(2)): Promise<RunManifest> {
  const config = parseLoadConfig(args);
  const payloads = await loadDataset(config.datasetPath);
  const source = collectSourceMetadata();
  const workload = describeWorkload(payloads, config.measuredRequests, config.seed);
  const server = await collectServerMetadata(config.endpoint);
  const runId = `${new Date().toISOString().replaceAll(":", "-")}_rps-${config.requestsPerSecond}_${randomUUID().slice(0, 8)}`;
  const outputDirectory = join(config.outputDir, runId);
  await mkdir(outputDirectory, { recursive: true });
  const draft = { schema: "loadgen-run-partial", schemaVersion: 2, runId, runKind: config.runKind,
    status: "running", config, workload, source, server, recordedAtUtc: new Date().toISOString(),
    note: "No final manifest means interrupted/incomplete; never infer missing outcomes as success." };
  await writeFile(join(outputDirectory, "run.partial.json"), JSON.stringify(draft, null, 2) + "\n", { flag: "wx" });
  const journal = await createJournal(outputDirectory);
  const controller = new AbortController();
  const interrupt = () => controller.abort("Process interrupted by signal");
  process.on("SIGINT", interrupt); process.on("SIGTERM", interrupt);
  let run: LoadRun;
  let requests: { count: number; sha256: string };
  try {
    run = await runLoad(config, payloads, { runId, signal: controller.signal, onResult: journal.append });
  } finally {
    requests = await journal.finish();
    process.removeListener("SIGINT", interrupt); process.removeListener("SIGTERM", interrupt);
  }
  const summary = summarizeRun(run);
  const healthHardware = object((server as unknown as Record<string, unknown>).hardware);
  const gpuModels = Array.isArray(healthHardware?.gpuModels) ? healthHardware.gpuModels.filter((value): value is string => typeof value === "string") : null;
  const hardware = { gpuModels, source: "server-health" };
  const reasons = runQualityReasons({ config, source, workload, server, summary, hardware, status: run.status, runKind: config.runKind });
  const manifest: RunManifest = {
    schema: "loadgen-run", schemaVersion: 2, runId, runKind: config.runKind,
    recordedAtUtc: new Date().toISOString(), status: run.status, stopReason: run.stopReason,
    timing: { clock: "performance.now", performanceTimeOriginUnixMs: performance.timeOrigin,
      plannedStartMs: run.plannedStartMs, plannedEndMs: run.plannedEndMs, finishedMs: run.finishedMs,
      plannedStartAtUtc: utc(run.plannedStartMs), plannedEndAtUtc: utc(run.plannedEndMs), finishedAtUtc: utc(run.finishedMs) },
    config, workload, source, hardware, server, summary,
    warmup: { requested: config.warmupRequests, completed: run.warmupResults.length,
      successful: run.warmupResults.filter((row) => row.outcome === "success").length, isolated: true, settleMs: config.settleMs },
    requests: { file: "requests.jsonl", schemaVersion: 3, ...requests! },
    quality: { reportable: reasons.length === 0, reasons, percentileConvention: "nearest-rank" },
  };
  await writeFile(join(outputDirectory, "run.json.tmp"), JSON.stringify(manifest, null, 2) + "\n", { flag: "wx" });
  await rename(join(outputDirectory, "run.json.tmp"), join(outputDirectory, "run.json"));
  console.log(JSON.stringify({ runId, outputDirectory, summary, status: run.status }));
  if (run.status === "aborted") process.exitCode = 2;
  return manifest;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await main();
}

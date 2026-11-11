/** Synthetic diagnostic: fixed-budget open arrivals versus a response-paced client.
 * Run: node --import tsx src/coordinated_omission_demo.ts --output <new-directory>
 * No model/GPU is involved. This file does not modify the canonical load generator.
 */
import { spawn, execFileSync, type ChildProcess } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  createJournal, runLoad, sendOne, summarizeRun, type DatasetPayload,
  type LoadConfig, type LoadRun, type RequestResult,
} from './loadgen.js';

const thisFile = fileURLToPath(import.meta.url);
const packageRoot = fileURLToPath(new URL('../', import.meta.url));
const repoRoot = resolve(packageRoot, '../..');
const sleep = (ms: number) => new Promise<void>((done) => setTimeout(done, Math.max(0, ms)));
const hash = (value: string | Buffer) => `sha256:${createHash('sha256').update(value).digest('hex')}`;
type Mode = 'open-loop' | 'closed-loop';
type Condition = { id: string; mode: Mode; targetRps: number };
const payload: DatasetPayload = {
  image_b64: 'c3ludGhldGlj', question: 'Synthetic service operation', workloadIndex: 0,
  questionId: 'synthetic-1', category: 'synthetic', sourceDataset: 'serial-service-calibration',
};

function row(runId: string, phase: 'warmup' | 'measurement', sequence: number, scheduledAtMs: number): RequestResult {
  return {
    requestId: `${runId}:${phase}:${sequence}`, phase, sequence, workloadIndex: 0,
    questionId: payload.questionId, category: payload.category, sourceDataset: payload.sourceDataset,
    scheduledAtMs, sentAtMs: null, completedAtMs: performance.now(), latencyMs: null,
    dispatchLatenessMs: null, plannedToCompleteMs: null, status: null, error: 'Not dispatched',
    outcome: 'cancelled', serverRequestId: null, serverMetrics: null, unattributedClientMs: null,
  };
}

async function waitUntil(target: number, signal: AbortSignal): Promise<void> {
  while (!signal.aborted && performance.now() < target) await sleep(Math.min(50, target - performance.now()));
}

/** Deliberate closed-loop control: at most one outstanding request, even when late.
 * scheduledAtMs preserves the reference schedule to expose missed arrivals. It is
 * not claimed to be an actual server arrival. The actual observation window ends
 * after the last terminal response, separately from the fixed reference horizon.
 */
export async function runClosedLoop(config: LoadConfig, runId: string, onResult: (result: RequestResult) => void): Promise<LoadRun> {
  const controller = new AbortController();
  const deadline = setTimeout(() => controller.abort('Diagnostic deadline reached'), config.runDeadlineMs ?? 120000);
  const warmupResults: RequestResult[] = [];
  const results: RequestResult[] = [];
  const signal = controller.signal;
  try {
    for (let sequence = 0; sequence < config.warmupRequests; sequence++) {
      const result = await sendOne(row(runId, 'warmup', sequence, performance.now()), config, payload, signal);
      warmupResults.push(result); onResult(result);
      if (result.outcome !== 'success') controller.abort('Warmup failed');
    }
    const plannedStartMs = performance.now();
    for (let sequence = 0; sequence < config.measuredRequests; sequence++) {
      const referenceAtMs = plannedStartMs + sequence * 1000 / config.requestsPerSecond;
      await waitUntil(referenceAtMs, signal);
      // This await is the intentional coordinated-omission mechanism.
      const result = await sendOne(row(runId, 'measurement', sequence, referenceAtMs), config, payload, signal);
      results.push(result); onResult(result);
    }
    const finishedMs = performance.now();
    return { plannedStartMs, plannedEndMs: finishedMs, finishedMs, results, warmupResults,
      status: signal.aborted ? 'aborted' : 'complete', stopReason: signal.aborted ? String(signal.reason) : null };
  } finally { clearTimeout(deadline); }
}

/** Runs in its own process so server timers do not share the client's event loop. */
async function serve(delayMs: number): Promise<void> {
  let queued = Promise.resolve();
  let outstanding = 0, maximumOutstanding = 0, measuredReceived = 0, warmupReceived = 0;
  const server = createServer((request, response) => {
    if (request.url === '/health') {
      response.setHeader('content-type', 'application/json');
      response.end(JSON.stringify({ pid: process.pid, service: 'synthetic-serial-queue', requestedServiceDelayMs: delayMs,
        outstanding, maximumOutstanding, measuredReceived, warmupReceived }));
      return;
    }
    if (request.url !== '/infer') { response.writeHead(404); response.end(); return; }
    let body = '';
    request.on('data', (chunk: Buffer) => { body += chunk.toString(); });
    request.on('end', () => {
      let data: { request_id?: string };
      try { data = JSON.parse(body) as { request_id?: string }; }
      catch { response.writeHead(400); response.end('Invalid JSON'); return; }
      const entered = performance.now();
      outstanding++; maximumOutstanding = Math.max(maximumOutstanding, outstanding);
      if (data.request_id?.includes(':measurement:')) measuredReceived++;
      else if (data.request_id?.includes(':warmup:')) warmupReceived++;
      queued = queued.then(async () => {
        const admitted = performance.now();
        await sleep(delayMs);
        const finished = performance.now();
        response.setHeader('content-type', 'application/json');
        response.end(JSON.stringify({ answer: 'synthetic', request_id: data.request_id, metrics: {
          schema_version: 1, queue_ms: admitted - entered, service_ms: finished - admitted,
          requested_service_delay_ms: delayMs,
        } }));
        outstanding--;
      });
    });
  });
  await new Promise<void>((done) => server.listen(0, '127.0.0.1', done));
  const address = server.address();
  if (address === null || typeof address === 'string') throw Error('No TCP address');
  console.log(JSON.stringify({ endpoint: `http://127.0.0.1:${address.port}/infer`, pid: process.pid }));
  process.on('SIGTERM', () => { server.close(); server.closeAllConnections(); });
}

async function startService(delayMs: number): Promise<{ child: ChildProcess; endpoint: string; closed: Promise<void> }> {
  const child = spawn(process.execPath, ['--import', 'tsx', thisFile, '--server', String(delayMs)], {
    cwd: packageRoot, stdio: ['ignore', 'pipe', 'pipe'],
  });
  const closed = new Promise<void>((done) => child.once('close', () => done()));
  let stderr = '';
  child.stderr!.on('data', (data: Buffer) => { stderr += data.toString(); });
  const endpoint = await new Promise<string>((done, reject) => {
    const timer = setTimeout(() => { child.kill('SIGTERM'); reject(Error(`Synthetic service startup timed out: ${stderr}`)); }, 10000);
    let stdout = '';
    child.once('error', (error) => { clearTimeout(timer); reject(error); });
    child.once('exit', () => { clearTimeout(timer); reject(Error(`Synthetic service exited: ${stderr}`)); });
    child.stdout!.on('data', (data: Buffer) => {
      stdout += data.toString();
      if (stdout.includes('\n')) {
        try { const ready = JSON.parse(stdout.split('\n')[0]!); clearTimeout(timer); done(ready.endpoint); }
        catch (error) { clearTimeout(timer); reject(error); }
      }
    });
  });
  return { child, endpoint, closed };
}

function schedule(rates: number[], seed: number): Condition[] {
  let state = seed >>> 0;
  const next = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; };
  const conditions = rates.flatMap((targetRps) => (['open-loop', 'closed-loop'] as Mode[]).map((mode) => ({ id: `${mode}-r${targetRps}`, mode, targetRps })));
  for (let i = conditions.length - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    [conditions[i], conditions[j]] = [conditions[j]!, conditions[i]!];
  }
  return conditions;
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  if (args[0] === '--server') { await serve(Number(args[1])); return; }
  const allowed = new Set(['--output', '--requests', '--rates', '--service-ms', '--seed']);
  const flags = new Map<string, string>();
  for (let i = 0; i < args.length; i += 2) {
    if (!allowed.has(args[i]!) || !args[i + 1] || flags.has(args[i]!)) throw Error('Usage: coordinated_omission_demo.ts --output NEW_DIRECTORY [--requests 10000] [--rates 250,750] [--service-ms 2] [--seed 465]');
    flags.set(args[i]!, args[i + 1]!);
  }
  const outputValue = flags.get('--output');
  if (!outputValue) throw Error('--output is required and must not already exist');
  const output = resolve(outputValue), count = Number(flags.get('--requests') ?? 10000);
  const rates = (flags.get('--rates') ?? '250,750').split(',').map(Number);
  const delayMs = Number(flags.get('--service-ms') ?? 2), seed = Number(flags.get('--seed') ?? 465);
  if (!Number.isSafeInteger(count) || count < 1 || count > 20000 || !rates.length || rates.some((rate) => !Number.isFinite(rate) || rate <= 0 || rate > 2000) || new Set(rates).size !== rates.length || !Number.isFinite(delayMs) || delayMs < 1 || delayMs > 100 || !Number.isSafeInteger(seed)) throw Error('Invalid bounded diagnostic configuration');
  await mkdir(output, { recursive: false });
  const source = await readFile(thisFile), canonical = await readFile(join(packageRoot, 'src/loadgen.ts'));
  let gitCommit: string | null = null, gitDirty: boolean | null = null;
  try { gitCommit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repoRoot, encoding: 'utf8' }).trim();
    gitDirty = execFileSync('git', ['status', '--porcelain=v1', '--untracked-files=normal'], { cwd: repoRoot, encoding: 'utf8' }).trim() !== '';
  } catch { /* Hashes still identify the exact executed sources. */ }
  await mkdir(join(output, 'source'));
  await writeFile(join(output, 'source/coordinated_omission_demo.ts'), source, { flag: 'wx' });
  await writeFile(join(output, 'source/loadgen.ts'), canonical, { flag: 'wx' });
  const conditions = schedule(rates, seed);
  const protocol = {
    schema: 'coordinated-omission-protocol', schemaVersion: 1, recordedBeforeCollectionUtc: new Date().toISOString(),
    synthetic: true, reportableGpuResult: false, requestedServiceDelayMs: delayMs,
    measuredRequestsPerCondition: count, warmupRequestsPerCondition: 10, seed, conditions,
    primaryComparison: 'Sent-to-complete p95 alongside delivered arrival rate and reference-horizon delivery counts',
    fixedBudget: true, independentReplicatesPerCondition: 1, requestTimeoutMs: 60000,
    source: { gitCommit, gitDirty, demoSha256: hash(source), canonicalLoadgenSha256: hash(canonical), node: process.version, platform: process.platform },
    limitations: [
      'Synthetic CPU serial service; no model, GPU speedup, production capacity, or accuracy conclusion.',
      'Requested timer delay is not exact service time. Actual queue and service durations are recorded separately.',
      'One trial per condition; correlated requests are descriptive samples, not independent experimental replications.',
      'Closed-loop sends every predeclared request eventually but suppresses arrivals during the reference N/RPS horizon.',
      '10,000 observations allow descriptive tail calculation; they do not guarantee precise or stationary tails.',
      'Client and server use separate processes on the same host; host scheduling and transport remain part of results.',
    ],
  };
  await writeFile(join(output, 'protocol.json'), JSON.stringify(protocol, null, 2) + '\n', { flag: 'wx' });
  const trials = [];
  for (const condition of conditions) {
    console.log(JSON.stringify({ event: 'trial-start', condition: condition.id, measuredRequests: count, referenceHorizonSeconds: count / condition.targetRps }));
    const directory = join(output, condition.id); await mkdir(directory);
    const service = await startService(delayMs);
    const config: LoadConfig = {
      endpoint: service.endpoint, requestsPerSecond: condition.targetRps, durationSeconds: count / condition.targetRps,
      measuredRequests: count, timeoutMs: 60000, datasetPath: 'synthetic-in-memory', warmupRequests: 10,
      seed, settleMs: 0, runKind: 'open-loop', outputDir: directory,
      runDeadlineMs: Math.ceil(count / condition.targetRps * 1000 + count * delayMs + 60000),
    };
    const journal = await createJournal(directory);
    let journalFinished = false;
    try {
      const run = condition.mode === 'open-loop'
        ? await runLoad(config, [payload], { runId: condition.id, onResult: journal.append })
        : await runClosedLoop(config, condition.id, journal.append);
      const raw = await journal.finish(); journalFinished = true;
      const summary = summarizeRun(run);
      const health = await fetch(new URL('/health', service.endpoint), { signal: AbortSignal.timeout(5000) }).then((response) => response.json());
      const referenceEnd = run.plannedStartMs + config.durationSeconds * 1000;
      const trial = {
        ...condition, schema: 'coordinated-omission-trial', schemaVersion: 1, status: run.status, stopReason: run.stopReason,
        summary, service: health, clock: 'performance.now',
        timing: { measuredStartMs: run.plannedStartMs, observationEndMs: run.plannedEndMs, finishedMs: run.finishedMs,
          observationWindowSeconds: (run.plannedEndMs - run.plannedStartMs) / 1000,
          actualElapsedIncludingDrainSeconds: (run.finishedMs - run.plannedStartMs) / 1000,
          referencePlannedEndMs: referenceEnd, referenceHorizonSeconds: config.durationSeconds },
        referenceHorizon: {
          prescribedRequests: count,
          deliveredRequests: run.results.filter((r) => r.sentAtMs !== null && r.sentAtMs <= referenceEnd).length,
          successfulCompletions: run.results.filter((r) => r.outcome === 'success' && r.completedAtMs <= referenceEnd).length,
        },
        requests: { file: 'requests.jsonl', ...raw },
        note: condition.mode === 'closed-loop'
          ? 'Actual observation window extends until all responses; it is not the nominal N/RPS horizon. Missed reference arrivals appear as dispatch lateness. Low HTTP latency coexists with reduced delivered load.'
          : 'Prescribed open-loop window is fixed at N/RPS; final drain is reported separately.',
      };
      await writeFile(join(directory, 'trial.json'), JSON.stringify(trial, null, 2) + '\n', { flag: 'wx' });
      trials.push({ ...trial, directory: condition.id });
      await writeFile(join(output, 'demo.partial.json'), JSON.stringify({ protocol, trials }, null, 2) + '\n');
      console.log(JSON.stringify({ event: 'trial-complete', condition: condition.id,
        actualDispatchRps: summary.achievedArrivalRateRps, successful: summary.successfulRequests, failed: summary.failedRequests,
        p95Ms: summary.successfulRequestLatencyMs.p95, referenceDelivered: trial.referenceHorizon.deliveredRequests }));
      if (run.status !== 'complete') throw Error('Diagnostic aborted; partial evidence retained');
    } finally {
      if (!journalFinished) await journal.finish();
      service.child.kill('SIGTERM'); await service.closed;
    }
  }
  await writeFile(join(output, 'demo.json'), JSON.stringify({ schema: 'coordinated-omission-demo', schemaVersion: 1,
    protocolFile: 'protocol.json', protocolSha256: hash(await readFile(join(output, 'protocol.json'))), protocol, trials }, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ event: 'demo-complete', outputDirectory: output, trials: trials.length }));
}

if (process.argv[1] && resolve(process.argv[1]) === thisFile) main().catch((error: unknown) => { console.error(error); process.exitCode = 1; });

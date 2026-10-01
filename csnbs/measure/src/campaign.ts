import { spawn, type ChildProcess } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, writeFile, rename, open, copyFile, unlink } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { hostname } from 'node:os';
import { collectServerMetadata } from './server_metadata.js';
import { requireIdleGpu, monitorResources } from './resources.js';
import { readVerifiedRun, type VerifiedRun, type ComparisonKind } from './analyze.js';

export type Variant = { id: string; command: string[]; env?: Record<string, string>; expected: Record<string, unknown> };
export type CampaignSpec = {
  campaignId: string; purpose: 'integration' | 'baseline' | 'aa' | 'ab';
  endpoint: string; datasetPath: string; rates: number[]; repetitions: number;
  measuredRequests: number; warmupRequests: number; seed: number; timeoutMs: number;
  startupTimeoutMs: number; variants: [Variant, Variant];
  runKind?: 'open-loop' | 'isolated'; settleMs?: number; gpuExclusive?: boolean;
  comparisonKind?: ComparisonKind;
};
export type TrialStatus = 'pending' | 'running' | 'complete' | 'failed' | 'interrupted';
export type ScheduledTrial = { trialId: string; blockId: string; repetition: number; rate: number; variantIndex: number };
export type TrialAttempt = {
  attemptId: string; directory: string; startedAtUtc: string; finishedAtUtc: string | null;
  status: TrialStatus; runDirectory: string | null; error: string | null;
};
export type Trial = ScheduledTrial & { status: TrialStatus; runDirectory: string | null; error: string | null; attempts?: TrialAttempt[] };
export type Campaign = {
  schema: 'loadgen-campaign'; schemaVersion: 2; campaignId: string; spec: CampaignSpec;
  specHash: string; createdAtUtc: string; schedule: ScheduledTrial[]; trials: Trial[];
};
const packageRoot = fileURLToPath(new URL('../', import.meta.url));
const repoRoot = resolve(packageRoot, '../..');
export const campaignSpecHash = (value: CampaignSpec): string => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const delay = (ms: number) => new Promise<void>((done) => setTimeout(done, ms));
export function random(seed: number): () => number {
  let n = seed >>> 0;
  return () => { n += 0x6D2B79F5; let t = n; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
function shuffle<T>(items: T[], rng: () => number): T[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [out[i], out[j]] = [out[j]!, out[i]!]; }
  return out;
}
export function validateSpec(s: CampaignSpec): void {
  if (!/^[a-zA-Z0-9_-]+$/.test(s.campaignId)) throw Error('campaignId must use letters, numbers, underscore or hyphen');
  if (!['integration', 'baseline', 'aa', 'ab'].includes(s.purpose)) throw Error('Unknown campaign purpose');
  if (!Array.isArray(s.rates) || !s.rates.length || s.rates.some((r) => !Number.isFinite(r) || r <= 0) || new Set(s.rates).size !== s.rates.length) throw Error('Rates must be unique positive finite values');
  for (const key of ['repetitions', 'measuredRequests', 'timeoutMs', 'startupTimeoutMs'] as const) if (!Number.isSafeInteger(s[key]) || s[key] <= 0) throw Error(`${key} must be a positive integer`);
  if (!Number.isSafeInteger(s.warmupRequests) || s.warmupRequests < 0 || !Number.isSafeInteger(s.seed) || s.seed < 0 || s.seed + s.repetitions - 1 > 0xffffffff) throw Error('Invalid warmup or seed');
  if (s.settleMs !== undefined && (!Number.isFinite(s.settleMs) || s.settleMs < 0)) throw Error('Invalid settling period');
  if (s.runKind !== undefined && !['open-loop', 'isolated'].includes(s.runKind)) throw Error('Invalid campaign run kind');
  if (s.comparisonKind !== undefined && !['token-count', 'implementation', 'instrumentation'].includes(s.comparisonKind)) throw Error('Invalid comparison kind');
  if (!s.datasetPath || typeof s.datasetPath !== 'string') throw Error('A workload dataset path is required');
  if (!Array.isArray(s.variants) || s.variants.length !== 2 || s.variants.some((v) => !v.id || !v.command?.length || !v.expected || typeof v.expected !== 'object' || Array.isArray(v.expected) || !Object.keys(v.expected).length || v.command.some((a) => typeof a !== 'string' || !a))) throw Error('Two owned server variants with expected configuration are required');
  if (s.variants[0].id === s.variants[1].id) throw Error('Variant IDs must be unique');
  if (s.purpose !== 'integration' && s.gpuExclusive !== true) throw Error('Model campaign requires a reserved exclusive GPU (gpuExclusive:true)');
  const url = new URL(s.endpoint);
  if (!['http:', 'https:'].includes(url.protocol) || !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname)) throw Error('Campaign must run on the server host against loopback HTTP');
  if (s.purpose !== 'integration' && s.warmupRequests < 10) throw Error('Model campaigns require at least ten warmups');
}
export function planCampaign(input: CampaignSpec): Campaign {
  validateSpec(input);
  const s = structuredClone(input), rng = random(s.seed), trials: Trial[] = [];
  for (const rate of shuffle(s.rates, rng)) {
    const first = Math.floor(rng() * 2);
    for (let repetition = 0; repetition < s.repetitions; repetition++) {
      const blockId = `r${rate}-p${repetition + 1}`;
      const order = (first + repetition) % 2 === 0 ? [0, 1] : [1, 0];
      for (const variantIndex of order) trials.push({ trialId: `${blockId}-v${variantIndex}`, blockId, repetition, rate, variantIndex, status: 'pending', runDirectory: null, error: null });
    }
  }
  return { schema: 'loadgen-campaign', schemaVersion: 2, campaignId: s.campaignId, spec: s,
    specHash: campaignSpecHash(s), createdAtUtc: new Date().toISOString(),
    schedule: trials.map(({ trialId, blockId, repetition, rate, variantIndex }) => ({ trialId, blockId, repetition, rate, variantIndex })), trials };
}
export function verifyCampaign(campaign: Campaign): void {
  if (campaign.schema !== 'loadgen-campaign' || campaign.schemaVersion !== 2) throw Error('Report requires a V2 campaign; use the historical plot reader for V1');
  validateSpec(campaign.spec);
  if (campaign.campaignId !== campaign.spec.campaignId || campaign.specHash !== campaignSpecHash(campaign.spec)) throw Error('Campaign specification hash/identity mismatch');
  const expected = planCampaign(campaign.spec).schedule;
  if (JSON.stringify(campaign.schedule) !== JSON.stringify(expected)) throw Error('Campaign immutable schedule differs from its specification');
  if (!Array.isArray(campaign.trials) || campaign.trials.length !== expected.length) throw Error('Campaign has missing or extra trial records');
  const directories = new Set<string>();
  campaign.trials.forEach((trial, index) => {
    const scheduled = expected[index]!;
    for (const key of ['trialId', 'blockId', 'repetition', 'rate', 'variantIndex'] as const) if (trial[key] !== scheduled[key]) throw Error('Trial identity/order differs from immutable schedule');
    if (!['pending', 'running', 'complete', 'failed', 'interrupted'].includes(trial.status)) throw Error('Invalid trial state');
    if (trial.status === 'complete' && (!trial.runDirectory || trial.error !== null)) throw Error('Completed trial lacks exact run identity or contains an error');
    if (trial.runDirectory) {
      if (directories.has(trial.runDirectory)) throw Error('One run directory is reused across trials');
      directories.add(trial.runDirectory);
    }
    const attempts = new Set<string>();
    for (const attempt of trial.attempts ?? []) {
      if (!attempt.attemptId || attempts.has(attempt.attemptId) || !attempt.directory || !['running', 'complete', 'failed', 'interrupted'].includes(attempt.status)) throw Error('Invalid attempt history');
      attempts.add(attempt.attemptId);
    }
    const latest = trial.attempts?.at(-1);
    if (latest && (latest.status !== trial.status || latest.runDirectory !== trial.runDirectory || latest.error !== trial.error)) throw Error('Trial state differs from its latest recorded attempt');
  });
}
export function configurationMatches(actual: unknown, expected: Record<string, unknown>): boolean {
  if (!actual || typeof actual !== 'object') return false;
  return Object.entries(expected).every(([key, value]) => JSON.stringify((actual as Record<string, unknown>)[key]) === JSON.stringify(value));
}
export function verifyTrialRun(campaign: Campaign, trial: Trial, run: VerifiedRun): void {
  const { manifest } = run, s = campaign.spec;
  if (manifest.status !== 'complete' || manifest.config.measuredRequests !== s.measuredRequests || manifest.config.requestsPerSecond !== trial.rate || manifest.config.warmupRequests !== s.warmupRequests || manifest.config.seed !== s.seed + trial.repetition || manifest.config.timeoutMs !== s.timeoutMs || manifest.config.settleMs !== (s.settleMs ?? 0) || manifest.runKind !== (s.runKind ?? 'open-loop')) throw Error(`Run does not satisfy the prescribed trial ${trial.trialId}`);
  if (manifest.summary.totalRequests !== s.measuredRequests || manifest.summary.dispatchedRequests !== s.measuredRequests) throw Error(`Incomplete delivered budget for trial ${trial.trialId}`);
  const expectedDuration = s.measuredRequests / trial.rate;
  if (!Number.isFinite(manifest.config.durationSeconds) || Math.abs(manifest.config.durationSeconds - expectedDuration) > Math.max(1e-9, expectedDuration * Number.EPSILON * 8)) throw Error(`Run duration differs from prescribed trial ${trial.trialId}`);
  if (!configurationMatches(manifest.server.configuration, s.variants[trial.variantIndex]!.expected)) throw Error(`Saved run configuration differs from trial ${trial.trialId}`);
}
async function atomicWrite(path: string, value: unknown): Promise<void> { await writeFile(`${path}.tmp`, JSON.stringify(value, null, 2) + '\n'); await rename(`${path}.tmp`, path); }
async function occupied(endpoint: string): Promise<boolean> {
  try { await fetch(new URL('/health', endpoint), { signal: AbortSignal.timeout(500) }); return true; }
  catch (error) { if ((error as Error).name === 'TimeoutError') return true; return false; }
}
type ManagedChild = { child: ChildProcess; closed: Promise<void>; error: () => Error | null };
function manage(child: ChildProcess): ManagedChild {
  let failure: Error | null = null;
  const closed = new Promise<void>((done) => { child.once('error', (error) => { failure = error; }); child.once('close', () => done()); });
  return { child, closed, error: () => failure };
}
async function stopOwned(managed: ManagedChild): Promise<void> {
  const { child, closed } = managed;
  if (child.pid === undefined || child.exitCode !== null || child.signalCode !== null) { await closed; return; }
  child.kill('SIGTERM');
  let timer: ReturnType<typeof setTimeout> | undefined;
  const stopped = await Promise.race([closed.then(() => true), new Promise<false>((done) => { timer = setTimeout(() => done(false), 5000); })]);
  if (timer) clearTimeout(timer);
  if (!stopped) {
    child.kill('SIGKILL');
    await Promise.race([closed, new Promise<never>((_, reject) => { timer = setTimeout(() => reject(Error('Owned process did not exit after SIGKILL')), 5000); })]).finally(() => { if (timer) clearTimeout(timer); });
  }
}
async function waitReady(managed: ManagedChild, s: CampaignSpec, variant: Variant, signal: AbortSignal) {
  const deadline = performance.now() + s.startupTimeoutMs;
  while (performance.now() < deadline) {
    if (signal.aborted) throw Error('Campaign interrupted during startup');
    if (managed.error()) throw managed.error();
    if (managed.child.exitCode !== null || managed.child.signalCode !== null) throw Error('Owned server exited before readiness');
    const metadata = await collectServerMetadata(s.endpoint);
    if (metadata.error === null) {
      if (metadata.pid !== managed.child.pid) throw Error('Health PID does not match the owned server process; launch commands must exec the server');
      if (!configurationMatches(metadata.configuration, variant.expected)) throw Error(`Effective server configuration mismatch for ${variant.id}`);
      if (s.purpose !== 'integration' && (metadata.mode !== 'model' || metadata.modelLoaded !== true)) throw Error('Expected a loaded real model');
      if (s.purpose !== 'integration' && (!metadata.source?.gitCommit || metadata.source.gitDirty !== false || !Array.isArray(metadata.hardware?.devices) || !metadata.hardware.devices.length)) throw Error('Model campaign requires clean server provenance and GPU metadata');
      return metadata;
    }
    await Promise.race([delay(200), managed.closed]);
  }
  throw Error('Server startup deadline expired');
}
class LoadExecutionError extends Error {
  constructor(message: string, readonly runDirectory: string | null) { super(message); }
}
async function executeLoad(args: string[], logPath: string, signal: AbortSignal): Promise<{ runId: string; outputDirectory: string }> {
  const log = await open(logPath, 'wx');
  const managed = manage(spawn(process.execPath, ['--import', 'tsx', join(packageRoot, 'src/loadgen.ts'), ...args], { cwd: packageRoot, stdio: ['ignore', 'pipe', 'pipe'] }));
  let stdout = '', writes = Promise.resolve(), writeError: unknown = null;
  const append = (bytes: Buffer) => { writes = writes.then(() => log.writeFile(bytes)).catch((error: unknown) => { writeError = error; }); };
  managed.child.stdout!.on('data', (bytes: Buffer) => { stdout += bytes.toString(); append(bytes); });
  managed.child.stderr!.on('data', append);
  let stopping: Promise<void> | null = null;
  const cancel = () => { stopping ??= stopOwned(managed); void stopping.catch(() => {}); };
  signal.addEventListener('abort', cancel, { once: true }); if (signal.aborted) cancel();
  try {
    await managed.closed; await writes;
    if (stopping) await stopping;
    if (managed.error()) throw managed.error();
    if (writeError !== null) throw writeError;
    const jsonLine = stdout.trim().split('\n').reverse().find((line) => line.startsWith('{'));
    let result: { runId?: unknown; outputDirectory?: unknown } | null = null;
    try { result = jsonLine ? JSON.parse(jsonLine) : null; } catch { /* Preserve the exact log for malformed child output. */ }
    const runDirectory = typeof result?.outputDirectory === 'string' ? result.outputDirectory : null;
    if (managed.child.exitCode !== 0) throw new LoadExecutionError(`Load generator exited ${managed.child.exitCode}; see ${logPath}`, runDirectory);
    if (typeof result?.runId !== 'string' || runDirectory === null) throw new LoadExecutionError('Load generator did not return exact run identity', runDirectory);
    return { runId: result.runId, outputDirectory: runDirectory };
  } finally { signal.removeEventListener('abort', cancel); await stopOwned(managed); await writes; await log.close(); }
}

async function acquireLock(path: string) {
  try { return await open(path, 'wx'); }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
    const existing = JSON.parse(await readFile(path, 'utf8')) as { pid?: number; host?: string };
    if (existing.host !== hostname() || !Number.isSafeInteger(existing.pid) || existing.pid! <= 0) throw Error('Campaign lock owner is unknown; inspect it before removing the lock');
    try { process.kill(existing.pid!, 0); }
    catch (probe) {
      if ((probe as NodeJS.ErrnoException).code !== 'ESRCH') throw probe;
      await unlink(path);
      return open(path, 'wx');
    }
    throw Error(`Campaign is already active in process ${existing.pid}`);
  }
}

export async function executeCampaign(s: CampaignSpec, directory: string): Promise<Campaign> {
  validateSpec(s); directory = resolve(directory);
  await mkdir(directory, { recursive: true }); const path = join(directory, 'campaign.json');
  const lockPath = join(directory, '.campaign.lock');
  const lock = await acquireLock(lockPath);
  await lock.writeFile(JSON.stringify({ pid: process.pid, host: hostname(), createdAtUtc: new Date().toISOString() }) + '\n');
  const controller = new AbortController();
  const onSignal = () => controller.abort('Campaign interrupted by signal');
  process.on('SIGINT', onSignal); process.on('SIGTERM', onSignal);
  try {
    let campaign: Campaign;
    let saved: string | null = null;
    try { saved = await readFile(path, 'utf8'); }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
    if (saved === null) campaign = planCampaign(s);
    else {
      campaign = JSON.parse(saved) as Campaign;
      verifyCampaign(campaign);
      if (campaign.specHash !== campaignSpecHash(s)) throw Error('Resume specification differs from saved campaign');
      for (const trial of campaign.trials) {
        if (trial.status === 'running') {
          trial.status = 'interrupted'; trial.error = 'Previous campaign process ended before recording completion';
          const attempt = trial.attempts?.at(-1);
          if (attempt) { attempt.status = trial.status; attempt.error = trial.error; attempt.finishedAtUtc = new Date().toISOString(); }
        }
        if (trial.status === 'complete') verifyTrialRun(campaign, trial, await readVerifiedRun(resolve(directory, trial.runDirectory!)));
      }
    }
    await atomicWrite(path, campaign!);
    if (await occupied(s.endpoint)) throw Error('Endpoint is occupied; refusing to stop or benchmark an unowned server');
    for (const trial of campaign!.trials) {
      if (trial.status === 'complete') continue;
      if (controller.signal.aborted) break;
      if (s.purpose !== 'integration') await requireIdleGpu();
      const variant = s.variants[trial.variantIndex]!;
      const attemptId = `${trial.trialId}-${randomUUID().slice(0, 8)}`, trialDir = join(directory, attemptId);
      await mkdir(trialDir);
      trial.attempts ??= [];
      // Preserve a legacy failed state even if its original writer had no attempt history.
      if (!trial.attempts.length && (trial.status === 'failed' || trial.status === 'interrupted')) trial.attempts.push({ attemptId: `${trial.trialId}-legacy`, directory, startedAtUtc: campaign!.createdAtUtc, finishedAtUtc: new Date().toISOString(), status: trial.status, runDirectory: trial.runDirectory, error: trial.error });
      const attempt: TrialAttempt = { attemptId, directory: trialDir, startedAtUtc: new Date().toISOString(), finishedAtUtc: null, status: 'running', runDirectory: null, error: null };
      trial.attempts.push(attempt); trial.status = 'running'; trial.error = null; trial.runDirectory = null;
      await atomicWrite(path, campaign!);
      const serverLog = await open(join(trialDir, 'server.log'), 'wx');
      const managed = manage(spawn(variant.command[0]!, variant.command.slice(1), { cwd: repoRoot, env: { ...process.env, ...variant.env }, stdio: ['ignore', serverLog.fd, serverLog.fd] }));
      let stopMonitor: () => Promise<void> = async () => {};
      const interruptServer = () => { managed.child.kill('SIGTERM'); };
      controller.signal.addEventListener('abort', interruptServer, { once: true });
      try {
        if (s.purpose !== 'integration') stopMonitor = await monitorResources(join(trialDir, 'resources.jsonl'));
        const started = performance.now(), metadata = await waitReady(managed, s, variant, controller.signal);
        await atomicWrite(join(trialDir, 'server.json'), { ...metadata, startupMs: performance.now() - started });
        const result = await executeLoad([
          '--endpoint', s.endpoint, '--rps', String(trial.rate), '--requests', String(s.measuredRequests),
          '--warmup', String(s.warmupRequests), '--seed', String(s.seed + trial.repetition), '--timeout', String(s.timeoutMs),
          '--dataset', resolve(s.datasetPath), '--run-kind', s.runKind ?? 'open-loop', '--settle-ms', String(s.settleMs ?? 0),
          '--output-dir', join(trialDir, 'runs'),
        ], join(trialDir, 'loadgen.log'), controller.signal);
        trial.runDirectory = result.outputDirectory;
        const run = await readVerifiedRun(result.outputDirectory);
        if (run.manifest.runId !== result.runId) throw Error('Child run identity differs from saved manifest');
        verifyTrialRun(campaign!, trial, run);
        if (controller.signal.aborted) throw Error('Campaign interrupted');
        trial.status = 'complete';
      } catch (error) {
        if (error instanceof LoadExecutionError) trial.runDirectory = error.runDirectory;
        trial.status = controller.signal.aborted ? 'interrupted' : 'failed'; trial.error = String(error);
      } finally {
        controller.signal.removeEventListener('abort', interruptServer);
        try {
          const cleanup = await Promise.allSettled([stopOwned(managed), stopMonitor(), serverLog.close()]);
          const failedCleanup = cleanup.find((result) => result.status === 'rejected');
          if (failedCleanup?.status === 'rejected') throw failedCleanup.reason;
          if (s.purpose !== 'integration' && trial.runDirectory) {
            const resourcePath = join(trial.runDirectory, 'resources.jsonl');
            await copyFile(join(trialDir, 'resources.jsonl'), resourcePath);
            const bytes = await readFile(resourcePath), manifestPath = join(trial.runDirectory, 'run.json');
            const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
            manifest.resourceSamples = { file: 'resources.jsonl', count: bytes.toString().trim().split('\n').filter(Boolean).length,
              sha256: `sha256:${createHash('sha256').update(bytes).digest('hex')}`, scope: 'sampled whole-device memory, not exact KV cache' };
            await atomicWrite(manifestPath, manifest);
          }
        } catch (error) { trial.status = 'failed'; trial.error = `Cleanup/resource persistence failed: ${String(error)}`; }
        Object.assign(attempt, { status: trial.status, runDirectory: trial.runDirectory, error: trial.error, finishedAtUtc: new Date().toISOString() });
        await atomicWrite(path, campaign!);
      }
      if (trial.status !== 'complete') break;
    }
    return campaign!;
  } finally {
    process.off('SIGINT', onSignal); process.off('SIGTERM', onSignal);
    await lock.close(); await unlink(lockPath);
  }
}
async function main(): Promise<void> {
  const args = process.argv.slice(2), specPath = args[0];
  if (!specPath) throw Error('Usage: campaign <spec.json> [--plan] [--output <directory>]');
  const spec: CampaignSpec = JSON.parse(await readFile(specPath, 'utf8'));
  spec.datasetPath = resolve(dirname(resolve(specPath)), spec.datasetPath);
  const index = args.indexOf('--output');
  if (index >= 0 && !args[index + 1]) throw Error('--output requires a directory');
  const directory = resolve(index >= 0 ? args[index + 1]! : join(repoRoot, 'results', 'campaigns', spec.campaignId));
  if (args.includes('--plan')) { console.log(JSON.stringify(planCampaign(spec), null, 2)); return; }
  const result = await executeCampaign(spec, directory);
  console.log(JSON.stringify({ campaignFile: join(directory, 'campaign.json'), completed: result.trials.filter((trial) => trial.status === 'complete').length, total: result.trials.length }));
  if (result.trials.some((trial) => trial.status !== 'complete')) process.exitCode = 1;
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch((error) => { console.error(error); process.exitCode = 1; });

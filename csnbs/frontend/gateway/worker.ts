import type { LiveAnswer, LiveGatewayConfig, LiveGenerateInput, LiveModelConfig } from '../src/live-contract';

/** Kept structural so the same gateway can run on loopback without Cloudflare tooling. */
export interface DurableStorage {
  get<T>(key: string): Promise<T | undefined>;
  put<T>(key: string, value: T): Promise<void>;
  delete(key: string): Promise<boolean | void>;
  list<T>(options?: { prefix?: string; limit?: number }): Promise<Map<string, T>>;
}
export interface DurableState { storage: DurableStorage }
export interface DurableNamespace {
  idFromName(name: string): unknown;
  get(id: unknown): { fetch(request: Request): Promise<Response> };
}
export interface GatewayEnv {
  LIVE_LIMITER: DurableNamespace;
  ASSETS?: { fetch(request: Request): Promise<Response> };
  LIVE_ENABLED?: string;
  LOCAL_DEVELOPMENT?: string;
  ALLOWED_ORIGINS?: string;
  MODEL_UPSTREAM_URL?: string;
  COOKIE_SECRET?: string;
  IP_HASH_SECRET?: string;
  TURNSTILE_SITE_KEY?: string;
  TURNSTILE_SECRET?: string;
  MODAL_KEY?: string;
  MODAL_SECRET?: string;
}
export interface GatewayDependencies { fetch?: typeof fetch }

export const LIVE_LIMITS = {
  imageBytes: 5 * 1024 * 1024,
  questionChars: 2_000,
  outputTokens: 128,
  upstreamTimeoutMs: 120_000,
  admissionLeaseMs: 240_000,
  modelConfigLifetimeMs: 10 * 60_000,
  idempotencyLifetimeMs: 48 * 60 * 60_000,
  generate: { browser: 5, ip: 10, global: 100 },
  warmup: { browser: 3, ip: 6, global: 50 },
  browserBurstMs: 3_000,
  ipBurstMs: 1_000,
} as const;

const COOKIE_NAME = 'live_visitor';
const COOKIE_SECONDS = 30 * 24 * 60 * 60;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const HASH = /^[a-f0-9]{64}$/;
const JSON_HEADERS = { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' };
const TURNSTILE_VERIFY_URL = 'https://challenges.cloudflare.com/turnstile/v0/siteverify';
const BODY_BYTES = Math.ceil(LIVE_LIMITS.imageBytes / 3) * 4 + 32_768;
const encoder = new TextEncoder();

class GatewayError extends Error {
  constructor(readonly status: number, readonly code: string, message: string, readonly retry?: number) { super(message); }
}
function json(value: unknown, status = 200, extra: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(value), { status, headers: { ...JSON_HEADERS, ...extra } });
}
function errorResponse(error: unknown): Response {
  const known = error instanceof GatewayError ? error : new GatewayError(503, 'gateway_unavailable', 'The live demo is temporarily unavailable. Please try again later.');
  return json({ error: { code: known.code, message: known.message, ...(known.retry === undefined ? {} : { retry_after_seconds: known.retry }) } }, known.status,
    known.retry === undefined ? {} : { 'retry-after': String(known.retry) });
}
function isRecord(value: unknown): value is Record<string, unknown> { return !!value && typeof value === 'object' && !Array.isArray(value); }
function loopback(host: string): boolean { return ['localhost', '127.0.0.1', '[::1]'].includes(host); }
function parseURL(value: string | undefined): URL | null { try { return value ? new URL(value) : null; } catch { return null; } }
function configured(request: Request, env: GatewayEnv): { local: boolean; upstream: URL; origins: Set<string> } {
  const upstream = parseURL(env.MODEL_UPSTREAM_URL);
  const requestURL = new URL(request.url);
  if (!upstream || upstream.username || upstream.password || upstream.search || upstream.hash || !['http:', 'https:'].includes(upstream.protocol)) {
    throw new GatewayError(503, 'gateway_unavailable', 'The live demo has not been configured.');
  }
  const requestedLocal = env.LOCAL_DEVELOPMENT === 'true';
  const local = requestedLocal && loopback(requestURL.hostname) && loopback(upstream.hostname);
  const origins = new Set((env.ALLOWED_ORIGINS || '').split(',').map(value => value.trim()).filter(Boolean));
  const validOrigins = [...origins].every(origin => {
    const url = parseURL(origin);
    return !!url && url.origin === origin && (local ? loopback(url.hostname) && ['http:', 'https:'].includes(url.protocol) : url.protocol === 'https:');
  });
  if ((requestedLocal && !local) || (!local && (requestURL.protocol !== 'https:' || upstream.protocol !== 'https:' || loopback(upstream.hostname))) ||
      !origins.size || !validOrigins || !env.LIVE_LIMITER || (env.COOKIE_SECRET?.length || 0) < 32 || (env.IP_HASH_SECRET?.length || 0) < 32 ||
      (!local && (!env.TURNSTILE_SITE_KEY || !env.TURNSTILE_SECRET || !env.MODAL_KEY || !env.MODAL_SECRET))) {
    throw new GatewayError(503, 'gateway_unavailable', 'The live demo has not been configured.');
  }
  return { local, upstream, origins };
}
function toBase64URL(bytes: Uint8Array): string { return btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, ''); }
function fromBase64URL(value: string): Uint8Array<ArrayBuffer> {
  return Uint8Array.from(atob(value.replace(/-/g, '+').replace(/_/g, '/')), char => char.charCodeAt(0));
}
async function hmacKey(secret: string): Promise<CryptoKey> {
  return crypto.subtle.importKey('raw', encoder.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign', 'verify']);
}
async function sign(secret: string, value: string): Promise<Uint8Array> {
  return new Uint8Array(await crypto.subtle.sign('HMAC', await hmacKey(secret), encoder.encode(value)));
}
async function browserID(request: Request, secret: string): Promise<string | null> {
  const value = request.headers.get('cookie')?.split(';').map(part => part.trim()).find(part => part.startsWith(`${COOKIE_NAME}=`))?.slice(COOKIE_NAME.length + 1);
  if (!value || value.length > 256) return null;
  const [id, expiration, signature, extra] = value.split('.');
  if (extra || !UUID.test(id || '') || !/^\d{13}$/.test(expiration || '') || Number(expiration) <= Date.now() || Number(expiration) > Date.now() + COOKIE_SECONDS * 1000 + 60_000 || !signature) return null;
  try {
    const verified = await crypto.subtle.verify('HMAC', await hmacKey(secret), fromBase64URL(signature), encoder.encode(`${id}.${expiration}`));
    return verified ? id : null;
  } catch { return null; }
}
async function newCookie(secret: string, local: boolean): Promise<string> {
  const payload = `${crypto.randomUUID()}.${Date.now() + COOKIE_SECONDS * 1000}`;
  return `${COOKIE_NAME}=${payload}.${toBase64URL(await sign(secret, payload))}; Path=/api/live; Max-Age=${COOKIE_SECONDS}; HttpOnly; SameSite=Strict${local ? '' : '; Secure'}`;
}
async function readJSON(request: Request | Response, maxBytes: number): Promise<unknown> {
  const declared = request.headers.get('content-length');
  if (declared && (!/^\d+$/.test(declared) || Number(declared) > maxBytes)) throw new GatewayError(413, 'payload_too_large', 'This request is too large.');
  if (!request.body) throw new GatewayError(400, 'invalid_request', 'A JSON request body is required.');
  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > maxBytes) { await reader.cancel(); throw new GatewayError(413, 'payload_too_large', 'This request is too large.'); }
      chunks.push(value);
    }
    const bytes = new Uint8Array(size);
    let cursor = 0;
    for (const chunk of chunks) { bytes.set(chunk, cursor); cursor += chunk.length; }
    return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
  } catch (error) {
    if (error instanceof GatewayError) throw error;
    throw new GatewayError(400, 'invalid_request', 'The request must contain valid JSON.');
  } finally { reader.releaseLock(); }
}
function stringValue(value: unknown, max: number): value is string { return typeof value === 'string' && value.length > 0 && value.length <= max; }
function requestID(value: unknown): asserts value is string {
  if (typeof value !== 'string' || !UUID.test(value)) throw new GatewayError(400, 'invalid_request', 'A valid request ID is required.');
}
function validateImage(base64: unknown, mime: unknown): asserts base64 is string {
  if (typeof base64 !== 'string' || !base64.length || base64.length % 4 !== 0 || !/^[A-Za-z0-9+/]+={0,2}$/.test(base64) ||
      !['image/jpeg', 'image/png', 'image/webp'].includes(String(mime))) {
    throw new GatewayError(400, 'invalid_image', 'Choose a JPEG, PNG, or WebP image.');
  }
  const byteLength = (base64.length / 4) * 3 - (base64.endsWith('==') ? 2 : base64.endsWith('=') ? 1 : 0);
  if (byteLength > LIVE_LIMITS.imageBytes) throw new GatewayError(413, 'image_too_large', 'Choose an image smaller than 5 MiB.');
  const signature = atob(base64.slice(0, 32));
  const valid = mime === 'image/jpeg' ? signature.startsWith('\xff\xd8\xff') : mime === 'image/png' ? signature.startsWith('\x89PNG\r\n\x1a\n') : signature.startsWith('RIFF') && signature.slice(8, 12) === 'WEBP';
  if (!valid) throw new GatewayError(400, 'invalid_image', 'The image contents do not match the selected image format.');
}
function generateInput(value: Record<string, unknown>): LiveGenerateInput {
  requestID(value.request_id);
  validateImage(value.image_b64, value.image_mime);
  if (!stringValue(value.question, LIVE_LIMITS.questionChars) || !value.question.trim() ||
      !Number.isInteger(value.visual_token_num) || Number(value.visual_token_num) < 1 || Number(value.visual_token_num) > 576 ||
      typeof value.important_ratio !== 'number' || !Number.isFinite(value.important_ratio) || value.important_ratio < 0 || value.important_ratio > 1 ||
      !Number.isInteger(value.max_output_tokens) || Number(value.max_output_tokens) < 1 || Number(value.max_output_tokens) > LIVE_LIMITS.outputTokens) {
    throw new GatewayError(400, 'invalid_request', 'Check the question, token setting, and output length.');
  }
  return { request_id: value.request_id, image_b64: value.image_b64, image_mime: value.image_mime as LiveGenerateInput['image_mime'], question: value.question.trim(),
    visual_token_num: Number(value.visual_token_num), important_ratio: value.important_ratio, max_output_tokens: Number(value.max_output_tokens) };
}
function modelConfig(value: unknown): LiveModelConfig {
  if (!isRecord(value) || !['model', 'fake'].includes(String(value.server_mode)) || !['OFF', 'STARTING', 'READY', 'BUSY'].includes(String(value.state)) ||
      !stringValue(value.model, 200) || !stringValue(value.method, 200) || !Array.isArray(value.gpu) || value.gpu.length > 8 || !value.gpu.every(item => stringValue(item, 100)) ||
      !(value.git_commit === null || stringValue(value.git_commit, 64)) || !Array.isArray(value.visual_token_options) || !value.visual_token_options.length || value.visual_token_options.length > 16 ||
      !value.visual_token_options.includes(576) || new Set(value.visual_token_options).size !== value.visual_token_options.length ||
      !value.visual_token_options.every(item => Number.isInteger(item) && item > 0 && item <= 576) || typeof value.important_ratio !== 'number' || !Number.isFinite(value.important_ratio) || value.important_ratio < 0 || value.important_ratio > 1 ||
      !Number.isInteger(value.max_output_tokens) || Number(value.max_output_tokens) < 1 || Number(value.max_output_tokens) > 4096 || typeof value.supports_streaming !== 'boolean') {
    throw new GatewayError(502, 'invalid_upstream', 'The model service returned an unsupported configuration.');
  }
  return { server_mode: value.server_mode as LiveModelConfig['server_mode'], state: value.state as LiveModelConfig['state'], model: value.model, method: value.method, gpu: value.gpu as string[],
    git_commit: value.git_commit as string | null, visual_token_options: value.visual_token_options as number[], important_ratio: value.important_ratio,
    max_output_tokens: Math.min(Number(value.max_output_tokens), LIVE_LIMITS.outputTokens), supports_streaming: false };
}
function modelAnswer(value: unknown, input: LiveGenerateInput, config: LiveModelConfig): LiveAnswer {
  const nullableNumber = (number: unknown) => number === null || (typeof number === 'number' && Number.isFinite(number) && number >= 0);
  const nullableCount = (number: unknown) => number === null || (typeof number === 'number' && Number.isSafeInteger(number) && number >= 0);
  if (!isRecord(value) || value.request_id !== input.request_id || typeof value.answer !== 'string' || value.answer.length > 32_768 || value.server_mode !== 'model' ||
      value.visual_token_num !== input.visual_token_num || value.important_ratio !== input.important_ratio || value.model !== config.model || value.method !== config.method ||
      !Array.isArray(value.gpu) || JSON.stringify([...value.gpu].sort()) !== JSON.stringify([...config.gpu].sort()) || value.git_commit !== config.git_commit ||
      !nullableNumber(value.ttft_ms) || !nullableNumber(value.total_ms) || !nullableCount(value.n_input_tokens) || !nullableCount(value.n_output_tokens) ||
      (value.n_output_tokens !== null && Number(value.n_output_tokens) > input.max_output_tokens) ||
      (typeof value.ttft_ms === 'number' && typeof value.total_ms === 'number' && value.ttft_ms > value.total_ms)) {
    throw new GatewayError(502, 'invalid_upstream', 'The model service returned an unsupported answer.');
  }
  return { request_id: input.request_id, answer: value.answer, server_mode: 'model', visual_token_num: input.visual_token_num, important_ratio: input.important_ratio,
    model: config.model, method: config.method, gpu: value.gpu as string[], git_commit: config.git_commit, ttft_ms: value.ttft_ms as number | null,
    total_ms: value.total_ms as number | null, n_input_tokens: value.n_input_tokens as number | null, n_output_tokens: value.n_output_tokens as number | null };
}
async function verifyTurnstile(token: unknown, origin: string, env: GatewayEnv, fetcher: typeof fetch): Promise<void> {
  if (!stringValue(token, 2048)) throw new GatewayError(403, 'verification_required', 'Complete the human verification before continuing.');
  let result: unknown;
  try {
    const response = await fetcher(TURNSTILE_VERIFY_URL, { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ secret: env.TURNSTILE_SECRET, response: token }), redirect: 'error', signal: AbortSignal.timeout(8_000) });
    if (!response.ok) throw new Error('verification unavailable');
    result = await readJSON(response, 8_192);
  } catch { throw new GatewayError(503, 'verification_unavailable', 'Human verification is temporarily unavailable. Please try again.'); }
  if (!isRecord(result) || result.success !== true || result.hostname !== new URL(origin).hostname || result.action !== 'live-demo') {
    throw new GatewayError(403, 'verification_failed', 'Human verification expired or failed. Please try again.');
  }
}
async function limiter(env: GatewayEnv, operation: string, body: unknown): Promise<Response> {
  return env.LIVE_LIMITER.get(env.LIVE_LIMITER.idFromName('global-live-admission-v1')).fetch(new Request(`https://admission.internal/${operation}`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  }));
}

/** Production Worker and the loopback adapter both call this exact handler. */
export async function handleLiveRequest(request: Request, env: GatewayEnv, dependencies: GatewayDependencies = {}): Promise<Response> {
  try {
    const url = new URL(request.url);
    const path = url.pathname;
    if (!path.startsWith('/api/live/') && env.ASSETS) return env.ASSETS.fetch(request);
    if (!['/api/live/config', '/api/live/warmup', '/api/live/generate'].includes(path)) throw new GatewayError(404, 'not_found', 'This endpoint does not exist.');
    if ((path.endsWith('/config') && request.method !== 'GET') || (!path.endsWith('/config') && request.method !== 'POST')) throw new GatewayError(405, 'method_not_allowed', 'This method is not supported.');
    const enabled = env.LIVE_ENABLED === 'true';
    if (!enabled) {
      if (!path.endsWith('/config')) throw new GatewayError(503, 'live_disabled', 'The live demo is currently off. Recorded examples are still available.');
      const config: LiveGatewayConfig = { enabled: false, turnstile_site_key: null, local_development: false, max_image_bytes: LIVE_LIMITS.imageBytes, max_question_chars: LIVE_LIMITS.questionChars, max_output_tokens: LIVE_LIMITS.outputTokens };
      return json(config);
    }
    const runtime = configured(request, env);
    if (path.endsWith('/config')) {
      const config: LiveGatewayConfig = { enabled: true, turnstile_site_key: runtime.local ? null : env.TURNSTILE_SITE_KEY!, local_development: runtime.local,
        max_image_bytes: LIVE_LIMITS.imageBytes, max_question_chars: LIVE_LIMITS.questionChars, max_output_tokens: LIVE_LIMITS.outputTokens };
      return json(config, 200, await browserID(request, env.COOKIE_SECRET!) ? {} : { 'set-cookie': await newCookie(env.COOKIE_SECRET!, runtime.local) });
    }
    const origin = request.headers.get('origin');
    if (!origin || !runtime.origins.has(origin)) throw new GatewayError(403, 'origin_rejected', 'Open the demo from its own website before continuing.');
    if (request.headers.get('content-type')?.split(';')[0].trim().toLowerCase() !== 'application/json') throw new GatewayError(415, 'unsupported_media_type', 'Send a JSON request.');
    const browser = await browserID(request, env.COOKIE_SECRET!);
    if (!browser) throw new GatewayError(403, 'session_required', 'Reload the demo to start a new anonymous session.');
    // CF sets this header on public Workers. The loopback adapter never trusts forwarded client headers.
    const ip = runtime.local ? 'loopback' : request.headers.get('cf-connecting-ip');
    if (!ip || ip.length > 64 || (!runtime.local && !/^[0-9a-fA-F:.]+$/.test(ip))) throw new GatewayError(503, 'gateway_unavailable', 'The live demo is temporarily unavailable.');
    const ipHash = [...await sign(env.IP_HASH_SECRET!, `live-ip-v1:${ip}`)].map(byte => byte.toString(16).padStart(2, '0')).join('');
    const raw = await readJSON(request, path.endsWith('/warmup') ? 4096 : BODY_BYTES);
    if (!isRecord(raw)) throw new GatewayError(400, 'invalid_request', 'Send a valid demo request.');
    requestID(raw.request_id);
    const kind = path.endsWith('/warmup') ? 'warmup' : 'generate';
    const input = kind === 'generate' ? generateInput(raw) : null;
    const fetcher = dependencies.fetch || fetch;
    if (!runtime.local) await verifyTurnstile(raw.turnstile_token, origin, env, fetcher);
    const admission = await limiter(env, 'admit', { kind, request_id: raw.request_id, browser, ip: ipHash,
      ...(input ? { settings: { visual_token_num: input.visual_token_num, important_ratio: input.important_ratio, max_output_tokens: input.max_output_tokens } } : {}) });
    if (!admission.ok) return admission;
    const admitted = await admission.json() as { config?: LiveModelConfig };
    let upstreamCompleted = false;
    let resultConfig: LiveModelConfig | undefined;
    try {
      const upstreamURL = new URL(`${runtime.upstream.pathname.replace(/\/$/, '')}/${kind}`, runtime.upstream.origin);
      const response = await fetcher(upstreamURL, { method: 'POST', headers: { 'content-type': 'application/json',
        ...(runtime.local ? {} : { 'Modal-Key': env.MODAL_KEY!, 'Modal-Secret': env.MODAL_SECRET! }) },
        body: JSON.stringify(input || { request_id: raw.request_id }), redirect: 'error', signal: AbortSignal.timeout(LIVE_LIMITS.upstreamTimeoutMs) });
      // A completed HTTP response (including an error) is the backend's completion acknowledgement.
      const result = await readJSON(response, kind === 'warmup' ? 16_384 : 65_536);
      upstreamCompleted = true;
      if (!response.ok) throw new GatewayError(response.status === 429 || response.status === 503 ? 503 : 502, 'model_unavailable', 'The model service is busy or unavailable. Please try again later.', 30);
      if (kind === 'warmup') {
        resultConfig = modelConfig(result);
        return json(resultConfig);
      }
      return json(modelAnswer(result, input!, admitted.config!));
    } catch (error) {
      if (error instanceof GatewayError && upstreamCompleted) throw error;
      throw new GatewayError(504, 'model_timeout', 'The model did not finish in time. Please wait before trying again.', Math.ceil(LIVE_LIMITS.admissionLeaseMs / 1000));
    } finally {
      // On disconnect/timeout there may still be a GPU call. Retain the durable lease; never retry it.
      if (upstreamCompleted) await limiter(env, 'finish', { request_id: raw.request_id, ...(resultConfig ? { config: resultConfig } : {}) });
    }
  } catch (error) { return errorResponse(error); }
}

type Kind = 'warmup' | 'generate';
type Counters = { global: number; browsers: Record<string, number>; ips: Record<string, number> };
type Ledger = {
  day: string;
  generate: Counters;
  warmup: Counters;
  lastBrowser: Record<string, number>;
  lastIP: Record<string, number>;
  active?: { request_id: string; until: number };
  config?: { value: LiveModelConfig; until: number };
  lastPrune?: number;
};
const emptyCounters = (): Counters => ({ global: 0, browsers: {}, ips: {} });

/** A single SQLite-backed object owns global admission and all exact anonymous counters. */
export class LiveAdmission {
  private queue: Promise<unknown> = Promise.resolve();
  constructor(private readonly state: DurableState, _env?: GatewayEnv, private readonly now: () => number = Date.now) {}
  async fetch(request: Request): Promise<Response> {
    // Serialize even with the tiny in-memory storage adapter used for local development/tests.
    const result = this.queue.then(() => this.perform(request));
    this.queue = result.catch(() => undefined);
    try { return await result; } catch (error) { return errorResponse(error); }
  }
  private async perform(request: Request): Promise<Response> {
    if (request.method !== 'POST') throw new GatewayError(405, 'method_not_allowed', 'This method is not supported.');
    const input = await readJSON(request, 16_384);
    if (!isRecord(input)) throw new GatewayError(400, 'invalid_request', 'Invalid admission request.');
    requestID(input.request_id);
    const now = this.now();
    const day = new Date(now).toISOString().slice(0, 10);
    let ledger = await this.state.storage.get<Ledger>('ledger');
    if (!ledger || ledger.day !== day) ledger = { day, generate: emptyCounters(), warmup: emptyCounters(), lastBrowser: {}, lastIP: {}, active: ledger?.active, config: ledger?.config, lastPrune: ledger?.lastPrune };
    const operation = new URL(request.url).pathname;
    if (operation === '/finish') {
      if (ledger.active?.request_id === input.request_id) {
        delete ledger.active;
        if (input.config) ledger.config = { value: modelConfig(input.config), until: now + LIVE_LIMITS.modelConfigLifetimeMs };
        await this.state.storage.put('ledger', ledger);
      }
      return json({ ok: true });
    }
    if (operation !== '/admit') throw new GatewayError(404, 'not_found', 'This endpoint does not exist.');
    if (!['warmup', 'generate'].includes(String(input.kind)) || typeof input.browser !== 'string' || !UUID.test(input.browser) || typeof input.ip !== 'string' || !HASH.test(input.ip)) {
      throw new GatewayError(400, 'invalid_request', 'Invalid admission request.');
    }
    // IDs expire after 48 hours; cleanup is activity-driven and daily traffic is hard-capped.
    // No prompt, image, answer, raw IP, or Turnstile token reaches storage.
    if (!ledger.lastPrune || ledger.lastPrune + 86_400_000 <= now) {
      const entries = await this.state.storage.list<number>({ prefix: 'request:', limit: 512 });
      for (const [key, expires] of entries) if (expires <= now) await this.state.storage.delete(key);
      ledger.lastPrune = now;
    }
    const priorExpiration = await this.state.storage.get<number>(`request:${input.request_id}`);
    if (priorExpiration && priorExpiration > now) throw new GatewayError(409, 'duplicate_request', 'This request has already been submitted. Start a new request to continue.');
    if (ledger.active && ledger.active.until > now) throw new GatewayError(503, 'model_busy', 'Another live request is running. Please try again shortly.', Math.min(5, Math.max(1, Math.ceil((ledger.active.until - now) / 1000))));
    const kind = input.kind as Kind;
    const counters = ledger[kind];
    const limits = LIVE_LIMITS[kind];
    if (counters.global >= limits.global || (counters.browsers[input.browser] || 0) >= limits.browser || (counters.ips[input.ip] || 0) >= limits.ip) {
      const midnight = new Date(`${day}T00:00:00.000Z`).getTime() + 86_400_000;
      throw new GatewayError(429, 'daily_limit', 'The daily live-demo allowance has been reached. Please use the recorded examples or return tomorrow.', Math.ceil((midnight - now) / 1000));
    }
    const burstRemaining = Math.max((ledger.lastBrowser[input.browser] || 0) + LIVE_LIMITS.browserBurstMs - now, (ledger.lastIP[input.ip] || 0) + LIVE_LIMITS.ipBurstMs - now);
    if (burstRemaining > 0) throw new GatewayError(429, 'rate_limited', 'Please wait a few seconds between live requests.', Math.ceil(burstRemaining / 1000));
    if (kind === 'generate') {
      const config = ledger.config?.value;
      if (!config || ledger.config!.until <= now || config.state !== 'READY' || config.server_mode !== 'model') throw new GatewayError(409, 'model_not_ready', 'Start the model before submitting a question.');
      const settings = input.settings;
      if (!isRecord(settings) || !config.visual_token_options.includes(Number(settings.visual_token_num)) || settings.important_ratio !== config.important_ratio ||
          !Number.isInteger(settings.max_output_tokens) || Number(settings.max_output_tokens) < 1 || Number(settings.max_output_tokens) > config.max_output_tokens) {
        throw new GatewayError(400, 'unsupported_setting', 'Choose one of the settings supplied by the model service.');
      }
    }
    // Write the tombstone first: an interrupted storage write can refuse a retry, never duplicate GPU work.
    await this.state.storage.put(`request:${input.request_id}`, now + LIVE_LIMITS.idempotencyLifetimeMs);
    counters.global += 1;
    counters.browsers[input.browser] = (counters.browsers[input.browser] || 0) + 1;
    counters.ips[input.ip] = (counters.ips[input.ip] || 0) + 1;
    ledger.lastBrowser[input.browser] = now;
    ledger.lastIP[input.ip] = now;
    ledger.active = { request_id: input.request_id, until: now + LIVE_LIMITS.admissionLeaseMs };
    await this.state.storage.put('ledger', ledger);
    return json({ admitted: true, ...(kind === 'generate' ? { config: ledger.config!.value } : {}) });
  }
}

export default { fetch: handleLiveRequest };

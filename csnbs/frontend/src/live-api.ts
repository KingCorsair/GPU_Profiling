import type { LiveAnswer, LiveGatewayConfig, LiveGenerateInput, LiveModelConfig } from './live-contract';
import { parseGatewayConfig, parseLiveAnswer, parseModelConfig } from './live-validation';

// Never expose backend response bodies, traces, URLs, or credentials in UI errors.
const messages: Record<string, string> = {
  disabled: 'The live demo is currently switched off. The recorded study remains available.',
  quota_exceeded: 'The live demo has reached its usage allowance. Please try again later or explore the recorded study.',
  rate_limited: 'Please wait before sending another request.',
  busy: 'The model is helping another visitor. Please try again shortly.',
  starting: 'The model is still starting. Please try connecting again shortly.',
  verification_failed: 'The verification expired or could not be accepted. Complete the new verification and try again.',
  invalid_request: 'The request could not be accepted. Check your image, question, and selected settings.',
  payload_too_large: 'The image is too large for this demo. Choose an image no larger than 5 MB.',
  upstream_timeout: 'The model took too long to respond. It may still be processing the request.',
  upstream_unavailable: 'The model service is not connected. You can explore the recorded study while it is unavailable.',
  duplicate_request: 'This request was already submitted. Wait for the current request to finish before trying again.',
  session_required: 'Your demo session expired. Reconnect to start a new session.',
  gateway_unavailable: 'The live gateway is not configured or could not be reached. The recorded study remains available.',
};

export class LiveDemoError extends Error {
  constructor(public code: string, public retryAfterSeconds?: number) {
    super(messages[code] ?? 'The live request could not be completed. Please try again later.');
    this.name = 'LiveDemoError';
  }
}

export function parseApiError(status: number, body: unknown): LiveDemoError {
  const nested = typeof body === 'object' && body !== null && 'error' in body ? body.error : null;
  const value = typeof nested === 'object' && nested !== null ? nested as Record<string, unknown> : {};
  const aliases: Record<string, string> = { live_disabled: 'disabled', verification_required: 'verification_failed',
    verification_unavailable: 'verification_failed', origin_rejected: 'verification_failed', invalid_image: 'invalid_request',
    image_too_large: 'payload_too_large', model_unavailable: 'upstream_unavailable', model_timeout: 'upstream_timeout',
    model_busy: 'busy', daily_limit: 'quota_exceeded', model_not_ready: 'starting', unsupported_setting: 'invalid_request',
    invalid_upstream: 'upstream_unavailable', DISABLED: 'disabled', DEMO_DISABLED: 'disabled', QUOTA_EXCEEDED: 'quota_exceeded',
    RATE_LIMITED: 'rate_limited', BUSY: 'busy', STARTING: 'starting', TURNSTILE_FAILED: 'verification_failed',
    VERIFICATION_FAILED: 'verification_failed', INVALID_REQUEST: 'invalid_request', PAYLOAD_TOO_LARGE: 'payload_too_large',
    UPSTREAM_TIMEOUT: 'upstream_timeout', UPSTREAM_UNAVAILABLE: 'upstream_unavailable' };
  const raw = typeof value.code === 'string' ? value.code : '';
  const candidate = aliases[raw] ?? raw;
  const code = Object.hasOwn(messages, candidate) ? candidate
    : status === 429 ? 'rate_limited' : status === 413 ? 'payload_too_large' : status === 403 ? 'verification_failed'
      : status === 409 ? 'busy' : status === 504 ? 'upstream_timeout' : [404, 502, 503].includes(status) ? 'upstream_unavailable' : 'unknown';
  const retry = value.retry_after_seconds;
  return new LiveDemoError(code, typeof retry === 'number' && Number.isFinite(retry) && retry >= 0 && retry <= 86400 ? Math.ceil(retry) : undefined);
}

async function json(path: 'config' | 'warmup' | 'generate', options: RequestInit = {}): Promise<unknown> {
  const timeout = AbortSignal.timeout(path === 'config' ? 15_000 : 240_000);
  const signal = options.signal ? AbortSignal.any([options.signal, timeout]) : timeout;
  try {
    const response = await fetch(`/api/live/${path}`, { ...options, signal, credentials: 'same-origin', redirect: 'error', cache: 'no-store' });
    let body: unknown;
    try { body = await response.json(); }
    catch { if (signal.aborted) throw signal.reason; if (!response.ok) throw parseApiError(response.status, null); throw new LiveDemoError('upstream_unavailable'); }
    if (!response.ok) throw parseApiError(response.status, body);
    return body;
  } catch (failure) {
    if (options.signal?.aborted) throw options.signal.reason;
    if (timeout.aborted) throw new LiveDemoError('upstream_timeout');
    if (failure instanceof LiveDemoError) throw failure;
    throw new LiveDemoError('upstream_unavailable');
  }
}

let pendingConfig: Promise<LiveGatewayConfig> | null = null;
export function readLiveConfig(): Promise<LiveGatewayConfig> {
  if (!pendingConfig) pendingConfig = json('config').then(parseGatewayConfig).finally(() => { pendingConfig = null; });
  return pendingConfig;
}

// One request across React StrictMode's mount cycle and quick route re-entry.
// No timers, polling, or periodic refresh: entering the Playground is the trigger.
let warmup: { promise: Promise<LiveModelConfig>; completedAt: number | null } | null = null;
export function warmLiveModel(turnstileToken?: string): Promise<LiveModelConfig> {
  if (warmup && (warmup.completedAt === null || Date.now() - warmup.completedAt < 30_000)) return warmup.promise;
  const entry: NonNullable<typeof warmup> = { completedAt: null, promise: Promise.resolve(null as unknown as LiveModelConfig) };
  entry.promise = json('warmup', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ request_id: crypto.randomUUID(), ...(turnstileToken ? { turnstile_token: turnstileToken } : {}) }),
  }).then(parseModelConfig).finally(() => {
    // Hold failures briefly as well: automatic remounts must not retry billable calls.
    entry.completedAt = Date.now();
  });
  warmup = entry;
  return entry.promise;
}

export async function generateLiveAnswer(input: LiveGenerateInput, config: LiveModelConfig, signal?: AbortSignal): Promise<LiveAnswer> {
  if (config.server_mode !== 'model' || config.state !== 'READY') throw new LiveDemoError('starting');
  if (!input.image_b64 || !input.question.trim() || !config.visual_token_options.includes(input.visual_token_num)
    || input.important_ratio !== config.important_ratio || !Number.isInteger(input.max_output_tokens)
    || input.max_output_tokens < 1 || input.max_output_tokens > config.max_output_tokens) throw new LiveDemoError('invalid_request');
  const answer = await json('generate', { method: 'POST', signal, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(input) });
  return parseLiveAnswer(answer, input, config);
}

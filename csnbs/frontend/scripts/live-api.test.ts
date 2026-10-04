import assert from 'node:assert/strict';
import test from 'node:test';
import { generateLiveAnswer, LiveDemoError, parseApiError, readLiveConfig, warmLiveModel } from '../src/live-api.ts';
import { imagePayload, matchingExecution, parseGatewayConfig, parseLiveAnswer, parseModelConfig } from '../src/live-validation.ts';
import type { LiveAnswer, LiveGatewayConfig, LiveGenerateInput, LiveModelConfig } from '../src/live-contract.ts';

const config = (): LiveModelConfig => ({ server_mode: 'model', state: 'READY', model: 'llava-v1.5-7b', method: 'vispruner', gpu: ['NVIDIA L4'],
  git_commit: 'abcdef0123456789', visual_token_options: [576, 384, 256, 128], important_ratio: 0.5, max_output_tokens: 128, supports_streaming: false });
const gateway = (): LiveGatewayConfig => ({ enabled: true, local_development: false, turnstile_site_key: 'public-site-key', max_image_bytes: 5 * 1024 * 1024, max_question_chars: 2000, max_output_tokens: 128 });
const input = (): LiveGenerateInput => ({ request_id: 'b22196ad-259c-440e-914b-f92ed5ac265a', image_b64: 'aW1hZ2U=', image_mime: 'image/png',
  question: 'What is in the image?', visual_token_num: 128, important_ratio: 0.5, max_output_tokens: 128, turnstile_token: 'one-use-token' });
const answer = (): LiveAnswer => ({ request_id: input().request_id, answer: 'An orange.', server_mode: 'model', visual_token_num: 128,
  important_ratio: 0.5, model: config().model, method: config().method, gpu: config().gpu, git_commit: config().git_commit,
  total_ms: 1420, ttft_ms: null, n_input_tokens: null, n_output_tokens: 4 });
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

test('gateway config requires verification in public mode and caps accepted image sizes', () => {
  assert.deepEqual(parseGatewayConfig(gateway()), gateway());
  assert.doesNotThrow(() => parseGatewayConfig({ ...gateway(), local_development: true, turnstile_site_key: null }));
  assert.doesNotThrow(() => parseGatewayConfig({ ...gateway(), enabled: false, turnstile_site_key: null }));
  for (const invalid of [null, {}, { ...gateway(), turnstile_site_key: null }, { ...gateway(), max_image_bytes: 6 * 1024 * 1024 },
    { ...gateway(), max_question_chars: -1 }, { ...gateway(), max_output_tokens: 0 }]) assert.throws(() => parseGatewayConfig(invalid));
});

test('model settings exclude fake inference and reject ambiguous or unsupported budgets', () => {
  assert.deepEqual(parseModelConfig(config()), config());
  for (const invalid of [{ ...config(), server_mode: 'fake' }, { ...config(), visual_token_options: [128] },
    { ...config(), visual_token_options: [576, 128, 128] }, { ...config(), visual_token_options: [576, 600] },
    { ...config(), important_ratio: -0.1 }, { ...config(), important_ratio: NaN }, { ...config(), gpu: [12] },
    { ...config(), state: 'UNKNOWN' }, { ...config(), max_output_tokens: 1.5 }]) assert.throws(() => parseModelConfig(invalid));
});

test('answers preserve server measurements and missing metrics remain null', () => {
  assert.deepEqual(parseLiveAnswer(answer(), input(), config()), answer());
  const absent = { ...answer(), total_ms: undefined, ttft_ms: undefined, n_input_tokens: undefined, n_output_tokens: undefined };
  const parsed = parseLiveAnswer(absent, input(), config());
  assert.equal(parsed.total_ms, null);
  assert.equal(parsed.ttft_ms, null);
  assert.equal(parsed.n_input_tokens, null);
  assert.equal(parsed.n_output_tokens, null);
});

test('answers from another request, setting, source, or fake mode cannot enter the comparison', () => {
  for (const change of [{ request_id: 'unrelated' }, { server_mode: 'fake' }, { visual_token_num: 576 }, { important_ratio: 0.8 },
    { model: 'another-model' }, { method: 'different-method' }, { git_commit: 'new-commit' }, { gpu: ['NVIDIA A100'] },
    { total_ms: -1 }, { ttft_ms: Infinity }, { n_input_tokens: 5.5 }, { n_output_tokens: 129 }, { ttft_ms: 2000 }, { answer: null }]) {
    assert.throws(() => parseLiveAnswer({ ...answer(), ...change }, input(), config()));
  }
});

test('comparison requires the same known source and configuration, while allowing another visual budget', () => {
  const base = { ...answer(), visual_token_num: 576 };
  assert.equal(matchingExecution(base, 128, answer(), 128), true);
  assert.equal(matchingExecution(base, 128, answer(), 64), false);
  for (const change of [{ git_commit: null }, { git_commit: 'other' }, { model: 'other' }, { method: 'other' }, { gpu: ['Other GPU'] }, { important_ratio: 0.8 }]) {
    assert.equal(matchingExecution(base, 128, { ...answer(), ...change }, 128), false);
  }
  assert.equal(matchingExecution({ ...base, git_commit: null }, 128, { ...answer(), git_commit: null }, 128), false);
});

test('image transport preserves PNG/JPEG/WebP encoding and limits decoded bytes', () => {
  for (const mime of ['image/png', 'image/jpeg', 'image/webp']) {
    assert.deepEqual(imagePayload(`data:${mime};base64,aW1hZ2U=`, 5), { image_mime: mime, image_b64: 'aW1hZ2U=' });
    assert.throws(() => imagePayload(`data:${mime};base64,aW1hZ2U=`, 4), /no larger/);
  }
  for (const invalid of ['data:image/svg+xml;base64,aW1hZ2U=', 'https://example.com/image.png', 'data:image/png;base64,abc', 'data:image/png;base64,']) {
    assert.throws(() => imagePayload(invalid, 1024), /Upload/);
  }
});

test('error rendering uses only fixed messages and bounded retry hints', () => {
  const failure = parseApiError(429, { error: { code: 'daily_limit', message: 'private token=secret and upstream URL', retry_after_seconds: 12.1 } });
  assert.equal(failure.code, 'quota_exceeded');
  assert.equal(failure.retryAfterSeconds, 13);
  assert.doesNotMatch(failure.message, /private|secret|upstream URL/);
  assert.equal(parseApiError(500, { error: { code: 'unknown-private-server-trace', message: 'secret', retry_after_seconds: Infinity } }).retryAfterSeconds, undefined);
  assert.equal(parseApiError(409, { error: { code: 'model_busy' } }).code, 'busy');
  assert.equal(parseApiError(503, { error: { code: 'live_disabled' } }).code, 'disabled');
  assert.equal(parseApiError(400, { error: { code: 'session_required' } }).code, 'session_required');
  assert.equal(parseApiError(403, { error: { code: 'verification_required' } }).code, 'verification_failed');
});

test('simultaneous configuration reads share one same-origin request and never warm the model', async (t) => {
  let finish!: (response: Response) => void;
  const calls: { url: string; options?: RequestInit }[] = [];
  t.mock.method(globalThis, 'fetch', (url: string, options?: RequestInit) => {
    calls.push({ url, options });
    return new Promise<Response>((resolve) => { finish = resolve; });
  });
  const first = readLiveConfig();
  const second = readLiveConfig();
  assert.equal(first, second);
  finish(json(gateway()));
  await Promise.all([first, second]);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, '/api/live/config');
  assert.equal(calls[0].options?.credentials, 'same-origin');
  assert.equal(calls[0].options?.redirect, 'error');
});

test('warmup is shared throughout slow startup and cached across immediate remounts without background refresh', async (t) => {
  let now = 1000;
  let finish!: (response: Response) => void;
  const bodies: Record<string, unknown>[] = [];
  t.mock.method(Date, 'now', () => now);
  t.mock.method(globalThis, 'fetch', (_url: string, options: RequestInit) => {
    bodies.push(JSON.parse(String(options.body)));
    return new Promise<Response>((resolve) => { finish = resolve; });
  });
  const first = warmLiveModel('first-token');
  now += 90_000;
  assert.equal(warmLiveModel('second-token'), first);
  assert.equal(bodies.length, 1);
  assert.equal(bodies[0].turnstile_token, 'first-token');
  assert.match(String(bodies[0].request_id), /^[0-9a-f-]{36}$/);
  finish(json(config()));
  await first;
  now += 29_999;
  assert.equal(warmLiveModel('unused-token'), first);
  assert.equal(bodies.length, 1);
  now += 2;
  const next = warmLiveModel('new-token');
  assert.notEqual(next, first);
  assert.equal(bodies.length, 2);
  finish(json(config()));
  await next;
});

test('generation sends the selected contract and validates the returned request and model snapshot', async (t) => {
  t.mock.method(globalThis, 'fetch', async (url: string, options: RequestInit) => {
    assert.equal(url, '/api/live/generate');
    assert.equal(options.method, 'POST');
    assert.equal(options.credentials, 'same-origin');
    assert.deepEqual(JSON.parse(String(options.body)), input());
    return json(answer());
  });
  assert.deepEqual(await generateLiveAnswer(input(), config()), answer());
});

test('invalid local settings prevent transmitting an image', async (t) => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => json(answer()));
  await assert.rejects(generateLiveAnswer({ ...input(), visual_token_num: 10 }, config()), /could not be accepted/);
  await assert.rejects(generateLiveAnswer(input(), { ...config(), server_mode: 'fake' }), /still starting/);
  await assert.rejects(generateLiveAnswer(input(), { ...config(), state: 'BUSY' }), /still starting/);
  assert.equal(fetch.mock.callCount(), 0);
});

test('abort stops waiting without replacing cancellation with an upstream error', async (t) => {
  const controller = new AbortController();
  t.mock.method(globalThis, 'fetch', async (_url: string, options: RequestInit) => new Promise<Response>((_resolve, reject) => {
    options.signal?.addEventListener('abort', () => reject(options.signal?.reason), { once: true });
  }));
  const pending = generateLiveAnswer(input(), config(), controller.signal);
  controller.abort();
  await assert.rejects(pending, (failure: Error) => failure.name === 'AbortError');
});

test('gateway failures and non-JSON static-host fallbacks never surface server traces', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response('<html>private server trace</html>', { status: 200 }));
  await assert.rejects(generateLiveAnswer(input(), config()), (failure: Error) => failure instanceof LiveDemoError && !failure.message.includes('private'));
  t.mock.method(globalThis, 'fetch', async () => json({ error: { code: 'model_busy', message: 'secret URL' } }, 409));
  await assert.rejects(generateLiveAnswer(input(), config()), /another visitor/);
});

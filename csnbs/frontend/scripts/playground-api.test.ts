import assert from 'node:assert/strict';
import test from 'node:test';
import { parsePlaygroundHealth, parsePlaygroundResult, readPlaygroundHealth, runPlaygroundInference } from '../src/playground-api.ts';

const healthFixture = () => ({
  service: 'csnbs-llava-server', mode: 'model', model_loaded: true,
  configuration: {
    visual_token_num: 128, important_ratio: 0.5, max_new_tokens: 64, batch_size: 1,
    model_id: 'liuhaotian/llava-v1.5-7b', implementation: 'vispruner-vendored-blocking-v1',
  },
  source: { gitCommit: 'abc123' }, hardware: { gpuModels: ['NVIDIA A40'] },
});
const request = { imageBase64: 'aW1hZ2U=', question: 'What is in the image?', expectedTokens: 128 };
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

test('reads the actual fixed model settings and preserves fake mode', () => {
  const model = parsePlaygroundHealth(healthFixture());
  assert.equal(model.visualTokens, 128);
  assert.equal(model.modelId, 'liuhaotian/llava-v1.5-7b');
  assert.equal(model.sourceCommit, 'abc123');
  assert.deepEqual(model.gpuModels, ['NVIDIA A40']);
  assert.equal(parsePlaygroundHealth({ ...healthFixture(), mode: 'fake', model_loaded: false }).mode, 'fake');
});

test('rejects unrelated services and invalid health contracts', () => {
  assert.throws(() => parsePlaygroundHealth({ ...healthFixture(), service: 'csnbs-gemma-server' }), /not the project/);
  for (const value of [null, [], {}, { ...healthFixture(), configuration: null }, { ...healthFixture(), mode: 'mystery' }]) {
    assert.throws(() => parsePlaygroundHealth(value));
  }
  for (const tokens of [0, 577, 1.2, '128', Infinity]) {
    const fixture = healthFixture();
    assert.throws(() => parsePlaygroundHealth({ ...fixture, configuration: { ...fixture.configuration, visual_token_num: tokens } }));
  }
});

test('retains server measurements without substituting the configured token budget for observations', () => {
  const health = parsePlaygroundHealth(healthFixture());
  const result = parsePlaygroundResult({ answer: 'A dog.', metrics: { schema_version: 2, service_ms: 345.2, generation_wall_ms: 301.5, visual_tokens: null, generated_text_tokens: 4, queue_ms: null, token_unavailable_reason: 'Unverified wrapper', queue_unavailable_reason: 'Not observed' } }, health);
  assert.equal(result.metrics.serviceMs, 345.2);
  assert.equal(result.metrics.generationMs, 301.5);
  assert.equal(result.metrics.visualTokens, null);
  assert.equal(result.metrics.queueMs, null);
  assert.equal(result.metrics.tokenUnavailableReason, 'Unverified wrapper');
  assert.equal(result.metrics.queueUnavailableReason, 'Not observed');
  assert.equal(parsePlaygroundResult({ answer: 'A dog.' }, health).metrics.serviceMs, null);
});

test('rejects invalid or contradictory server measurements', () => {
  const health = parsePlaygroundHealth(healthFixture());
  for (const metrics of [
    { schema_version: 3 }, { schema_version: 1, service_ms: -1 },
    { schema_version: 2, generation_wall_ms: Infinity }, { schema_version: 2, visual_tokens: 576 },
    { schema_version: 2, generated_text_tokens: 0.2 }, { schema_version: 2, queue_ms: 'unknown' },
  ]) assert.throws(() => parsePlaygroundResult({ answer: 'A dog.', metrics }, health));
});

test('rechecks readiness and the exact budget before inference, sends the real contract, and returns only saved server timing', async (t) => {
  const calls: { url: string; options?: RequestInit }[] = [];
  t.mock.method(globalThis, 'fetch', async (url: string, options?: RequestInit) => {
    calls.push({ url, options });
    if (url.endsWith('/health')) return json(healthFixture());
    const body = JSON.parse(String(options?.body));
    assert.deepEqual(Object.keys(body).sort(), ['image_b64', 'question', 'request_id']);
    assert.equal(body.image_b64, request.imageBase64);
    assert.equal(body.question, request.question);
    return json({ answer: 'A dog.', request_id: body.request_id, metrics: { schema_version: 1, service_ms: 12.25 } });
  });
  const result = await runPlaygroundInference('pruned', request);
  assert.deepEqual(calls.map((call) => call.url), ['/api/playground/pruned/health', '/api/playground/pruned/infer']);
  assert.equal(calls[1].options?.method, 'POST');
  assert.equal(result.answer, 'A dog.');
  assert.equal(result.metrics.serviceMs, 12.25);
  assert.equal(result.metrics.visualTokens, null);
});

for (const [name, override, pattern] of [
  ['fake service', { mode: 'fake', model_loaded: false }, /fake service/],
  ['loading model', { model_loaded: false }, /still loading/],
  ['changed token budget', { configuration: { ...healthFixture().configuration, visual_token_num: 64 } }, /uses 64 tokens/],
] as const) {
  test(`blocks inference on ${name} before the image is sent`, async (t) => {
    let requests = 0;
    t.mock.method(globalThis, 'fetch', async () => { requests += 1; return json({ ...healthFixture(), ...override }); });
    await assert.rejects(runPlaygroundInference('pruned', request), pattern);
    assert.equal(requests, 1);
  });
}

test('does not accept a pruned setting for the baseline', async (t) => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => json(healthFixture()));
  await assert.rejects(runPlaygroundInference('baseline', request), /baseline must use all 576/);
  assert.equal(fetch.mock.callCount(), 0);
});

test('does not display an answer from a different request', async (t) => {
  t.mock.method(globalThis, 'fetch', async (url: string) => url.endsWith('/health') ? json(healthFixture()) : json({ answer: 'Unrelated answer', request_id: 'other-request' }));
  await assert.rejects(runPlaygroundInference('pruned', request), /could not be matched/);
});

test('handles disconnected and static-host endpoints clearly', async (t) => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => json({ error: 'No connection' }, 503));
  await assert.rejects(readPlaygroundHealth('pruned'), /not connected/);
  fetch.mock.mockImplementation(async () => new Response('<html>Static app</html>'));
  await assert.rejects(readPlaygroundHealth('pruned'), /did not return model data/);
  fetch.mock.mockImplementation(async () => { throw new TypeError('Failed to fetch'); });
  await assert.rejects(readPlaygroundHealth('pruned'), /could not be reached/);
});

test('preserves cancellation without reporting a service failure', async (t) => {
  const controller = new AbortController();
  controller.abort();
  t.mock.method(globalThis, 'fetch', async (_url: string, options?: RequestInit) => { throw options?.signal?.reason; });
  await assert.rejects(readPlaygroundHealth('pruned', controller.signal), { name: 'AbortError' });
});

test('bounds connection checks and reports a timeout separately from cancellation', async (t) => {
  const controller = new AbortController();
  controller.abort(new DOMException('Connection timed out', 'TimeoutError'));
  t.mock.method(AbortSignal, 'timeout', (milliseconds: number) => {
    assert.equal(milliseconds, 8_000);
    return controller.signal;
  });
  t.mock.method(globalThis, 'fetch', async (_url: string, options?: RequestInit) => { throw options?.signal?.reason; });
  await assert.rejects(readPlaygroundHealth('pruned'), /connection check timed out/);
});

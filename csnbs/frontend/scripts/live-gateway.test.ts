import assert from 'node:assert/strict';
import test from 'node:test';
import { handleLiveRequest, LiveAdmission, LIVE_LIMITS, type DurableStorage, type GatewayEnv } from '../gateway/worker';
import type { LiveAnswer, LiveGenerateInput, LiveModelConfig } from '../src/live-contract';

class MemoryStorage implements DurableStorage {
  entries = new Map<string, unknown>();
  async get<T>(key: string): Promise<T | undefined> { return structuredClone(this.entries.get(key)) as T | undefined; }
  async put<T>(key: string, value: T): Promise<void> { this.entries.set(key, structuredClone(value)); }
  async delete(key: string): Promise<boolean> { return this.entries.delete(key); }
  async list<T>(options: { prefix?: string; limit?: number } = {}): Promise<Map<string, T>> {
    return new Map([...this.entries].filter(([key]) => !options.prefix || key.startsWith(options.prefix)).sort(([a], [b]) => a.localeCompare(b)).slice(0, options.limit || Infinity).map(([key, value]) => [key, structuredClone(value) as T]));
  }
}
const model: LiveModelConfig = { server_mode: 'model', state: 'READY', model: 'LLaVA-1.5-7B', method: 'VisPruner', gpu: ['A40'], git_commit: 'abc1234',
  visual_token_options: [576, 128], important_ratio: 0.5, max_output_tokens: 128, supports_streaming: false };
const png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a31IAAAAASUVORK5CYII=';
const newInput = (): LiveGenerateInput => ({ request_id: crypto.randomUUID(), image_b64: png, image_mime: 'image/png', question: 'What is in this image?', visual_token_num: 128, important_ratio: 0.5, max_output_tokens: 32 });
const answer = (input: LiveGenerateInput): LiveAnswer => ({ request_id: input.request_id, answer: 'A test answer from the backend.', server_mode: 'model', visual_token_num: input.visual_token_num, important_ratio: input.important_ratio,
  model: model.model, method: model.method, gpu: model.gpu, git_commit: model.git_commit, ttft_ms: 20, total_ms: 100, n_input_tokens: 148, n_output_tokens: 12 });
const hash = (number: number) => number.toString(16).padStart(64, '0');

function fixture(local = false) {
  let clock = Date.now();
  const storage = new MemoryStorage();
  const admission = new LiveAdmission({ storage }, undefined, () => clock);
  const site = local ? 'http://127.0.0.1:5173' : 'https://demo.example.com';
  const worker = local ? 'http://127.0.0.1:8787' : site;
  const env: GatewayEnv = { LIVE_ENABLED: 'true', LOCAL_DEVELOPMENT: String(local), ALLOWED_ORIGINS: site,
    MODEL_UPSTREAM_URL: local ? 'http://127.0.0.1:8000' : 'https://private.example.modal.run', COOKIE_SECRET: 'cookie-secret-at-least-thirty-two-characters', IP_HASH_SECRET: 'ip-secret-at-least-thirty-two-characters',
    TURNSTILE_SITE_KEY: 'public-key', TURNSTILE_SECRET: 'secret-turnstile', MODAL_KEY: 'secret-modal-key', MODAL_SECRET: 'secret-modal-token',
    LIVE_LIMITER: { idFromName: name => name, get: () => admission } };
  const calls: { url: string; init?: RequestInit }[] = [];
  let implementation: (url: string, init?: RequestInit) => Promise<Response> = async (url, init) => {
    if (url.includes('siteverify')) return Response.json({ success: true, hostname: 'demo.example.com', action: 'live-demo' });
    if (url.endsWith('/warmup')) return Response.json(model);
    if (url.endsWith('/generate')) return Response.json(answer(JSON.parse(String(init?.body))));
    throw new Error('unexpected external request');
  };
  const fetcher: typeof fetch = async (url, init) => {
    calls.push({ url: String(url), init });
    return implementation(String(url), init);
  };
  const send = (path: string, body?: unknown, cookie?: string, headers: Record<string, string> = {}) => handleLiveRequest(new Request(`${worker}/api/live/${path}`, {
    method: body === undefined ? 'GET' : 'POST', headers: { origin: site, 'cf-connecting-ip': '203.0.113.10', ...(body === undefined ? {} : { 'content-type': 'application/json' }), ...(cookie ? { cookie } : {}), ...headers },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  }), env, { fetch: fetcher });
  const cookie = async () => (await send('config')).headers.get('set-cookie')!.split(';')[0];
  const post = (path: string, body: unknown, value: string, headers?: Record<string, string>) => send(path, { ...(body as object), turnstile_token: 'single-use-token' }, value, headers);
  const advance = (ms = 4000) => { clock += ms; };
  return { env, storage, admission, calls, send, post, cookie, advance, replaceFetch: (next: typeof implementation) => { implementation = next; } };
}
const upstreamCalls = (f: ReturnType<typeof fixture>) => f.calls.filter(call => !call.url.includes('siteverify'));
const code = async (response: Response) => (await response.json() as { error: { code: string } }).error.code;

test('kill switch refuses warmup and generation without network or session writes', async () => {
  const f = fixture(); f.env.LIVE_ENABLED = 'false';
  assert.equal((await (await f.send('config')).json()).enabled, false);
  assert.equal(await code(await f.send('warmup', { request_id: crypto.randomUUID() })), 'live_disabled');
  assert.equal(await code(await f.send('generate', newInput())), 'live_disabled');
  assert.equal(f.calls.length, 0);
  assert.equal(f.storage.entries.size, 0);
});

test('config is GPU-free, exposes only public settings, and issues a signed secure HttpOnly cookie', async () => {
  const f = fixture(); const response = await f.send('config');
  assert.equal(response.status, 200);
  assert.match(response.headers.get('set-cookie')!, /HttpOnly; SameSite=Strict; Secure$/);
  const text = await response.text();
  assert.doesNotMatch(text, /secret|private\.example|modal/i);
  assert.equal(JSON.parse(text).max_image_bytes, 5 * 1024 * 1024);
  assert.equal(f.calls.length, 0);
});

test('missing production secrets and publicly enabled local bypass fail closed', async () => {
  const f = fixture(); delete f.env.MODAL_SECRET;
  assert.equal(await code(await f.send('config')), 'gateway_unavailable');
  const g = fixture(); g.env.LOCAL_DEVELOPMENT = 'true';
  assert.equal(await code(await g.send('config')), 'gateway_unavailable');
  const h = fixture(true); h.env.MODEL_UPSTREAM_URL = 'https://private.example.modal.run';
  assert.equal(await code(await h.send('config')), 'gateway_unavailable');
  assert.equal(f.calls.length + g.calls.length + h.calls.length, 0);
});

test('origin, cookie and human verification are mandatory for every GPU route', async () => {
  for (const path of ['warmup', 'generate']) {
    const f = fixture(); const cookie = await f.cookie(); const body = path === 'warmup' ? { request_id: crypto.randomUUID() } : newInput();
    assert.equal(await code(await f.post(path, body, cookie, { origin: 'https://attacker.example' })), 'origin_rejected');
    assert.equal(await code(await f.post(path, body, cookie.replace('live_visitor=', 'live_visitor=forged'))), 'session_required');
    assert.equal(await code(await f.send(path, body, cookie)), 'verification_required');
    f.replaceFetch(async () => Response.json({ success: true, hostname: 'attacker.example', action: 'live-demo' }));
    assert.equal(await code(await f.post(path, body, cookie)), 'verification_failed');
    f.replaceFetch(async () => Response.json({ success: true, hostname: 'demo.example.com', action: 'wrong-action' }));
    assert.equal(await code(await f.post(path, body, cookie)), 'verification_failed');
    assert.equal(upstreamCalls(f).length, 0);
  }
});

test('loopback-only mode runs the same gateway without Turnstile or proxy credentials', async () => {
  const f = fixture(true); delete f.env.TURNSTILE_SECRET; delete f.env.MODAL_KEY; delete f.env.MODAL_SECRET;
  const cookie = await f.cookie();
  assert.equal((await f.send('warmup', { request_id: crypto.randomUUID() }, cookie)).status, 200);
  assert.equal(f.calls.length, 1);
  assert.equal(new Headers(f.calls[0].init?.headers).has('Modal-Key'), false);
});

test('warmup and generation forward only the contract with server-only auth and return real metadata', async () => {
  const f = fixture(); const cookie = await f.cookie();
  assert.equal((await f.post('warmup', { request_id: crypto.randomUUID() }, cookie)).status, 200);
  f.advance();
  const input = newInput(); const response = await f.post('generate', { ...input, arbitrary: 'discarded' }, cookie);
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), answer(input));
  for (const call of upstreamCalls(f)) {
    assert.equal(new Headers(call.init?.headers).get('Modal-Key'), 'secret-modal-key');
    assert.equal(new Headers(call.init?.headers).get('Modal-Secret'), 'secret-modal-token');
    assert.equal(call.init?.redirect, 'error');
    assert.doesNotMatch(String(call.init?.body), /turnstile_token|arbitrary/);
  }
  const stored = JSON.stringify([...f.storage.entries]);
  assert.doesNotMatch(stored, /203\.0\.113\.10|What is in|test answer|iVBORw|secret|single-use-token/);
});

test('body and setting validation happen before any GPU call', async () => {
  const f = fixture(); const cookie = await f.cookie();
  const invalidInputs = [
    { ...newInput(), image_b64: btoa('<svg>bad</svg>') },
    { ...newInput(), image_mime: 'image/svg+xml' },
    { ...newInput(), question: 'x'.repeat(2001) },
    { ...newInput(), max_output_tokens: 129 },
    { ...newInput(), image_b64: 'A'.repeat(Math.ceil((LIVE_LIMITS.imageBytes + 3) / 3) * 4) },
  ];
  for (const input of invalidInputs) assert.ok((await f.post('generate', input, cookie)).status >= 400);
  assert.equal(upstreamCalls(f).length, 0);
  await f.post('warmup', { request_id: crypto.randomUUID() }, cookie); f.advance();
  assert.equal(await code(await f.post('generate', { ...newInput(), visual_token_num: 64 }, cookie)), 'unsupported_setting');
  assert.equal(await code(await f.post('generate', { ...newInput(), important_ratio: 0.7 }, cookie)), 'unsupported_setting');
  assert.equal(upstreamCalls(f).length, 1);
});

test('one concurrent GPU admission and request-ID tombstones prevent replay', async () => {
  const f = fixture(); const cookie = await f.cookie();
  let release!: () => void;
  let started!: () => void;
  const startedPromise = new Promise<void>(resolve => { started = resolve; });
  f.replaceFetch(async url => {
    if (url.includes('siteverify')) return Response.json({ success: true, hostname: 'demo.example.com', action: 'live-demo' });
    started(); await new Promise<void>(resolve => { release = resolve; }); return Response.json(model);
  });
  const id = crypto.randomUUID();
  const first = f.post('warmup', { request_id: id }, cookie);
  await startedPromise;
  assert.equal(await code(await f.post('warmup', { request_id: id }, cookie)), 'duplicate_request');
  const busy = await f.post('warmup', { request_id: crypto.randomUUID() }, cookie);
  assert.equal(await code(busy), 'model_busy');
  assert.equal(busy.headers.get('retry-after'), '5');
  assert.equal(upstreamCalls(f).length, 1);
  release(); assert.equal((await first).status, 200);
  f.advance();
  assert.equal(await code(await f.post('warmup', { request_id: id }, cookie)), 'duplicate_request');
  assert.equal(upstreamCalls(f).length, 1);
});

test('backend failures are sanitized and uncertain completion keeps its durable lease', async () => {
  const f = fixture(true); const cookie = await f.cookie();
  f.replaceFetch(async () => { throw new Error('secret-modal-key private.example.modal.run'); });
  const id = crypto.randomUUID();
  const failed = await f.send('warmup', { request_id: id }, cookie);
  assert.equal(failed.status, 504);
  assert.doesNotMatch(await failed.text(), /secret-modal|private\.example/);
  f.advance(120_001);
  assert.equal(await code(await f.send('warmup', { request_id: crypto.randomUUID() }, cookie)), 'model_busy');
  assert.equal(await code(await f.send('warmup', { request_id: id }, cookie)), 'duplicate_request');
  assert.equal(upstreamCalls(f).length, 1);
  const g = fixture(true); const otherCookie = await g.cookie();
  g.replaceFetch(async () => Response.json({ secret: 'do not expose', endpoint: 'private.example.modal.run' }, { status: 500 }));
  const backendError = await g.send('warmup', { request_id: crypto.randomUUID() }, otherCookie);
  assert.equal(backendError.status, 502);
  assert.doesNotMatch(await backendError.text(), /do not expose|private\.example/);
});

test('fake mode is disclosed by warmup and cannot enter generation', async () => {
  const f = fixture(true); const cookie = await f.cookie();
  f.replaceFetch(async () => Response.json({ ...model, server_mode: 'fake' }));
  assert.equal((await (await f.send('warmup', { request_id: crypto.randomUUID() }, cookie)).json()).server_mode, 'fake');
  f.advance();
  assert.equal(await code(await f.send('generate', newInput(), cookie)), 'model_not_ready');
  assert.equal(upstreamCalls(f).length, 1);
});

test('storage failure fails closed before GPU work and consumed IDs stay non-replayable', async () => {
  const f = fixture(true); const cookie = await f.cookie();
  const originalPut = f.storage.put.bind(f.storage);
  f.storage.put = async (key, value) => { if (key === 'ledger') throw new Error('storage unavailable'); await originalPut(key, value); };
  const id = crypto.randomUUID();
  assert.equal(await code(await f.send('warmup', { request_id: id }, cookie)), 'gateway_unavailable');
  assert.equal(await code(await f.send('warmup', { request_id: id }, cookie)), 'duplicate_request');
  assert.equal(upstreamCalls(f).length, 0);
});

test('server config requires a unique baseline and narrows its output cap', async () => {
  for (const patch of [{ visual_token_options: [128] }, { visual_token_options: [576, 128, 128] }, { max_output_tokens: 0 }]) {
    const f = fixture(true); const cookie = await f.cookie();
    f.replaceFetch(async () => Response.json({ ...model, ...patch }));
    assert.equal(await code(await f.send('warmup', { request_id: crypto.randomUUID() }, cookie)), 'invalid_upstream');
  }
  const f = fixture(true); const cookie = await f.cookie();
  f.replaceFetch(async () => Response.json({ ...model, max_output_tokens: 16 }));
  assert.equal((await (await f.send('warmup', { request_id: crypto.randomUUID() }, cookie)).json()).max_output_tokens, 16);
  f.advance();
  assert.equal(await code(await f.send('generate', newInput(), cookie)), 'unsupported_setting');
  assert.equal(upstreamCalls(f).length, 1);
});

test('output provenance and actual token bounds are validated; an empty EOS answer is allowed', async () => {
  for (const patch of [{ gpu: ['wrong GPU'] }, { n_output_tokens: 1000 }, { n_input_tokens: 1.5 }, { ttft_ms: 200 }, { request_id: crypto.randomUUID() }, { server_mode: 'fake' }]) {
    const f = fixture(true); const cookie = await f.cookie();
    await f.send('warmup', { request_id: crypto.randomUUID() }, cookie); f.advance();
    f.replaceFetch(async (_url, init) => Response.json({ ...answer(JSON.parse(String(init?.body))), ...patch }));
    assert.equal(await code(await f.send('generate', newInput(), cookie)), 'invalid_upstream');
  }
  const f = fixture(true); const cookie = await f.cookie();
  await f.send('warmup', { request_id: crypto.randomUUID() }, cookie); f.advance();
  f.replaceFetch(async (_url, init) => Response.json({ ...answer(JSON.parse(String(init?.body))), answer: '', n_output_tokens: 0 }));
  assert.equal((await f.send('generate', newInput(), cookie)).status, 200);
});

const direct = (admission: LiveAdmission, path: string, body: unknown) => admission.fetch(new Request(`https://internal/${path}`, { method: 'POST', body: JSON.stringify(body) }));
async function warmDirect(f: ReturnType<typeof fixture>) {
  const id = crypto.randomUUID();
  assert.equal((await direct(f.admission, 'admit', { kind: 'warmup', request_id: id, browser: crypto.randomUUID(), ip: hash(1) })).status, 200);
  await direct(f.admission, 'finish', { request_id: id, config: model }); f.advance();
}
const settings = { visual_token_num: 128, important_ratio: 0.5, max_output_tokens: 32 };

test('exact browser, IP and global daily generation limits cannot race', async () => {
  for (const scope of ['browser', 'ip', 'global'] as const) {
    const f = fixture(); await warmDirect(f);
    const browser = crypto.randomUUID();
    const max = LIVE_LIMITS.generate[scope];
    for (let index = 0; index <= max; index++) {
      const request_id = crypto.randomUUID();
      const response = await direct(f.admission, 'admit', { kind: 'generate', request_id, browser: scope === 'browser' ? browser : crypto.randomUUID(), ip: scope === 'ip' ? hash(2) : hash(index + 10), settings });
      if (index === max) assert.equal(await code(response), 'daily_limit', scope);
      else { assert.equal(response.status, 200, `${scope} ${index}`); await direct(f.admission, 'finish', { request_id }); }
      f.advance();
    }
  }
  const f = fixture(); await warmDirect(f);
  const results = await Promise.all(Array.from({ length: 20 }, () => direct(f.admission, 'admit', { kind: 'generate', request_id: crypto.randomUUID(), browser: crypto.randomUUID(), ip: hash(20), settings })));
  assert.equal(results.filter(response => response.ok).length, 1);
  assert.equal(results.filter(response => response.status === 503).length, 19);
});

test('warmups have independent limits and both paths share the burst guard', async () => {
  const f = fixture(); const browser = crypto.randomUUID();
  for (let index = 0; index < 4; index++) {
    const request_id = crypto.randomUUID();
    const response = await direct(f.admission, 'admit', { kind: 'warmup', request_id, browser, ip: hash(40) });
    if (index === 3) assert.equal(await code(response), 'daily_limit');
    else {
      assert.equal(response.status, 200); await direct(f.admission, 'finish', { request_id, config: model });
      assert.equal(await code(await direct(f.admission, 'admit', { kind: 'generate', request_id: crypto.randomUUID(), browser, ip: hash(40), settings })), 'rate_limited');
    }
    f.advance();
  }
});

test('model config expires, stale finish cannot unlock a new request, and old IDs are cleaned', async () => {
  const f = fixture(); await warmDirect(f);
  f.advance(LIVE_LIMITS.modelConfigLifetimeMs);
  assert.equal(await code(await direct(f.admission, 'admit', { kind: 'generate', request_id: crypto.randomUUID(), browser: crypto.randomUUID(), ip: hash(10), settings })), 'model_not_ready');
  const oldID = crypto.randomUUID();
  await direct(f.admission, 'admit', { kind: 'warmup', request_id: oldID, browser: crypto.randomUUID(), ip: hash(12) });
  f.advance(LIVE_LIMITS.admissionLeaseMs + 1);
  const newID = crypto.randomUUID();
  assert.equal((await direct(f.admission, 'admit', { kind: 'warmup', request_id: newID, browser: crypto.randomUUID(), ip: hash(13) })).status, 200);
  await direct(f.admission, 'finish', { request_id: oldID, config: model });
  assert.equal(await code(await direct(f.admission, 'admit', { kind: 'warmup', request_id: crypto.randomUUID(), browser: crypto.randomUUID(), ip: hash(14) })), 'model_busy');
  f.advance(LIVE_LIMITS.idempotencyLifetimeMs + 1);
  await direct(f.admission, 'admit', { kind: 'warmup', request_id: crypto.randomUUID(), browser: crypto.randomUUID(), ip: hash(15) });
  assert.equal(f.storage.entries.has(`request:${oldID}`), false);
  assert.equal(f.storage.entries.has(`request:${newID}`), false);
});

test('non-API paths use optional static assets while unknown live API paths stay JSON 404', async () => {
  const f = fixture(); let assets = 0;
  f.env.ASSETS = { fetch: async () => { assets++; return new Response('site'); } };
  assert.equal(await (await handleLiveRequest(new Request('https://demo.example.com/'), f.env)).text(), 'site');
  assert.equal((await handleLiveRequest(new Request('https://demo.example.com/api/live/unknown'), f.env)).status, 404);
  assert.equal(assets, 1);
});

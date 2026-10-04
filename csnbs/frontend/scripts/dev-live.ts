/** Local development only. Reuses the production handler; never starts a GPU. */
import { createServer as createHTTPServer } from 'node:http';
import { randomBytes } from 'node:crypto';
import { Readable } from 'node:stream';
import { fileURLToPath } from 'node:url';
import { createServer as createViteServer, loadEnv } from 'vite';
import { handleLiveRequest, LiveAdmission } from '../gateway/worker.ts';
import type { DurableStorage, GatewayEnv } from '../gateway/worker.ts';

class MemoryStorage implements DurableStorage {
  private values = new Map<string, unknown>();
  async get<T>(key: string): Promise<T | undefined> { return structuredClone(this.values.get(key)) as T | undefined; }
  async put<T>(key: string, value: T): Promise<void> { this.values.set(key, structuredClone(value)); }
  async delete(key: string): Promise<boolean> { return this.values.delete(key); }
  async list<T>(options: { prefix?: string; limit?: number } = {}): Promise<Map<string, T>> {
    return new Map([...this.values.entries()].filter(([key]) => key.startsWith(options.prefix ?? '')).sort(([a], [b]) => a.localeCompare(b))
      .slice(0, options.limit ?? Infinity).map(([key, value]) => [key, structuredClone(value) as T]));
  }
}

const directory = fileURLToPath(new URL('../', import.meta.url));
const settings = { ...loadEnv('development', directory, ''), ...process.env };
const port = Number(settings.LIVE_DEV_PORT || '5180');
if (!Number.isInteger(port) || port < 1024 || port > 65535 || port === 8787) throw new Error('LIVE_DEV_PORT must be an available port from 1024 to 65535, other than 8787.');
const upstream = settings.LIVE_LOCAL_UPSTREAM_URL?.trim();
if (upstream) {
  const url = new URL(upstream);
  if (!['localhost', '127.0.0.1', '[::1]'].includes(url.hostname) || !['http:', 'https:'].includes(url.protocol)
      || url.username || url.password || url.search || url.hash) throw new Error('LIVE_LOCAL_UPSTREAM_URL must be a loopback HTTP(S) URL without credentials, query, or fragment.');
}

// Production environment variables cannot turn this loopback helper into a public gateway.
const admission = new LiveAdmission({ storage: new MemoryStorage() });
const env: GatewayEnv = {
  LIVE_ENABLED: upstream ? 'true' : 'false', LOCAL_DEVELOPMENT: 'true',
  ALLOWED_ORIGINS: [`http://127.0.0.1:${port}`, `http://localhost:${port}`].join(','),
  MODEL_UPSTREAM_URL: upstream,
  COOKIE_SECRET: randomBytes(32).toString('hex'), IP_HASH_SECRET: randomBytes(32).toString('hex'),
  LIVE_LIMITER: { idFromName: name => name, get: () => admission },
};

const gateway = createHTTPServer(async (incoming, outgoing) => {
  try {
    // Never honor a caller-supplied Host or forwarded address for the development bypass.
    const path = incoming.url || '/';
    if (!path.startsWith('/') || path.startsWith('//')) { outgoing.writeHead(400); outgoing.end(); return; }
    const headers = new Headers();
    for (const [key, value] of Object.entries(incoming.headers)) {
      if (value && !['host', 'cf-connecting-ip', 'x-forwarded-for', 'x-forwarded-host'].includes(key)) headers.set(key, Array.isArray(value) ? value.join(', ') : value);
    }
    const request = new Request(`http://127.0.0.1:8787${path}`, {
      method: incoming.method, headers,
      ...(!['GET', 'HEAD'].includes(incoming.method || 'GET') ? { body: Readable.toWeb(incoming), duplex: 'half' } : {}),
    } as RequestInit);
    const response = await handleLiveRequest(request, env);
    outgoing.writeHead(response.status, Object.fromEntries(response.headers.entries()));
    outgoing.end(Buffer.from(await response.arrayBuffer()));
  } catch {
    if (!outgoing.headersSent) outgoing.writeHead(503, { 'content-type': 'application/json', 'cache-control': 'no-store' });
    outgoing.end(JSON.stringify({ error: { code: 'gateway_unavailable', message: 'The local demo connection is unavailable.' } }));
  }
});
gateway.requestTimeout = 30_000;
gateway.headersTimeout = 15_000;

await new Promise<void>((resolve, reject) => { gateway.once('error', reject); gateway.listen(8787, '127.0.0.1', resolve); });
try {
  const vite = await createViteServer({ root: directory, mode: 'live', server: { host: '127.0.0.1', port, strictPort: true } });
  await vite.listen();
  console.log(`Live Playground: http://127.0.0.1:${port}/#/playground`);
  console.log(upstream ? 'Local model connection configured; only visiting Playground can trigger model work.' : 'No model connected. The interface will show its unavailable-service state.');
  console.log('Local gateway uses temporary in-memory quotas. No cloud resources have been created.');
  const stop = async () => { await vite.close(); gateway.closeAllConnections(); gateway.close(); };
  process.once('SIGINT', () => void stop());
  process.once('SIGTERM', () => void stop());
} catch (error) {
  gateway.closeAllConnections(); gateway.close();
  throw error;
}

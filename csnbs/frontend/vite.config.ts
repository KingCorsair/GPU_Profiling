import { defineConfig, loadEnv } from 'vite';
import type { ProxyOptions } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', ['PLAYGROUND_', 'VITE_']);
  const demoMode = mode === 'live' ? 'live' : mode === 'showcase' ? 'showcase' : env.VITE_DEMO_MODE || 'showcase';
  if (!['live', 'showcase'].includes(demoMode)) throw new Error('VITE_DEMO_MODE must be live or showcase.');
  const proxy: Record<string, ProxyOptions> = {};
  if (demoMode === 'live') {
    // Loopback only. Production secrets and upstream routing belong to the gateway.
    proxy['/api/live'] = { target: 'http://127.0.0.1:8787', changeOrigin: false };
  }
  for (const target of demoMode === 'live' ? ['baseline', 'pruned'] as const : []) {
    const name = `PLAYGROUND_${target.toUpperCase()}_URL`;
    const value = env[name]?.trim();
    if (!value) continue;
    const url = new URL(value);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
      throw new Error(`${name} must be an HTTP(S) model-service URL without credentials, a query, or a fragment.`);
    }
    const prefix = `/api/playground/${target}`;
    proxy[prefix] = {
      target: url.href.replace(/\/$/, ''), changeOrigin: true,
      rewrite: (path) => path.slice(prefix.length),
    };
  }
  return {
    define: { __LIVE_DEMO__: JSON.stringify(demoMode === 'live') },
    plugins: [react(), {
      name: 'demo-build-mode',
      transformIndexHtml: (html) => html.replace('</head>', `<meta name="demo-mode" content="${demoMode}" /></head>`),
    }, {
      name: 'local-playground-connections',
      configureServer(server) {
        server.middlewares.use((request, response, next) => {
          const { url, method } = request as typeof request & { url?: string; method?: string };
          if (!url?.startsWith('/api/playground/')) return next();
          const match = /^\/api\/playground\/(baseline|pruned)\/(health|infer)$/.exec(url);
          const configured = match && proxy[`/api/playground/${match[1]}`];
          const expectedMethod = match?.[2] === 'health' ? 'GET' : 'POST';
          if (configured && method === expectedMethod) return next();
          response.statusCode = !match ? 404 : !configured ? 503 : 405;
          response.setHeader('Content-Type', 'application/json');
          response.end(JSON.stringify({ error: !configured ? 'Model connection is not configured.' : 'Unsupported model request.' }));
        });
      },
    }],
    server: { port: 5173, strictPort: true, proxy },
    build: {
      rollupOptions: {
        output: { manualChunks: { charts: ['recharts'] } },
      },
    },
  };
});

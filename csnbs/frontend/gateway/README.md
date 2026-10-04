# Live demo gateway

This Worker serves the live build and its same-origin `/api/live/*` API. It is off by
default. Nothing in this folder provisions a GPU, adds a payment method, or deploys
anything automatically. The separate showcase build never needs this gateway.

The frontend's local adapter reuses `handleLiveRequest` and `LiveAdmission`. No
Cloudflare dependency or account is required for local development. Wrangler is
needed only to validate/deploy this configuration later. Build `dist-live-share`
before running Wrangler from this folder.

## Public endpoints

- `GET /api/live/config` reports gateway availability and upload/output limits. When
  enabled and configured, it sets a signed anonymous `HttpOnly; SameSite=Strict`
  cookie. In production the cookie also requires HTTPS. This endpoint never contacts
  the GPU.
- `POST /api/live/warmup` accepts `{request_id, turnstile_token}` and returns the
  backend's validated `LiveModelConfig`. It is subject to all protections below.
- `POST /api/live/generate` accepts `LiveGenerateInput` and returns a complete,
  validated `LiveAnswer`. The shared definitions live in `src/live-contract.ts`.

The UI must use fresh UUIDs, a Turnstile widget with `action: 'live-demo'`, and a
fresh token for each POST. There is no SSE emulation, automatic retry, GPU health
polling, or heartbeat. Entering the live Playground can trigger one bounded warmup
after verification (immediately in loopback development). Opening the overview or
reading gateway config alone never starts the GPU. The showcase build has no such
warmup or gateway connection.

## Production configuration and account ownership

Before a teammate chooses to deploy, that teammate must supply their hosting
accounts, budget/spend controls, an authenticated model service, a domain/route,
and a Turnstile widget configured for the final hostname. Account ownership and
billing are outside this source code. Keep `LIVE_ENABLED=false` until verified.

Set `ALLOWED_ORIGINS` to the exact HTTPS site origin (or comma-separated origins).
Set `MODEL_UPSTREAM_URL` to the private backend base URL and
`TURNSTILE_SITE_KEY` to the widget's public key. Configure these **Worker secrets**,
never Vite variables or committed files:

- `COOKIE_SECRET`: independent random secret, at least 32 characters.
- `IP_HASH_SECRET`: a different random secret, at least 32 characters.
- `TURNSTILE_SECRET`: Turnstile server secret.
- `MODAL_KEY`, `MODAL_SECRET`: Modal proxy-token credentials.

The Worker only sends those Modal credentials server-to-server. Responses and
errors never return them or upstream URLs. Redirects are refused. Native Modal
Web Functions must enable `requires_proxy_auth=True`; protecting only this Worker
would leave a directly callable, billable endpoint. The checked-in configuration
disables the workers.dev hostname and needs an explicitly selected production
route/custom domain. Static assets come only from the filtered live share build.

`LOCAL_DEVELOPMENT=true` skips Turnstile/proxy authentication only when the incoming
URL, configured upstream, and every allowed origin are loopback addresses. A
public URL with that flag fails closed. The local adapter must bind only loopback
and discard/ignore spoofed forwarded IP headers. It must never be exposed through
a public tunnel. Production requires HTTPS, authentication secrets, and Cloudflare's
client IP header. No CORS access is enabled.

## Admission, retention, and limits

One SQLite-backed Durable Object serializes admission globally. It keeps one
current UTC-day counter record, one model config cached for ten minutes, and
request-ID tombstones with a 48-hour idempotency window. Tombstones have no request
contents and expire logically after 48 hours; expired records are removed during
the next daily admission cleanup. Traffic is capped at 150 admitted GPU calls a
day (100 generation attempts plus 50 warmups). Anonymous browser and keyed IP-hash
counters rotate at UTC midnight. No raw IPs, prompts, images, answers, challenge
tokens, or credentials are stored or logged by application code. Platform-level
retention and logging still require the account owner's review.

| Limit | Generation | Warmup |
| --- | ---: | ---: |
| Per signed browser cookie / UTC day | 5 | 3 |
| Per HMAC-hashed IP / UTC day | 10 | 6 |
| Global / UTC day | 100 | 50 |

Both kinds also share a three-second browser and one-second IP burst gap. Only
one admitted GPU request can be in flight; the next gets a busy response rather
than joining a billable queue. Requests consume allowance when admitted, including
backend errors or client disconnects. Reusing an admitted UUID within 48 hours
returns a duplicate error and **does not replay the GPU call or retain its answer**.
A new UUID is a new billable attempt. A cookie reset cannot bypass the IP/global
limits. These are request limits, **not a dollar spending cap**.

Uploads are bounded while reading the body, not just via Content-Length. Decoded
files are capped at 5 MiB; only JPEG/PNG/WebP signatures are accepted. Questions are
at most 2,000 characters. Output requests are at most 128 tokens and never exceed
the backend-advertised limit. The chosen visual-token count and fixed important
ratio must match the current server configuration. The baseline 576-token option
must be present. A fake/not-ready backend cannot admit generation.

## Backend obligations and timeout limits

Amay's backend must implement authenticated `POST /warmup` and `POST /generate`.
`/warmup` must return the model config when ready; `/generate` must return the exact
request ID plus actual execution metadata. Config must report real token options,
fixed ratio, hardware, model, method, git revision, and `server_mode`. The gateway
does not modify serving or measurement code and does not fabricate timing results.

Backend configuration must enforce one container/one concurrent execution, bounded
image dimensions and safe decoding, output-token limits, **a hard execution deadline
of at most 180 seconds**, and scale-to-zero after the agreed **120 seconds idle**.
The Worker times out upstream I/O after 120 seconds. A timeout or broken response
holds the admission lease for 240 seconds from admission. A fully read backend
response acknowledges completion and releases admission sooner. A disconnected
browser or aborted HTTP request does **not** prove that GPU work stopped. Only the
backend's deadline/serialization and the hosting owner's spend cap can bound that
work; the lease alone is not sufficient. The gateway never automatically retries.

Turning `LIVE_ENABLED=false` immediately refuses new calls, including warmups. It
cannot cancel an already running backend operation. The account owner must retain
a backend shutdown/kill switch and monitor actual charges.

## References used

- [SQLite-backed Durable Object storage](https://developers.cloudflare.com/durable-objects/api/sqlite-storage-api/)
- [SQLite namespace configuration](https://developers.cloudflare.com/durable-objects/best-practices/access-durable-objects-storage/)
- [Turnstile server-side validation](https://developers.cloudflare.com/turnstile/get-started/server-side-validation/)
- [Modal proxy authentication](https://modal.com/docs/guide/webhook-proxy-auth)

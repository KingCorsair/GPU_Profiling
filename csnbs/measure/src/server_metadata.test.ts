import assert from "node:assert/strict";
import { createServer } from "node:http";
import { test, type TestContext } from "node:test";
import { collectServerMetadata } from "./server_metadata.js";

async function healthServer(t: TestContext, body: unknown, status = 200): Promise<string> {
  const server = createServer((req, res) => {
    assert.equal(req.url, "/health");
    res.writeHead(status, { "content-type": "application/json" });
    res.end(JSON.stringify(body));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => new Promise<void>((resolve, reject) => {
    server.close((error) => error ? reject(error) : resolve());
    server.closeAllConnections();
  }));
  const address = server.address();
  assert.ok(address && typeof address !== "string");
  return `http://127.0.0.1:${address.port}/infer`;
}

test("captures model identity and revision from the serving process", async (t) => {
  const configuration = {
    model_id: "unsloth/gemma-3-4b-it",
    download_provenance: { revision: "pinned-checkpoint" },
    dtype: "bfloat16", max_new_tokens: 64,
  };
  const endpoint = await healthServer(t, { service: "csnbs-gemma-server", pid: 123, configuration });
  const result = await collectServerMetadata(endpoint);
  assert.equal(result.modelId, configuration.model_id);
  assert.equal(result.checkpointRevision, "pinned-checkpoint");
  assert.equal(result.pid, 123);
  assert.deepEqual(result.configuration, configuration);
  assert.equal(result.error, null);
});

test("legacy health responses leave unknown model identity explicit", async (t) => {
  const endpoint = await healthServer(t, { service: "csnbs-llava-server", model_loaded: true });
  const result = await collectServerMetadata(endpoint);
  assert.equal(result.modelId, null);
  assert.equal(result.checkpointRevision, null);
  assert.equal(result.configuration, null);
  assert.equal(result.error, null);
});

test("unavailable metadata is recorded without inventing a model", async (t) => {
  const endpoint = await healthServer(t, { detail: "Not Found" }, 404);
  const result = await collectServerMetadata(endpoint);
  assert.equal(result.modelId, null);
  assert.match(result.error ?? "", /HTTP 404/);
});

test("invalid endpoint metadata does not throw away the loadgen's error records", async () => {
  const result = await collectServerMetadata("not-an-absolute-url");
  assert.equal(result.healthUrl, null);
  assert.equal(result.modelId, null);
  assert.ok(result.error);
});

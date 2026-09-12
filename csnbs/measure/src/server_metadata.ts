export type ServerMetadata = {
  healthUrl: string;
  service: string | null;
  pid: number | null;
  modelId: string | null;
  checkpointRevision: string | null;
  configuration: Record<string, unknown> | null;
  error: string | null;
};

function objectOrNull(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

/* Read server-reported provenance before timed requests. Missing data stays null. */
export async function collectServerMetadata(endpoint: string): Promise<ServerMetadata> {
  const healthUrl = new URL("/health", endpoint).href;
  const result: ServerMetadata = {
    healthUrl, service: null, pid: null, modelId: null,
    checkpointRevision: null, configuration: null, error: null,
  };
  try {
    const response = await fetch(healthUrl, { signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw new Error(`Health endpoint returned HTTP ${response.status}`);
    const health = objectOrNull(await response.json());
    if (health === null) throw new Error("Health response is not an object");
    result.service = typeof health.service === "string" ? health.service : null;
    result.pid = typeof health.pid === "number" ? health.pid : null;
    result.configuration = objectOrNull(health.configuration);
    const config = result.configuration;
    result.modelId = typeof config?.model_id === "string" ? config.model_id : null;
    const provenance = objectOrNull(config?.download_provenance);
    result.checkpointRevision = typeof provenance?.revision === "string" ? provenance.revision : null;
  } catch (error: unknown) {
    result.error = error instanceof Error ? error.message : String(error);
  }
  return result;
}

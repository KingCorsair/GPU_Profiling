/** Live demo transport only. These single answers are not benchmark or accuracy results. */
export type PlaygroundTarget = 'baseline' | 'pruned';

export type PlaygroundHealth = {
  mode: 'model' | 'fake';
  modelLoaded: boolean;
  visualTokens: number;
  importantRatio: number;
  maxNewTokens: number;
  batchSize: number;
  modelId: string | null;
  implementation: string;
  sourceCommit: string | null;
  gpuModels: string[];
};

export type PlaygroundResult = {
  answer: string;
  requestId: string | null;
  health: PlaygroundHealth;
  metrics: {
    serviceMs: number | null;
    generationMs: number | null;
    queueMs: number | null;
    visualTokens: number | null;
    generatedTextTokens: number | null;
    tokenUnavailableReason: string | null;
    queueUnavailableReason: string | null;
  };
};

const object = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value);
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const positiveInteger = (value: unknown): value is number => finite(value) && Number.isInteger(value) && value > 0;
const nullableString = (value: unknown): value is string | null | undefined => value === undefined || value === null || typeof value === 'string';
const targetName = (target: PlaygroundTarget) => target === 'baseline' ? 'baseline' : 'pruned';

export function parsePlaygroundHealth(input: unknown): PlaygroundHealth {
  if (!object(input) || input.service !== 'csnbs-llava-server') throw new Error('This endpoint is not the project’s LLaVA service.');
  const config = input.configuration;
  if ((input.mode !== 'model' && input.mode !== 'fake') || typeof input.model_loaded !== 'boolean' || !object(config)
    || !positiveInteger(config.visual_token_num) || config.visual_token_num > 576
    || !finite(config.important_ratio) || config.important_ratio < 0 || config.important_ratio > 1
    || !positiveInteger(config.max_new_tokens) || !positiveInteger(config.batch_size)
    || !nullableString(config.model_id) || typeof config.implementation !== 'string' || !config.implementation) {
    throw new Error('The service returned incomplete or unsupported model settings.');
  }
  const sourceCommit = object(input.source) && typeof input.source.gitCommit === 'string' ? input.source.gitCommit : null;
  const gpuModels = object(input.hardware) && Array.isArray(input.hardware.gpuModels)
    ? input.hardware.gpuModels.filter((model): model is string => typeof model === 'string') : [];
  return {
    mode: input.mode, modelLoaded: input.model_loaded, visualTokens: config.visual_token_num,
    importantRatio: config.important_ratio, maxNewTokens: config.max_new_tokens, batchSize: config.batch_size,
    modelId: config.model_id ?? null, implementation: config.implementation, sourceCommit, gpuModels,
  };
}

export function parsePlaygroundResult(input: unknown, health: PlaygroundHealth): PlaygroundResult {
  if (!object(input) || typeof input.answer !== 'string' || !nullableString(input.request_id)) {
    throw new Error('The model returned an unsupported answer format.');
  }
  const metrics = input.metrics;
  if (metrics !== undefined && metrics !== null && (!object(metrics) || (metrics.schema_version !== 1 && metrics.schema_version !== 2))) {
    throw new Error('The model returned an unsupported metrics format.');
  }
  const saved = object(metrics) ? metrics : {};
  const number = (key: string, integer = false): number | null => {
    const value = saved[key];
    if (value === undefined || value === null) return null;
    if (!finite(value) || value < 0 || (integer && !Number.isInteger(value))) throw new Error('The model returned invalid measurement values.');
    return value;
  };
  const reason = (key: string): string | null => {
    const value = saved[key];
    if (!nullableString(value)) throw new Error('The model returned invalid metric availability details.');
    return value ?? null;
  };
  const visualTokens = number('visual_tokens', true);
  if (visualTokens !== null && (visualTokens < 1 || visualTokens > 576 || visualTokens !== health.visualTokens)) {
    throw new Error('The answer’s observed token count does not match the checked model settings. Check the service and try again.');
  }
  return {
    answer: input.answer, requestId: input.request_id ?? null, health,
    metrics: {
      serviceMs: number('service_ms'), generationMs: number('generation_wall_ms'), queueMs: number('queue_ms'),
      visualTokens, generatedTextTokens: number('generated_text_tokens', true),
      tokenUnavailableReason: reason('token_unavailable_reason'), queueUnavailableReason: reason('queue_unavailable_reason'),
    },
  };
}

async function requestJson(target: PlaygroundTarget, path: 'health' | 'infer', options: RequestInit): Promise<unknown> {
  const timeout = AbortSignal.timeout(path === 'health' ? 8_000 : 120_000);
  const signal = options.signal ? AbortSignal.any([options.signal, timeout]) : timeout;
  const transportFailure = (error: unknown): never => {
    if (options.signal?.aborted) throw error;
    if (timeout.aborted) throw new Error(`The ${targetName(target)} model ${path === 'health' ? 'connection check' : 'request'} timed out. Check the service and try again.`);
    if (error instanceof Error && error.name === 'AbortError') throw error;
    throw new Error(`The ${targetName(target)} model service could not be reached. Check the local connection and try again.`);
  };
  let response: Response;
  try {
    response = await fetch(`/api/playground/${target}/${path}`, { ...options, signal, cache: 'no-store', credentials: 'omit', redirect: 'error' });
  } catch (error) {
    return transportFailure(error);
  }
  if (!response.ok) {
    if ([404, 502, 503, 504].includes(response.status)) throw new Error(`The ${targetName(target)} model is not connected. Configure its local endpoint and restart the development server.`);
    throw new Error(`The ${targetName(target)} service could not complete this request (HTTP ${response.status}).`);
  }
  try { return await response.json(); }
  catch (error) {
    if (signal.aborted) return transportFailure(error);
    throw new Error(`The ${targetName(target)} endpoint did not return model data. Check its local connection.`);
  }
}

export async function readPlaygroundHealth(target: PlaygroundTarget, signal?: AbortSignal): Promise<PlaygroundHealth> {
  return parsePlaygroundHealth(await requestJson(target, 'health', { method: 'GET', signal }));
}

export async function runPlaygroundInference(
  target: PlaygroundTarget,
  input: { imageBase64: string; question: string; expectedTokens: number },
  signal?: AbortSignal,
): Promise<PlaygroundResult> {
  if (!input.imageBase64 || !input.question.trim()) throw new Error('Choose an image and enter a question first.');
  if (!positiveInteger(input.expectedTokens) || input.expectedTokens > 576) throw new Error('Choose a valid image token budget.');
  if (target === 'baseline' && input.expectedTokens !== 576) throw new Error('The baseline must use all 576 image tokens.');

  // Settings belong to the running process. Never pretend the UI can change them per request.
  const health = await readPlaygroundHealth(target, signal);
  if (health.mode !== 'model') throw new Error('This endpoint is running a fake service. Connect a real model to ask image questions.');
  if (!health.modelLoaded) throw new Error('The model is still loading. Try connecting again when it is ready.');
  if (health.visualTokens !== input.expectedTokens) {
    throw new Error(`The ${targetName(target)} service uses ${health.visualTokens} tokens; this request expects ${input.expectedTokens}. Update the model connection or select its actual budget.`);
  }
  const requestId = crypto.randomUUID();
  const result = parsePlaygroundResult(await requestJson(target, 'infer', {
    method: 'POST', signal, headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ request_id: requestId, image_b64: input.imageBase64, question: input.question.trim() }),
  }), health);
  if (result.requestId !== requestId) throw new Error('The model response could not be matched to this request. Try again.');
  return result;
}

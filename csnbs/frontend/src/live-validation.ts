import type { LiveAnswer, LiveGatewayConfig, LiveGenerateInput, LiveModelConfig } from './live-contract';

const object = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value);
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const integer = (value: unknown, min = 1, max = Number.MAX_SAFE_INTEGER): value is number => finite(value) && Number.isInteger(value) && value >= min && value <= max;
const text = (value: unknown): value is string => typeof value === 'string' && value.trim().length > 0;
const strings = (value: unknown): value is string[] => Array.isArray(value) && value.every(text);
const source = (value: unknown): value is string | null => value === null || text(value);
const modelError = () => new Error('The live service returned incomplete or unsupported model settings.');

export function parseGatewayConfig(value: unknown): LiveGatewayConfig {
  if (!object(value) || typeof value.enabled !== 'boolean' || typeof value.local_development !== 'boolean'
    || !source(value.turnstile_site_key) || !integer(value.max_image_bytes, 1, 5 * 1024 * 1024)
    || !integer(value.max_question_chars, 1, 10_000) || !integer(value.max_output_tokens, 1, 4096)
    || (value.enabled && !value.local_development && !text(value.turnstile_site_key))) {
    throw new Error('The live demo connection is not configured correctly. You can still explore the recorded study.');
  }
  return { enabled: value.enabled, local_development: value.local_development, turnstile_site_key: value.turnstile_site_key,
    max_image_bytes: value.max_image_bytes, max_question_chars: value.max_question_chars, max_output_tokens: value.max_output_tokens };
}

export function parseModelConfig(value: unknown): LiveModelConfig {
  if (!object(value) || value.server_mode !== 'model') throw new Error('A real model is not connected. Test responses are not shown as live answers.');
  if (!['OFF', 'STARTING', 'READY', 'BUSY'].includes(String(value.state)) || !text(value.model) || !text(value.method)
    || !strings(value.gpu) || !source(value.git_commit) || !Array.isArray(value.visual_token_options)
    || value.visual_token_options.length === 0 || !value.visual_token_options.every((item) => integer(item, 1, 576))
    || new Set(value.visual_token_options).size !== value.visual_token_options.length
    || !value.visual_token_options.includes(576) || !finite(value.important_ratio) || value.important_ratio < 0 || value.important_ratio > 1
    || !integer(value.max_output_tokens, 1, 4096) || typeof value.supports_streaming !== 'boolean') throw modelError();
  return { server_mode: 'model', state: value.state as LiveModelConfig['state'], model: value.model, method: value.method,
    gpu: value.gpu, git_commit: value.git_commit, visual_token_options: value.visual_token_options as number[],
    important_ratio: value.important_ratio, max_output_tokens: value.max_output_tokens, supports_streaming: value.supports_streaming };
}

export function parseLiveAnswer(value: unknown, request: LiveGenerateInput, config: LiveModelConfig): LiveAnswer {
  if (!object(value) || typeof value.answer !== 'string' || value.answer.length > 100_000 || value.server_mode !== 'model') {
    throw new Error('The service did not return a real model answer in the supported format.');
  }
  if (value.request_id !== request.request_id) throw new Error('The answer could not be matched to your request. Please try again.');
  if (value.visual_token_num !== request.visual_token_num || value.important_ratio !== request.important_ratio) {
    throw new Error('The model used different token settings than requested. This answer cannot be used for this comparison.');
  }
  if (!text(value.model) || !text(value.method) || !strings(value.gpu) || !source(value.git_commit)
    || value.model !== config.model || value.method !== config.method || value.git_commit !== config.git_commit
    || [...value.gpu].sort().join('\0') !== [...config.gpu].sort().join('\0')) {
    throw new Error('The model configuration changed during this request. Reconnect before comparing answers.');
  }
  const metric = (key: string, count = false): number | null => {
    const item = value[key];
    if (item === null || item === undefined) return null;
    if (!finite(item) || item < 0 || (count && !Number.isSafeInteger(item))) throw new Error('The service reported an invalid request metric.');
    return item;
  };
  const total = metric('total_ms');
  const ttft = metric('ttft_ms');
  const output = metric('n_output_tokens', true);
  if ((total !== null && ttft !== null && ttft > total) || (output !== null && output > request.max_output_tokens)) {
    throw new Error('The service reported request metrics that do not match this request.');
  }
  return { request_id: request.request_id, answer: value.answer, server_mode: 'model', visual_token_num: request.visual_token_num,
    important_ratio: request.important_ratio, model: value.model, method: value.method, gpu: value.gpu, git_commit: value.git_commit,
    ttft_ms: ttft, total_ms: total, n_input_tokens: metric('n_input_tokens', true), n_output_tokens: output };
}

/** Only compatible, attributable executions may share a pinned comparison. */
export function matchingExecution(a: LiveAnswer, aCap: number, b: LiveAnswer, bCap: number): boolean {
  return a.git_commit !== null && b.git_commit !== null && a.git_commit === b.git_commit && a.model === b.model
    && a.method === b.method && a.important_ratio === b.important_ratio && aCap === bCap
    && [...a.gpu].sort().join('\0') === [...b.gpu].sort().join('\0');
}

export function imagePayload(src: string, maxBytes: number): Pick<LiveGenerateInput, 'image_b64' | 'image_mime'> {
  const match = /^data:(image\/(?:jpeg|png|webp));base64,([A-Za-z0-9+/]+={0,2})$/.exec(src);
  if (!match || match[2].length % 4 !== 0) throw new Error('Upload a JPG, PNG, or WebP image before running the model.');
  const bytes = match[2].length * 3 / 4 - (match[2].endsWith('==') ? 2 : match[2].endsWith('=') ? 1 : 0);
  if (bytes > maxBytes) throw new Error(`Choose an image no larger than ${Math.round(maxBytes / 1024 / 1024)} MB for the live demo.`);
  return { image_mime: match[1] as LiveGenerateInput['image_mime'], image_b64: match[2] };
}

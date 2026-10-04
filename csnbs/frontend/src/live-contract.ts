/** Public live-demo contract. Values describe individual requests, never benchmark evidence. */
export type LiveState = 'OFF' | 'STARTING' | 'READY' | 'BUSY';

export type LiveModelConfig = {
  server_mode: 'model' | 'fake';
  state: LiveState;
  model: string;
  method: string;
  gpu: string[];
  git_commit: string | null;
  visual_token_options: number[];
  important_ratio: number;
  max_output_tokens: number;
  supports_streaming: boolean;
};

export type LiveGatewayConfig = {
  enabled: boolean;
  turnstile_site_key: string | null;
  local_development: boolean;
  max_image_bytes: number;
  max_question_chars: number;
  max_output_tokens: number;
};

export type LiveGenerateInput = {
  request_id: string;
  image_b64: string;
  image_mime: 'image/jpeg' | 'image/png' | 'image/webp';
  question: string;
  visual_token_num: number;
  important_ratio: number;
  max_output_tokens: number;
  turnstile_token?: string;
};

export type LiveAnswer = {
  request_id: string;
  answer: string;
  server_mode: 'model' | 'fake';
  visual_token_num: number;
  important_ratio: number;
  model: string;
  method: string;
  gpu: string[];
  git_commit: string | null;
  ttft_ms: number | null;
  total_ms: number | null;
  n_input_tokens: number | null;
  n_output_tokens: number | null;
};

export type LiveApiError = { error: { code: string; message: string; retry_after_seconds?: number } };

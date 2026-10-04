import { useEffect, useRef, useState } from 'react';
import { ArrowRight, Check, ChevronDown, RefreshCw, Send, Square } from 'lucide-react';
import { ImagePicker, TokenImage, useDemoImage } from './TokenImageDemo';
import { generateLiveAnswer, LiveDemoError, readLiveConfig, warmLiveModel } from './live-api';
import { imagePayload, matchingExecution } from './live-validation';
import type { LiveAnswer, LiveGatewayConfig, LiveGenerateInput, LiveModelConfig } from './live-contract';
import Turnstile from './Turnstile';
import './playground.css';
import './live-playground.css';

type SavedAnswer = { answer: LiveAnswer; imageSrc: string; imageName: string; question: string; outputCap: number };
type Operation = 'connecting' | 'warming' | 'generating' | null;
const formatMs = (value: number | null) => value === null ? 'Not reported' : `${value.toLocaleString('en-US', { maximumFractionDigits: 1 })} ms`;
const message = (failure: unknown) => failure instanceof Error ? failure.message : 'The live request could not be completed. Please try again.';
const suggestions = ['Describe the scene.', 'What text can you read?', 'How many objects are there?'];

function Answer({ saved, baseline, pruningUnavailable = false }: { saved?: SavedAnswer; baseline: boolean; pruningUnavailable?: boolean }) {
  const answer = saved?.answer;
  return <article className={`play-answer live-answer${saved ? '' : ' live-answer-empty'}`} aria-label={baseline ? 'Baseline answer' : 'Pruned answer'}>
    <div className="play-answer-heading"><span className="live-answer-title">{baseline ? 'Full image' : 'Pruned image'}</span>
      <span>{baseline ? '576 tokens · baseline' : answer ? `${answer.visual_token_num} tokens` : 'Fewer visual tokens'}</span></div>
    {answer && saved ? <>
      <span className="play-answer-tag live-answer-tag"><Check size={13} aria-hidden="true" />Real model response</span>
      <p className="play-answer-text">{answer.answer || '(The model returned an empty answer.)'}</p>
      <dl className="play-response-metrics">
        <div><dt>Server total time</dt><dd>{formatMs(answer.total_ms)}</dd></div>
        <div><dt>Time to first token</dt><dd>{formatMs(answer.ttft_ms)}</dd></div>
        <div><dt>Input tokens reported</dt><dd>{answer.n_input_tokens ?? 'Not reported'}</dd></div>
        <div><dt>Output tokens reported</dt><dd>{answer.n_output_tokens ?? 'Not reported'}</dd></div>
      </dl>
      <details className="play-response-details"><summary>Response details <ChevronDown size={14} aria-hidden="true" /></summary>
        <dl><div><dt>Image</dt><dd>{saved.imageName}</dd></div><div><dt>Question</dt><dd>{saved.question}</dd></div>
          <div><dt>Model</dt><dd>{answer.model}</dd></div><div><dt>Method</dt><dd>{answer.method}</dd></div>
          <div><dt>Important-token ratio</dt><dd>{answer.important_ratio}</dd></div><div><dt>Output token cap</dt><dd>{saved.outputCap}</dd></div>
          <div><dt>GPU</dt><dd>{answer.gpu.join(', ') || 'Not reported'}</dd></div><div><dt>Source revision</dt><dd>{answer.git_commit ?? 'Not reported'}</dd></div>
          <div><dt>Request ID</dt><dd>{answer.request_id}</dd></div></dl>
        <p>Metrics are reported by the server for this request. Unavailable metrics are left blank; no browser timing is substituted. Answers arrive together when generation completes.</p>
      </details>
    </> : <div className="live-answer-placeholder"><p>{baseline ? 'Start with the full image.' : pruningUnavailable ? 'Pruned settings unavailable.' : 'Then try fewer tokens.'}</p>
      <span>{baseline ? 'Run 576 tokens to pin a baseline answer here.' : pruningUnavailable ? 'The connected service supports only the 576-token baseline. A lower token setting is needed to compare answers.' : 'Keep the same image and question to compare the actual answers.'}</span></div>}
  </article>;
}

export default function LivePlayground() {
  const [image, setImage] = useDemoImage();
  const [gateway, setGateway] = useState<LiveGatewayConfig | null>(null);
  const [model, setModel] = useState<LiveModelConfig | null>(null);
  const [operation, setOperation] = useState<Operation>('connecting');
  const [question, setQuestion] = useState('Describe the scene.');
  const [tokens, setTokens] = useState(576);
  const [prunedTokens, setPrunedTokens] = useState(128);
  const [verification, setVerification] = useState<string | null>(null);
  const [verificationKey, setVerificationKey] = useState(0);
  const [imageLoading, setImageLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availabilityError, setAvailabilityError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [answers, setAnswers] = useState<SavedAnswer[]>([]);
  const [retryWait, setRetryWait] = useState(0);
  const mounted = useRef(false);
  const attemptedWarmup = useRef(false);
  const controller = useRef<AbortController | null>(null);
  const running = useRef(false);
  const answerHeading = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; controller.current?.abort(); };
  }, []);

  useEffect(() => {
    let active = true;
    void readLiveConfig().then((config) => { if (active) { setGateway(config); setOperation(null); } })
      .catch((failure) => { if (active) { setError(message(failure)); setOperation(null); } });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (retryWait === 0) return;
    const timer = window.setTimeout(() => setRetryWait(0), retryWait * 1000);
    return () => clearTimeout(timer);
  }, [retryWait]);

  useEffect(() => {
    if (!gateway?.enabled || attemptedWarmup.current || (!gateway.local_development && !verification)) return;
    attemptedWarmup.current = true;
    setOperation('warming');
    setError(null);
    const token = verification ?? undefined;
    setVerification(null);
    setVerificationKey((value) => value + 1);
    void warmLiveModel(token).then((config) => {
      if (!mounted.current) return;
      setModel(config);
      setTokens((current) => config.visual_token_options.includes(current) ? current : 576);
      setPrunedTokens((current) => config.visual_token_options.includes(current) && current !== 576 ? current : config.visual_token_options.find((value) => value < 576) ?? 128);
      if (config.state !== 'READY') setRetryWait(30);
    }).catch((failure) => {
      if (!mounted.current) return;
      setError(message(failure));
      setRetryWait(Math.max(30, failure instanceof LiveDemoError ? failure.retryAfterSeconds ?? 0 : 0));
    }).finally(() => { if (mounted.current) setOperation(null); });
  }, [gateway, verification]);

  const busy = operation !== null || imageLoading;
  const verified = gateway?.local_development || !!verification;
  const ready = gateway?.enabled && model?.state === 'READY' && !availabilityError;
  const canRun = !!ready && !!verified && !!image.base64 && !!question.trim() && !busy && retryWait === 0;
  const currentAnswers = answers.filter((entry) => entry.imageSrc === image.src && entry.question === question.trim());
  const baseline = currentAnswers.find((entry) => entry.answer.visual_token_num === 576);
  const pruned = currentAnswers.find((entry) => entry.answer.visual_token_num === prunedTokens);
  const inputCap = Math.min(gateway?.max_output_tokens ?? 128, model?.max_output_tokens ?? 128);
  const pruningUnavailable = model !== null && !model.visual_token_options.some((value) => value < 576);
  const status = operation === 'connecting' ? 'Connecting' : operation === 'warming' ? 'Starting' : operation === 'generating' ? 'Running'
    : availabilityError === 'quota_exceeded' ? 'Usage limit reached' : availabilityError === 'busy' ? 'Busy'
      : availabilityError ? 'Connection needed' : !gateway?.enabled ? 'Offline' : model?.state === 'READY' ? 'Ready' : model?.state === 'BUSY' ? 'Busy' : model?.state === 'STARTING' ? 'Starting'
      : !gateway.local_development && !verification && !attemptedWarmup.current ? 'Verify to start' : 'Offline';
  const statusDetail = operation === 'connecting' ? 'Checking whether the live demo is available.'
    : operation === 'warming' ? 'Starting the model. A cold start can take a few minutes. You can prepare an image and question while it loads.'
    : operation === 'generating' ? 'The model is generating an answer. The full response will appear when it finishes.'
    : availabilityError === 'quota_exceeded' ? 'The demo has reached its allowance. The measured study remains available.'
    : availabilityError === 'busy' ? 'The model is helping another visitor. Try connecting again shortly.'
    : gateway?.enabled === false ? 'Live inference is switched off. The project and measured study are still available.'
    : ready ? 'The model was ready at the last check. It may sleep between requests to conserve resources.'
    : model?.state === 'BUSY' ? 'The model is helping another visitor. Try connecting again shortly.'
    : model?.state === 'STARTING' ? 'The model has not finished loading. Try connecting again shortly.'
    : gateway && !attemptedWarmup.current ? 'Complete visitor verification to start the model once. Your image is sent only when you run a request.'
    : 'A running model needs to be connected. You can explore the measured study in the meantime.';
  const runHint = !ready ? 'Connect to a ready model to run your image.' : !image.base64 ? 'Upload a JPG, PNG, or WebP image. The illustration is a preview only.'
    : !question.trim() ? 'Enter a question about your image.' : retryWait ? 'Please wait before sending another request.'
    : !verified ? 'Complete the visitor verification to continue.'
    : `${tokens === 576 ? 'Baseline' : 'Pruned'} request · ${tokens} visual tokens · up to ${inputCap} output tokens.`;

  function clearComparison() { setAnswers([]); setNotice(null); setError(null); }
  function consumeVerification() { const token = verification ?? undefined; setVerification(null); setVerificationKey((value) => value + 1); return token; }

  async function reconnect() {
    if (busy || retryWait > 0 || running.current) return;
    running.current = true;
    setOperation('connecting');
    setError(null);
    setAvailabilityError(null);
    setModel(null);
    try {
      const config = await readLiveConfig();
      if (!mounted.current) return;
      attemptedWarmup.current = false;
      setGateway(config);
    } catch (failure) { if (mounted.current) setError(message(failure)); }
    finally { running.current = false; if (mounted.current) setOperation(null); }
  }

  async function run() {
    if (!canRun || !gateway || !model || running.current) return;
    let payload: Pick<LiveGenerateInput, 'image_b64' | 'image_mime'>;
    try { payload = imagePayload(image.src, gateway.max_image_bytes); }
    catch (failure) { setError(message(failure)); return; }
    if (question.trim().length > gateway.max_question_chars) { setError('Shorten your question before sending it.'); return; }
    const abort = new AbortController();
    controller.current = abort;
    running.current = true;
    setOperation('generating');
    setError(null);
    setNotice(null);
    const input: LiveGenerateInput = { request_id: crypto.randomUUID(), ...payload, question: question.trim(), visual_token_num: tokens,
      important_ratio: model.important_ratio, max_output_tokens: inputCap, turnstile_token: consumeVerification() };
    const snapshot = { imageSrc: image.src, imageName: image.name, question: input.question, outputCap: inputCap };
    try {
      const answer = await generateLiveAnswer(input, model, abort.signal);
      if (abort.signal.aborted || !mounted.current) return;
      const saved = { ...snapshot, answer };
      const matching = currentAnswers.filter((entry) => matchingExecution(entry.answer, entry.outputCap, answer, inputCap));
      setAnswers([...matching.filter((entry) => entry.answer.visual_token_num !== answer.visual_token_num), saved]);
      if (tokens !== 576) setPrunedTokens(tokens);
      if (answer.git_commit === null) setNotice('Answer received. The service did not report its source revision, so other answers cannot be pinned as a verified comparison.');
      else if (matching.length !== currentAnswers.length) setNotice('Answer received. Earlier answers used a different or unreported source configuration and were cleared.');
      else setNotice(tokens === 576 ? pruningUnavailable ? 'Baseline answer received. This service supports only 576 tokens, so pruning comparisons are unavailable.' : 'Baseline pinned. Choose fewer tokens and run the same image and question to compare.' : 'Answer received. Compare the content below; request timings are not benchmark results.');
      requestAnimationFrame(() => answerHeading.current?.focus({ preventScroll: true }));
    } catch (failure) {
      if (abort.signal.aborted || !mounted.current) return;
      setError(message(failure));
      if (failure instanceof LiveDemoError && ['quota_exceeded', 'disabled', 'busy', 'upstream_unavailable', 'gateway_unavailable', 'session_required', 'starting'].includes(failure.code)) setAvailabilityError(failure.code);
      if (failure instanceof LiveDemoError && failure.retryAfterSeconds) setRetryWait(failure.retryAfterSeconds);
    } finally {
      if (controller.current === abort) { controller.current = null; running.current = false; if (mounted.current) setOperation(null); }
    }
  }

  function cancel() {
    controller.current?.abort();
    setNotice('Stopped waiting. The server may still finish processing this request; cancellation does not guarantee GPU work has stopped.');
  }

  return <div className="playground-page live-playground">
    <section className="intro play-intro" aria-labelledby="live-title"><p className="study-label">Live playground</p><h1 id="live-title">Ask about an image.</h1>
      <p className="intro-copy">Ask the real model a question. Keep the full image as a baseline, then see what changes when it has fewer visual tokens.</p></section>

    <section className="live-connection" aria-labelledby="live-connection-title">
      <div className="live-connection-copy"><div className="live-status-heading"><span className="live-status-dot" data-ready={!!ready && operation !== 'generating'} aria-hidden="true" />
        <h2 id="live-connection-title" role="status">{status}</h2><span className="live-status-caption">Live GPU demo</span></div><p>{statusDetail}</p></div>
      {operation !== 'warming' && operation !== 'generating' && <button className="play-secondary-button" type="button" disabled={busy || retryWait > 0} onClick={() => void reconnect()}><RefreshCw size={14} aria-hidden="true" />{operation === 'connecting' ? 'Connecting…' : retryWait > 0 ? 'Wait before retrying' : ready ? 'Reconnect' : 'Try connecting'}</button>}
    </section>
    {gateway?.local_development && <p className="live-local-note">Local development · visitor verification is disabled by the local gateway.</p>}
    {gateway?.enabled && !gateway.local_development && gateway.turnstile_site_key && <Turnstile siteKey={gateway.turnstile_site_key} resetKey={verificationKey} onToken={setVerification} />}
    {!ready && operation !== 'generating' && <p className="live-fallback"><a href="#/october">Explore the measured October study <ArrowRight size={14} aria-hidden="true" /></a><span>The study is available while the live model is offline or starting.</span></p>}

    <div className="play-workspace">
      <section className="play-image-panel" aria-labelledby="live-image-title"><div className="play-panel-heading"><h2 id="live-image-title">1. Choose an image</h2><span className="live-field-note">Original image</span></div>
        <ImagePicker image={image} onChange={(next) => { setImage(next); clearComparison(); }} disabled={operation === 'generating'} onLoadingChange={setImageLoading} maxBytes={gateway?.max_image_bytes ?? 5 * 1024 * 1024} />
        <div className="play-image-preview"><TokenImage image={image} tokens={576} overlay={false} grid={false} /></div>
        <p className="play-overlay-note">The image stays unchanged. Token pruning happens inside the model; this preview does not claim to show its retained tokens.</p>
      </section>

      <section className="play-composer-panel" aria-labelledby="live-question-title"><div className="play-panel-heading"><h2 id="live-question-title">2. Ask and compare</h2></div>
        <form onSubmit={(event) => { event.preventDefault(); void run(); }}><label className="sr-only" htmlFor="live-question">Question about your image</label>
          <textarea id="live-question" value={question} maxLength={gateway?.max_question_chars ?? 2000} disabled={operation === 'generating'} rows={4} placeholder="What would you like to know about this image?" onChange={(event) => { setQuestion(event.target.value); clearComparison(); }} />
          <div className="play-suggestions" aria-label="Suggested questions">{suggestions.map((suggestion) => <button type="button" key={suggestion} disabled={operation === 'generating'} onClick={() => { setQuestion(suggestion); clearComparison(); }}>{suggestion}</button>)}</div>
          <fieldset className="live-token-fieldset" disabled={!ready || busy}><legend>Visual token budget</legend><div className="live-token-options">{(model?.visual_token_options ?? [576, 384, 256, 128]).map((value) => <button type="button" key={value} aria-pressed={tokens === value} onClick={() => { setTokens(value); if (value !== 576) setPrunedTokens(value); setError(null); }}><strong>{value}</strong><span>{value === 576 ? 'Full image' : `${Math.round((1 - value / 576) * 100)}% fewer`}</span>{value === 576 && <small>Baseline</small>}</button>)}</div>
            <p>{pruningUnavailable ? 'This service supports the baseline only; pruned settings are unavailable.' : model ? 'These settings are supported by the connected model.' : 'Available settings will be confirmed by the model.'}</p></fieldset>
          <div className="live-fixed-settings"><span>Important-token ratio <b>{model?.important_ratio ?? 'Awaiting model'}</b></span><span>Output limit <b>{model ? `${inputCap} tokens` : 'Awaiting model'}</b></span></div>
          <div className="play-submit-row"><button type="submit" className="play-submit" disabled={!canRun}><Send size={15} aria-hidden="true" />{operation === 'generating' ? 'Generating answer…' : tokens === 576 ? 'Run baseline' : `Run with ${tokens} tokens`}</button>
            {operation === 'generating' && <button type="button" className="play-secondary-button" onClick={cancel}><Square size={12} aria-hidden="true" />Stop waiting</button>}</div>
          <p className="play-run-hint">{runHint}</p><p className="play-upload-notice">Running sends this image and question to the team’s model service. Use an image you are comfortable sharing. Results stay in this page until you leave or change the input.</p>
        </form>
      </section>
    </div>

    <div className="play-request-status" aria-live="polite">{error && <p className="play-error" role="alert">{error}</p>}{notice && <p>{notice}</p>}{retryWait > 0 && <p>Retry becomes available after {retryWait} seconds.</p>}</div>

    <section className="play-history live-comparison" aria-labelledby="live-comparison-title"><div className="play-history-heading"><div><h2 id="live-comparison-title" ref={answerHeading} tabIndex={-1}>Compare the answers.</h2><p>Same image and question. Source and settings checked before answers are paired.</p></div>
      {answers.length > 0 && <button className="play-secondary-button" type="button" disabled={busy} onClick={clearComparison}>Clear answers</button>}</div>
      <div className="play-answer-list"><Answer baseline saved={baseline} /><Answer baseline={false} saved={pruningUnavailable ? undefined : pruned} pruningUnavailable={pruningUnavailable} /></div>
      <p className="play-history-note">A live request demonstrates model behavior. It does not measure accuracy or establish a speedup. For controlled throughput and latency results, <a href="#/october">read the measured study</a>.</p>
    </section>
  </div>;
}

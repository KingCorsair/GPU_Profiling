import { useEffect, useRef, useState } from 'react';
import { ArrowRight, Check, ChevronDown, Layers, MessageSquare, RefreshCw, Send, Square } from 'lucide-react';
import { ImagePicker, TokenBudget, TokenImage, useDemoImage } from './TokenImageDemo';
import { readPlaygroundHealth, runPlaygroundInference } from './playground-api';
import type { PlaygroundHealth, PlaygroundResult, PlaygroundTarget } from './playground-api';
import './playground.css';

type Connection = { health?: PlaygroundHealth; error?: string };
type SavedAnswer = { id: number; imageName: string; question: string; target: PlaygroundTarget; result: PlaygroundResult };
const targets: PlaygroundTarget[] = ['baseline', 'pruned'];
const targetNames: Record<PlaygroundTarget, string> = { baseline: 'Baseline', pruned: 'Pruned' };
const suggestions = ['Describe the scene.', 'What text can you read?', 'How many objects are there?'];
const formatMs = (value: number | null) => value === null ? 'Unavailable' : `${value.toLocaleString('en-US', { maximumFractionDigits: 1 })} ms`;
const ready = (health?: PlaygroundHealth) => health?.mode === 'model' && health.modelLoaded;
const usable = (target: PlaygroundTarget, health?: PlaygroundHealth) => ready(health) && (target !== 'baseline' || health?.visualTokens === 576);
const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'The request could not be completed.';

function Answer({ entry }: { entry: SavedAnswer }) {
  const { result } = entry;
  return <article className="play-answer">
    <div className="play-answer-heading"><span className="play-answer-tag"><Check size={13} aria-hidden="true" />Live response</span><span>{targetNames[entry.target]} · {result.health.visualTokens} configured tokens</span></div>
    <p className="play-answer-question">{entry.question}</p>
    <p className="play-answer-image">Image: {entry.imageName}</p>
    <p className="play-answer-text">{result.answer || '(The model returned an empty answer.)'}</p>
    <dl className="play-response-metrics">
      <div><dt>Server service time</dt><dd>{formatMs(result.metrics.serviceMs)}</dd></div>
      <div><dt>Queue time</dt><dd>{formatMs(result.metrics.queueMs)}</dd></div>
      <div><dt>Visual tokens observed</dt><dd>{result.metrics.visualTokens ?? 'Unavailable'}</dd></div>
      <div><dt>Time to first token</dt><dd>Unavailable</dd></div>
    </dl>
    <details className="play-response-details">
      <summary>Response details <ChevronDown size={14} aria-hidden="true" /></summary>
      <dl>
        <div><dt>Model</dt><dd>{result.health.modelId ?? 'Not reported'}</dd></div>
        <div><dt>Important-token ratio</dt><dd>{result.health.importantRatio}</dd></div>
        <div><dt>Output token cap</dt><dd>{result.health.maxNewTokens}</dd></div>
        <div><dt>Batch size</dt><dd>{result.health.batchSize}</dd></div>
        <div><dt>GPU</dt><dd>{result.health.gpuModels.join(', ') || 'Not reported'}</dd></div>
        <div><dt>Generation time</dt><dd>{formatMs(result.metrics.generationMs)}</dd></div>
        <div><dt>Generated text tokens</dt><dd>{result.metrics.generatedTextTokens ?? 'Unavailable'}</dd></div>
        <div><dt>Implementation</dt><dd>{result.health.implementation}</dd></div>
        <div><dt>Source revision</dt><dd>{result.health.sourceCommit ?? 'Not reported'}</dd></div>
        <div><dt>Request ID</dt><dd>{result.requestId ?? 'Not reported'}</dd></div>
      </dl>
      {result.metrics.queueUnavailableReason && <p>Queue time: {result.metrics.queueUnavailableReason}</p>}
      {result.metrics.tokenUnavailableReason && <p>Token counts: {result.metrics.tokenUnavailableReason}</p>}
      <p>This endpoint returns a complete answer; it does not report time to first token. Service time starts inside the server handler and does not include all waiting or network time.</p>
    </details>
  </article>;
}

export default function Playground() {
  const [image, setImage] = useDemoImage();
  const [tokens, setTokens] = useState(128);
  const [overlay, setOverlay] = useState(true);
  const [mode, setMode] = useState<'local' | 'live'>('local');
  const [question, setQuestion] = useState('Describe the scene.');
  const [target, setTarget] = useState<PlaygroundTarget>('pruned');
  const [connections, setConnections] = useState<Partial<Record<PlaygroundTarget, Connection>>>({});
  const [checking, setChecking] = useState(false);
  const [running, setRunning] = useState(false);
  const [imageLoading, setImageLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [history, setHistory] = useState<SavedAnswer[]>([]);
  const controller = useRef<AbortController | null>(null);
  const sequence = useRef(0);
  const historyHeading = useRef<HTMLHeadingElement>(null);
  const connectionHeading = useRef<HTMLHeadingElement>(null);
  const activeHealth = connections[target]?.health;
  const isReady = usable(target, activeHealth);
  const validBaseline = target !== 'baseline' || activeHealth?.visualTokens === 576;
  const liveTokens = ready(activeHealth) ? activeHealth!.visualTokens : null;
  const previewTokens = mode === 'local' ? tokens : liveTokens ?? 576;
  const availableModels = targets.filter((item) => usable(item, connections[item]?.health)).length;
  const checked = targets.every((item) => connections[item] !== undefined);
  const busy = checking || running || imageLoading;
  const canRun = mode === 'live' && isReady && !!image.base64 && question.trim().length > 0 && !busy;
  const runHint = checking ? 'Checking for an available model…'
    : !isReady ? 'Choose an available model above to continue. You can check again after a service comes online.'
    : !image.base64 ? 'Upload your image to continue. The sample illustration is for preview only.'
    : !question.trim() ? 'Enter a question about your image to continue.'
    : `Ready to send your image and question to ${targetNames[target]}.`;

  useEffect(() => () => controller.current?.abort(), []);

  async function checkConnections() {
    const abort = new AbortController();
    controller.current?.abort();
    controller.current = abort;
    setChecking(true);
    setConnections({});
    setError(null);
    setNotice(null);
    const results = await Promise.allSettled(targets.map((item) => readPlaygroundHealth(item, abort.signal)));
    if (abort.signal.aborted) return;
    const next: Partial<Record<PlaygroundTarget, Connection>> = {};
    results.forEach((result, index) => {
      next[targets[index]] = result.status === 'fulfilled' ? { health: result.value } : { error: errorMessage(result.reason) };
    });
    setConnections(next);
    if (!usable(target, next[target]?.health)) {
      const available = targets.find((item) => usable(item, next[item]?.health));
      if (available) setTarget(available);
    }
    setChecking(false);
    if (controller.current === abort) controller.current = null;
  }

  function changeMode(next: 'local' | 'live', focusConnection = false) {
    if (running || imageLoading || (next === 'live' && checking) || mode === next) return;
    if (next === 'local' && checking) {
      controller.current?.abort();
      controller.current = null;
      setChecking(false);
    }
    setMode(next);
    setError(null);
    setNotice(null);
    if (next === 'live') {
      void checkConnections();
      if (focusConnection) requestAnimationFrame(() => connectionHeading.current?.focus());
    }
  }

  async function run() {
    if (!canRun || !image.base64 || liveTokens === null) return;
    const abort = new AbortController();
    controller.current = abort;
    const request = { imageBase64: image.base64, question: question.trim(), expectedTokens: liveTokens };
    const imageName = image.name;
    const requestTarget = target;
    setRunning(true);
    setError(null);
    setNotice(null);
    try {
      const result = await runPlaygroundInference(requestTarget, request, abort.signal);
      if (abort.signal.aborted) return;
      setConnections((old) => ({ ...old, [requestTarget]: { health: result.health } }));
      setHistory((old) => [{ id: ++sequence.current, imageName, question: request.question, target: requestTarget, result }, ...old].slice(0, 8));
      setNotice('Answer received. Your session results are below.');
      requestAnimationFrame(() => historyHeading.current?.focus({ preventScroll: true }));
    } catch (failure) {
      if (abort.signal.aborted) return;
      setError(errorMessage(failure));
    } finally {
      if (controller.current === abort) {
        controller.current = null;
        setRunning(false);
      }
    }
  }

  function cancel() {
    controller.current?.abort();
    setNotice('Stopped waiting for the answer. The server may still finish processing this request.');
  }

  return <div className="playground-page">
    <section className="intro play-intro" aria-labelledby="playground-title">
      <p className="study-label">Interactive playground</p>
      <h1 id="playground-title">Ask about an image.</h1>
      <p className="intro-copy">Explore how image tokens work, or send an image and question to a live model.</p>
    </section>

    <section className="play-mode-picker" aria-labelledby="play-mode-title">
      <h2 id="play-mode-title">Choose how to try it</h2>
      <div className="play-mode-options" role="group" aria-labelledby="play-mode-title">
        <button className="play-mode-card" type="button" aria-pressed={mode === 'local'} disabled={running || imageLoading} onClick={() => changeMode('local')}>
          <Layers className="play-mode-icon" size={21} aria-hidden="true" />
          <span className="play-mode-copy"><span className="play-mode-name">Explore image tokens</span><span className="play-mode-description">Move the slider to preview image patches. No AI answer is generated.</span><span className="play-mode-caption">Works in your browser · no connection needed</span></span>
          <span className="play-selection-mark" aria-hidden="true">{mode === 'local' && <Check size={13} />}</span>
        </button>
        <button className="play-mode-card" type="button" aria-pressed={mode === 'live'} disabled={busy} onClick={() => changeMode('live')}>
          <MessageSquare className="play-mode-icon" size={21} aria-hidden="true" />
          <span className="play-mode-copy"><span className="play-mode-name">Ask a live model</span><span className="play-mode-description">Get a real answer about your image using a connected model.</span><span className="play-mode-caption">{checking ? 'Checking availability…' : checked ? availableModels > 0 ? `${availableModels} ${availableModels === 1 ? 'model' : 'models'} available` : 'No models available · connection needed' : 'Requires a running model · check availability'}</span></span>
          <span className="play-selection-mark" aria-hidden="true">{mode === 'live' && <Check size={13} />}</span>
        </button>
      </div>
    </section>

    {mode === 'live' && <section className="play-connections" aria-label="Model connections">
      <div className="play-connection-heading"><div><h2 ref={connectionHeading} tabIndex={-1}>Choose a live model</h2><p>Availability is checked automatically. Your image is only sent when you ask a question.</p></div><button className="play-secondary-button" type="button" disabled={busy} onClick={() => void checkConnections()}><RefreshCw size={14} aria-hidden="true" />{checking ? 'Checking…' : 'Check again'}</button></div>
      <div className="play-targets" role="group" aria-label="Model service">
        {targets.map((item) => {
          const connection = connections[item];
          const health = connection?.health;
          return <button className="play-target" type="button" key={item} disabled={busy} aria-pressed={target === item} onClick={() => { setTarget(item); setError(null); setNotice(null); }}>
            <span className="play-target-name">{item === 'baseline' ? 'Full image' : 'Pruned image'}<span className="play-selection-mark" aria-hidden="true">{target === item && <Check size={13} />}</span></span>
            <span className="play-target-description">{item === 'baseline' ? 'Baseline · keeps all 576 image tokens' : 'Pruned · uses the model’s configured token budget'}</span>
            <span className="play-target-status" data-ready={usable(item, health)}>{checking ? 'Checking connection…' : health ? health.mode === 'fake' ? 'Test server · no live answers' : !health.modelLoaded ? 'Model is still loading' : !usable(item, health) ? 'Needs setup · baseline must use 576 tokens' : `Ready · ${health.visualTokens} tokens` : connection?.error ? 'Offline · connection needed' : 'Not checked'}</span>
          </button>;
        })}
      </div>
      <div aria-live="polite">
        {checking ? <p className="play-connection-note">Looking for connected models…</p> : checked && availableModels === 0 ? <div className="play-connection-empty"><h3>No live model is available yet.</h3><p>A running model service needs to be connected to this Playground. Once it’s online, choose “Check again.” You can explore image tokens in the meantime.</p><button className="play-secondary-button" type="button" disabled={busy} onClick={() => changeMode('local')}>Explore image tokens <ArrowRight size={14} aria-hidden="true" /></button></div> : isReady && <p className="play-connection-note">{targetNames[target]} selected · {activeHealth!.visualTokens} image tokens. Next, upload an image and ask a question.</p>}
      </div>
      {connections[target]?.error && <details className="play-connection-details"><summary>Connection details</summary><p>{connections[target]?.error}</p></details>}
      {activeHealth?.mode === 'fake' && <p className="play-connection-note">This service generates synthetic test responses. Live questions are disabled until a real model is loaded.</p>}
      {isReady && activeHealth && <details className="play-connection-details"><summary>Model settings</summary><p>{activeHealth.modelId ?? 'Connected model'} · important-token ratio {activeHealth.importantRatio} · output cap {activeHealth.maxNewTokens}</p></details>}
      {ready(activeHealth) && !validBaseline && <p className="play-connection-note">The baseline service must use 576 tokens. Update its server configuration and check the connection again before running.</p>}
    </section>}

    <div className="play-workspace">
      <section className="play-image-panel" aria-labelledby="play-image-title">
        <div className="play-panel-heading"><h2 id="play-image-title">1. Choose an image</h2><label className="play-overlay-toggle"><input type="checkbox" checked={overlay} onChange={(event) => setOverlay(event.target.checked)} />Token overlay</label></div>
        <ImagePicker image={image} onChange={setImage} disabled={busy} onLoadingChange={setImageLoading} />
        <div className="play-image-preview"><TokenImage image={image} tokens={previewTokens} overlay={overlay && (mode === 'local' || liveTokens !== null)} /></div>
        <p className="play-overlay-note">Illustrative patch selection. This overlay is not the model’s attention map or its actual retained tokens.</p>
        <div className="play-budget">{mode === 'local' ? <TokenBudget tokens={tokens} onChange={setTokens} disabled={busy} /> : <div className="play-live-budget"><span>Live image token budget</span><strong>{liveTokens === null ? 'Awaiting connection' : `${liveTokens} / 576`}</strong><p>{liveTokens === null ? 'The token budget appears when the selected model is ready.' : `Set by the ${targetNames[target]} service. Selecting another model updates this preview automatically.`}</p></div>}</div>
        <p className="play-budget-note">{mode === 'local' ? '576 tokens is the unpruned baseline. A smaller budget gives the model less visual information.' : 'To freely adjust the illustrative token budget, switch to Explore image tokens.'}</p>
      </section>

      <section className="play-composer-panel" aria-labelledby="play-question-title">
        <div className="play-panel-heading"><h2 id="play-question-title">{mode === 'local' ? '2. Prepare a question' : '2. Ask a question'}</h2></div>
        <form onSubmit={(event) => { event.preventDefault(); if (mode === 'local') changeMode('live', true); else void run(); }}>
          <label className="sr-only" htmlFor="play-question">Question about your image</label>
          <textarea id="play-question" value={question} disabled={busy} maxLength={2000} onChange={(event) => { setQuestion(event.target.value); setError(null); }} placeholder="What would you like to know about this image?" rows={4} />
          <div className="play-suggestions" aria-label="Suggested questions">{suggestions.map((suggestion) => <button type="button" key={suggestion} disabled={busy} onClick={() => { setQuestion(suggestion); setError(null); }}>{suggestion}</button>)}</div>
          <div className="play-submit-row">
            <button className="play-submit" type="submit" disabled={mode === 'local' ? busy : !canRun} aria-describedby="play-run-hint">{mode === 'local' ? <ArrowRight size={15} aria-hidden="true" /> : <Send size={15} aria-hidden="true" />}{mode === 'local' ? 'Continue to live model' : running ? 'Waiting for model…' : `Ask ${targetNames[target]}`}</button>
            {running && <button className="play-secondary-button" type="button" onClick={cancel}><Square size={12} aria-hidden="true" />Cancel</button>}
          </div>
          <p className="play-run-hint" id="play-run-hint">{mode === 'local' ? 'Getting an answer requires a live model. We’ll check for one next.' : running ? 'Your question is being processed.' : runHint}</p>
          <div className="play-request-status" aria-live="polite">
            {error && <p className="play-error" role="alert">{error} Check the connection and try again.</p>}
            {notice && <p>{notice}</p>}
            {running && <p>The full answer will appear when the model finishes. There is no token streaming.</p>}
          </div>
          {mode === 'live' && <p className="play-upload-notice">{image.base64 ? 'Running sends the uploaded image and question to the selected model service.' : 'Upload a JPG, PNG, or WebP to run a real question. The built-in illustration is for local exploration.'} Responses stay in this tab’s session.</p>}
        </form>

        {mode === 'local' ? <div className="play-empty-answer">
          <span className="play-small-label">Preview only · no AI answer</span>
          <h3>See what fewer tokens could look like.</h3>
          <p>Move the token slider to explore an illustrative selection of image patches. Your image stays in this browser.</p>
        </div> : <div className="play-live-guide">
          <h3>Compare answers at two budgets</h3>
          <p>Keep the image and question the same. Run on Baseline, then Pruned, using each service’s active token budget. Both answers will appear below.</p>
          <p>These are individual requests. Judge the answers yourself; this playground does not produce accuracy scores or benchmark speedups.</p>
        </div>}
      </section>
    </div>

    {history.length > 0 && <section className="play-history" aria-labelledby="play-history-title">
      <div className="play-history-heading"><div><h2 id="play-history-title" ref={historyHeading} tabIndex={-1}>Session results</h2><p>Latest first · up to 8 responses · cleared when you leave this page</p></div><button className="play-secondary-button" type="button" disabled={running} onClick={() => setHistory([])}>Clear results</button></div>
      <p className="play-history-note">Answers and metrics below came from the live service. Different model settings or background load can affect comparisons; use the <a href="#/october">October study</a> for controlled measurements.</p>
      <div className="play-answer-list">{history.map((entry) => <Answer key={entry.id} entry={entry} />)}</div>
    </section>}
  </div>;
}

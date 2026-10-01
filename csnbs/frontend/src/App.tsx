import { useEffect, useState } from 'react';
import { Moon, Sun } from 'lucide-react';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { parseCampaign } from './campaign';
import type { Campaign } from './campaign';
import RunArchive from './RunArchive';

type Percentile = 'p50' | 'p95' | 'p99';
type Theme = 'light' | 'dark' | 'system';
const metricNames = { p50: 'Median', p95: 'p95', p99: 'p99' };
const seconds = (value: number) => (value / 1000).toFixed(2);
const colors = { baseline: '#8792a2', pruned: '#3773d1', offered: '#a2aab5' };

function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem('gpu-profiling-theme');
    if (saved === 'light' || saved === 'dark') return saved;
  } catch { /* The page also works without local storage. */ }
  return 'system';
}

function Presentation({ data }: { data: Campaign }) {
  const [view, setView] = useState<'study' | 'archive'>('study');
  const [percentile, setPercentile] = useState<Percentile>('p50');
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [dark, setDark] = useState(false);
  const baseline = data.series.find((series) => series.visualTokenNum === 576)!;
  const pruned = data.series.find((series) => series.visualTokenNum === 128)!;
  const control = baseline.runs.find((run) => run.offeredRps === 3)!;
  const reduced = pruned.runs.find((run) => run.offeredRps === 3)!;
  const metric = `${percentile}LatencyMs` as const;
  // Display saved summaries. Only the latency unit is converted, from ms to s.
  const points = baseline.runs.map((run) => {
    const paired = pruned.runs.find((other) => other.offeredRps === run.offeredRps)!;
    return {
      rps: run.offeredRps,
      baselineLatency: run[metric] / 1000,
      prunedLatency: paired[metric] / 1000,
      baselineThroughput: run.successfulThroughputRps,
      prunedThroughput: paired.successfulThroughputRps,
    };
  });
  const runs = baseline.runs.flatMap((run) => [
    { ...run, tokens: 576 },
    { ...pruned.runs.find((other) => other.offeredRps === run.offeredRps)!, tokens: 128 },
  ]);

  useEffect(() => {
    const system = window.matchMedia('(prefers-color-scheme: dark)');
    const apply = () => {
      const isDark = theme === 'system' ? system.matches : theme === 'dark';
      setDark(isDark);
      document.documentElement.dataset.theme = isDark ? 'dark' : 'light';
    };
    apply();
    try { localStorage.setItem('gpu-profiling-theme', theme); } catch { /* Optional preference. */ }
    system.addEventListener('change', apply);
    return () => system.removeEventListener('change', apply);
  }, [theme]);

  const tickStyle = { fill: 'var(--muted)', fontSize: 13 };
  const tooltipStyle = { background: 'var(--paper)', border: '1px solid var(--line)', borderRadius: 6, color: 'var(--text)', fontSize: 14 };
  const chart = (throughput: boolean) => <div className="plot" role="img" aria-label={throughput ? 'Successful throughput against offered requests per second. Exact values are in Run details below.' : `${metricNames[percentile]} request latency against offered requests per second. Exact values are in Run details below.`}>
    <ResponsiveContainer width="100%" height="100%" minWidth={0}>
      <LineChart data={points} margin={{ top: 22, right: 18, bottom: 14, left: -12 }} accessibilityLayer>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="rps" type="number" domain={[0.5, 3]} ticks={[0.5, 1, 1.5, 2, 3]} tick={tickStyle} tickLine={false} axisLine={false} dy={12} />
        <YAxis tick={tickStyle} tickLine={false} axisLine={false} width={55} domain={throughput ? [0, 3] : [0, 'auto']} tickFormatter={(value: number) => value.toLocaleString('en-US', { maximumFractionDigits: 1 })} />
        <Tooltip contentStyle={tooltipStyle} labelFormatter={(value) => `${value} requests/s offered`} formatter={(value) => [`${Number(value).toFixed(3)} ${throughput ? 'requests/s' : 's'}`]} />
        {throughput && <Line dataKey="rps" name="Offered load" stroke={colors.offered} strokeDasharray="2 6" dot={false} isAnimationActive={false} />}
        <Line type="linear" dataKey={throughput ? 'baselineThroughput' : 'baselineLatency'} name="576 tokens" stroke={colors.baseline} strokeWidth={2.5} strokeDasharray="7 5" dot={{ r: 5, strokeWidth: 2, fill: 'var(--paper)' }} activeDot={{ r: 6 }} isAnimationActive={false} />
        <Line type="linear" dataKey={throughput ? 'prunedThroughput' : 'prunedLatency'} name="128 tokens" stroke={colors.pruned} strokeWidth={2.8} dot={{ r: 3.5, strokeWidth: 2, fill: colors.pruned }} activeDot={{ r: 6 }} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  </div>;

  return <div className="page">
    <a href="#main" className="skip-link">Skip to results</a>
    <header className="site-header">
      <a className="wordmark" href="#main">GPU <span>profiling</span></a>
      <div className="header-right"><nav className="view-options" aria-label="Benchmark views"><button aria-pressed={view === 'study'} onClick={() => setView('study')}>Historical study</button><button aria-pressed={view === 'archive'} onClick={() => setView('archive')}>Run archive</button></nav><button className="theme-button" onClick={() => setTheme(dark ? 'light' : 'dark')} aria-label={`Switch to ${dark ? 'light' : 'dark'} theme`} title={`Switch to ${dark ? 'light' : 'dark'} theme`}>{dark ? <Sun size={17} /> : <Moon size={17} />}</button></div>
    </header>

    <main id="main">
      {view === 'archive' ? <RunArchive /> : <>
      <section className="intro" aria-labelledby="page-title">
        <p className="study-label">LLaVA-1.5-7B / VisPruner</p>
        <h1 id="page-title">Visual token pruning<br /><span>under load.</span></h1>
        <p className="intro-copy">We kept 128 of the model’s 576 image tokens and measured latency and throughput as requests arrived.</p>
        <p className="setup-line">NVIDIA A40 <span>·</span> September 1, 2026 <span>·</span> Preliminary results</p>
      </section>

      <section className="figure-section" aria-labelledby="latency-title">
        <div className="figure-heading"><div><h2 id="latency-title">Request latency</h2><p>Time from sending a request to receiving its response, in seconds.</p></div><div className="metric-options" role="group" aria-label="Latency percentile">{(['p50', 'p95', 'p99'] as const).map((value) => <button key={value} aria-pressed={percentile === value} onClick={() => setPercentile(value)}>{metricNames[value]}</button>)}</div></div>
        <div className="plot-legend"><span><i className="baseline-key" />576 tokens</span><span><i className="pruned-key" />128 tokens</span></div>
        <figure>
          {chart(false)}
          <div className="axis-label">Offered load (requests / second)</div>
          <figcaption>At 3 requests/s, {percentile === 'p50' ? 'median' : percentile} latency was <strong>{seconds(control[metric])} s</strong> with 576 tokens and <strong>{seconds(reduced[metric])} s</strong> with 128.</figcaption>
        </figure>
        {percentile === 'p99' && <p className="tail-note">Only 5–80 requests per point. These p99 estimates are unstable; roughly 1,000+ samples are needed.</p>}
        <p className="figure-note">The question mix changes across arrival rates. These curves do not establish a capacity limit.</p>
      </section>

      <section className="figure-section" aria-labelledby="throughput-title">
        <div className="figure-heading"><div><h2 id="throughput-title">Successful throughput</h2><p>Completed requests per second.</p></div></div>
        <div className="plot-legend"><span><i className="baseline-key" />576 tokens</span><span><i className="pruned-key" />128 tokens</span><span><i className="offered-key" />Offered load</span></div>
        <figure>
          {chart(true)}
          <div className="axis-label">Offered load (requests / second)</div>
          <figcaption>At 3 requests/s, throughput was <strong>{control.successfulThroughputRps.toFixed(2)} requests/s</strong> with 576 tokens and <strong>{reduced.successfulThroughputRps.toFixed(2)} requests/s</strong> with 128.</figcaption>
        </figure>
        <p className="figure-note">The server processed one request at a time. Continuous batching was not measured.</p>
      </section>

      <section className="comparison-section" aria-labelledby="comparison-title">
        <div className="section-intro"><h2 id="comparison-title">The comparison so far</h2><p>Median latency reduction. Positive means faster.</p></div>
        <dl className="comparison-list">
          <div><dt>No pruning <span>576 tokens</span></dt><dd>0% <span>reference</span></dd></div>
          <div><dt>Stock VisPruner <span>no concurrent load</span></dt><dd className="unmeasured">Not measured</dd></div>
          <div><dt>Stock VisPruner <span>3 requests/s</span></dt><dd className="measured-result">{data.conditions[2].improvementPercent?.toFixed(1)}%</dd></div>
          <div><dt>Optimized VisPruner <span>under matched load</span></dt><dd className="unmeasured">Not measured</dd></div>
        </dl>
        <p className="comparison-caption">Pruning did not lower median latency in this run. With one run per setting, we can’t establish a speedup or a regression.</p>
        <p className="figure-note">An accuracy–throughput comparison still needs matched, validated accuracy results.</p>
      </section>

      <section className="notes-section" aria-labelledby="notes-title">
        <h2 id="notes-title">About these runs</h2>
        <div className="notes-copy">
          <p>Each point comes from one run with 30 seconds of scheduled arrivals. After the first ten requests were excluded, 5–80 successful requests remained. There are no confidence intervals, and the p99 estimates are especially noisy.</p>
          <p>The question mix changed with load: five OCR questions at 0.5 requests/s, and 80 questions across categories at 3 requests/s. Each 576/128-token pair used the same subset at its arrival rate.</p>
          <p>Batch size was 1, the output cap was 64 tokens, and the important-token ratio stayed at 0.5. Warmup requests were excluded by sequence rather than run as a separate phase. Some settings were recorded only in the campaign metadata.</p>
        </div>
        <details className="run-details">
          <summary>Run details <span aria-hidden="true">+</span></summary>
          <div className="details-body">
            <p>Revision <code>{data.campaign.gitCommit.slice(0, 12)}</code>{data.campaign.gitDirty ? ' · uncommitted changes present' : ''}. NVIDIA A40. All ten source-file hashes are checked when this page is built.</p>
            <div className="table-scroll" tabIndex={0} role="region" aria-label="Saved run measurements">
              <table><caption className="sr-only">Saved latency summaries in milliseconds. p99 values are unstable because of small samples.</caption><thead><tr><th scope="col">Tokens</th><th scope="col">Offered / s</th><th scope="col">Samples</th><th scope="col">p50 (ms)</th><th scope="col">p95 (ms)</th><th scope="col">p99 (ms)</th><th scope="col">Completed / s</th><th scope="col">Source</th></tr></thead><tbody>{runs.map((run) => <tr key={run.runId}><th scope="row">{run.tokens}</th><td>{run.offeredRps}</td><td>{run.successfulRequests}</td><td>{run.p50LatencyMs.toFixed(1)}</td><td>{run.p95LatencyMs.toFixed(1)}</td><td>{run.p99LatencyMs.toFixed(1)}</td><td>{run.successfulThroughputRps.toFixed(3)}</td><td><a href={`/data/runs/${run.runId}.json`} download aria-label={`Download ${run.tokens}-token run at ${run.offeredRps} requests per second`}>JSON</a></td></tr>)}</tbody></table>
            </div>
            <p>0 failed requests in these saved summaries. Workload identity:</p><code className="workload-hash">{data.campaign.workloadId}</code>
          </div>
        </details>
        <div className="downloads"><span>Downloads</span><a href="/plots/vispruner_under_load.svg" download>Load plots (SVG)</a><a href="/plots/vispruner_under_load_story.svg" download>Comparison (SVG)</a><a href="/plots/vispruner_under_load.png" download>Load plots (PNG)</a><a href="/plots/vispruner_under_load_story.png" download>Comparison (PNG)</a><a href="/data/campaign.json" download>Source data</a></div>
        <p className="download-note">The original figures use these same runs; the workload and warmup notes above also apply.</p>
      </section>
      </>}
    </main>
    <footer className="site-footer"><span>GPU Profiling</span><a href="#main">Back to top ↑</a></footer>
  </div>;
}

export default function App() {
  const [data, setData] = useState<Campaign | null>(null);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const abort = new AbortController();
    fetch('/data/campaign.json', { signal: abort.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error('Campaign unavailable');
        setData(parseCampaign(await response.json()));
      }).catch(() => { if (!abort.signal.aborted) setError(true); });
    return () => abort.abort();
  }, [attempt]);
  if (error) return <main className="page-state"><h1>Couldn’t load the results.</h1><button onClick={() => { setError(false); setAttempt(attempt + 1); }}>Try again</button></main>;
  if (!data) return <main className="page-state" aria-live="polite"><p>Loading results…</p></main>;
  return <Presentation data={data} />;
}

import { useEffect, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import PublicPreviewNote from './PublicPreviewNote';
import { parseCurrentStudy } from './current-study';
import type { CurrentStudy as CurrentStudyData } from './current-study';

const number = (value: number) => value.toLocaleString('en-US');
const seconds = (value: number) => (value / 1000).toFixed(2);
const measurementSource = 'https://github.com/KingCorsair/GPU_Profiling/blob/31c92e89606908002decffdd89a41d68c76891f5/csnbs/measure';

export function CurrentStudyView({ data }: { data: CurrentStudyData }) {
  const [rate, setRate] = useState<1 | 3>(3);
  const primary = data.primary;
  const throughput = primary.comparisons.find((comparison) => comparison.offeredRps === 3)!.throughput;
  const latency = primary.comparisons.find((comparison) => comparison.offeredRps === 1)!.latency;
  const trials = primary.trials.filter((trial) => trial.offeredRps === rate);
  const points = (tokens: number) => trials.filter((trial) => trial.visualTokenNum === tokens).map((trial) => ({
    pair: Number(trial.trialId.match(/-p(\d+)-/)![1]),
    throughput: trial.withinWindowThroughputRps,
  }));
  const tick = { fill: 'var(--muted)', fontSize: 12 };

  return <>
    <section className="intro current-intro" aria-labelledby="page-title">
      <p className="study-label">LLaVA-1.5-7B / VisPruner / October 1, 2026</p>
      <h1 id="page-title">Visual token pruning<br /><span>under load.</span></h1>
      <p className="intro-copy">LLaVA reads an image as a sequence of visual tokens. We tested whether reducing the budget from 576 to 128 makes responses faster and increases completed requests under load.</p>
      <p className="setup-line">{primary.gpu} <span>·</span> Batch size {primary.batchSize} <span>·</span> {primary.workloadRecords}-question dev workload</p>
      <p className="figure-note">Requests arrive independently while earlier requests can still be outstanding. The server processes one request at a time; matched accuracy evaluation is pending.</p>
    </section>

    <section className="figure-section" aria-labelledby="paired-results-title">
      <div className="figure-heading"><div><h2 id="paired-results-title">Primary comparison</h2><p>128 visual tokens relative to the 576-token baseline at 1 and 3 requests/s.</p></div></div>
      <div className="effect-grid">
        <article className="effect-card">
          <p className="effect-label">3 requests/s · primary outcome</p>
          <h3>More requests completed in 90 seconds</h3>
          <p className="effect-value">+{throughput.effectPercent.toFixed(2)}<span>%</span></p>
          <p>95% paired-block bootstrap interval: +{throughput.ci95[0].toFixed(2)}% to +{throughput.ci95[1].toFixed(2)}%.</p>
          <p className="effect-boundary">Within the 90-second window, the 128-token configuration completed about {throughput.effectPercent.toFixed(0)}% more requests (median paired change). Both configurations still had outstanding requests; sustainable capacity is not established.</p>
        </article>
        <article className="effect-card">
          <p className="effect-label">1 request/s · secondary outcome</p>
          <h3>Lower median response time</h3>
          <p className="effect-value">{latency.effectPercent.toFixed(2)}<span>%</span></p>
          <p>95% paired-block bootstrap interval: {latency.ci95[0].toFixed(2)}% to {latency.ci95[1].toFixed(2)}% reduction.</p>
          <p className="effect-boundary">HTTP latency measures time from sending a request to receiving its response. Both configurations completed the offered 1 request/s; equal throughput here does not establish equal capacity.</p>
        </article>
      </div>
      <p className="figure-note">Estimates are median paired percentage changes. Intervals use 10,000 bootstrap resamples of whole paired blocks. Five blocks limit uncertainty resolution; requests within a trial are not independent replications.</p>
      <dl className="study-totals" aria-label="All seven October campaigns">
        <div><dt>Campaigns</dt><dd>{number(data.totals.campaigns)}</dd></div>
        <div><dt>Completed trials</dt><dd>{number(data.totals.trials)}</dd></div>
        <div><dt>Measured requests</dt><dd>{number(data.totals.measuredRequests)}</dd></div>
        <div><dt>Request failures</dt><dd>{number(data.totals.failedMeasuredRequests)}</dd></div>
      </dl>
      <p className="figure-note">Collection totals cover seven campaigns, plus {number(data.totals.warmupRequests)} warmup requests. The primary comparison contains {primary.trials.length} trials.</p>
    </section>

    <section className="figure-section measurement-section" aria-labelledby="measurement-title">
      <h2 id="measurement-title">Measurement approach</h2>
      <dl className="measurement-list">
        <div><dt>Independent arrivals</dt><dd>Requests are scheduled independently of response times, so a slow response does not reduce the intended load.</dd></div>
        <div><dt>Paired comparisons</dt><dd>Each pair uses the same checkpoint, question order and decoding settings. Five pairs are run at each rate.</dd></div>
        <div><dt>Traceable results</dt><dd>Reports are checked against saved request records, with source revisions, settings and file hashes retained.</dd></div>
      </dl>
      <details className="archive-provenance current-disclosure measurement-explanation">
        <summary><span className="disclosure-label">Why schedule requests independently?</span></summary>
        <p className="figure-note">A client that waits for each response sends less traffic when the server slows down. That can make an overloaded server appear faster by leaving out requests that should have arrived. The harness keeps an independent arrival schedule and records dispatch lateness separately.</p>
      </details>
      <div className="downloads"><a href={`${measurementSource}/README.md`}>Measurement code and reproduction guide ↗</a><a href={`${measurementSource}/OCT01_PROTOCOL.md`}>Read the fixed study protocol ↗</a></div>
    </section>

    <section className="figure-section trials-section" aria-labelledby="current-throughput-title">
      <div className="figure-heading">
        <div><h2 id="current-throughput-title">Trial-level throughput</h2><p>Successful completions per second within the arrival window. Each point represents one trial; completions during subsequent draining are excluded.</p></div>
        <div className="metric-options trial-rate-options" role="group" aria-label="Offered request rate">
          {([1, 3] as const).map((value) => <button key={value} aria-pressed={rate === value} onClick={() => setRate(value)}>{value} request{value === 1 ? '' : 's'}/s</button>)}
        </div>
      </div>
      <div className="plot-legend"><span><i className="scatter-baseline-key" aria-hidden="true" />576 tokens</span><span><i className="scatter-pruned-key" aria-hidden="true" />128 tokens</span><span><i className="offered-key" aria-hidden="true" />Offered load</span></div>
      <figure>
        <div className="plot" role="img" aria-label={`Within-window completions per second for five paired trials at ${rate} requests per second. Exact values and outstanding work are in the table below.`}>
          <ResponsiveContainer width="100%" height="100%" minWidth={0}>
            <ScatterChart margin={{ top: 24, right: 25, bottom: 18, left: 4 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis type="number" dataKey="pair" name="Paired block" domain={[0.5, 5.5]} ticks={[1, 2, 3, 4, 5]} tick={tick} tickLine={false} axisLine={false} />
              <YAxis type="number" dataKey="throughput" name="Within-window completions" unit=" req/s" domain={[0, 3.2]} ticks={[0, 1, 2, 3]} tick={tick} tickLine={false} axisLine={false} width={76} />
              <ReferenceLine y={rate} stroke="var(--chart-offered)" strokeDasharray="3 6" />
              <Tooltip contentStyle={{ background: 'var(--paper)', border: '1px solid var(--line)', borderRadius: 6, color: 'var(--text)', fontSize: 13 }} formatter={(value, name) => [name === 'Paired block' ? value : Number(value).toFixed(3), name]} />
              <Scatter name="576 tokens" data={points(576)} fill="var(--chart-baseline)" shape="square" isAnimationActive={false} />
              <Scatter name="128 tokens" data={points(128)} fill="var(--chart-pruned)" shape="circle" isAnimationActive={false} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="axis-label">Paired trial block</div>
        <figcaption>{rate === 3 ? 'Pruning increased completions, but both configurations accumulated outstanding work.' : 'Both configurations completed all prescribed arrivals inside the window. Their throughput points overlap at 1 request/s.'}</figcaption>
      </figure>
      <details className="run-details current-disclosure trial-disclosure">
        <summary><ChevronDown className="trial-chevron" size={16} aria-hidden="true" /><span className="disclosure-label">Trial measurements · {trials.length} trials at {rate} request{rate === 1 ? '' : 's'}/s</span></summary>
        <div className="details-body">
          <p>Each trial has {primary.measuredRequestsPerTrial} measured requests and {primary.warmupsPerTrial} completed warmups. Drain is additional time after the arrival window. Latencies below are conditional on successful requests.</p>
          <div className="table-scroll" role="region" tabIndex={0} aria-label="October primary trial measurements">
            <table>
              <caption className="sr-only">All primary trials at the selected offered rate. Latencies and drain are in seconds.</caption>
              <thead><tr><th scope="col">Trial</th><th scope="col">Tokens</th><th scope="col">Window (s)</th><th scope="col">Completed / s</th><th scope="col">Outstanding</th><th scope="col">Drain (s)</th><th scope="col">p50 (s)</th><th scope="col">p95 (s)</th><th scope="col">Raw run</th></tr></thead>
              <tbody>{trials.map((trial) => <tr key={trial.runId}>
                <th scope="row">{trial.trialId}</th><td>{trial.visualTokenNum}</td><td>{trial.measurementWindowSeconds}</td><td>{trial.withinWindowThroughputRps.toFixed(3)}</td><td>{trial.outstandingAtWindowEnd}</td><td>{seconds(trial.drainMs)}</td><td>{seconds(trial.latencyMs.p50)}</td><td>{trial.latencyMs.p95 === null ? 'Unavailable' : seconds(trial.latencyMs.p95)}</td><td>{trial.downloadPath ? <a href={trial.downloadPath} download aria-label={`Download ${trial.trialId} raw run JSON`}>JSON ↓</a> : 'Local only'}</td>
              </tr>)}</tbody>
            </table>
          </div>
          <p>p99 is unavailable: each trial has fewer than 1,000 observations. Client outstanding work includes transport and service; it is not a direct server-queue measurement.</p>
        </div>
      </details>
    </section>

    <section className="comparison-section" aria-labelledby="accuracy-title">
      <div className="section-intro"><p className="study-label">Accuracy · pending</p><h2 id="accuracy-title">Matched accuracy evaluation</h2><p>Matched, validated accuracy results are not yet available. The accuracy–throughput comparison remains pending.</p></div>
      <div className="handoff-grid">
        <div><h3>Required evaluation evidence</h3><p>Validated results for the {data.accuracy.requiredConfigurationCount} measured configurations require a locked evaluation split, a validated scorer, per-category scores and random-control evidence.</p></div>
        <div><h3>Configuration matching</h3><p>Accuracy scores must match the execution identities of the serving trials before inclusion in the joint analysis. Accuracy points are withheld until these requirements are met.</p></div>
      </div>
      {data.accuracy.download && <div className="downloads"><a href={data.accuracy.download.downloadPath} download>Download required execution identities (JSON) ↓</a></div>}
    </section>

    <section className="notes-section" aria-labelledby="current-notes-title">
      <h2 id="current-notes-title">Scope and limitations</h2>
      <div className="notes-copy">
        <p>This is the recorded, instrumented, batch-one serving path. Continuous batching, sustainable capacity, matched accuracy and a causal GPU prefill/decode breakdown remain unestablished.</p>
        <p>The primary revision did not record actual output lengths. In a separate follow-up, longer instructions produced a median of one generated text token; that study did not establish long-output behavior.</p>
      </div>
      <details className="archive-provenance current-disclosure"><summary><span className="disclosure-label">Study settings and provenance</span></summary>
        <dl className="provenance-grid">
          <div><dt>Model</dt><dd>{primary.modelId}</dd></div><div><dt>Retained tokens</dt><dd>576 → 128</dd></div><div><dt>Important-token ratio</dt><dd>{primary.importantRatio}</dd></div>
          <div><dt>Output cap</dt><dd>{primary.maxNewTokens} tokens, natural EOS</dd></div><div><dt>Batch size</dt><dd>{primary.batchSize}</dd></div>{primary.gitCommit && <div><dt>Source commit</dt><dd><code>{primary.gitCommit}</code></dd></div>}
          <div><dt>Checkpoint revision</dt><dd><code>{primary.checkpointRevision}</code></dd></div><div><dt>Workload identity</dt><dd><code>{primary.workloadId}</code></dd></div>{data.downloads.report && <div><dt>Primary report SHA-256</dt><dd><code>{data.downloads.report.sha256}</code></dd></div>}
        </dl>
        <ul className="evidence-notes">{primary.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul>
      </details>
      <details className="archive-provenance current-disclosure"><summary><span className="disclosure-label">All {data.totals.campaigns} October campaigns</span></summary><div className="table-scroll" tabIndex={0} role="region" aria-label="October campaign collection"><table><thead><tr><th scope="col">Campaign</th><th scope="col">Trials</th><th scope="col">Measured requests</th><th scope="col">Report</th></tr></thead><tbody>{data.campaigns.map((campaign) => <tr key={campaign.campaignId}><th scope="row">{campaign.campaignId}</th><td>{campaign.trialCount}</td><td>{number(campaign.measuredRequests)}</td><td>{campaign.report ? <a href={campaign.report.downloadPath} download>JSON ↓</a> : 'Local only'}</td></tr>)}</tbody></table></div></details>
      <>
        <div className="downloads"><span>Source artifacts</span>{([
          ['reportMarkdown', 'Primary report (Markdown)'], ['report', 'Primary report (JSON)'],
          ['plotSvg', 'Original figure (SVG)'], ['plotPng', 'Original figure (PNG)'], ['collection', 'Collection index (JSON)'],
        ] as const).map(([key, label]) => { const item = data.downloads[key]; return item && <a key={key} href={item.downloadPath} download>{label}</a>; })}</div>
        <p className="download-note">Downloads preserve the original source bytes. The original figure’s distribution panels illustrate only the first recorded pair at 3 requests/s; the interactive plot above shows every trial at the selected rate.</p>
        {data.publicPreview && <PublicPreviewNote />}
      </>
    </section>
  </>;
}

export default function CurrentStudy() {
  const [data, setData] = useState<CurrentStudyData | null>(null);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const abort = new AbortController();
    fetch('/data/current-study.json', { signal: abort.signal }).then(async (response) => {
      if (!response.ok) throw new Error('October study unavailable');
      setData(parseCurrentStudy(await response.json()));
    }).catch(() => { if (!abort.signal.aborted) setError(true); });
    return () => abort.abort();
  }, [attempt]);
  if (error) return <section className="page-state"><h1>Couldn’t load the October results.</h1><p>The saved evidence is missing or incompatible.</p><button onClick={() => { setError(false); setAttempt(attempt + 1); }}>Try again</button></section>;
  if (!data) return <section className="page-state" aria-live="polite"><p>Loading October results…</p></section>;
  return <CurrentStudyView data={data} />;
}

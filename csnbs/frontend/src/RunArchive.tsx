import { useEffect, useMemo, useState } from 'react';
import { comparisonWarnings, parseArchive } from './archive';
import type { ArchivedRun, RunArchive as Archive } from './archive';

const value = (number: number | null, digits = 1) => number === null ? 'Unavailable' : number.toLocaleString('en-US', { maximumFractionDigits: digits });
const date = (timestamp: string | null) => timestamp ? timestamp.replace('T', ' ').replace(/\.\d+Z$/, ' UTC') : 'Date not recorded';
const evidence = { exploratory: 'Exploratory', integration: 'Integration only', eligible: 'Run checks passed', review: 'Needs review' };

function RunDetails({ run }: { run: ArchivedRun }) {
  return <div className="archive-detail">
    <div className="detail-heading"><div><p className="study-label">Selected run · {evidence[run.evidence]}</p><h2>{run.modelId ?? 'Model not recorded'}</h2></div><a href={run.downloadPath} download>Download run JSON ↓</a></div>
    <p className="archive-run-id">{run.runId}</p>
    <dl className="provenance-grid">
      <div><dt>Recorded</dt><dd>{date(run.recordedAtUtc)}</dd></div>
      <div><dt>Run kind / status</dt><dd>{run.runKind} / {run.status ?? 'Legacy'}</dd></div>
      <div><dt>GPU</dt><dd>{run.gpuModels.join(', ') || 'Not recorded'}</dd></div>
      <div><dt>Offered load</dt><dd>{run.runKind === 'isolated' ? 'Sequential (no arrival schedule)' : `${value(run.offeredRps, 3)} requests/s`}</dd></div>
      <div><dt>Visual tokens</dt><dd>{value(run.visualTokenNum, 0)}{run.tokenSource === 'campaign' ? ' (campaign metadata)' : ''}</dd></div>
      <div><dt>Output cap / batch size</dt><dd>{value(run.maxOutputTokens, 0)} / {value(run.batchSize, 0)}</dd></div>
      <div><dt>Source revision</dt><dd><code>{run.gitCommit?.slice(0, 12) ?? 'Not recorded'}</code>{run.gitDirty === true ? ' · modified' : run.gitDirty === false ? ' · clean' : ''}</dd></div>
    </dl>
    <div className="table-scroll" role="region" aria-label="Saved timing summaries" tabIndex={0}>
      <table><caption>Saved timing summaries, milliseconds</caption><thead><tr><th scope="col">Measurement</th><th scope="col">p50</th><th scope="col">p95</th><th scope="col">p99</th></tr></thead><tbody>{[
        ['Send to response', run.summary.latencyMs],
        ['Dispatch lateness', run.summary.dispatchLatenessMs],
        ['Scheduled to completion', run.summary.scheduledToCompleteMs],
        ['Server service', run.summary.serverServiceMs],
        ['Server queue', run.summary.serverQueueMs],
      ].map(([label, metrics]) => typeof metrics !== 'string' && <tr key={String(label)}><th scope="row">{String(label)}</th><td>{value(metrics.p50)}</td><td>{value(metrics.p95)}</td><td>{value(metrics.p99)}</td></tr>)}</tbody></table>
    </div>
    <p className="figure-note">Send-to-response excludes client dispatch delay. Scheduled-to-completion includes it. Server service and queue values appear only when separately reported by the server. Missing or sample-limited percentiles remain unavailable.</p>
    {run.schemaVersion === 2 && <dl className="provenance-grid">
      <div><dt>Completed within arrival window / s</dt><dd>{value(run.summary.withinWindowThroughputRps, 3)}</dd></div>
      <div><dt>Completed including drain / s</dt><dd>{value(run.summary.includingDrainThroughputRps, 3)}</dd></div>
      <div><dt>Outstanding at window end</dt><dd>{value(run.summary.outstandingAtWindowEnd, 0)}</dd></div>
      <div><dt>Drain duration</dt><dd>{value(run.summary.drainMs)} ms</dd></div>
    </dl>}
    <ul className="evidence-notes">{run.notes.map((note, index) => <li key={index}>{note}</li>)}</ul>
    <details className="archive-provenance"><summary>Full provenance and request records</summary><dl>
      <dt>Campaign</dt><dd>{run.campaign ?? 'Not assigned'}</dd>
      <dt>Workload identity</dt><dd><code>{run.workloadId ?? 'Not recorded'}</code></dd>
      <dt>Measured question order</dt><dd><code>{run.measuredOrderHash ?? 'Not recorded'}</code></dd>
      <dt>Checkpoint revision</dt><dd><code>{run.checkpointRevision ?? 'Not recorded'}</code></dd>
      <dt>Run source</dt><dd><code>{run.sourcePath}</code></dd>
      <dt>Run SHA-256</dt><dd><code>{run.sha256}</code></dd>
      {run.requestDownloadPath && <><dt>Request records</dt><dd><a href={run.requestDownloadPath} download>Download original JSONL ↓</a><code>{run.requestsSha256}</code></dd></>}
    </dl></details>
  </div>;
}

function Comparison({ runs }: { runs: ArchivedRun[] }) {
  const [left, right] = runs;
  if (!left || !right) return null;
  const rows: [string, string, string][] = [
    ['Evidence', evidence[left.evidence], evidence[right.evidence]],
    ['Model', left.modelId ?? 'Not recorded', right.modelId ?? 'Not recorded'],
    ['Visual tokens', value(left.visualTokenNum, 0), value(right.visualTokenNum, 0)],
    ['Offered requests/s', value(left.offeredRps, 3), value(right.offeredRps, 3)],
    ['Successful samples', value(left.summary.successfulRequests, 0), value(right.summary.successfulRequests, 0)],
    ['Failed requests', value(left.summary.failedRequests, 0), value(right.summary.failedRequests, 0)],
    ['Completed requests/s', value(left.summary.successfulThroughputRps, 3), value(right.summary.successfulThroughputRps, 3)],
    ...(['p50', 'p95', 'p99'] as const).map((key): [string, string, string] => [`${key} latency (ms)`, value(left.summary.latencyMs[key]), value(right.summary.latencyMs[key])]),
  ];
  return <section className="archive-comparison" aria-labelledby="saved-comparison-title"><h2 id="saved-comparison-title">Saved summaries side by side</h2><p className="figure-note">Descriptive comparison. No speedup or significance is inferred.</p><div className="table-scroll" tabIndex={0} role="region" aria-label="Two selected run summaries"><table><thead><tr><th scope="col">Metric</th><th scope="col"><code>{left.runId}</code></th><th scope="col"><code>{right.runId}</code></th></tr></thead><tbody>{rows.map(([label, a, b]) => <tr key={label}><th scope="row">{label}</th><td>{a}</td><td>{b}</td></tr>)}</tbody></table></div><ul className="evidence-notes">{comparisonWarnings(left, right).map((warning) => <li key={warning}>{warning}</li>)}</ul></section>;
}

export default function RunArchive() {
  const [data, setData] = useState<Archive | null>(null);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [query, setQuery] = useState('');
  const [evidenceFilter, setEvidenceFilter] = useState('all');
  const [campaignFilter, setCampaignFilter] = useState('all');
  const [selected, setSelected] = useState<string | null>(null);
  const [comparison, setComparison] = useState<string[]>([]);
  useEffect(() => {
    const abort = new AbortController();
    fetch('/data/archive.json', { signal: abort.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error('Archive unavailable');
        setData(parseArchive(await response.json()));
      }).catch(() => { if (!abort.signal.aborted) setError(true); });
    return () => abort.abort();
  }, [attempt]);
  const filtered = useMemo(() => (data?.runs ?? []).filter((run) =>
    (evidenceFilter === 'all' || run.evidence === evidenceFilter) &&
    (campaignFilter === 'all' || (run.campaign ?? 'unassigned') === campaignFilter) &&
    [run.runId, run.modelId, run.recordedAtUtc, run.gitCommit, ...run.gpuModels].join(' ').toLowerCase().includes(query.toLowerCase().trim())), [data, query, evidenceFilter, campaignFilter]);
  const selectedRun = data?.runs.find((run) => run.sha256 === selected);
  const comparedRuns = comparison.flatMap((sha256) => data?.runs.filter((run) => run.sha256 === sha256) ?? []);
  function toggleCompare(sha256: string) {
    setComparison((current) => current.includes(sha256) ? current.filter((item) => item !== sha256) : current.length < 2 ? [...current, sha256] : current);
  }
  return <section className="archive-section" aria-labelledby="archive-title">
    <div className="intro archive-intro"><p className="study-label">Measurement workspace</p><h1 id="archive-title">Run archive.</h1><p className="intro-copy">Inspect the saved evidence behind every run. Compare recorded summaries, check provenance, and download the original request records.</p><p className="figure-note">Accuracy–throughput remains pending until matched, validated accuracy and controlled serving measurements are available.</p></div>
    {error ? <div className="archive-empty" role="alert"><p>Couldn’t load the run archive.</p><button onClick={() => { setError(false); setAttempt(attempt + 1); }}>Try again</button></div> : !data ? <p aria-live="polite">Loading saved runs…</p> : <>
      <div className="archive-filters"><label>Search runs<input type="search" placeholder="Model, run ID, GPU, date, commit…" value={query} onChange={(event) => setQuery(event.target.value)} /></label><label>Evidence<select value={evidenceFilter} onChange={(event) => setEvidenceFilter(event.target.value)}><option value="all">All evidence</option><option value="exploratory">Exploratory</option><option value="integration">Integration only</option><option value="eligible">Run checks passed</option><option value="review">Needs review</option></select></label><label>Campaign<select value={campaignFilter} onChange={(event) => setCampaignFilter(event.target.value)}><option value="all">All campaigns</option>{[...new Set(data.runs.flatMap((run) => run.campaign ? [run.campaign] : []))].map((campaign) => <option key={campaign} value={campaign}>{campaign}</option>)}<option value="unassigned">Not assigned</option></select></label></div>
      <div className="archive-count"><p aria-live="polite">{filtered.length} of {data.runs.length} saved runs · select two to compare</p>{comparison.length > 0 && <button onClick={() => setComparison([])}>Clear comparison ({comparison.length}/2)</button>}</div>
      {filtered.length ? <div className="table-scroll archive-table" tabIndex={0} role="region" aria-label="Saved run archive"><table><caption className="sr-only">Saved run summaries. p99 estimates with fewer than 1,000 successful samples are unstable.</caption><thead><tr><th scope="col">Compare</th><th scope="col">Run / model</th><th scope="col">Evidence</th><th scope="col">Offered / s</th><th scope="col">Success / fail</th><th scope="col">p50 (ms)</th><th scope="col">p95 (ms)</th><th scope="col">p99 (ms)</th><th scope="col">Completed / s</th></tr></thead><tbody>{filtered.map((run) => <tr key={run.sha256} data-selected={selected === run.sha256}><td><input type="checkbox" checked={comparison.includes(run.sha256)} disabled={comparison.length === 2 && !comparison.includes(run.sha256)} onChange={() => toggleCompare(run.sha256)} aria-label={`Compare run ${run.runId}`} /></td><th scope="row"><button className="run-select" aria-expanded={selected === run.sha256} aria-controls="selected-run-details" onClick={() => setSelected(selected === run.sha256 ? null : run.sha256)}>{run.modelId ?? 'Model not recorded'}</button><small>{run.recordedAtUtc?.slice(0, 10) ?? 'Unknown date'} · {run.visualTokenNum === null ? run.runKind : `${run.visualTokenNum} visual tokens`}</small><small>{run.runId}</small></th><td><span className={`evidence-badge ${run.evidence}`}>{evidence[run.evidence]}</span></td><td>{value(run.offeredRps, 3)}</td><td>{run.summary.successfulRequests} / {run.summary.failedRequests}</td><td>{value(run.summary.latencyMs.p50)}</td><td>{value(run.summary.latencyMs.p95)}</td><td>{value(run.summary.latencyMs.p99)}{run.summary.successfulRequests < 1000 && <small>{run.schemaVersion === 1 ? 'Unstable tail' : 'Insufficient samples'}</small>}</td><td>{value(run.summary.successfulThroughputRps, 3)}</td></tr>)}</tbody></table></div> : <div className="archive-empty"><p>No saved runs match these filters.</p><button onClick={() => { setQuery(''); setEvidenceFilter('all'); setCampaignFilter('all'); }}>Reset filters</button></div>}
      <Comparison runs={comparedRuns} />
      <div id="selected-run-details">{selectedRun && <RunDetails run={selectedRun} />}</div>
      {data.issues.length > 0 && <div className="archive-issues" role="status"><h2>Files needing review</h2><p className="figure-note">These files were not included in the archive. Unsupported or incomplete evidence is never silently converted into a benchmark result.</p><ul className="evidence-notes">{data.issues.map((issue) => <li key={issue.sourcePath}><code>{issue.sourcePath}</code>: {issue.message}</li>)}</ul></div>}
      <p className="download-note">Snapshot generated {date(data.generatedAtUtc)}. Refresh the saved evidence with the dashboard’s data-sync or build command. The archive displays saved summaries; it does not recompute percentiles or statistical comparisons.</p>
    </>}
  </section>;
}

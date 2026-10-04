import { useEffect, useMemo, useRef, useState } from 'react';
import { comparisonWarnings, parseArchive } from './archive';
import type { ArchivedRun, RunArchive as Archive } from './archive';
import WorkloadDetails from './WorkloadDetails';
import CampaignReports from './CampaignReports';
import ArchiveSelect from './ArchiveSelect';
import PublicPreviewNote from './PublicPreviewNote';
import { archiveLocationHash, parseArchiveLocation } from './archive-routing';
import type { ArchiveLocation } from './archive-routing';

const value = (number: number | null, digits = 1) => number === null ? 'Unavailable' : number.toLocaleString('en-US', { maximumFractionDigits: digits });
const date = (timestamp: string | null) => timestamp ? timestamp.replace('T', ' ').replace(/\.\d+Z$/, ' UTC') : 'Date not recorded';
const evidence = { exploratory: 'Exploratory', integration: 'Integration only', eligible: 'Run checks passed', review: 'Needs review' };

export function RunDetails({ run }: { run: ArchivedRun }) {
  const isolated = run.runKind === 'isolated';
  return <div className="archive-detail">
    <div className="detail-heading"><div><p className="study-label">Selected run · {evidence[run.evidence]}</p><h2 id="selected-run-title">{run.modelId ?? 'Model not recorded'}</h2></div>{run.downloadPath && <a href={run.downloadPath} download>Download run JSON ↓</a>}</div>
    <p className="archive-run-id">{run.runId}</p>
    <dl className="provenance-grid">
      <div><dt>Recorded</dt><dd>{date(run.recordedAtUtc)}</dd></div>
      <div><dt>Run kind / status</dt><dd>{run.runKind} / {run.status ?? 'Legacy'}</dd></div>
      <div><dt>GPU</dt><dd>{run.gpuModels.join(', ') || 'Not recorded'}</dd></div>
      <div><dt>Offered load</dt><dd>{run.runKind === 'isolated' ? 'Sequential (no arrival schedule)' : `${value(run.offeredRps, 3)} requests/s`}</dd></div>
      <div><dt>Configured visual-token budget</dt><dd>{value(run.visualTokenNum, 0)}{run.tokenSource === 'campaign' ? ' (campaign metadata)' : ''}</dd></div>
      <div><dt>Output cap / batch size</dt><dd>{value(run.maxOutputTokens, 0)} / {value(run.batchSize, 0)}</dd></div>
      <div><dt>Source revision</dt><dd><code>{run.gitCommit?.slice(0, 12) ?? 'Not recorded'}</code>{run.gitDirty === true ? ' · modified' : run.gitDirty === false ? ' · clean' : ''}</dd></div>
    </dl>
    <div className="table-scroll" role="region" aria-label="Saved timing summaries" tabIndex={0}>
      <table><caption>Saved timing summaries, milliseconds</caption><thead><tr><th scope="col">Measurement</th><th scope="col">p50</th><th scope="col">p95</th><th scope="col">p99</th></tr></thead><tbody>{[
        ['Send to response', run.summary.latencyMs],
        ...(!isolated ? [['Dispatch lateness', run.summary.dispatchLatenessMs], ['Scheduled to completion', run.summary.scheduledToCompleteMs]] : []),
        ['Server service', run.summary.serverServiceMs],
        ['Server queue', run.summary.serverQueueMs],
      ].map(([label, metrics]) => typeof metrics !== 'string' && <tr key={String(label)}><th scope="row">{String(label)}</th><td>{value(metrics.p50)}</td><td>{value(metrics.p95)}</td><td>{value(metrics.p99)}</td></tr>)}</tbody></table>
    </div>
    <p className="figure-note">{isolated ? 'Each isolated request starts after the previous response completes. There is no independent arrival schedule or offered request rate.' : 'Send-to-response excludes client dispatch delay. Scheduled-to-completion includes it.'} Server service and queue values appear only when separately reported by the server. Missing or sample-limited percentiles remain unavailable.</p>
    {run.schemaVersion === 2 && <dl className="provenance-grid">
      {!isolated && <div><dt>Completed within arrival window / s</dt><dd>{value(run.summary.withinWindowThroughputRps, 3)}</dd></div>}
      <div><dt>{isolated ? 'Serial completions including drain / s' : 'Completed incl. drain / s'}</dt><dd>{value(run.summary.includingDrainThroughputRps, 3)}</dd></div>
      <div><dt>{isolated ? 'Outstanding at measurement end' : 'Outstanding at window end'}</dt><dd>{value(run.summary.outstandingAtWindowEnd, 0)}</dd></div>
      <div><dt>Drain duration</dt><dd>{value(run.summary.drainMs)} ms</dd></div>
    </dl>}
    <WorkloadDetails data={run.workloadCharacterization} />
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
    ['Request mode', left.runKind, right.runKind],
    ['Configured visual-token budget', value(left.visualTokenNum, 0), value(right.visualTokenNum, 0)],
    ['Offered requests/s', left.runKind === 'isolated' ? 'Not applicable (sequential)' : value(left.offeredRps, 3), right.runKind === 'isolated' ? 'Not applicable (sequential)' : value(right.offeredRps, 3)],
    ['Successful samples', value(left.summary.successfulRequests, 0), value(right.summary.successfulRequests, 0)],
    ['Failed requests', value(left.summary.failedRequests, 0), value(right.summary.failedRequests, 0)],
    [left.schemaVersion === 2 && right.schemaVersion === 2 ? 'Completed incl. drain / s' : 'Saved completion rate / s', value(left.summary.successfulThroughputRps, 3), value(right.summary.successfulThroughputRps, 3)],
    ['Completion-rate boundary', left.schemaVersion === 2 ? 'Includes drain' : 'Legacy saved rate', right.schemaVersion === 2 ? 'Includes drain' : 'Legacy saved rate'],
    ...(['p50', 'p95', 'p99'] as const).map((key): [string, string, string] => [`${key} latency (ms)`, value(left.summary.latencyMs[key]), value(right.summary.latencyMs[key])]),
  ];
  return <section className="archive-comparison" aria-labelledby="saved-comparison-title"><h2 id="saved-comparison-title">Saved summaries side by side</h2><p className="figure-note">Descriptive comparison. No speedup or significance is inferred.</p><div className="table-scroll" tabIndex={0} role="region" aria-label="Two selected run summaries"><table><thead><tr><th scope="col">Metric</th><th scope="col"><code>{left.runId}</code></th><th scope="col"><code>{right.runId}</code></th></tr></thead><tbody>{rows.map(([label, a, b]) => <tr key={label}><th scope="row">{label}</th><td>{a}</td><td>{b}</td></tr>)}</tbody></table></div><ul className="evidence-notes">{comparisonWarnings(left, right).map((warning) => <li key={warning}>{warning}</li>)}</ul></section>;
}

export default function RunArchive() {
  const [data, setData] = useState<Archive | null>(null);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [location, setLocation] = useState(() => parseArchiveLocation(window.location.hash));
  const { query, evidence: evidenceFilter, campaign: campaignFilter, run: selected, compare: comparison } = location;
  const detailsPanel = useRef<HTMLDivElement>(null);
  const comparisonPanel = useRef<HTMLDivElement>(null);
  const runTable = useRef<HTMLDivElement>(null);
  const pendingFocus = useRef<'details' | 'comparison' | 'table' | null>(selected ? 'details' : comparison.length === 2 ? 'comparison' : null);
  useEffect(() => {
    const restore = () => {
      if (window.location.hash.split('?')[0] !== '#/archive') return;
      const next = parseArchiveLocation(window.location.hash);
      pendingFocus.current = next.run ? 'details' : next.compare.length === 2 ? 'comparison' : null;
      setLocation(next);
    };
    window.addEventListener('hashchange', restore);
    return () => window.removeEventListener('hashchange', restore);
  }, []);
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
  const campaigns = [...new Set((data?.runs ?? []).flatMap((run) => run.campaign ? [run.campaign] : []))];
  const missingCampaign = data !== null && campaignFilter !== 'all' && campaignFilter !== 'unassigned' && !campaigns.includes(campaignFilter);
  const selectedRun = data?.runs.find((run) => run.runId === selected);
  const comparedRuns = comparison.flatMap((runId) => data?.runs.filter((run) => run.runId === runId) ?? []);
  const missingRuns = data ? [selected, ...comparison].filter((runId): runId is string => Boolean(runId) && !data.runs.some((run) => run.runId === runId)) : [];
  function reveal(target: 'details' | 'comparison' | 'table') {
    const element = (target === 'details' ? detailsPanel : target === 'comparison' ? comparisonPanel : runTable).current;
    if (!element) return;
    element.focus({ preventScroll: true });
    element.scrollIntoView({ block: 'start', behavior: 'instant' });
  }
  useEffect(() => {
    if (!data || !pendingFocus.current) return;
    reveal(pendingFocus.current);
    pendingFocus.current = null;
  }, [data, location]);
  function updateLocation(change: Partial<ArchiveLocation>, replace = false, focus?: 'details' | 'comparison' | 'table') {
    const next = { ...location, ...change };
    const hash = archiveLocationHash(next);
    if (window.location.hash !== hash) {
      if (replace) window.history.replaceState(null, '', hash);
      else window.history.pushState(null, '', hash);
    }
    pendingFocus.current = focus ?? null;
    setLocation(next);
  }
  function toggleCompare(runId: string) {
    const next = comparison.includes(runId) ? comparison.filter((item) => item !== runId) : comparison.length < 2 ? [...comparison, runId] : comparison;
    updateLocation({ compare: next }, false, next.length === 2 ? 'comparison' : undefined);
  }
  return <section className="archive-section" aria-labelledby="archive-title">
    <div className="intro archive-intro"><p className="study-label">Measurement workspace</p><h1 id="archive-title">Run archive.</h1><p className="intro-copy">Inspect saved run summaries and compare recorded measurements across configurations.</p><p className="figure-note">The October serving comparison is complete. The joint accuracy–throughput analysis still requires matching, validated accuracy results.</p></div>
    {error ? <div className="archive-empty" role="alert"><p>Couldn’t load the run archive.</p><button onClick={() => { setError(false); setAttempt(attempt + 1); }}>Try again</button></div> : !data ? <p aria-live="polite">Loading saved runs…</p> : <>
      <div className="archive-filters">
        <label>Search runs<input type="search" placeholder="Model, run ID, GPU, date, commit…" value={query} onChange={(event) => updateLocation({ query: event.target.value }, true)} /></label>
        <ArchiveSelect label="Evidence" value={evidenceFilter} onChange={(evidence) => updateLocation({ evidence }, true)} options={[
          { value: 'all', label: 'All evidence' },
          ...Object.entries(evidence).map(([value, label]) => ({ value, label })),
        ]} />
        <ArchiveSelect label="Campaign" value={campaignFilter} onChange={(campaign) => updateLocation({ campaign }, true)} options={[
          { value: 'all', label: 'All campaigns' },
          ...(missingCampaign ? [{ value: campaignFilter, label: `Unavailable: ${campaignFilter}` }] : []),
          ...campaigns.map((campaign) => ({ value: campaign, label: campaign })),
          { value: 'unassigned', label: 'Not assigned' },
        ]} />
      </div>
      {missingCampaign && <div className="archive-count" role="status"><p>The campaign “{campaignFilter}” requested by this link is not available in this archive.</p><button onClick={() => updateLocation({ campaign: 'all' }, true)}>Clear campaign filter</button></div>}
      <div className="archive-count"><p aria-live="polite">{filtered.length} of {data.runs.length} saved runs · {comparison.length === 0 ? 'select two to compare' : `${comparison.length} of 2 selected for comparison`}</p>{comparison.length > 0 && <button onClick={() => updateLocation({ compare: [] }, false, 'table')}>Clear comparison selection</button>}</div>
      {missingRuns.length > 0 && <p className="figure-note" role="status">This archive does not contain {missingRuns.length === 1 ? 'a run' : 'some runs'} requested by the link. Choose a saved run below.</p>}
      {comparedRuns.length === 2 && <div className="archive-reveal" ref={comparisonPanel} tabIndex={-1} role="region" aria-labelledby="saved-comparison-title">
        <div className="archive-panel-actions"><button aria-label="Back to run list from comparison" onClick={() => reveal('table')}>Back to run list ↓</button><button onClick={() => updateLocation({ compare: [] }, false, 'table')}>Clear comparison</button></div>
        <Comparison runs={comparedRuns} />
      </div>}
      {selectedRun && <div id="selected-run-details" className="archive-reveal" ref={detailsPanel} tabIndex={-1} role="region" aria-labelledby="selected-run-title">
        <div className="archive-panel-actions"><button aria-label="Back to run list from run details" onClick={() => reveal('table')}>Back to run list ↓</button><button onClick={() => updateLocation({ run: null }, false, 'table')}>Close run details</button></div>
        <RunDetails run={selectedRun} />
      </div>}
      <p className="figure-note" id="archive-throughput-note">V2 completion rates include drain time. Rows marked “Legacy rate” preserve the original saved throughput; check the run notes before comparing timing boundaries.</p>
      <div ref={runTable} tabIndex={-1} className="archive-reveal" role="group" aria-label="Run list">
        {filtered.length ? <div className="table-scroll archive-table" tabIndex={0} role="region" aria-label="Saved run archive" aria-describedby="archive-throughput-note"><table><caption className="sr-only">Saved run summaries. p99 estimates with fewer than 1,000 successful samples are unstable.</caption><thead><tr><th scope="col">Compare</th><th scope="col">Run / model</th><th scope="col">Evidence</th><th scope="col">Offered / s</th><th scope="col">Success / fail</th><th scope="col">p50 (ms)</th><th scope="col">p95 (ms)</th><th scope="col">p99 (ms)</th><th scope="col">Completed incl. drain / s</th></tr></thead><tbody>{filtered.map((run) => <tr key={run.sha256} data-selected={selected === run.runId}>
          <td><input type="checkbox" checked={comparison.includes(run.runId)} disabled={comparison.length === 2 && !comparison.includes(run.runId)} onChange={() => toggleCompare(run.runId)} aria-label={`Compare run ${run.runId}`} /></td>
          <th scope="row"><span className="archive-run-model">{run.modelId ?? 'Model not recorded'}</span><button className="run-select" aria-label={`${selected === run.runId ? 'Hide' : 'View'} details for run ${run.runId}`} aria-expanded={selected === run.runId} aria-controls={selected === run.runId ? 'selected-run-details' : undefined} onClick={() => updateLocation({ run: selected === run.runId ? null : run.runId }, false, selected === run.runId ? undefined : 'details')}>{selected === run.runId ? 'Hide details' : 'View details'} →</button><small>{run.recordedAtUtc?.slice(0, 10) ?? 'Unknown date'} · {run.visualTokenNum === null ? run.runKind : `${run.visualTokenNum} visual-token budget`}</small><small>{run.runId}</small></th>
          <td><span className={`evidence-badge ${run.evidence}`}>{evidence[run.evidence]}</span></td><td>{run.runKind === 'isolated' ? 'Sequential' : value(run.offeredRps, 3)}</td><td>{run.summary.successfulRequests} / {run.summary.failedRequests}</td><td>{value(run.summary.latencyMs.p50)}</td><td>{value(run.summary.latencyMs.p95)}</td><td>{value(run.summary.latencyMs.p99)}{run.summary.successfulRequests < 1000 && <small>{run.schemaVersion === 1 ? 'Unstable tail' : 'Insufficient samples'}</small>}</td><td>{value(run.summary.successfulThroughputRps, 3)}{run.schemaVersion === 1 && <small>Legacy rate</small>}</td>
        </tr>)}</tbody></table></div> : <div className="archive-empty"><p>No saved runs match these filters.</p><button onClick={() => updateLocation({ query: '', evidence: 'all', campaign: 'all' }, true)}>Reset filters</button></div>}
      </div>
      <CampaignReports reports={data.campaignReports.filter((report) => campaignFilter === 'all' || campaignFilter === report.campaignId)} />
      {data.issues.length > 0 && <div className="archive-issues" role="status"><h2>Files needing review</h2><p className="figure-note">These files were not included in the archive. Unsupported or incomplete evidence is never silently converted into a benchmark result.</p><ul className="evidence-notes">{data.issues.map((issue) => <li key={issue.sourcePath}><code>{issue.sourcePath}</code>: {issue.message}</li>)}</ul></div>}
      {data.publicPreview && <PublicPreviewNote />}
      <p className="download-note">Snapshot generated {date(data.generatedAtUtc)}. The archive displays saved summaries; it does not recompute percentiles or statistical comparisons.</p>
    </>}
  </section>;
}

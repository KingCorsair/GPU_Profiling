import { useId, useState } from 'react';
import { ArrowDown, ArrowRight, ArrowUpRight, ChevronDown, Code2, Layers3, Activity, ScanLine } from 'lucide-react';
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import { demoCategories, demoRates, illustrativeStudy } from './demo-data';
import type { DemoPath } from './demo-data';
import TokenImageDemo from './TokenImageDemo';

const repo = 'https://github.com/KingCorsair/GPU_Profiling';
const recordedSource = `${repo}/tree/31c92e89606908002decffdd89a41d68c76891f5`;
const tick = { fill: 'var(--muted)', fontSize: 12 };
const tooltip = { background: 'var(--paper)', border: '1px solid var(--line)', borderRadius: 8, color: 'var(--text)', fontSize: 14 };

function ExplorerSelector<T extends string | number>({ value, options, onChange, compact = false }: {
  value: T;
  options: { value: T; label: string; caption?: string }[];
  onChange: (value: T) => void;
  compact?: boolean;
}) {
  const name = useId();
  return <div className={`segmented-control${compact ? ' scenario-control' : ''}`}>
    {options.map((option) => <label className="segmented-option" key={option.value}>
      <input className="sr-only" type="radio" name={name} value={option.value} checked={value === option.value} onChange={() => onChange(option.value)} />
      <span className="segmented-option-label">{option.label}{option.caption && <span className="segmented-option-caption">{option.caption}</span>}</span>
    </label>)}
  </div>;
}

export default function ProjectOverview() {
  const [tokens, setTokens] = useState(128);
  const [rate, setRate] = useState(3);
  const [path, setPath] = useState<DemoPath>('reference');
  const [showSampleValues, setShowSampleValues] = useState(false);
  const configs = illustrativeStudy.configurations;
  const selected = configs.find((config) => config.tokens === tokens)!;
  const baseline = configs[0];
  const rateIndex = demoRates.indexOf(rate as typeof demoRates[number]);
  const readings = selected[path];
  const frontier = configs.map((config) => ({ tokens: config.tokens, throughput: config[path].throughput[rateIndex], accuracy: config.accuracy }));
  const latency = demoRates.map((rps, index) => ({ rps, baseline: baseline[path].p95[index] / 1000, selected: readings.p95[index] / 1000 }));
  const jumpToExplorer = () => { const target = document.getElementById('image-demo-title'); target?.focus({ preventScroll: true }); target?.scrollIntoView({ behavior: 'auto', block: 'start' }); };

  return <>
    <section className="project-hero" aria-labelledby="page-title">
      <div className="hero-copy">
        <p className="project-label">LLaVA-1.5-7B / VisPruner</p>
        <h1 id="page-title">Visual token pruning<br /><span>under load</span></h1>
        <p className="intro-copy">LLaVA represents an image with 576 visual tokens. We use VisPruner to keep fewer of them, then measure how this changes response times, throughput, and answer quality.</p>
        <div className="hero-actions"><button className="primary-action" onClick={jumpToExplorer}>Try the image demo <ArrowDown size={16} /></button><a className="text-action" href="#/playground">Open Playground <ArrowRight size={16} /></a></div>
        <p className="hero-context">October campaign complete. Matched accuracy and batching comparisons are still pending.</p>
      </div>
      <div className="token-illustration" aria-label="Schematic of 576 image tokens with 128 highlighted. This is not a measured importance map." role="img">
        <div className="token-visual-label"><ScanLine size={17} /><span>Image representation</span><span>24 × 24</span></div>
        <div className="token-grid" aria-hidden="true">{Array.from({ length: 576 }, (_, index) => <i key={index} className={(index * 137) % 576 < 128 ? 'kept' : ''} />)}</div>
        <div className="token-visual-footer"><span><b>576</b> original</span><ArrowRight size={18} /><span><b>128</b> retained</span></div>
        <p>Schematic only · not an attention map</p>
      </div>
    </section>

    <div className="project-question-strip"><p><strong>Research question</strong> How much of the speedup from pruning survives concurrent requests?</p><p><strong>Final comparison</strong> Accuracy against measured throughput for matching model settings.</p></div>

    <TokenImageDemo />

    <section className="explorer-section" aria-labelledby="explorer-title">
      <div className="section-heading-row"><div><h2 id="explorer-title" tabIndex={-1}>Results preview</h2></div></div>
      <p className="section-description">Change the token budget, arrival rate, and serving path to explore the planned results view.</p>
      <p className="demo-notice"><strong>Sample data.</strong> These values are made up for the preview. <a href="#/october">View measured October results <ArrowUpRight size={14} /></a></p>

      <div className="explorer-controls">
        <fieldset><legend>Image tokens retained</legend><ExplorerSelector value={tokens} onChange={setTokens} options={configs.map(({ tokens: value }) => ({ value, label: String(value), caption: value === 576 ? 'baseline' : undefined }))} /><p>Only this budget changes; the importance ratio stays fixed.</p></fieldset>
        <fieldset><legend>Arrival rate</legend><ExplorerSelector value={rate} onChange={setRate} options={[1, 3, 5].map((value) => ({ value, label: String(value), caption: 'req/s' }))} /><p>Requests arrive independently of response times.</p></fieldset>
        <fieldset><legend>Serving path</legend><ExplorerSelector<DemoPath> value={path} onChange={setPath} compact options={[{ value: 'reference', label: 'Batch size 1' }, { value: 'proposed', label: 'Batching (planned)' }]} /><p>Both paths use sample data.</p></fieldset>
      </div>

      <div className="demo-stat-strip" aria-live="polite" aria-atomic="true">
        <div><span>Sample completions / second</span><strong>{readings.throughput[rateIndex].toFixed(1)}<small>req/s</small></strong></div>
        <div><span>Sample accuracy</span><strong>{selected.accuracy}<small>%</small></strong></div>
        <div><span>Sample p95 response time</span><strong>{(readings.p95[rateIndex] / 1000).toFixed(2)}<small>s</small></strong></div>
        <div><span>Image tokens removed</span><strong>{((576 - tokens) / 576 * 100).toFixed(0)}<small>%</small></strong></div>
      </div>

      <div className="explorer-plots">
        <figure className="explorer-chart"><div className="chart-title-row"><h3>Accuracy vs. throughput</h3><span className="fixture-label">Sample data</span></div><p className="chart-subtitle">Each point is one token budget. The outlined point is your selection.</p>
          <div className="plot frontier-plot" role="img" aria-label={`Synthetic accuracy versus throughput at ${rate} offered requests per second. Selected ${tokens} tokens: ${readings.throughput[rateIndex]} completions per second, ${selected.accuracy}% accuracy. Exact demo values are available below.`}>
            <ResponsiveContainer width="100%" height="100%" minWidth={0}><ScatterChart margin={{ top: 25, right: 23, bottom: 5, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} /><XAxis dataKey="throughput" type="number" domain={[0, 5.2]} ticks={[0, 1, 2, 3, 4, 5]} tick={tick} tickLine={false} axisLine={false} /><YAxis dataKey="accuracy" type="number" domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} unit="%" width={48} tick={tick} tickLine={false} axisLine={false} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }) => { const point = payload?.[0]?.payload as typeof frontier[number] | undefined; return active && point ? <div className="demo-tooltip"><b>{point.tokens} tokens · demo</b><span>{point.throughput} req/s · {point.accuracy}% accuracy</span></div> : null; }} />
              <Scatter data={frontier} fill="var(--chart-baseline)" shape="square" isAnimationActive={false} />
              <Scatter data={frontier.filter((point) => point.tokens === tokens)} fill="var(--chart-pruned)" shape={(props: { cx?: number; cy?: number }) => <circle cx={props.cx} cy={props.cy} r={8} fill="var(--chart-pruned)" stroke="var(--text)" strokeWidth={2} />} isAnimationActive={false} />
            </ScatterChart></ResponsiveContainer>
          </div><div className="axis-label">Successful completions / second (demo)</div><figcaption>Compare completed requests per second with accuracy at the selected arrival rate.</figcaption>
        </figure>
        <figure className="explorer-chart"><div className="chart-title-row"><h3>Response time as traffic rises</h3><span className="fixture-label">Sample data</span></div><p className="chart-subtitle">p95: 95% of successful responses finish within this time.</p>
          <div className="plot-legend compact-legend"><span><i className="baseline-key" />576-token baseline</span><span><i className="pruned-key" />{tokens} tokens selected</span></div>
          <div className="plot latency-demo-plot" role="img" aria-label={`Synthetic p95 latency across offered traffic for baseline and ${tokens} tokens. Exact values are in the demo table below.`}>
            <ResponsiveContainer width="100%" height="100%" minWidth={0}><LineChart data={latency} margin={{ top: 20, right: 23, bottom: 5, left: 0 }} accessibilityLayer>
              <CartesianGrid stroke="var(--grid)" vertical={false} /><XAxis dataKey="rps" tick={tick} tickLine={false} axisLine={false} /><YAxis unit=" s" tick={tick} tickLine={false} axisLine={false} width={48} /><ReferenceLine x={rate} stroke="var(--chart-offered)" strokeDasharray="3 6" />
              <Tooltip contentStyle={tooltip} labelFormatter={(value) => `${value} req/s · synthetic data`} formatter={(value, name) => [`${Number(value).toFixed(2)} s`, name]} />
              <Line dataKey="baseline" name="576-token baseline" stroke="var(--chart-baseline)" strokeWidth={2} strokeDasharray="6 5" dot={{ r: 4 }} isAnimationActive={false} /><Line dataKey="selected" name={`${tokens} tokens selected`} stroke="var(--chart-pruned)" strokeWidth={2.5} dot={{ r: 4 }} isAnimationActive={false} />
            </LineChart></ResponsiveContainer>
          </div><div className="axis-label">Offered requests / second (demo)</div><figcaption>Compare the selected token budget with the 576-token baseline as arrivals increase.</figcaption>
        </figure>
      </div>

      <div className="category-section"><div><h3>Accuracy by category</h3><p>OCR, counting, and spatial reasoning need their own scores. A single average can hide a weak category.</p><span className="fixture-label">Sample category scores</span></div><div className="category-bars"><div className="category-bars-heading"><span>Task</span><span>576 tokens → {tokens} tokens</span></div>{demoCategories.map((category, index) => <div className="category-row" key={category}><span>{category}</span><div className="category-track" aria-hidden="true"><i style={{ width: `${baseline.categories[index]}%` }} /><b style={{ width: `${selected.categories[index]}%` }} /></div><span>{baseline.categories[index]}% → <strong>{selected.categories[index]}%</strong></span></div>)}</div></div>
      <div className="run-details demo-table">
        <button type="button" className="sample-values-toggle" aria-expanded={showSampleValues} aria-controls="sample-values-panel" onClick={() => setShowSampleValues((shown) => !shown)}>
          <span>{showSampleValues ? 'Hide sample values' : 'View all sample values'}</span>
          <span className="sample-values-hint">{configs.length * demoRates.length} rows <ChevronDown size={16} aria-hidden="true" /></span>
        </button>
        <div id="sample-values-panel" className="details-body" hidden={!showSampleValues}>
          <p>Sample values for the selected serving path, across all token budgets and arrival rates.</p>
          <div className="table-scroll" tabIndex={0} role="region" aria-label="Synthetic explorer values">
            <table><caption className="sr-only">Invented data for the selected serving scenario, across all token budgets and rates.</caption><thead><tr><th scope="col">Tokens</th><th scope="col">Offered / s</th><th scope="col">Demo completed / s</th><th scope="col">Demo accuracy</th><th scope="col">Demo p50 (s)</th><th scope="col">Demo p95 (s)</th></tr></thead><tbody>{configs.flatMap((config) => demoRates.map((rps, index) => <tr key={`${config.tokens}-${rps}`}><th scope="row">{config.tokens}</th><td>{rps}</td><td>{config[path].throughput[index].toFixed(1)}</td><td>{config.accuracy}%</td><td>{(config[path].p50[index] / 1000).toFixed(2)}</td><td>{(config[path].p95[index] / 1000).toFixed(2)}</td></tr>))}</tbody></table>
          </div>
        </div>
      </div>
    </section>

    <section className="project-method" aria-labelledby="method-title"><h2 id="method-title">How we measure</h2><p className="section-description">Performance and accuracy must refer to the same model and pruning settings. We keep the records behind each claim, so a chart can be traced back to its experiment.</p>
      <ol className="pipeline"><li><ScanLine /><span>01</span><h3>Represent the image</h3><p>LLaVA’s vision encoder produces 576 patch tokens.</p></li><li><Layers3 /><span>02</span><h3>Keep fewer tokens</h3><p>VisPruner selects image information before the language model. The 576-token setting is the unpruned baseline.</p></li><li><Activity /><span>03</span><h3>Measure under traffic</h3><p>Schedule independent HTTP arrivals, warm up first, and repeat matched trials with saved request records.</p></li><li><Code2 /><span>04</span><h3>Join with accuracy</h3><p>Match validated, per-category scores to the serving execution. This final join is pending.</p></li></ol>
      <div className="method-detail-grid"><article><h3>Independent arrivals</h3><p>A client that waits for each answer sends less traffic when the server slows down. This can hide queueing delays, a problem called coordinated omission.</p><a className="text-action" href={`${repo}/blob/31c92e89606908002decffdd89a41d68c76891f5/results/calibration/2026-10-01/coordinated-omission-fixed-10000/report.md`}>Our coordinated-omission experiment <ArrowUpRight size={15} /></a></article><article><h3>Throughput and capacity</h3><p>Finishing more requests during a short window is useful evidence. It does not prove a server can sustain that rate indefinitely, especially if requests are still outstanding.</p><a className="text-action" href="#/october">October results and limitations <ArrowRight size={15} /></a></article></div>
    </section>

    <section className="project-progress" aria-labelledby="progress-title"><div className="section-heading-row"><div><h2 id="progress-title">Measurements so far</h2></div><a className="text-action" href="#/archive">Browse saved runs <ArrowRight size={16} /></a></div><p className="section-description">These studies contain recorded measurements. Their workloads and protocols differ; they are milestones, not a controlled before-and-after comparison.</p>
      <div className="milestone-grid"><a className="milestone" href="#/september"><span className="milestone-status">Exploratory · September 1</span><h3>Initial measurements</h3><p>One run per setting compared 576 and 128 tokens. Small samples and changing question mixes limited the conclusions.</p><span className="text-action">Read September study <ArrowRight size={16} /></span></a><a className="milestone" href="#/october"><span className="milestone-status measured-status">Measured · October 1</span><h3>Repeated, paired trials</h3><p>Seven campaigns with saved evidence, uncertainty intervals, and explicit boundaries on what the results establish.</p><span className="text-action">Read October study <ArrowRight size={16} /></span></a><article className="milestone milestone-pending"><span className="milestone-status">Pending · next evidence</span><h3>Accuracy and serving changes</h3><p>Matched accuracy with validated scoring, followed by controlled measurements of verified serving changes.</p><span className="milestone-footnote">No measured joint curve yet</span></article></div>
    </section>

    <section className="project-next" aria-labelledby="next-title"><h2 id="next-title">Remaining experiments</h2><div className="research-questions"><article><span>Implementation</span><h3>Masking vs. real removal</h3><p>Compare explicitly pinned paths. FastV’s masking and in-place removal implementations must not be conflated.</p></article><article><span>Workload</span><h3>Prefill vs. decode</h3><p>Measure actual output lengths and GPU stage boundaries to understand when reduced input work matters.</p></article><article><span>Serving</span><h3>Batching and memory</h3><p>Test padding, packed or bucketed execution, and KV-cache allocation. October measured a batch-one path.</p></article><article><span>Quality</span><h3>Accuracy by category</h3><p>Use a locked split, validated scoring, and a random control before concluding which information can be discarded.</p></article></div></section>
    <section className="project-source" aria-label="Project source"><div><h2>Source code and records</h2><p>Protocols, source code, recorded results, and the remaining questions.</p></div><a className="primary-action" href={recordedSource}>Open repository <ArrowUpRight size={17} /></a></section>
  </>;
}

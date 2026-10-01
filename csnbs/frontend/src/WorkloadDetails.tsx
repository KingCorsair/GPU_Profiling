import type { WorkloadCharacterization } from './archive';

export default function WorkloadDetails({ data }: { data: WorkloadCharacterization | null }) {
  if (!data) return <p className="figure-note">Workload characterization is unavailable without the original request records.</p>;
  return <section className="workload-details" aria-labelledby="workload-details-title">
    <h3 id="workload-details-title">What this run actually measured</h3>
    {data.scope === 'measurement-phase' ? <>
      <p className="figure-note">{data.measuredRecords} measured request records; {data.successfulRecords} successful. Warmups are excluded. Counts describe the saved traffic, including repeated questions.</p>
      <div className="workload-grid">{[
        ['Task category', data.categoryCounts], ['Source dataset', data.sourceDatasetCounts],
      ].map(([name, rows]) => typeof rows !== 'string' && <table key={String(name)}><caption>{String(name)}</caption><thead><tr><th scope="col">Label in saved records</th><th scope="col">Requests</th></tr></thead><tbody>{rows.map((row) => <tr key={row.label}><th scope="row">{row.label}</th><td>{row.count}</td></tr>)}</tbody></table>)}</div>
      <div className="table-scroll" role="region" aria-label="Recorded response and token lengths" tabIndex={0}><table><caption>Actual length observations from successful responses</caption><thead><tr><th scope="col">Measurement</th><th scope="col">Recorded responses</th><th scope="col">Minimum</th><th scope="col">Maximum</th></tr></thead><tbody>{[
        [`Output characters (${data.outputCharacters.unit})`, data.outputCharacters],
        ['Generated text tokens', data.actualTokens.generated_text_tokens],
        ['Text prompt tokens (server-reported)', data.actualTokens.prompt_text_tokens],
        ['Visual tokens', data.actualTokens.visual_tokens],
      ].map(([name, observation]) => typeof observation !== 'string' && <tr key={String(name)}><th scope="row">{String(name)}</th><td>{observation.observedCount} / {data.successfulRecords}</td><td>{observation.min ?? 'Unavailable'}</td><td>{observation.max ?? 'Unavailable'}</td></tr>)}</tbody></table></div>
      {data.outputCharacters.reason && <p className="figure-note">Output character lengths: {data.outputCharacters.reason}</p>}
    </> : <p className="figure-note">The saved legacy records cannot establish measurement-phase category or source proportions.</p>}
    <ul className="evidence-notes">{data.limitations.map((note) => <li key={note}>{note}</li>)}</ul>
    <p className="figure-note">Still needed from the serving path: verified actual text/visual token counts when absent, CUDA-event prefill/decode timing, actual batch occupancy and padded/packed work, and exact KV-cache allocation. Accuracy comparisons require the accuracy owner’s matched scores and validated scorer. A longer output cap alone does not measure longer-output sensitivity.</p>
  </section>;
}

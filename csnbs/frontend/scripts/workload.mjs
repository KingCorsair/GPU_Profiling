// Descriptive counts from saved records only. Never reconstruct missing tokens
// from characters, configured budgets, or an unrelated dataset on this machine.
const object = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const nonnegativeInteger = (value) => Number.isSafeInteger(value) && value >= 0;
const label = (value) => typeof value === 'string' && value.trim() ? value.trim() : 'Not recorded';
const range = (values) => ({ observedCount: values.length, min: values.reduce((a, b) => Math.min(a, b), Infinity), max: values.reduce((a, b) => Math.max(a, b), -Infinity) });

export function characterizeRecords(raw, records) {
  const result = {
    scope: raw.schemaVersion === 2 ? 'measurement-phase' : 'unavailable',
    measuredRecords: null,
    successfulRecords: null,
    categoryCounts: [],
    sourceDatasetCounts: [],
    outputCharacters: { observedCount: 0, min: null, max: null, unit: 'Unicode code points', reason: 'No output character counts were recorded by the server; saved answers are unavailable.' },
    actualTokens: Object.fromEntries(['generated_text_tokens', 'prompt_text_tokens', 'visual_tokens'].map((key) => [key, { observedCount: 0, min: null, max: null }])),
    limitations: [],
  };
  if (raw.schemaVersion !== 2) {
    result.limitations.push('Legacy request records do not identify the measurement phase or preserve category/source metadata. No workload distribution is inferred.');
    return result;
  }
  if (!Array.isArray(records) || records.some((record) => !object(record) || !['measurement', 'warmup'].includes(record.phase))) throw new Error('V2 workload characterization requires explicit request phases');
  const measured = records.filter((record) => record.phase === 'measurement');
  if (measured.length !== raw.summary.totalRequests) throw new Error('V2 measured request count differs from the saved summary');
  result.measuredRecords = measured.length;
  const successful = measured.filter((record) => record.outcome === 'success');
  if (successful.length !== raw.summary.successfulRequests) throw new Error('V2 successful request count differs from the saved summary');
  result.successfulRecords = successful.length;
  const counts = (field) => {
    const values = new Map();
    for (const record of measured) {
      const key = label(record[field]);
      values.set(key, (values.get(key) ?? 0) + 1);
    }
    return [...values].sort(([a], [b]) => a.localeCompare(b)).map(([label, count]) => ({ label, count }));
  };
  result.categoryCounts = counts('category');
  result.sourceDatasetCounts = counts('sourceDataset');
  // Optional explicit instrumentation only. Output caps, token counts, and
  // transport bytes cannot substitute for server-recorded character counts.
  const characters = successful.flatMap((record) => nonnegativeInteger(record.serverMetrics?.output_characters) ? [record.serverMetrics.output_characters] : []);
  if (characters.length) result.outputCharacters = { ...range(characters), unit: 'Unicode code points', reason: characters.length < successful.length ? 'Only some successful responses include a recorded character count.' : null };
  for (const key of Object.keys(result.actualTokens)) {
    const values = successful.flatMap((record) => nonnegativeInteger(record.serverMetrics?.[key]) ? [record.serverMetrics[key]] : []);
    if (values.length) result.actualTokens[key] = range(values);
  }
  result.limitations.push('Category and source counts include every measured outcome, including failed and cancelled requests. Warmups are excluded.');
  result.limitations.push('Length observations describe successful responses only. Missing observations remain unavailable; output caps are not actual lengths.');
  result.limitations.push('Characters are not model tokens. Text prompt tokens, generated text tokens and visual tokens require separate server measurements.');
  result.limitations.push('This describes the saved workload, not a workload-sensitivity experiment or broad population coverage.');
  return result;
}

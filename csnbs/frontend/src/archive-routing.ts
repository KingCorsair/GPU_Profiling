export type ArchiveLocation = {
  run: string | null;
  compare: string[];
  query: string;
  evidence: string;
  campaign: string;
};

const evidenceValues = new Set(['all', 'exploratory', 'integration', 'eligible', 'review']);

export function parseArchiveLocation(hash: string): ArchiveLocation {
  const separator = hash.indexOf('?');
  const path = separator === -1 ? hash : hash.slice(0, separator);
  const query = separator === -1 ? '' : hash.slice(separator + 1);
  const params = new URLSearchParams(path === '#/archive' ? query : '');
  const evidence = params.get('evidence') ?? 'all';
  return {
    run: params.get('run') || null,
    compare: [...new Set((params.get('compare') ?? '').split(',').filter(Boolean))].slice(0, 2),
    query: params.get('q') ?? '',
    evidence: evidenceValues.has(evidence) ? evidence : 'all',
    campaign: params.get('campaign') || 'all',
  };
}

export function archiveLocationHash(location: ArchiveLocation): string {
  const params = new URLSearchParams();
  if (location.run) params.set('run', location.run);
  if (location.compare.length) params.set('compare', location.compare.join(','));
  if (location.query) params.set('q', location.query);
  if (location.evidence !== 'all') params.set('evidence', location.evidence);
  if (location.campaign !== 'all') params.set('campaign', location.campaign);
  const query = params.toString();
  return `#/archive${query ? `?${query}` : ''}`;
}

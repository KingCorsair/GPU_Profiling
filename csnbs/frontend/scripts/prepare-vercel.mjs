// Package only the privacy-filtered build for `vercel deploy --prebuilt`.
import assert from 'node:assert/strict';
import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const source = new URL('../dist-share/', import.meta.url);
const output = new URL('../.vercel/output/', import.meta.url);
assert((await readFile(new URL('index.html', source), 'utf8')).includes('name="demo-mode" content="showcase"'), 'Personal Vercel sharing requires the showcase build; use a separate hosting project for live.');
for (const name of ['current-study', 'archive', 'campaign']) {
  const data = JSON.parse(await readFile(new URL(`data/${name}.json`, source), 'utf8'));
  assert.equal(data.publicPreview, true, 'Vercel requires the privacy-filtered sharing build.');
}

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
await cp(source, new URL('static/', output), { recursive: true });
await writeFile(new URL('config.json', output), `${JSON.stringify({
  version: 3,
  routes: [{ src: '/.*', headers: { 'X-Robots-Tag': 'noindex, nofollow' }, continue: true }],
}, null, 2)}\n`);
console.log(`Vercel static output prepared: ${fileURLToPath(output)}`);

import { readFile, writeFile, rename, rm } from 'node:fs/promises';
import { loadPublicCorpus } from './content/build.mjs';
import { auditArtifact, prepareCorpus, syncPilot } from './knowledge/phase2a.mjs';
import { chatProvider } from './knowledge/model.mjs';

const output = new URL('../content/related.json', import.meta.url);
const ctx = prepareCorpus(await loadPublicCorpus());
let previous;
try {
  previous = JSON.parse(await readFile(output, 'utf8'));
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
}
if (process.argv.includes('--check')) {
  auditArtifact(ctx, previous);
  console.log('Verified recommendation snapshot', ctx.corpusHash);
} else {
  const snapshot = await syncPilot(ctx, chatProvider(), previous);
  // A publisher may pull or edit posts while model calls are in flight.
  const latest = prepareCorpus(await loadPublicCorpus());
  auditArtifact(latest, snapshot);
  const tmp = new URL(output.href + '.' + process.pid + '.tmp');
  try {
    await writeFile(tmp, JSON.stringify(snapshot, null, 2) + '\n');
    await rename(tmp, output);
  } finally {
    await rm(tmp, { force: true });
  }
  for (const [id, entry] of Object.entries(snapshot.entries)) {
    console.log(
      id,
      snapshot.facts[id].status,
      entry.status,
      entry.recommendations.map((r) => r.id).join(', '),
    );
  }
  if (
    [...Object.values(snapshot.entries), ...Object.values(snapshot.facts)].some(
      (r) => r.status === 'fallback',
    )
  )
    process.exitCode = 2;
}

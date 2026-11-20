import { mkdir, readFile, readdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

export const VERSION = 2;
const ID = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
// Older than any single write; a younger temp file may belong to a sync running in parallel.
const STALE_TEMP_MS = 10 * 60 * 1000;
let sequence = 0;

export function assertId(id) {
  if (typeof id !== 'string' || !ID.test(id))
    throw new Error('Article ID cannot be used as a file name: ' + id);
}

/**
 * Every ID must be a file name, and no two may differ by case alone: on a case-insensitive
 * volume (macOS, Windows) they would be one file, and the two publishing servers would disagree.
 */
export function assertIds(ids) {
  const seen = new Map();
  for (const id of ids) {
    assertId(id);
    const folded = id.toLowerCase();
    if (seen.has(folded))
      throw new Error(`Article IDs differ only by case and would share one file: ${seen.get(folded)}, ${id}`);
    seen.set(folded, id);
  }
}

/** Fixed key order so that the same result is the same bytes on both publishing servers. */
export function serialize(record) {
  const facts = record.facts && {
    inputHash: record.facts.inputHash,
    status: record.facts.status,
    facts: record.facts.facts,
    review: record.facts.review || [],
  };
  const entry = record.entry && {
    inputHash: record.entry.inputHash,
    status: record.entry.status,
    ...(record.entry.fallbackFrom ? { fallbackFrom: record.entry.fallbackFrom } : {}),
    candidates: record.entry.candidates,
    recommendations: record.entry.recommendations,
  };
  const ordered = {
    version: VERSION,
    id: record.id,
    model: record.model,
    settings: record.settings,
    embeddingProfile: record.embeddingProfile ?? null,
    facts: facts || null,
    entry: entry || null,
  };
  return JSON.stringify(ordered, null, 2) + '\n';
}

/**
 * One record per article. A file that cannot be parsed or has another schema version is returned
 * as `{ id, invalid: reason }` so that it only affects its own article.
 */
export async function readRecords(dir) {
  const records = new Map();
  let names;
  try {
    names = await readdir(dir);
  } catch (error) {
    if (error.code === 'ENOENT') return records;
    throw error;
  }
  for (const name of names.filter((n) => n.endsWith('.json')).sort()) {
    const id = name.slice(0, -5);
    try {
      const record = JSON.parse(await readFile(join(dir, name), 'utf8'));
      records.set(id, record?.version === VERSION ? record : { id, invalid: '스키마 버전 불일치' });
    } catch {
      records.set(id, { id, invalid: 'JSON 오류' });
    }
  }
  return records;
}

export async function writeRecord(dir, record) {
  assertId(record.id);
  await mkdir(dir, { recursive: true });
  const path = join(dir, record.id + '.json');
  // A reader or a concurrent Git operation never sees a half-written record.
  const tmp = `${path}.${process.pid}.${++sequence}.tmp`;
  try {
    await writeFile(tmp, serialize(record));
    await rename(tmp, path);
  } finally {
    await rm(tmp, { force: true });
  }
}

/** Delete records of articles that are no longer public, and temp files an interrupted run left behind. */
export async function removeOrphans(dir, publicIds, now = Date.now()) {
  const keep = new Set(publicIds);
  const removed = [];
  let names;
  try {
    names = await readdir(dir);
  } catch (error) {
    if (error.code === 'ENOENT') return removed;
    throw error;
  }
  for (const name of names.sort()) {
    if (name.endsWith('.tmp')) {
      // A fresh temp file is another process between writeFile and rename; removing it fails that write.
      const modified = await stat(join(dir, name)).then(
        (s) => s.mtimeMs,
        () => Infinity,
      );
      if (now - modified > STALE_TEMP_MS) await rm(join(dir, name), { force: true });
    } else if (name.endsWith('.json') && !keep.has(name.slice(0, -5))) {
      await rm(join(dir, name), { force: true });
      removed.push(name.slice(0, -5));
    }
  }
  return removed;
}

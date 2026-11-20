// Shared fixtures for the recommendation tests: a mock model, a deterministic embedder and a
// temporary record directory. Not a test file itself (`npm test` runs tests/*.test.mjs).
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { embeddingProfile, loadVectors, memoryStore, unitVector } from '../tools/knowledge/embeddings.mjs';
import { generateRelated } from '../tools/knowledge/related.mjs';
import { readRecords } from '../tools/knowledge/store.mjs';
import { auditAll, factsInstruction, prepareCorpus, syncAll } from '../tools/knowledge/phase2a.mjs';

export const concepts = [
  { id: 'ocr', label: 'OCR', description: 'Transcription', aliases: ['문자 인식'], broaderIds: [] },
];
export function article(id, html, conceptRefs = [], description = '', title = id) {
  return {
    id,
    title,
    path: '/' + id + '/',
    description,
    html,
    plainText: html.replace(/<[^>]*>/g, ' '),
    conceptRefs,
    publishedAt: '2026-01-01',
    language: 'en',
  };
}
export const corpus = (articles, extra = {}) =>
  prepareCorpus({ concepts, series: [], overrides: {}, articles, ...extra });

// A deterministic embedding: one axis per topic word, so unrelated texts have cosine exactly 0.
// A word counts wherever it occurs, also inside a longer token ("xalpha").
const AXES = { alpha: 0, beta: 1, gamma: 2, delta: 3, feline: 4, 고양이: 4 };
export function embedder() {
  const { profile, profileId } = embeddingProfile({ EMBEDDINGS_DIMENSIONS: '8' });
  const batches = [];
  return {
    profile,
    profileId,
    batches,
    async embed(texts) {
      batches.push(texts);
      return texts.map((text) => {
        const values = new Array(8).fill(0);
        for (const [word, axis] of Object.entries(AXES))
          values[axis] += text.toLowerCase().split(word).length - 1;
        // A text without any topic word stays the zero vector, which unitVector refuses.
        return values.some(Boolean) ? unitVector(values, 8) : new Float32Array(8);
      });
    },
  };
}

export const first = (input) => ({
  recommendations: input.candidates.slice(0, 1).map((c) => ({
    id: c.id,
    relation: c.relationOptions[0],
    reason: c.id + ' 추천',
    sourceBlock: 0,
    targetBlock: 0,
  })),
});
/** Calls are logged as `facts:<title>` and `rank:<id>`; most fixtures use the ID as the title. */
export function model({ facts = () => ({ facts: [] }), rank = first, name = 'fixture', delay } = {}) {
  const calls = [];
  return {
    model: name,
    settings: { seed: 7 },
    usage: { requests: 0, promptTokens: 0, completionTokens: 0 },
    calls,
    async complete(instruction, input) {
      const kind = instruction === factsInstruction ? 'facts' : 'rank';
      const id = kind === 'facts' ? input.title : input.source.id;
      calls.push(kind + ':' + id);
      if (delay) await delay(id);
      return kind === 'facts' ? facts(input) : rank(input);
    },
  };
}
export const silent = () => {};

export async function workspace(t) {
  const dir = await mkdtemp(join(tmpdir(), 'related-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const embed = embedder();
  const store = memoryStore();
  return {
    dir,
    embed,
    store,
    vectors: (ctx) => loadVectors(ctx, embed, store),
    async sync(ctx, provider, options = {}) {
      const vectors = await loadVectors(ctx, embed, store);
      return syncAll(ctx, provider, { dir, vectors, log: silent, ...options });
    },
    async files() {
      const out = {};
      for (const name of (await readdir(dir)).sort()) out[name] = await readFile(join(dir, name), 'utf8');
      return out;
    },
    record: async (id) => JSON.parse(await readFile(join(dir, id + '.json'), 'utf8')),
    audit: async (ctx) => auditAll(ctx, await readRecords(dir)),
  };
}
export const rules = (ctx) => generateRelated(ctx.articles, ctx.concepts, ctx.series, ctx.overrides);

// ── A small library: titles differ from IDs, excerpts are chosen out of document order, three
// articles carry a reviewed OCR role and the model finds a fourth. ─────────────────────────────

const refs = (...pairs) => pairs.map(([id, role]) => ({ id, role }));
export const TITLES = {
  scan: 'Scanning pages at scale',
  reader: 'Reading text lines',
  index: 'Index shards',
  voice: 'Voice synthesis',
  tone: 'Tone length',
  blank: 'No text',
};
export const HTML = {
  // The last Eval block has no topic word: it is never an excerpt, so editing it changes the
  // article's own facts hash and nothing else.
  scan:
    '<h2 id="intro">Intro</h2><p>delta delta field notes</p>' +
    '<h2 id="method">Method</h2><p>alpha gamma pipeline stage</p>' +
    '<p>alpha alpha OCR runs on every page</p><p>alpha beta beta fallback path</p>' +
    '<h2 id="eval">Eval</h2><p>gamma gamma delta results</p><p>plain closing remark</p>' +
    '<h2 id="refs">References</h2><p>alpha cited work</p>',
  reader:
    '<h2 id="engine">Engine</h2><p>The alpha engine uses OCR for each line</p>' +
    '<h2 id="decode">Decode</h2><p>alpha gamma decoding loop</p><p>beta speech aside</p>' +
    '<h2 id="tune">Tune</h2><p>gamma tuning sweep</p><p>plain trailing remark</p>',
  index: '<p>alpha shard layout</p><p>alpha delta merge policy</p>',
  voice: '<p>beta 음성 합성</p><p>beta gamma prosody</p>',
  tone: '<p>beta 발화 길이</p>',
  blank: '<p>😂😂😂</p>',
};
const REFS = { scan: refs(['ocr', 'about']), index: refs(['ocr', 'applies']), voice: refs(['ocr', 'about']) };
// `change` replaces the HTML, title or conceptRefs of single articles.
export function library(change = {}) {
  return Object.keys(HTML).map((id) =>
    article(
      id,
      change[id]?.html ?? HTML[id],
      change[id]?.refs ?? REFS[id] ?? [],
      '',
      change[id]?.title ?? TITLES[id],
    ),
  );
}
// The model finds that `reader` applies OCR; no editor reviewed that.
export const readerApplies = (input) => ({
  facts: input.title === TITLES.reader ? [{ conceptId: 'ocr', role: 'applies', block: 0 }] : [],
});
// scan recommends reader as an application of OCR, which only reader's LLM fact allows.
export const pick = (input) => {
  if (input.source.id === 'tone') return { recommendations: [] };
  if (input.source.id !== 'scan') return first(input);
  const target = input.candidates.find((c) => c.id === 'reader');
  return {
    recommendations: [
      {
        id: 'reader',
        relation: 'application_example',
        reason: 'OCR 적용 사례',
        sourceBlock: 0,
        targetBlock: 0,
      },
    ].filter(() => target),
  };
};

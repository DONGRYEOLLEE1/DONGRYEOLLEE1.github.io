import assert from 'node:assert/strict';
import test from 'node:test';
import { evidenceBlocks } from '../tools/knowledge/evidence.mjs';
import { generateRelated } from '../tools/knowledge/related.mjs';
import {
  auditArtifact,
  factsInstruction,
  prepareCorpus,
  projectRecommendations,
  syncPilot,
} from '../tools/knowledge/phase2a.mjs';

const concepts = [
  { id: 'ocr', label: 'OCR', description: 'Transcription', aliases: ['문자 인식'], broaderIds: [] },
];
function article(id, html, conceptRefs = []) {
  return {
    id,
    title: id,
    path: '/' + id + '/',
    description: 'OCR recognition',
    html,
    plainText: html.replace(/<[^>]*>/g, ' '),
    conceptRefs,
    publishedAt: '2026-01-01',
    language: 'en',
  };
}
function corpus(extra = []) {
  return prepareCorpus({
    concepts,
    series: [],
    overrides: {},
    articles: [
      article('example', '<p>OCR transcribes visible characters.</p>'),
      article('guide', '<p>OCR recognition and transcription metrics.</p>', [{ id: 'ocr', role: 'about' }]),
      ...extra,
    ],
  });
}
function model(rank) {
  const calls = [];
  return {
    model: 'fixture',
    calls,
    async complete(instruction, input) {
      calls.push(instruction);
      if (instruction === factsInstruction)
        return { facts: [{ conceptId: 'ocr', role: 'applies', block: 0 }] };
      return rank(input);
    },
  };
}
const recommend = () => ({
  recommendations: [
    {
      id: 'guide',
      relation: 'concept_explanation',
      reason: '문자 전사 평가',
      sourceBlock: 0,
      targetBlock: 0,
    },
  ],
});
const silent = () => {};
const ids = ['example'];

test('headingless code, preamble and section evidence retain exact text and optional anchor', () => {
  const blocks = evidenceBlocks(
    article('example', '<p>intro</p><h2 id="ocr">OCR</h2><pre><code>x = 1</code></pre>'),
  );
  assert.deepEqual(
    blocks.map((b) => [b.anchor, b.quote]),
    [
      [null, 'intro'],
      ['ocr', 'x = 1'],
    ],
  );
  assert.deepEqual(evidenceBlocks(article('image', '<p>😂<img src="/fig.png"></p>')), []);
});

test('independent facts feed a directional rerank and verified references are projected without the prompt', async () => {
  const ctx = corpus();
  const judge = model(recommend);
  const artifact = await syncPilot(ctx, judge, null, ids, silent);
  auditArtifact(ctx, artifact, ids);
  assert.equal(artifact.entries.example.status, 'verified');
  assert.equal(artifact.facts.example.facts[0].role, 'applies');
  const baseline = generateRelated(ctx.articles, concepts, []);
  const visible = projectRecommendations(baseline, artifact, {}).find((a) => a.id === 'example');
  assert.deepEqual(visible.related, [{ id: 'guide', reason: '문자 전사 평가' }]);
  const repeat = await syncPilot(ctx, judge, artifact, ids, silent);
  assert.deepEqual(artifact, repeat);
  assert.equal(judge.calls.length, 2);
  const corrupt = structuredClone(artifact);
  corrupt.entries.example.recommendations[0].targetEvidence.quote = 'different statement';
  assert.throws(() => auditArtifact(ctx, corrupt, ids));
});

test('explicit abstention is cached; invalid blocks or provider failure are retryable', async () => {
  const ctx = corpus();
  const empty = await syncPilot(
    ctx,
    model(() => ({ recommendations: [] })),
    null,
    ids,
    silent,
  );
  assert.equal(empty.entries.example.status, 'abstained');
  const fail = model(() => {
    throw Error('offline');
  });
  await syncPilot(ctx, fail, empty, ids, silent);
  assert.equal(fail.calls.length, 0);
  for (const rank of [
    () => ({
      recommendations: Array.from({ length: 4 }, () => ({
        ...recommend().recommendations[0],
        sourceBlock: 99,
      })),
    }),
    () => ({ recommendations: [{ ...recommend().recommendations[0], id: 'not-a-candidate' }] }),
    () => {
      throw Error('timeout');
    },
  ]) {
    const result = await syncPilot(ctx, model(rank), null, ids, silent);
    assert.equal(result.entries.example.status, 'fallback');
    assert.equal(result.entries.example.fallbackFrom, 'rules');
    const retried = await syncPilot(ctx, model(recommend), result, ids, silent);
    assert.equal(retried.entries.example.status, 'verified');
  }
});

test('publication rejects a stale merge snapshot and missing or changed target evidence', async () => {
  const initial = corpus(),
    first = await syncPilot(initial, model(recommend), null, ids, silent);
  const merged = corpus([article('server-b', '<p>Extra article.</p>')]);
  assert.throws(() => auditArtifact(merged, first, ids), /stale/);
  const updated = await syncPilot(merged, model(recommend), first, ids, silent);
  auditArtifact(merged, updated, ids);
  for (const articles of [
    initial.articles.filter((a) => a.id !== 'guide'),
    [initial.articles[0], article('guide', '<p>Different OCR experiment.</p>')],
  ]) {
    const changed = prepareCorpus({ ...initial, articles });
    const failed = await syncPilot(
      changed,
      model(() => {
        throw Error('offline');
      }),
      first,
      ids,
      silent,
    );
    assert.equal(failed.entries.example.recommendations.length, 0);
    auditArtifact(changed, failed, ids);
  }
});

test('image-only pilot can abstain without inference and preserves the previous series ordering', async () => {
  const articles = [article('example', '<p>😂😂😂</p>'), article('guide', '<pre>model.generate()</pre>')];
  const series = [{ id: 'notes', orderedArticleIds: ['example', 'guide'] }];
  const ctx = prepareCorpus({ articles, concepts, series, overrides: {} });
  const judge = model(recommend);
  const artifact = await syncPilot(ctx, judge, null, ids, silent);
  auditArtifact(ctx, artifact, ids);
  assert.equal(judge.calls.length, 0);
  const projected = projectRecommendations(generateRelated(articles, concepts, series), artifact, {});
  assert.deepEqual(projected[0].related, []);
  assert.equal(projected[0].seriesOrder, 1);
  assert.equal(projected[1].seriesOrder, 2);
});

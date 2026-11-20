import assert from 'node:assert/strict';
import test from 'node:test';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { evidenceBlocks } from '../tools/knowledge/evidence.mjs';
import { loadVectors, memoryStore } from '../tools/knowledge/embeddings.mjs';
import { generateRelated } from '../tools/knowledge/related.mjs';
import {
  auditSummary,
  manifest,
  prepareCorpus,
  prepareSearch,
  projectRecommendations,
  rankHash,
  rankTask,
  storedRankHash,
} from '../tools/knowledge/phase2a.mjs';
import { article, concepts, corpus, embedder, first, model, rules, workspace } from './fixtures.mjs';

// Two topics with disjoint vocabulary: nothing links an "alpha" article to a "beta" one.
const alpha = () => [
  article('a1', '<p>alpha retrieval scoring</p><p>alpha ranking depth</p>'),
  article('a2', '<p>alpha index shards</p>'),
  article('a3', '<p>alpha query latency</p>'),
];
const beta = () => [article('b1', '<p>beta 음성 합성</p>'), article('b2', '<p>beta 발화 길이</p>')];

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

test('independent facts feed a directional rerank and only verified id and reason are projected', async (t) => {
  const ws = await workspace(t);
  const ctx = corpus([
    article('example', '<p>OCR transcribes visible characters.</p>', [], 'OCR recognition'),
    article(
      'guide',
      '<p>OCR recognition and transcription metrics.</p>',
      [{ id: 'ocr', role: 'about' }],
      'OCR recognition',
    ),
  ]);
  const judge = model({
    facts: (input) => ({
      facts: input.title === 'example' ? [{ conceptId: 'ocr', role: 'applies', block: 0 }] : [],
    }),
    rank: (input) =>
      input.source.id === 'example'
        ? {
            recommendations: [
              {
                id: 'guide',
                relation: 'concept_explanation',
                reason: '문자 전사 평가',
                sourceBlock: 0,
                targetBlock: 0,
              },
            ],
          }
        : { recommendations: [] },
  });
  const summary = await ws.sync(ctx, judge);
  assert.deepEqual([summary.facts.called, summary.rank.called, summary.rank.abstained], [2, 2, 1]);
  const record = await ws.record('example');
  assert.deepEqual(Object.keys(record), [
    'version',
    'id',
    'model',
    'settings',
    'embeddingProfile',
    'facts',
    'entry',
  ]);
  assert.equal(record.version, 2);
  assert.equal(record.entry.status, 'verified');
  assert.equal(record.facts.facts[0].role, 'applies');
  const audit = await ws.audit(ctx);
  assert.equal(audit.valid.size, 2);
  const visible = projectRecommendations(rules(ctx), audit.valid, {}).find((a) => a.id === 'example');
  assert.deepEqual(visible.related, [{ id: 'guide', reason: '문자 전사 평가' }]);

  const before = await ws.files();
  await ws.sync(ctx, judge);
  assert.equal(judge.calls.length, 4);
  assert.deepEqual(await ws.files(), before);
});

test('an embedding cache hit costs no request and an edited block is the only one embedded again', async (t) => {
  const ws = await workspace(t);
  const articles = [...alpha(), ...beta()];
  await ws.sync(corpus(articles), model());
  const requests = ws.embed.batches.length;
  assert.ok(requests > 0);
  const repeat = await ws.sync(corpus(articles), model());
  assert.equal(ws.embed.batches.length, requests);
  assert.deepEqual([repeat.embeddings.embedded, repeat.embeddings.cached], [0, 6]);

  articles[0] = article('a1', '<p>alpha retrieval scoring</p><p>alpha ranking depth, revised</p>');
  const edited = await ws.sync(corpus(articles), model());
  assert.equal(edited.embeddings.embedded, 1);
  assert.deepEqual(ws.embed.batches.at(-1), ['a1\n\nalpha ranking depth, revised']);
});

test('an article with no shared concept or word becomes a candidate through embeddings alone', async () => {
  const ctx = corpus([
    article('en', '<p>feline whiskers purr</p>'),
    article('ko', '<p>고양이 수염 관찰</p>'),
    article('other', '<p>delta river mouth</p>'),
  ]);
  const facts = new Map();
  assert.deepEqual([...rankTask(ctx, prepareSearch(ctx, facts), 'en').candidates.keys()], []);
  const vectors = await loadVectors(ctx, embedder(), memoryStore());
  assert.deepEqual([...rankTask(ctx, prepareSearch(ctx, facts, vectors), 'en').candidates.keys()], ['ko']);
});

test('excerpts take a direct link first, then fact evidence, then the closest remaining block', async () => {
  const source = article(
    'source',
    '<h2 id="one">One</h2><p>gamma gamma study</p>' +
      '<h2 id="two">Two</h2><p>delta unrelated notes</p>' +
      '<h2 id="three">Three</h2><p>OCR is applied to pages delta</p>' +
      '<h2 id="four">Four</h2><p>See <a href="/target/">the guide</a> delta</p>' +
      '<h2 id="five">Five</h2><p>gamma delta mixed</p>',
  );
  const target = article('target', '<p>gamma explains OCR</p>', [{ id: 'ocr', role: 'about' }]);
  const ctx = corpus([source, target]);
  const block = ctx.usable.get('source')[2];
  const facts = new Map([
    [
      'source',
      {
        facts: [
          {
            id: 'ocr',
            role: 'applies',
            evidence: {
              articleId: 'source',
              blockId: block.id,
              anchor: block.anchor,
              quote: block.quote,
              hash: block.hash,
            },
          },
        ],
      },
    ],
  ]);
  const vectors = await loadVectors(ctx, embedder(), memoryStore());
  const anchors = (search) =>
    rankTask(ctx, search, 'source')
      .candidates.get('target')
      .sourceBlocks.map((b) => b.anchor);
  assert.deepEqual(anchors(prepareSearch(ctx, facts, vectors)), ['four', 'three', 'one']);
  // Without the link and the fact, similarity alone decides: cosine 1, then 0.71, then 0.
  const plain = corpus([
    article('source', source.html.replace('<a href="/target/">the guide</a>', 'the guide')),
    target,
  ]);
  const plainVectors = await loadVectors(plain, embedder(), memoryStore());
  assert.deepEqual(
    rankTask(plain, prepareSearch(plain, new Map(), plainVectors), 'source')
      .candidates.get('target')
      .sourceBlocks.map((b) => b.anchor),
    ['one', 'five', 'two'],
  );
  const pair = rankTask(ctx, prepareSearch(ctx, facts, vectors), 'source').candidates.get('target');
  assert.deepEqual(pair.relationOptions, ['concept_explanation', 'related_method', 'evaluation_comparison']);
});

test('the rerank hash ignores candidate order but not the candidate set or an excerpt', async () => {
  const articles = alpha();
  const ctx = corpus(articles);
  const facts = new Map();
  const vectors = await loadVectors(ctx, embedder(), memoryStore());
  const task = rankTask(ctx, prepareSearch(ctx, facts, vectors), 'a1');
  const settings = { seed: 7 };
  const base = rankHash('a1', task, 'fixture', settings);
  assert.equal(task.candidates.size, 2);

  const reversed = { ...task, candidates: new Map([...task.candidates].reverse()) };
  assert.equal(rankHash('a1', reversed, 'fixture', settings), base);
  // The stored candidate list alone reproduces the hash, in either order and without vectors.
  for (const candidates of [manifest(task), manifest(reversed)])
    assert.equal(storedRankHash(ctx, 'a1', { candidates }, 'fixture', settings, facts), base);

  const fewer = { ...task, candidates: new Map([...task.candidates].slice(1)) };
  assert.notEqual(rankHash('a1', fewer, 'fixture', settings), base);
  assert.notEqual(rankHash('a1', task, 'other-model', settings), base);
  assert.notEqual(rankHash('a1', task, 'fixture', { seed: 8 }), base);

  articles[1] = article('a2', '<p>alpha index shards, rebuilt</p>');
  const edited = corpus(articles);
  const editedVectors = await loadVectors(edited, embedder(), memoryStore());
  const editedTask = rankTask(edited, prepareSearch(edited, facts, editedVectors), 'a1');
  // Same candidates and block IDs (in whatever order); only the text of one excerpt differs.
  const sorted = (rows) => rows.toSorted((a, b) => a.id.localeCompare(b.id));
  assert.deepEqual(sorted(manifest(editedTask)), sorted(manifest(task)));
  assert.notEqual(rankHash('a1', editedTask, 'fixture', settings), base);
});

test('a new dictionary concept does not re-extract facts; reextract does', async (t) => {
  const ws = await workspace(t);
  const articles = [...alpha(), article('image', '<p>😂😂😂</p>')];
  await ws.sync(corpus(articles), model());
  const grown = corpus(articles, {
    concepts: [...concepts, { id: 'tts', label: 'TTS', description: 'Speech', aliases: [], broaderIds: [] }],
  });
  const judge = model();
  const summary = await ws.sync(grown, judge);
  assert.deepEqual(judge.calls, []);
  assert.equal(summary.facts.reused, 4);

  const forced = await ws.sync(grown, judge, { reextract: true });
  assert.deepEqual(judge.calls.sort(), ['facts:a1', 'facts:a2', 'facts:a3']);
  assert.ok(forced.facts.items.every((item) => item.reason === 'reextract'));
});

test('a stored fact whose concept left the dictionary is extracted again', async (t) => {
  const ws = await workspace(t);
  const articles = [article('example', '<p>OCR transcribes visible characters.</p>')];
  const facts = () => ({ facts: [{ conceptId: 'ocr', role: 'applies', block: 0 }] });
  await ws.sync(corpus(articles), model({ facts }));
  const judge = model();
  await ws.sync(corpus(articles, { concepts: [] }), judge);
  assert.deepEqual(judge.calls, ['facts:example']);
});

test('publishing one article costs 2 + k calls and leaves unaffected records byte-identical', async (t) => {
  const ws = await workspace(t);
  const existing = [...alpha(), ...beta()];
  const initial = await ws.sync(corpus(existing), model());
  assert.equal(initial.facts.called + initial.rank.called, 10);
  const before = await ws.files();

  const judge = model();
  const grown = corpus([...existing, article('x', '<p>alpha cache warmup</p>')]);
  const summary = await ws.sync(grown, judge, { concurrency: 3 });
  const reranked = summary.rank.items.filter((item) => item.call && item.id !== 'x');
  const k = reranked.length;
  assert.equal(judge.calls.length, 2 + k);
  assert.deepEqual(
    judge.calls.filter((c) => c.startsWith('facts:')),
    ['facts:x'],
  );
  assert.ok(judge.calls.includes('rank:x'));
  assert.equal(summary.rank.items.find((item) => item.id === 'x').reason, 'new');
  // x joins the candidates of the three alpha articles and of no beta article.
  assert.deepEqual(reranked.map((item) => [item.id, item.reason]).sort(), [
    ['a1', 'candidate-set'],
    ['a2', 'candidate-set'],
    ['a3', 'candidate-set'],
  ]);
  assert.equal(summary.embeddings.embedded, 1);

  const after = await ws.files();
  const changed = Object.keys(after).filter((name) => after[name] !== before[name]);
  assert.deepEqual(changed, ['a1.json', 'a2.json', 'a3.json', 'x.json']);
  assert.equal(after['b1.json'], before['b1.json']);
  assert.equal((await ws.audit(grown)).valid.size, 6);

  const plan = await ws.sync(grown, model(), { dryRun: true });
  assert.deepEqual([plan.facts.items, plan.rank.items], [[], []]);
});

test('a dry run names the articles to call and writes nothing', async (t) => {
  const ws = await workspace(t);
  const existing = alpha();
  await ws.sync(corpus(existing), model());
  const before = await ws.files();
  const judge = model();
  const plan = await ws.sync(corpus([...existing, article('x', '<p>alpha cache warmup</p>')]), judge, {
    dryRun: true,
  });
  assert.deepEqual(judge.calls, []);
  assert.deepEqual(plan.facts.items, [{ id: 'x', reason: 'new', call: true }]);
  assert.deepEqual(plan.rank.items.map((item) => item.id).sort(), ['a1', 'a2', 'a3', 'x']);
  assert.deepEqual(await ws.files(), before);
});

test('only restricts facts and reranking to the named articles while searching the whole corpus', async (t) => {
  const ws = await workspace(t);
  const ctx = corpus(alpha());
  const judge = model();
  await ws.sync(ctx, judge, { only: ['a1'] });
  assert.deepEqual(judge.calls, ['facts:a1', 'rank:a1']);
  assert.deepEqual(Object.keys(await ws.files()), ['a1.json']);
  assert.deepEqual((await ws.record('a1')).entry.candidates.map((c) => c.id).sort(), ['a2', 'a3']);
  const audit = await ws.audit(ctx);
  assert.deepEqual([[...audit.valid.keys()], audit.missing], [['a1'], ['a2', 'a3']]);
  await assert.rejects(ws.sync(ctx, judge, { only: ['missing'] }), /Not a public article/);
});

test('removing a recommended article invalidates its referrers until the next sync', async (t) => {
  const ws = await workspace(t);
  const ctx = corpus(alpha());
  await ws.sync(ctx, model());
  const target = (await ws.record('a1')).entry.recommendations[0].id;
  const remaining = corpus(ctx.articles.filter((a) => a.id !== target));

  const audit = await ws.audit(remaining);
  assert.ok(audit.invalid.has('a1'));
  assert.deepEqual(audit.orphans, [target]);
  const baseline = rules(remaining);
  const projected = projectRecommendations(baseline, audit.valid, {});
  assert.deepEqual(
    projected.find((a) => a.id === 'a1'),
    baseline.find((a) => a.id === 'a1'),
  );

  const judge = model();
  const summary = await ws.sync(remaining, judge);
  assert.ok(judge.calls.includes('rank:a1'));
  assert.ok(judge.calls.every((call) => call.startsWith('rank:')));
  assert.deepEqual(summary.orphans, [target]);
  assert.ok(!(target + '.json' in (await ws.files())));
  const healed = await ws.audit(remaining);
  assert.equal(healed.invalid.size + healed.missing.length, 0);
  assert.ok(!(await ws.record('a1')).entry.recommendations.some((row) => row.id === target));
});

test('a changed excerpt invalidates the record; a failed retry falls back to the rules', async (t) => {
  const ws = await workspace(t);
  const articles = alpha();
  await ws.sync(corpus(articles), model());
  const target = (await ws.record('a1')).entry.recommendations[0].id;
  const edited = corpus(
    articles.map((a) => (a.id === target ? article(target, '<p>alpha rewritten text</p>') : a)),
  );
  assert.ok((await ws.audit(edited)).invalid.has('a1'));

  const offline = model({
    rank: () => {
      throw Error('offline');
    },
  });
  const summary = await ws.sync(edited, offline);
  assert.equal(summary.rank.items.find((item) => item.id === 'a1').reason, 'evidence');
  const entry = (await ws.record('a1')).entry;
  assert.deepEqual([entry.status, entry.fallbackFrom, entry.recommendations], ['fallback', 'rules', []]);
  const audit = await ws.audit(edited);
  assert.equal(audit.invalid.size, 0);
  assert.equal(auditSummary(audit).fallback, 3);
  const baseline = rules(edited);
  assert.deepEqual(projectRecommendations(baseline, audit.valid, {}), baseline);
});

test('a broken record sends only its own article back to the rules', async (t) => {
  const ws = await workspace(t);
  const ctx = corpus([...alpha(), ...beta()]);
  await ws.sync(ctx, model());
  await writeFile(join(ws.dir, 'a2.json'), '{ "version": 2, ');
  const tampered = await ws.record('a3');
  tampered.entry.recommendations[0].targetEvidence.quote = 'different statement';
  await writeFile(join(ws.dir, 'a3.json'), JSON.stringify(tampered, null, 2) + '\n');
  const stale = await ws.record('b2');
  stale.version = 1;
  await writeFile(join(ws.dir, 'b2.json'), JSON.stringify(stale));

  const audit = await ws.audit(ctx);
  assert.deepEqual([...audit.invalid.keys()].sort(), ['a2', 'a3', 'b2']);
  assert.deepEqual([...audit.valid.keys()].sort(), ['a1', 'b1']);
  const summary = auditSummary(audit);
  assert.deepEqual([summary.llm, summary.rules, summary.missing, summary.invalid], [2, 3, 0, 3]);
  const baseline = rules(ctx);
  const projected = projectRecommendations(baseline, audit.valid, {});
  for (const id of ['a1', 'b1']) assert.match(projected.find((a) => a.id === id).related[0].reason, /추천$/);
  for (const id of ['a2', 'a3', 'b2'])
    assert.deepEqual(
      projected.find((a) => a.id === id),
      baseline.find((a) => a.id === id),
    );

  const judge = model();
  await ws.sync(ctx, judge);
  assert.deepEqual(judge.calls.sort(), ['facts:a2', 'facts:b2', 'rank:a2', 'rank:a3', 'rank:b2']);
  assert.equal((await ws.audit(ctx)).valid.size, 5);
});

test('provider errors, bad block numbers and truncation become a fallback retried alone', async (t) => {
  const failures = [
    () => {
      throw Error('Inference HTTP 503');
    },
    () => {
      throw Error('Inference output truncated');
    },
    (input) => ({ recommendations: [{ ...first(input).recommendations[0], sourceBlock: 99 }] }),
    (input) => ({ recommendations: [{ ...first(input).recommendations[0], id: 'not-a-candidate' }] }),
  ];
  for (const failure of failures) {
    const ws = await workspace(t);
    const ctx = corpus(alpha());
    const flaky = model({
      rank: (input) =>
        input.source.id === 'a1'
          ? failure(input)
          : input.source.id === 'a2'
            ? { recommendations: [] }
            : first(input),
    });
    const summary = await ws.sync(ctx, flaky);
    assert.deepEqual([summary.rank.failed, summary.rank.abstained], [1, 1]);
    const entry = (await ws.record('a1')).entry;
    assert.deepEqual([entry.status, entry.fallbackFrom], ['fallback', 'rules']);
    assert.equal((await ws.record('a2')).entry.status, 'abstained');
    assert.equal((await ws.audit(ctx)).valid.size, 3);

    // The explicit abstention of a2 is a result and is reused; only the failure is retried.
    const judge = model();
    const retry = await ws.sync(ctx, judge);
    assert.deepEqual(judge.calls, ['rank:a1']);
    assert.deepEqual(retry.rank.items, [{ id: 'a1', reason: 'fallback-retry', call: true }]);
    assert.equal((await ws.record('a1')).entry.status, 'verified');
  }
});

test('a failed fact extraction is retried alone, and a failed rerank keeps still-valid previous picks', async (t) => {
  const ws = await workspace(t);
  const ctx = corpus(alpha());
  const broken = model({
    facts: (input) => {
      if (input.title === 'a1') throw Error('Inference HTTP 500');
      return { facts: [] };
    },
  });
  const summary = await ws.sync(ctx, broken);
  assert.equal(summary.facts.failed, 1);
  assert.equal((await ws.record('a1')).facts.status, 'fallback');
  const judge = model();
  await ws.sync(ctx, judge);
  assert.deepEqual(judge.calls, ['facts:a1']);

  // A model change re-ranks everything; a1 fails but its earlier picks still cite current excerpts.
  const previous = (await ws.record('a1')).entry.recommendations;
  const upgraded = model({
    name: 'fixture-2',
    rank: (input) => {
      if (input.source.id === 'a1') throw Error('timeout');
      return first(input);
    },
  });
  const changed = await ws.sync(ctx, upgraded);
  assert.ok(changed.rank.items.every((item) => item.reason === 'model'));
  const entry = (await ws.record('a1')).entry;
  assert.deepEqual([entry.status, entry.fallbackFrom], ['fallback', 'previous']);
  assert.deepEqual(entry.recommendations, previous);
  const audit = await ws.audit(ctx);
  assert.equal(audit.valid.size, 3);
  assert.equal(
    projectRecommendations(rules(ctx), audit.valid, {})[0].related[0].reason,
    previous[0].id + ' 추천',
  );
});

test('the records do not depend on concurrency or completion order', async (t) => {
  const ctx = corpus([...alpha(), ...beta(), article('image', '<p>😂😂😂</p>')]);
  const results = [];
  for (const concurrency of [1, 8]) {
    const ws = await workspace(t);
    // With eight workers the later an article starts, the earlier it finishes.
    let order = 12;
    const delay = () => new Promise((resolve) => setTimeout(resolve, concurrency === 1 ? 0 : order--));
    const summary = await ws.sync(ctx, model({ delay }), { concurrency });
    assert.equal(summary.facts.called + summary.rank.called, 10);
    results.push(await ws.files());
  }
  assert.equal(Object.keys(results[0]).length, 6);
  assert.deepEqual(results[0], results[1]);
});

test('an image-only article is not judged: no inference, rule ranking and series order kept', async (t) => {
  const ws = await workspace(t);
  const articles = [article('example', '<p>😂😂😂</p>'), article('guide', '<pre>model.generate()</pre>')];
  const series = [{ id: 'notes', orderedArticleIds: ['example', 'guide'] }];
  const ctx = prepareCorpus({ articles, concepts, series, overrides: {} });
  const judge = model();
  await ws.sync(ctx, judge, { only: ['example'] });
  assert.deepEqual(judge.calls, []);
  const record = await ws.record('example');
  assert.deepEqual([record.facts.status, record.entry.status], ['abstained', 'abstained']);
  const audit = await ws.audit(ctx);
  const baseline = generateRelated(articles, concepts, series);
  const projected = projectRecommendations(baseline, audit.valid, {});
  // Nothing was sent to the model, so its silence is not a verdict: the rule ranking stays.
  assert.equal(baseline[0].related[0].reason, '같은 시리즈의 다음 글');
  assert.deepEqual(projected[0].related, baseline[0].related);
  assert.deepEqual([auditSummary(audit).llm, auditSummary(audit).noEvidenceIds], [0, ['example']]);
  assert.equal(projected[0].seriesOrder, 1);
  assert.equal(projected[1].seriesOrder, 2);
  assert.equal(projected[1].related[0].reason, '같은 시리즈의 이전 글');
});

test('pins stay ahead of model picks and excluded articles are never candidates', async (t) => {
  const ws = await workspace(t);
  const ctx = corpus(alpha(), { overrides: { pins: { a1: ['a3'] }, excludes: { a2: ['a1'] } } });
  await ws.sync(
    ctx,
    model({
      rank: (input) => ({
        recommendations: first({ candidates: input.candidates.slice(-1) }).recommendations,
      }),
    }),
  );
  assert.deepEqual(
    (await ws.record('a1')).entry.candidates.map((c) => c.id),
    ['a3', 'a2'],
  );
  assert.deepEqual(
    (await ws.record('a2')).entry.candidates.map((c) => c.id),
    ['a3'],
  );
  const audit = await ws.audit(ctx);
  const projected = projectRecommendations(rules(ctx), audit.valid, ctx.overrides);
  assert.deepEqual(
    projected[0].related.map((row) => row.id),
    ['a3', 'a2'],
  );
  assert.deepEqual(Object.keys(projected[0].related[1]), ['id', 'reason']);
});

test('a role the model cites twice keeps its first citation instead of failing the article', async () => {
  const { factTask, readFacts } = await import('../tools/knowledge/phase2a.mjs');
  const ctx = corpus([article('scan', '<p>OCR reads text.</p><p>OCR again here.</p>')]);
  const task = factTask(ctx, 'scan');
  const facts = readFacts(ctx, 'scan', task, {
    facts: [
      { conceptId: 'ocr', role: 'applies', block: 1 },
      { conceptId: 'ocr', role: 'applies', block: 0 },
      { conceptId: 'ocr', role: 'about', block: 0 },
    ],
  });
  assert.deepEqual(
    facts.map((f) => [f.id, f.role, f.evidence.quote]),
    [
      ['ocr', 'applies', 'OCR again here.'],
      ['ocr', 'about', 'OCR reads text.'],
    ],
  );
  // An unknown concept or a block out of range is still a rejected answer.
  assert.throws(() => readFacts(ctx, 'scan', task, { facts: [{ conceptId: 'nope', role: 'about', block: 0 }] }));
  assert.throws(() => readFacts(ctx, 'scan', task, { facts: [{ conceptId: 'ocr', role: 'about', block: 9 }] }));
});

test('a reason that is not written in Korean is a rejected answer', async () => {
  const { readRanking } = await import('../tools/knowledge/phase2a.mjs');
  const ctx = corpus([
    article('source', '<p>OCR reads text from scans.</p>', [{ id: 'ocr', role: 'applies' }]),
    article('target', '<p>OCR is transcription of images.</p>', [{ id: 'ocr', role: 'about' }]),
  ]);
  const task = rankTask(ctx, prepareSearch(ctx, new Map()), 'source');
  const answer = (reason) => ({
    recommendations: [{ id: 'target', relation: 'concept_explanation', reason, sourceBlock: 0, targetBlock: 0 }],
  });
  assert.equal(readRanking('source', task, answer('OCR의 원리를 설명합니다.'))[0].id, 'target');
  // Observed with the fixed model on an English article.
  assert.throws(() => readRanking('source', task, answer('OCRの基本的な仕組みを理解できます。')), /Invalid recommendation/);
  assert.throws(() => readRanking('source', task, answer('Explains how OCR works.')), /Invalid recommendation/);
  assert.throws(() => readRanking('source', task, answer('책 등뼈 검출을どう解釈するか')), /Invalid recommendation/);
});

test('series neighbours stay in the list and a hand-corrected reason replaces the model sentence', () => {
  const baseline = [
    {
      id: 'part2',
      related: [
        { id: 'part3', reason: '같은 시리즈의 다음 글', score: 0.6 },
        { id: 'part1', reason: '같은 시리즈의 이전 글', score: 0.6 },
        { id: 'far', reason: '같은 시리즈의 다른 글', score: 0.3 },
        { id: 'topic', reason: '공통 개념 · OCR', score: 0.2 },
      ],
    },
  ];
  const entry = (recommendations) =>
    new Map([['part2', { entry: { status: recommendations.length ? 'verified' : 'abstained', candidates: [{ id: 'x' }], recommendations } }]]);
  const picked = [
    { id: 'guide', reason: '모델이 쓴 문장' },
    { id: 'part3', reason: '모델이 고른 다음 편' },
  ];
  assert.deepEqual(projectRecommendations(baseline, entry(picked), {}, { part2: { guide: '작성자가 고친 문장' } })[0].related, [
    { id: 'guide', reason: '작성자가 고친 문장' },
    { id: 'part3', reason: '모델이 고른 다음 편' },
    { id: 'part1', reason: '같은 시리즈의 이전 글' },
  ]);
  // A judged abstention still shows the adjacent parts, and only those.
  assert.deepEqual(
    projectRecommendations(baseline, entry([]), {})[0].related.map((row) => row.id),
    ['part3', 'part1'],
  );
  // Four picks leave no room: the model's list is not displaced.
  const four = ['a', 'b', 'c', 'd'].map((id) => ({ id, reason: '이유' }));
  assert.deepEqual(projectRecommendations(baseline, entry(four), {})[0].related.map((row) => row.id), ['a', 'b', 'c', 'd']);
});

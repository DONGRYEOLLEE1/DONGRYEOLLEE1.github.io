// Verification of stored records, hash composition and candidate merging on a corpus where titles
// differ from IDs, excerpts are chosen out of document order and one article has an LLM fact.
import assert from 'node:assert/strict';
import test from 'node:test';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { evidenceBlocks, hash } from '../tools/knowledge/evidence.mjs';
import { loadVectors, memoryStore } from '../tools/knowledge/embeddings.mjs';
import { readRecords } from '../tools/knowledge/store.mjs';
import {
  auditAll,
  auditSummary,
  factHash,
  factTask,
  factsInstruction,
  manifest,
  prepareSearch,
  projectRecommendations,
  rankHash,
  rankInstruction,
  rankTask,
  storedRankHash,
} from '../tools/knowledge/phase2a.mjs';
import {
  HTML,
  TITLES,
  article,
  corpus,
  embedder,
  library,
  model,
  pick,
  readerApplies,
  rules,
  workspace,
} from './fixtures.mjs';

const judge = (overrides = {}) => model({ facts: readerApplies, rank: pick, ...overrides });
const SETTINGS = { seed: 7 };

async function synced(t, extra) {
  const ws = await workspace(t);
  const ctx = corpus(library(), extra);
  await ws.sync(ctx, judge());
  return { ws, ctx };
}
const reasons = (audit) => Object.fromEntries(audit.invalid);
async function rewrite(ws, id, change) {
  const record = await ws.record(id);
  await writeFile(join(ws.dir, id + '.json'), JSON.stringify(change(record) ?? record, null, 2) + '\n');
}

test('after a sync every record verifies and its stored candidates reproduce the rerank hash', async (t) => {
  const { ws, ctx } = await synced(t);
  const audit = await ws.audit(ctx);
  assert.deepEqual(reasons(audit), {});
  assert.equal(audit.valid.size, 6);

  const search = prepareSearch(ctx, audit.facts, await ws.vectors(ctx));
  let outOfOrder = 0;
  for (const { id, title } of ctx.articles) {
    assert.notEqual(title, id);
    const task = rankTask(ctx, search, id);
    const expected = rankHash(id, task, 'fixture', SETTINGS);
    assert.equal(
      storedRankHash(ctx, id, { candidates: manifest(task) }, 'fixture', SETTINGS, audit.facts),
      expected,
    );
    assert.equal((await ws.record(id)).entry.inputHash, expected);
    const order = ctx.usable.get(id).map((block) => block.id);
    for (const pair of task.candidates.values()) {
      const positions = pair.sourceBlocks.map((block) => order.indexOf(block.id));
      if (positions.some((position, i) => i && position < positions[i - 1])) outOfOrder++;
    }
  }
  // The fixture must exercise what the hash promises: excerpts keep their selection order.
  assert.ok(outOfOrder > 0);
  assert.notEqual(
    storedRankHash(ctx, 'scan', (await ws.record('scan')).entry, 'other', SETTINGS, audit.facts),
    (await ws.record('scan')).entry.inputHash,
  );
});

test('the two input hashes are exactly the documented compositions', async (t) => {
  const { ws, ctx } = await synced(t);
  const audit = await ws.audit(ctx);
  const scan = ctx.byId.get('scan');
  const blocks = ctx.usable.get('scan');
  // The reference list is not evidence and is not hashed; the dictionary is deliberately left out.
  assert.equal(blocks.length, 6);
  assert.ok(!blocks.some((block) => block.heading === 'References'));
  assert.equal(
    factHash(ctx, 'scan', 'm', SETTINGS),
    hash(['m', SETTINGS, factsInstruction, scan.title, scan.conceptRefs, blocks.map((block) => block.hash)]),
  );

  const task = rankTask(ctx, prepareSearch(ctx, audit.facts, await ws.vectors(ctx)), 'scan');
  const rows = [...task.candidates]
    .map(([id, pair]) => ({
      id,
      title: TITLES[id],
      paths: pair.paths,
      relationOptions: pair.relationOptions,
      sourceBlocks: pair.sourceBlocks.map((block) => [block.id, block.hash]),
      targetBlocks: pair.targetBlocks.map((block) => [block.id, block.hash]),
    }))
    .sort((a, b) => (a.id < b.id ? -1 : 1));
  assert.deepEqual(
    rows.map((row) => row.id),
    ['index', 'reader', 'tone', 'voice'],
  );
  assert.deepEqual(rows[1].paths, [{ concept: 'ocr', sourceRoles: ['about'], targetRoles: ['applies'] }]);
  assert.deepEqual(rows[1].relationOptions, [
    'application_example',
    'related_method',
    'evaluation_comparison',
  ]);
  assert.equal(rankHash('scan', task, 'm', SETTINGS), hash(['m', SETTINGS, rankInstruction, 'scan', rows]));
});

test('an edit that no stored quote covers is still caught by the input hash', async (t) => {
  const { ws, ctx } = await synced(t);
  const records = await readRecords(ws.dir);
  const scan = records.get('scan').entry;
  assert.deepEqual(
    [scan.recommendations[0].sourceEvidence.blockId, scan.recommendations[0].targetEvidence.blockId],
    ['["method",1]', '["engine",1]'],
  );
  assert.equal(records.get('tone').entry.status, 'abstained');
  const audited = (change) => reasons(auditAll(corpus(library(change)), records));

  // reader's Tune block is an excerpt shown to scan but not the one its recommendation quotes.
  const tuned = audited({
    reader: { html: HTML.reader.replace('gamma tuning sweep', 'gamma tuning sweep, redone') },
  });
  assert.equal(tuned.scan, '추천 입력 해시 불일치');
  assert.equal(tuned.reader, '관계 입력 해시 불일치');
  // An abstained record quotes nothing at all; its candidates' excerpts are still hashed.
  assert.equal(tuned.tone, '추천 입력 해시 불일치');
  assert.equal(
    audited({ voice: { html: HTML.voice.replace('음성 합성', '음성 합성 개정') } }).tone,
    '추천 입력 해시 불일치',
  );

  // A candidate's title is part of the prompt. tone and blank do not have index as a candidate.
  assert.deepEqual(audited({ index: { title: 'Index shards, second edition' } }), {
    scan: '추천 입력 해시 불일치',
    reader: '추천 입력 해시 불일치',
    voice: '추천 입력 해시 불일치',
    index: '관계 입력 해시 불일치',
  });

  const tampered = (id, change) => {
    const copy = new Map(records);
    const record = structuredClone(records.get(id));
    change(record);
    copy.set(id, record);
    return reasons(auditAll(ctx, copy));
  };
  assert.deepEqual(
    tampered('scan', (r) => (r.entry.inputHash = '0'.repeat(64))),
    { scan: '추천 입력 해시 불일치' },
  );
  // The model and settings are hashed into both steps.
  assert.deepEqual(
    tampered('scan', (r) => (r.model = 'other')),
    { scan: '관계 입력 해시 불일치' },
  );
  for (const change of [(r) => (r.model = 'other'), (r) => (r.settings = { seed: 8 })])
    assert.deepEqual(
      tampered('scan', (r) => {
        change(r);
        r.facts.inputHash = factHash(ctx, 'scan', r.model, r.settings);
      }),
      { scan: '추천 입력 해시 불일치' },
    );
  // A candidate that no longer exists makes the hash unrebuildable; a missing stored hash must
  // not be taken for a match.
  assert.deepEqual(
    tampered('tone', (r) => {
      r.entry.inputHash = null;
      r.entry.candidates.push({ id: 'no-such-article', sourceBlocks: [], targetBlocks: [] });
    }),
    { tone: '추천 입력 해시 불일치' },
  );
  assert.deepEqual(
    tampered('tone', (r) => delete r.entry.inputHash),
    { tone: '추천 입력 해시 불일치' },
  );
  assert.deepEqual(
    tampered('tone', (r) => (r.facts.inputHash = null)),
    { tone: '관계 입력 해시 불일치' },
  );
});

test('each check of a stored record rejects that record alone', async (t) => {
  const { ws, ctx } = await synced(t);
  const records = await readRecords(ws.dir);
  const facts = auditAll(ctx, records).facts;
  const block = (id, index) => {
    const found = ctx.usable.get(id)[index];
    return { articleId: id, blockId: found.id, anchor: found.anchor, quote: found.quote, hash: found.hash };
  };
  // Recompute the stored hash after a change, so that only the row checks can reject the record.
  const rehash = (id, record) =>
    (record.entry.inputHash = storedRankHash(ctx, id, record.entry, record.model, record.settings, facts));
  const row = (id, relation = 'related_method') => ({
    id,
    relation,
    reason: '이유',
    sourceEvidence: block('scan', 1),
    targetEvidence: block(id, 0),
  });
  const cases = [
    ['id of another article', 'scan', (r) => (r.id = 'reader'), 'ID 불일치'],
    ['schema version', 'scan', (r) => (r.version = 1), '스키마 버전 불일치'],
    ['settings missing', 'scan', (r) => delete r.settings, '모델 정보 없음'],
    ['model missing', 'scan', (r) => delete r.model, '모델 정보 없음'],
    ['entry missing', 'scan', (r) => (r.entry = null), '추천 없음'],
    ['unknown status', 'scan', (r) => (r.entry.status = 'done'), '추천 상태 오류'],
    ['verified without rows', 'scan', (r) => (r.entry.recommendations = []), '추천 상태 불일치'],
    ['abstained with a row', 'scan', (r) => (r.entry.status = 'abstained'), '추천 상태 불일치'],
    ['fallback without origin', 'scan', (r) => (r.entry.status = 'fallback'), '추천 상태 불일치'],
    [
      'rules fallback with rows',
      'scan',
      (r) => Object.assign(r.entry, { status: 'fallback', fallbackFrom: 'rules' }),
      '추천 상태 불일치',
    ],
    [
      'five rows',
      'scan',
      (r) => (r.entry.recommendations = ['reader', 'index', 'voice', 'tone', 'reader'].map((id) => row(id))),
      '추천 근거 불일치',
    ],
    [
      'a row twice',
      'scan',
      (r) => r.entry.recommendations.push(r.entry.recommendations[0]),
      '추천 근거 불일치',
    ],
    [
      'self reference',
      'scan',
      (r) => {
        r.entry.candidates.push({
          id: 'scan',
          sourceBlocks: [block('scan', 1).blockId],
          targetBlocks: [block('scan', 0).blockId],
        });
        r.entry.recommendations = [row('scan')];
        rehash('scan', r);
      },
      '추천 근거 불일치',
    ],
    [
      'relation the pair does not allow',
      'scan',
      (r) => (r.entry.recommendations[0].relation = 'concept_explanation'),
      '추천 근거 불일치',
    ],
    ['unknown relation', 'scan', (r) => (r.entry.recommendations[0].relation = 'sequel'), '추천 근거 불일치'],
    ['empty reason', 'scan', (r) => (r.entry.recommendations[0].reason = '  '), '추천 근거 불일치'],
    ['reason of another type', 'scan', (r) => (r.entry.recommendations[0].reason = 7), '추천 근거 불일치'],
    [
      // A real block of the source, quoted correctly, but not one the model was shown for this pair.
      'quote outside the stored excerpts',
      'scan',
      (r) => (r.entry.recommendations[0].sourceEvidence = block('scan', 3)),
      '추천 근거 불일치',
    ],
    [
      'target quote outside the stored excerpts',
      'scan',
      (r) => (r.entry.recommendations[0].targetEvidence = block('reader', 2)),
      '추천 근거 불일치',
    ],
    [
      'target outside the stored candidates',
      'scan',
      (r) => {
        r.entry.candidates = r.entry.candidates.filter((c) => c.id !== 'reader');
        rehash('scan', r);
      },
      '추천 근거 불일치',
    ],
    ['row that is not an object', 'scan', (r) => (r.entry.recommendations = [null]), '추천 근거 불일치'],
    [
      'more than six facts',
      'reader',
      (r) => (r.facts.facts = Array(7).fill(r.facts.facts[0])),
      '관계 형식 오류',
    ],
    ['a fact twice', 'reader', (r) => r.facts.facts.push(r.facts.facts[0]), '관계 중복'],
    ['verified without facts', 'reader', (r) => (r.facts.facts = []), '관계 상태 불일치'],
    ['abstained with facts', 'reader', (r) => (r.facts.status = 'abstained'), '관계 상태 불일치'],
    ['unknown facts status', 'reader', (r) => (r.facts.status = 'ok'), '관계 상태 불일치'],
    ['fact with an unknown role', 'reader', (r) => (r.facts.facts[0].role = 'mentions'), '관계 근거 불일치'],
    ['fact quoting other text', 'reader', (r) => (r.facts.facts[0].evidence.quote = 'x'), '관계 근거 불일치'],
    // Shapes nobody writes on purpose (hand edits, a botched merge): invalid, never an exception.
    [
      'candidate blocks that are a number',
      'scan',
      (r) => (r.entry.candidates.find((c) => c.id === 'reader').sourceBlocks = 5),
      '추천 근거 불일치',
    ],
    [
      'candidate blocks that are an object',
      'scan',
      (r) => (r.entry.candidates.find((c) => c.id === 'reader').targetBlocks = {}),
      '추천 근거 불일치',
    ],
    ['candidates that are a string', 'scan', (r) => (r.entry.candidates = 'reader'), '추천 근거 불일치'],
    ['a null candidate', 'tone', (r) => r.entry.candidates.push(null), '추천 입력 해시 불일치'],
    ['entry that is a string', 'scan', (r) => (r.entry = 'verified'), '추천 상태 오류'],
    ['facts that are an array', 'scan', (r) => (r.facts = []), '관계 형식 오류'],
    ['a null fact', 'reader', (r) => (r.facts.facts = [null]), '관계 근거 불일치'],
    ['settings that are null', 'scan', (r) => (r.settings = null), '모델 정보 없음'],
  ];
  for (const [name, id, change, reason] of cases) {
    const copy = new Map(records);
    const record = structuredClone(records.get(id));
    change(record);
    copy.set(id, record);
    let audit;
    assert.doesNotThrow(() => (audit = auditAll(ctx, copy)), name);
    assert.equal(audit.invalid.get(id), reason, name);
    // reader's fact feeds the relations of its pairs with scan, index and voice (they all have an
    // OCR role), so losing it invalidates those three; no other change reaches a second record.
    const collateral = [...audit.invalid.keys()].filter((other) => other !== id);
    assert.deepEqual(collateral, id === 'reader' ? ['scan', 'index', 'voice'] : [], name);
  }
  assert.deepEqual(reasons(auditAll(ctx, new Map([...records, ['scan', ['not', 'a', 'record']]]))), {
    scan: '스키마 버전 불일치',
  });
  // Whatever a record does to the checks, the damage stays with its own article.
  const hostile = structuredClone(records.get('tone'));
  Object.defineProperty(hostile, 'entry', {
    get() {
      throw new TypeError('unexpected');
    },
  });
  const contained = auditAll(ctx, new Map([...records, ['tone', hostile]]));
  assert.deepEqual(reasons(contained), { tone: '레코드 형식 오류' });
  assert.equal(contained.valid.size, 5);

  // Five rows, each valid on its own: only the limit of four rejects them.
  const peers = corpus(
    ['p0', 'p1', 'p2', 'p3', 'p4', 'p5'].map((id) => article(id, `<p>alpha note ${id}</p>`)),
  );
  const many = await workspace(t);
  const four = (input) => ({
    recommendations: input.candidates.slice(0, 4).map((c) => ({
      id: c.id,
      relation: 'related_method',
      reason: '이유',
      sourceBlock: 0,
      targetBlock: 0,
    })),
  });
  await many.sync(peers, model({ rank: four }));
  assert.deepEqual(reasons(await many.audit(peers)), {});
  await rewrite(many, 'p0', (r) => {
    const extra = r.entry.candidates.find(
      (c) => !r.entry.recommendations.some((picked) => picked.id === c.id),
    );
    const quote = peers.usable.get(extra.id)[0];
    r.entry.recommendations.push({
      ...r.entry.recommendations[0],
      id: extra.id,
      targetEvidence: {
        ...r.entry.recommendations[0].targetEvidence,
        articleId: extra.id,
        quote: quote.quote,
        hash: quote.hash,
      },
    });
  });
  assert.deepEqual(reasons(await many.audit(peers)), { p0: '추천 근거 불일치' });

  // An exclusion added after the sync is not part of any hash; the row check alone enforces it.
  const excluded = corpus(library(), { overrides: { excludes: { scan: ['reader'] } } });
  assert.deepEqual(reasons(auditAll(excluded, records)), { scan: '추천 근거 불일치' });
  // Excluding a candidate that was not recommended leaves the published record valid.
  const unrelated = corpus(library(), { overrides: { excludes: { scan: ['tone'] } } });
  assert.deepEqual(reasons(auditAll(unrelated, records)), {});
});

test('an invalid or failed record shows the rule-based list, which is not empty', async (t) => {
  const { ws, ctx } = await synced(t);
  const baseline = rules(ctx);
  const ruled = (id) => baseline.find((a) => a.id === id).related;
  assert.deepEqual(
    ruled('scan').map((r) => [r.id, r.reason]),
    [
      ['voice', '공통 개념 · OCR'],
      ['index', '적용 사례 · OCR'],
    ],
  );
  assert.ok(ruled('index').length > 0 && ruled('voice').length > 0);

  const whole = await ws.audit(ctx);
  const shown = (audit, id) =>
    projectRecommendations(baseline, audit.valid, {}).find((a) => a.id === id).related;
  assert.deepEqual(shown(whole, 'scan'), [{ id: 'reader', reason: 'OCR 적용 사례' }]);
  assert.deepEqual(
    [auditSummary(whole).llm, auditSummary(whole).rules, auditSummary(whole).fallback],
    // The blockless article was never judged and stays with the rules.
    [5, 1, 0],
  );

  await rewrite(ws, 'scan', (r) => void (r.entry.recommendations[0].targetEvidence.quote = 'changed'));
  await rewrite(ws, 'voice', (r) => void (r.version = 1));
  const broken = await ws.audit(ctx);
  assert.deepEqual(reasons(broken), { scan: '추천 근거 불일치', voice: '스키마 버전 불일치' });
  assert.deepEqual(shown(broken, 'scan'), ruled('scan'));
  assert.deepEqual(shown(broken, 'voice'), ruled('voice'));
  assert.notDeepEqual(shown(broken, 'index'), ruled('index'));
  const summary = auditSummary(broken);
  assert.deepEqual([summary.llm, summary.rules, summary.missing, summary.invalid], [3, 3, 0, 2]);

  // A failed call with nothing to fall back on is a valid record that shows the rules.
  const failing = judge({
    rank: (input) => {
      if (input.source.id === 'scan') throw Error('Inference HTTP 503');
      return pick(input);
    },
  });
  await ws.sync(ctx, failing);
  const entry = (await ws.record('scan')).entry;
  assert.deepEqual([entry.status, entry.fallbackFrom, entry.recommendations], ['fallback', 'rules', []]);
  const failed = await ws.audit(ctx);
  assert.deepEqual(reasons(failed), {});
  assert.deepEqual(shown(failed, 'scan'), ruled('scan'));
  const counts = auditSummary(failed);
  assert.deepEqual([counts.llm, counts.rules, counts.fallback, counts.fallbackIds], [4, 2, 1, ['scan']]);
});

test('pins come first, a pinned pick is not repeated and the list stops at four', () => {
  const baseline = [
    {
      id: 's',
      related: [
        { id: 'rule', reason: '규칙', score: 1 },
        { id: 'pin', reason: '작성자가 선택한 이어 읽기', score: 3 },
      ],
    },
  ];
  const picks = ['m1', 'pin', 'm2', 'm3'].map((id) => ({
    id,
    reason: id + ' 이유',
    relation: 'related_method',
  }));
  const valid = new Map([['s', { entry: { status: 'verified', recommendations: picks } }]]);
  const pinned = projectRecommendations(baseline, valid, { pins: { s: ['pin'] } })[0].related;
  assert.deepEqual(
    pinned.map((r) => r.id),
    ['pin', 'm1', 'm2', 'm3'],
  );
  // Neither a pinned rule row nor a model row carries anything but the ID and the reason.
  assert.deepEqual(pinned.slice(0, 2), [
    { id: 'pin', reason: '작성자가 선택한 이어 읽기' },
    { id: 'm1', reason: 'm1 이유' },
  ]);

  const four = ['m1', 'm2', 'm3', 'm4'].map((id) => ({ id, reason: id }));
  valid.set('s', { entry: { status: 'verified', recommendations: four } });
  assert.deepEqual(
    projectRecommendations(baseline, valid, { pins: { s: ['pin'] } })[0].related.map((r) => r.id),
    ['pin', 'm1', 'm2', 'm3'],
  );
  assert.deepEqual(
    projectRecommendations(baseline, valid, {})[0].related.map((r) => r.id),
    ['m1', 'm2', 'm3', 'm4'],
  );
});

test('stored facts of an edited article still serve other pairs while their evidence stands', async (t) => {
  const { ws, ctx } = await synced(t);
  const records = await readRecords(ws.dir);
  const audited = (change) => auditAll(corpus(library(change)), records);
  const baseline = rules(ctx);

  // reader is edited away from its fact and from every stored excerpt: only reader is stale.
  const aside = audited({
    reader: { html: HTML.reader.replace('plain trailing remark', 'plain trailing remark, edited') },
  });
  assert.deepEqual(reasons(aside), { reader: '관계 입력 해시 불일치' });
  assert.deepEqual(projectRecommendations(baseline, aside.valid, {}).find((a) => a.id === 'scan').related, [
    { id: 'reader', reason: 'OCR 적용 사례' },
  ]);

  // The block that proves the fact changes: the fact is gone and so is the relation scan relied on.
  const proof = audited({
    reader: { html: HTML.reader.replace('uses OCR for each line', 'reads each line') },
  });
  assert.equal(reasons(proof).reader, '관계 입력 해시 불일치');
  assert.equal(reasons(proof).scan, '추천 근거 불일치');

  // index quotes that very block; voice was only shown it.
  assert.deepEqual(
    [reasons(proof).index, reasons(proof).voice],
    ['추천 근거 불일치', '추천 입력 해시 불일치'],
  );

  // An unreadable file contributes no facts at all, which is the same loss for the three articles
  // whose pair with reader had an OCR path. tone also lists reader as a candidate but shares no
  // concept with it, and is unaffected.
  const unreadable = auditAll(ctx, new Map([...records, ['reader', { id: 'reader', invalid: 'JSON 오류' }]]));
  assert.deepEqual(reasons(unreadable), {
    scan: '추천 근거 불일치',
    reader: 'JSON 오류',
    index: '추천 입력 해시 불일치',
    voice: '추천 입력 해시 불일치',
  });
  assert.ok(unreadable.valid.has('tone') && unreadable.valid.has('blank'));
});

test('each hashed input re-runs exactly the calls that depend on it', async (t) => {
  const run = async (change, options) => {
    const { ws } = await synced(t);
    const next = corpus(library(change));
    const before = reasons(await ws.audit(next));
    const provider = judge(options?.provider);
    const summary = await ws.sync(next, provider, options?.sync);
    assert.deepEqual(reasons(await ws.audit(next)), {});
    return { before, calls: provider.calls.toSorted(), summary };
  };

  // A block that is no excerpt anywhere: the article's facts are extracted again, nothing else.
  const body = await run({
    scan: { html: HTML.scan.replace('plain closing remark', 'plain closing remark, edited') },
  });
  assert.deepEqual(body.before, { scan: '관계 입력 해시 불일치' });
  assert.deepEqual(body.calls, ['facts:' + TITLES.scan]);
  assert.deepEqual(body.summary.facts.items, [{ id: 'scan', reason: 'changed', call: true }]);

  // The reviewed concept roles are sent with the facts prompt and decide the pair relations.
  const roles = await run({ voice: { refs: [] } });
  assert.deepEqual(roles.calls, [
    'facts:' + TITLES.voice,
    'rank:index',
    'rank:reader',
    'rank:scan',
    'rank:voice',
  ]);

  // A new title: that article's facts, and the rerank of every article that lists it as a candidate.
  const title = await run({ index: { title: 'Index shards, second edition' } });
  assert.deepEqual(title.calls, [
    'facts:Index shards, second edition',
    'rank:reader',
    'rank:scan',
    'rank:voice',
  ]);
  assert.ok(title.summary.rank.items.every((item) => item.reason === 'evidence'));

  // An excerpt of the source itself, quoted by nobody.
  const excerpt = await run({ tone: { html: '<p>beta 발화 길이 개정</p>' } });
  assert.ok(excerpt.calls.includes('rank:tone') && excerpt.calls.includes('facts:' + TITLES.tone));

  // reader's fact disappears on re-extraction: only pairs whose relation options it fed are re-ranked.
  const relation = await run({}, { provider: { facts: () => ({ facts: [] }) }, sync: { reextract: true } });
  assert.deepEqual(
    relation.calls.filter((call) => call.startsWith('rank:')),
    ['rank:index', 'rank:reader', 'rank:scan', 'rank:voice'],
  );
  assert.equal(relation.calls.filter((call) => call.startsWith('facts:')).length, 5);
});

test('a dry run leaves a record untouched even when its facts are due and its ranking is not', async (t) => {
  const ws = await workspace(t);
  // The model proposes a role against the reviewed one, so index has a review entry to lose.
  const facts = (input) =>
    input.title === TITLES.index
      ? { facts: [{ conceptId: 'ocr', role: 'about', block: 0 }] }
      : readerApplies(input);
  await ws.sync(corpus(library()), judge({ facts }));
  assert.deepEqual((await ws.record('index')).facts.review, [{ id: 'ocr', proposedRole: 'about' }]);
  const before = await ws.files();

  const forced = await ws.sync(corpus(library()), judge(), { dryRun: true, reextract: true });
  assert.equal(forced.facts.items.filter((item) => item.call).length, 5);
  assert.deepEqual(forced.rank.items, []);
  assert.deepEqual(await ws.files(), before);

  const edited = corpus(
    library({ scan: { html: HTML.scan.replace('plain closing remark', 'plain closing remark, edited') } }),
  );
  const provider = judge();
  const plan = await ws.sync(edited, provider, { dryRun: true });
  assert.deepEqual([plan.facts.items.map((item) => item.id), plan.rank.items], [['scan'], []]);
  assert.deepEqual(provider.calls, []);
  assert.deepEqual(await ws.files(), before);
});

test('a failed extraction keeps the facts that still hold instead of erasing them', async (t) => {
  const { ws } = await synced(t);
  const failing = () =>
    judge({
      facts: (input) => {
        if (input.title === TITLES.reader) throw Error('Inference HTTP 503');
        return readerApplies(input);
      },
    });
  const before = await ws.files();

  // Forced re-extraction of facts that are current: the failure changes nothing on disk.
  const again = failing();
  const forced = await ws.sync(corpus(library()), again, { reextract: true });
  assert.equal(forced.facts.failed, 1);
  assert.ok(!again.calls.some((call) => call.startsWith('rank:')));
  assert.deepEqual(await ws.files(), before);

  // reader is edited outside its fact: the failed call leaves the fact in place as a fallback, no
  // other article is re-ranked, and the next sync retries reader alone.
  const edited = corpus(
    library({
      reader: { html: HTML.reader.replace('plain trailing remark', 'plain trailing remark, edited') },
    }),
  );
  const down = failing();
  const summary = await ws.sync(edited, down);
  assert.deepEqual(down.calls, ['facts:' + TITLES.reader]);
  assert.equal(summary.facts.failed, 1);
  const stored = (await ws.record('reader')).facts;
  assert.deepEqual(
    [stored.status, stored.facts.map((f) => f.id + '/' + f.role)],
    ['fallback', ['ocr/applies']],
  );
  assert.deepEqual(reasons(await ws.audit(edited)), {});
  const after = await ws.files();
  assert.deepEqual(
    Object.keys(after).filter((name) => after[name] !== before[name]),
    ['reader.json'],
  );
  const retry = judge();
  const healed = await ws.sync(edited, retry);
  assert.deepEqual(retry.calls, ['facts:' + TITLES.reader]);
  assert.deepEqual(healed.facts.items, [{ id: 'reader', reason: 'fallback-retry', call: true }]);
  assert.equal((await ws.record('reader')).facts.status, 'verified');

  // The block that proved the fact is rewritten: nothing survives the failed call.
  const rewritten = corpus(
    library({ reader: { html: HTML.reader.replace('uses OCR for each line', 'reads each line') } }),
  );
  await ws.sync(rewritten, failing());
  const lost = (await ws.record('reader')).facts;
  assert.deepEqual([lost.status, lost.facts], ['fallback', []]);
});

// ── Candidate search ────────────────────────────────────────────────────────

// Six articles per ranking, each visible to that ranking alone: g* share reviewed concepts with
// the source, l* share its words, e* point along its embedding axis inside tokens of their own.
function rankings() {
  const dictionary = [1, 2, 3, 4, 5, 6].map((n) => ({
    id: 'c' + n,
    label: 'Zeta' + n,
    description: 'Concept ' + n,
    aliases: [],
    broaderIds: [],
  }));
  const words = ['kappaone', 'kappatwo', 'kappathree', 'kappafour', 'kappafive', 'kappasix'];
  const about = (count) => dictionary.slice(0, count).map((c) => ({ id: c.id, role: 'about' }));
  const articles = [article('s', `<p>salpha ${words.join(' ')}</p>`, about(6), '', 'Source')];
  for (let k = 1; k <= 6; k++) {
    articles.push(article('g' + k, `<p>graph${k} filler${k}</p>`, about(7 - k), '', 'Gee' + k));
    articles.push(article('l' + k, `<p>${words.slice(0, 7 - k).join(' ')}</p>`, [], '', 'Lex' + k));
    const text = Array(7 - k)
      .fill(`e${k}alpha`)
      .concat(Array(k).fill(`e${k}beta`))
      .join(' ');
    articles.push(article('e' + k, `<p>${text}</p>`, [], '', 'Emb' + k));
  }
  articles.push(article('pin', '<p>unrelated pinned note</p>', [], '', 'Pinned'));
  return { articles, concepts: dictionary };
}

test('candidates alternate between the graph, lexical and embedding rankings and stop at fifteen', async () => {
  const { articles, concepts: dictionary } = rankings();
  const candidates = async (overrides) => {
    const ctx = corpus(articles, { concepts: dictionary, overrides });
    const vectors = await loadVectors(ctx, embedder(), memoryStore());
    const search = prepareSearch(ctx, new Map(), vectors);
    return { search, keys: [...rankTask(ctx, search, 's').candidates.keys()] };
  };
  const { search, keys } = await candidates({});
  // Each ranking sees its own six articles, best first, and none of the others.
  assert.deepEqual(search.graph.get('s'), ['g1', 'g2', 'g3', 'g4', 'g5', 'g6']);
  assert.deepEqual(search.similarity.ranked('s'), ['e1', 'e2', 'e3', 'e4', 'e5', 'e6']);
  assert.ok(
    ['l1', 'l2', 'l3', 'l4', 'l5', 'l6'].every(
      (id, i, all) => !i || search.lexical('s', all[i - 1]) > search.lexical('s', id),
    ),
  );
  assert.equal(search.lexical('s', 'g1') + search.lexical('s', 'e1') + search.lexical('s', 'pin'), 0);
  assert.deepEqual(keys, [
    ...['g1', 'l1', 'e1', 'g2', 'l2', 'e2', 'g3', 'l3', 'e3'],
    ...['g4', 'l4', 'e4', 'g5', 'l5', 'e5'],
  ]);

  const pinned = await candidates({ pins: { s: ['pin'] }, excludes: { s: ['l1'] } });
  assert.equal(pinned.keys.length, 15);
  assert.equal(pinned.keys[0], 'pin');
  assert.ok(!pinned.keys.includes('l1') && pinned.keys.includes('l2'));
});

test('similarity excerpts cover different sections first and skip evidence of an unshared concept', async () => {
  const source = article(
    'source',
    '<h2 id="one">One</h2><p>alpha first</p><p>alpha alpha beta second</p><p>alpha beta third</p>' +
      '<h2 id="two">Two</h2><p>delta OCR remark</p>' +
      '<h2 id="three">Three</h2><p>alpha gamma gamma gamma fourth</p>',
  );
  const target = article('target', '<p>alpha guide</p>');
  const ctx = corpus([source, target]);
  const blocks = ctx.usable.get('source');
  const picked = async (facts) => {
    const vectors = await loadVectors(ctx, embedder(), memoryStore());
    return rankTask(ctx, prepareSearch(ctx, facts, vectors), 'source')
      .candidates.get('target')
      .sourceBlocks.map((block) => block.quote);
  };
  // Cosines to the target: 1, 0.89, 0.71 under One; 0 under Two; 0.32 under Three. The second
  // pick is the best block of another section, not the second-best block overall.
  assert.deepEqual(await picked(new Map()), [
    'alpha first',
    'alpha gamma gamma gamma fourth',
    'delta OCR remark',
  ]);

  // source applies OCR in its Two section, but target has nothing to do with OCR: that evidence
  // has no priority for this pair.
  const fact = {
    id: 'ocr',
    role: 'applies',
    evidence: {
      articleId: 'source',
      blockId: blocks[3].id,
      anchor: 'two',
      quote: blocks[3].quote,
      hash: blocks[3].hash,
    },
  };
  const facts = new Map([['source', { facts: [fact] }]]);
  assert.deepEqual(await picked(facts), [
    'alpha first',
    'alpha gamma gamma gamma fourth',
    'delta OCR remark',
  ]);
  const shared = corpus([source, article('target', '<p>alpha guide</p>', [{ id: 'ocr', role: 'about' }])]);
  const vectors = await loadVectors(shared, embedder(), memoryStore());
  assert.equal(
    rankTask(shared, prepareSearch(shared, facts, vectors), 'source').candidates.get('target').sourceBlocks[0]
      .quote,
    'delta OCR remark',
  );
});

test('a reference list is never embedded, shown to the model or used as an excerpt', async () => {
  const cited = article(
    'cited',
    '<h2 id="body">Body</h2><p>alpha body text</p><h2 id="references">References</h2><p>alpha alpha cited paper</p>' +
      '<h2 id="ko">3. 참고 문헌</h2><p>alpha 인용 논문</p>',
  );
  const ctx = corpus([cited, article('other', '<p>alpha alpha other text</p>')]);
  assert.equal(ctx.blocks.get('cited').length, 3);
  assert.deepEqual(
    ctx.usable.get('cited').map((block) => block.quote),
    ['alpha body text'],
  );
  const embed = embedder();
  const vectors = await loadVectors(ctx, embed, memoryStore());
  assert.deepEqual(embed.batches.flat().sort(), [
    'cited\nBody\nalpha body text',
    'other\n\nalpha alpha other text',
  ]);
  assert.deepEqual(factTask(ctx, 'cited').input.blocks, [[0, 'Body', 'alpha body text']]);
  const search = prepareSearch(ctx, new Map(), vectors);
  assert.deepEqual(
    rankTask(ctx, search, 'other')
      .candidates.get('cited')
      .targetBlocks.map((block) => block.quote),
    ['alpha body text'],
  );
});

test('a link to an article with a non-ASCII path is the first excerpt', async () => {
  const target = { ...article('regex', '<p>delta 정규식 안내</p>'), path: '/python/정규표현식.html' };
  const source = article(
    'source',
    '<p>delta delta closest block</p><p>See <a href="/python/정규표현식.html">정규식 글</a></p>' +
      '<p>See <a href="/python/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D.html">같은 글</a> again</p>',
  );
  const ctx = corpus([source, target]);
  const vectors = await loadVectors(ctx, embedder(), memoryStore());
  const pair = rankTask(ctx, prepareSearch(ctx, new Map(), vectors), 'source').candidates.get('regex');
  assert.deepEqual(
    pair.sourceBlocks.map((block) => block.quote),
    ['See 정규식 글', 'See 같은 글 again', 'delta delta closest block'],
  );
  // The article's own path may be the escaped spelling as well.
  const escaped = corpus([
    source,
    { ...target, path: '/python/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D.html' },
  ]);
  const same = rankTask(escaped, prepareSearch(escaped, new Map(), vectors), 'source').candidates.get(
    'regex',
  );
  assert.equal(same.sourceBlocks[0].quote, 'See 정규식 글');
});

test('block IDs stay unique under headings without an id or with a repeated id', async (t) => {
  const html =
    '<p>intro text</p><h2>No id</h2><p>alpha first body</p><p>alpha second body</p>' +
    '<h2 id="dup">Dup</h2><p>alpha third body</p><h2 id="dup">Dup again</h2><p>alpha fourth body</p>';
  const ids = evidenceBlocks(article('loose', html)).map((block) => block.id);
  assert.deepEqual(ids, ['[null,1]', '[null,2]', '[null,3]', '["dup",1]', '["dup",2]']);
  // With unique anchors nothing changes: numbering restarts under each heading.
  assert.deepEqual(
    evidenceBlocks(article('tidy', '<p>a</p><h2 id="x">X</h2><p>b</p><p>c</p><h2 id="y">Y</h2><p>d</p>')).map(
      (block) => block.id,
    ),
    ['[null,1]', '["x",1]', '["x",2]', '["y",1]'],
  );

  // Such an article used to be invalid right after every sync, and was called again each time.
  const ws = await workspace(t);
  const ctx = corpus([article('loose', html), article('peer', '<p>alpha peer text</p>')]);
  const rank = (input) => ({
    recommendations: input.candidates.slice(0, 1).map((c) => ({
      id: c.id,
      relation: 'related_method',
      reason: '이유',
      sourceBlock: c.sourceBlocks.length - 1,
      targetBlock: c.targetBlocks.length - 1,
    })),
  });
  await ws.sync(ctx, model({ rank }));
  assert.equal((await ws.audit(ctx)).valid.size, 2);
  const repeat = model({ rank });
  await ws.sync(ctx, repeat);
  assert.deepEqual(repeat.calls, []);
});

test('article IDs that differ only by case are refused before any call', async (t) => {
  const ws = await workspace(t);
  const provider = model();
  const ctx = corpus([article('Post', '<p>alpha one</p>'), article('post', '<p>alpha two</p>')]);
  await assert.rejects(ws.sync(ctx, provider), /differ only by case.*Post, post/);
  await assert.rejects(ws.sync(corpus([article('a/b', '<p>alpha</p>')]), provider), /file name/);
  assert.deepEqual(provider.calls, []);
  assert.deepEqual(await ws.files(), {});
});

// ── Refused requests ────────────────────────────────────────────────────────

const refusal = (status, detail) =>
  Object.assign(Error('Inference HTTP ' + status), { status, detail, rejected: true });

test('a stage whose first calls are all refused stops without writing a fallback everywhere', async (t) => {
  const ws = await workspace(t);
  const ctx = corpus(library());
  const logged = [];
  const locked = model({
    facts: () => {
      throw refusal(401, '{"error":{"message":"No auth credentials found"}}');
    },
  });
  await assert.rejects(
    ws.sync(ctx, locked, { concurrency: 1, log: (line) => logged.push(line) }),
    (error) => {
      assert.equal(error.code, 'INFERENCE_REFUSED');
      assert.match(error.message, /관계 추출 요청이 처음 3건 모두 거부/);
      assert.match(
        error.message,
        /Inference HTTP 401: \{"error":\{"message":"No auth credentials found"\}\}/,
      );
      return true;
    },
  );
  assert.equal(locked.calls.length, 3);
  assert.deepEqual(await ws.files(), {});
  // The provider's answer is in every log line, not only the status.
  assert.match(logged[0], /^scan facts: Inference HTTP 401: \{"error"/);

  // Facts succeed, then the rerank schema is refused: the facts are kept, no entry is written.
  const schema = model({
    facts: readerApplies,
    rank: () => {
      throw refusal(400, 'Invalid schema for response_format');
    },
  });
  await assert.rejects(
    ws.sync(ctx, schema, { concurrency: 2 }),
    /재정렬 요청이 처음 3건 모두 거부.*Invalid schema/,
  );
  const written = await readRecords(ws.dir);
  assert.equal(written.size, 6);
  assert.ok([...written.values()].every((record) => record.entry === null));
  assert.equal(written.get('reader').facts.status, 'verified');

  // An unreachable host is the same for every article.
  const offline = model({
    facts: () => {
      throw Object.assign(Error('Inference request failed (ECONNREFUSED)'), { unreachable: true });
    },
  });
  await assert.rejects(ws.sync(ctx, offline, { reextract: true }), /ECONNREFUSED/);
});

test('refusals that are not the whole story are recorded as ordinary fallbacks', async (t) => {
  // One success first: later refusals concern single articles (say, a prompt that is too long).
  const ws = await workspace(t);
  const ctx = corpus(library());
  const mixed = model({
    facts: readerApplies,
    rank: (input) => {
      if (input.source.id === 'scan') return pick(input);
      throw refusal(400, 'context length exceeded');
    },
  });
  const summary = await ws.sync(ctx, mixed, { concurrency: 1 });
  assert.equal(summary.rank.failed, 4);
  assert.equal((await ws.record('tone')).entry.status, 'fallback');
  assert.equal((await ws.record('scan')).entry.status, 'verified');

  // Fewer refusals than the limit: they are written once the stage ends.
  const single = await workspace(t);
  const denied = model({
    rank: () => {
      throw refusal(404, 'No endpoints found');
    },
  });
  const one = await single.sync(ctx, denied, { only: ['scan', 'tone'] });
  assert.equal(one.rank.failed, 2);
  assert.deepEqual(
    [(await single.record('scan')).entry.status, (await single.record('tone')).entry.status],
    ['fallback', 'fallback'],
  );

  // A rate limit or a server error is transient however often it repeats.
  const busy = await workspace(t);
  const limited = model({
    facts: () => {
      throw Object.assign(Error('Inference HTTP 429'), { status: 429, rejected: false });
    },
  });
  const throttled = await busy.sync(ctx, limited);
  assert.equal(throttled.facts.failed, 5);
  assert.equal(Object.keys(await busy.files()).length, 6);
});

test('a sync repairs a record of an unexpected shape instead of stopping on it', async (t) => {
  const { ws, ctx } = await synced(t);
  await rewrite(ws, 'scan', (r) => void (r.entry.candidates.find((c) => c.id === 'reader').sourceBlocks = 5));
  await rewrite(ws, 'index', (r) => void (r.entry.recommendations = [null]));
  await rewrite(ws, 'voice', (r) => void (r.facts.facts = [null]));
  assert.deepEqual(reasons(await ws.audit(ctx)), {
    scan: '추천 근거 불일치',
    index: '추천 근거 불일치',
    voice: '관계 상태 불일치',
  });
  const plan = await ws.sync(ctx, judge(), { dryRun: true });
  assert.deepEqual(
    plan.rank.items.map((item) => item.id),
    ['scan', 'index'],
  );
  const provider = judge();
  await ws.sync(ctx, provider);
  assert.deepEqual(provider.calls.toSorted(), ['facts:' + TITLES.voice, 'rank:index', 'rank:scan']);
  assert.deepEqual(reasons(await ws.audit(ctx)), {});
});

test('model calls run in parallel up to the concurrency limit and no further', async (t) => {
  const peak = async (concurrency) => {
    const ws = await workspace(t);
    let running = 0;
    let most = 0;
    const delay = async () => {
      most = Math.max(most, ++running);
      await new Promise((resolve) => setTimeout(resolve, 2));
      running--;
    };
    await ws.sync(corpus(library()), judge({ delay }), { concurrency });
    return most;
  };
  assert.equal(await peak(1), 1);
  assert.equal(await peak(3), 3);
  assert.equal(await peak(50), 5);
});

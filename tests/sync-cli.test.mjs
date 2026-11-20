// The command line (exit codes, what it writes and prints) and the build projection, driven through
// their entry points with a fake corpus loader and a fake fetch. Nothing leaves the process.
import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { applyRelated } from '../tools/content/build.mjs';
import { factsInstruction } from '../tools/knowledge/phase2a.mjs';
import { generateRelated } from '../tools/knowledge/related.mjs';
import { run } from '../tools/sync-recommendations.mjs';
import { TITLES, concepts, library, pick, readerApplies } from './fixtures.mjs';

const KEY = 'sk-or-test-key';
const AXES = ['alpha', 'beta', 'gamma', 'delta'];
const reply = (payload, status = 200) => ({
  ok: status < 300,
  status,
  json: async () => payload,
  text: async () => (typeof payload === 'string' ? payload : JSON.stringify(payload)),
});
const answer = (content) => ({
  choices: [{ finish_reason: 'stop', message: { content: JSON.stringify(content) } }],
  usage: { prompt_tokens: 10, completion_tokens: 2 },
});

/** An OpenAI-compatible endpoint. `chat(kind, input)` may return a payload or a ready response. */
function service(chat = (kind, input) => (kind === 'facts' ? readerApplies(input) : pick(input))) {
  const calls = [];
  const keys = new Set();
  const fetch = async (url, init) => {
    keys.add(init.headers.Authorization);
    const body = JSON.parse(init.body);
    if (url.endsWith('/embeddings')) {
      calls.push('embed');
      // The last axis is shared so that no text is the zero vector.
      const vector = (text) => [...AXES.map((word) => text.split(word).length - 1), 0, 0, 0, 0.05];
      return reply({
        data: body.input.map((text, index) => ({ index, embedding: vector(text) })),
        usage: { prompt_tokens: body.input.length },
      });
    }
    const input = JSON.parse(body.messages[1].content);
    const kind = body.messages[0].content === factsInstruction ? 'facts' : 'rank';
    calls.push(kind + ':' + (kind === 'facts' ? input.title : input.source.id));
    const result = chat(kind, input);
    return typeof result?.json === 'function' ? result : reply(answer(result));
  };
  return { calls, keys, fetch, chat: () => calls.filter((call) => call !== 'embed') };
}

async function bench(t) {
  const root = await mkdtemp(join(tmpdir(), 'cli-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const dir = join(root, 'related');
  const state = { change: {}, overrides: {} };
  const lines = [];
  const print = (line) => lines.push(String(line));
  return {
    root,
    dir,
    state,
    lines,
    text: () => lines.join('\n'),
    /** Runs the command against `remote`; returns the exit code. */
    async cli(args, remote = service(), env = { OPENROUTER_API_KEY: KEY }) {
      lines.length = 0;
      return run(args, {
        env: { EMBEDDINGS_DIMENSIONS: '8', ...env },
        dir,
        cache: join(root, 'cache'),
        loadCorpus: async () => {
          const articles = library(state.change);
          // What a publisher's pull or edit does while the model calls are in flight.
          if (state.afterLoad) Object.assign(state, { change: state.afterLoad, afterLoad: null });
          return { articles, concepts, series: [], overrides: state.overrides };
        },
        fetch: remote.fetch,
        out: print,
        warn: print,
        err: print,
      });
    },
    async files() {
      const out = {};
      for (const name of (await readdir(dir).catch(() => [])).sort())
        out[name] = await readFile(join(dir, name), 'utf8');
      return out;
    },
    record: async (id) => JSON.parse(await readFile(join(dir, id + '.json'), 'utf8')),
    async rewrite(id, change) {
      const record = JSON.parse(await readFile(join(dir, id + '.json'), 'utf8'));
      change(record);
      await writeFile(join(dir, id + '.json'), JSON.stringify(record, null, 2) + '\n');
    },
  };
}

test('sync writes one record per article, a second run calls nothing and the checks pass', async (t) => {
  const b = await bench(t);
  const remote = service();
  const report = join(b.root, 'report.json');
  assert.equal(await b.cli(['--json', report], remote), 0);
  assert.deepEqual(
    Object.keys(await b.files()),
    ['blank', 'index', 'reader', 'scan', 'tone', 'voice'].map((id) => id + '.json'),
  );
  assert.equal(remote.chat().length, 10);
  assert.match(b.text(), /글 6편, 모델 openai\/gpt-6-luna/);
  assert.match(b.text(), /facts: 호출 5 \/ 재사용 0 \/ 실패 0/);
  assert.match(b.text(), /연관 글: LLM 5편, 규칙 1편\(파일 없음 0, 무효 0, 근거 없음 1\)/);
  assert.deepEqual([...remote.keys], ['Bearer ' + KEY]);
  const summary = JSON.parse(await readFile(report, 'utf8'));
  assert.deepEqual(
    summary.reranked.map((item) => [item.id, item.reason]),
    ['scan', 'reader', 'index', 'voice', 'tone'].map((id) => [id, 'new']),
  );
  assert.equal(summary.embeddings.requests, 1);
  assert.equal((await b.record('scan')).model, 'openai/gpt-6-luna');
  assert.ok(!JSON.stringify(await b.files()).includes(KEY) && !JSON.stringify(summary).includes(KEY));

  const before = await b.files();
  const again = service();
  assert.equal(await b.cli([], again), 0);
  assert.deepEqual(again.calls, []);
  assert.deepEqual(await b.files(), before);

  // The gate: no model call, no embedding, nothing written.
  const gate = service();
  assert.equal(await b.cli(['--check'], gate), 0);
  assert.match(b.text(), /모든 글의 저장 해시가 현재 입력과 같습니다/);
  assert.equal(await b.cli(['--check', '--offline'], gate), 0);
  assert.equal(await b.cli(['--dry-run'], gate), 0);
  assert.deepEqual(gate.calls, []);
  assert.deepEqual(await b.files(), before);
  // Neither check needs a key.
  assert.equal(await b.cli(['--check'], gate, {}), 0);
  assert.equal(await b.cli(['--check', '--offline'], gate, {}), 0);
});

test('check fails for anything the build would not publish as stored', async (t) => {
  const b = await bench(t);
  assert.equal(await b.cli([]), 0);
  const pristine = await b.files();
  const restore = async () => {
    await rm(b.dir, { recursive: true, force: true });
    await mkdir(b.dir);
    for (const [name, text] of Object.entries(pristine)) await writeFile(join(b.dir, name), text);
    b.state.change = {};
  };
  const gate = service();

  // A header the hash comparison never looks at: the plan finds nothing to call, the build
  // would still send both articles back to the rules.
  await b.rewrite('voice', (record) => delete record.settings);
  await b.rewrite('tone', (record) => (record.id = 'tonex'));
  assert.equal(await b.cli(['--check'], gate), 1);
  assert.match(b.text(), /rank 호출 대상 0편/);
  assert.match(b.text(), /무효: voice \(모델 정보 없음\)/);
  assert.match(b.text(), /낡은 글 2편: voice, tone/);
  assert.equal(await b.cli(['--check', '--offline'], gate), 1);
  await restore();

  // A record of an article that is no longer public.
  await writeFile(join(b.dir, 'retired.json'), pristine['tone.json']);
  assert.equal(await b.cli(['--check'], gate), 1);
  assert.match(b.text(), /공개 글이 아닌 추천 파일 1개/);
  // The build ignores it, so the offline check does not fail on it.
  assert.equal(await b.cli(['--check', '--offline'], gate), 0);
  await restore();

  // A missing record is stale for the gate and merely absent for the build.
  await rm(join(b.dir, 'tone.json'));
  assert.equal(await b.cli(['--check'], gate), 1);
  assert.match(b.text(), /낡은 글 1편: tone/);
  assert.equal(await b.cli(['--check', '--offline'], gate), 0);
  await restore();

  // An edited article.
  b.state.change = { tone: { html: '<p>beta 발화 길이 개정</p>' } };
  // Its new block has no vector yet: the check says so instead of embedding it.
  assert.equal(await b.cli(['--check'], gate), 1);
  assert.match(b.text(), /임베딩이 필요합니다: 1 blocks/);
  assert.equal(await b.cli(['--check', '--offline'], gate), 1);
  assert.deepEqual(gate.calls, []);
  // A dry run embeds it, names the calls and still writes no record.
  assert.equal(await b.cli(['--dry-run'], gate), 0);
  assert.deepEqual(gate.calls, ['embed']);
  assert.match(b.text(), /tone \(changed\)/);
  assert.equal(await b.cli(['--check'], gate), 1);
  assert.match(b.text(), /낡은 글 \d편: .*tone/);
  assert.deepEqual(await b.files(), pristine);
});

test('a failed call exits with 2, is stored as a fallback and is retried alone', async (t) => {
  const b = await bench(t);
  const flaky = service((kind, input) => {
    if (kind === 'rank' && input.source.id === 'voice')
      return reply({ error: { message: 'upstream busy' } }, 503);
    return kind === 'facts' ? readerApplies(input) : pick(input);
  });
  assert.equal(await b.cli([], flaky), 2);
  assert.match(b.text(), /voice ranking: Inference HTTP 503: \{"error":\{"message":"upstream busy"\}\}/);
  assert.match(b.text(), /rank: 호출 5 \/ 재사용 0 \/ 보류 2 \/ 실패 1/);
  assert.match(b.text(), /추론 실패 1/);
  assert.deepEqual(
    [(await b.record('voice')).entry.status, (await b.record('voice')).entry.fallbackFrom],
    ['fallback', 'rules'],
  );
  // A stored fallback is a valid state: the gate passes, the next sync retries it.
  assert.equal(await b.cli(['--check']), 0);
  const retry = service();
  assert.equal(await b.cli([], retry), 0);
  assert.deepEqual(retry.calls, ['rank:voice']);
});

test('an article edited while the sync was running is reported and the run fails', async (t) => {
  const b = await bench(t);
  b.state.afterLoad = { tone: { html: '<p>beta 발화 길이 개정</p>' } };
  assert.equal(await b.cli([]), 1);
  assert.match(b.text(), /무효: tone \(관계 입력 해시 불일치\)/);
  assert.match(b.text(), /동기화 중 원문이 바뀌었습니다. 다시 실행하세요: .*tone/);
  // The next run sees the edit from the start and settles it.
  assert.equal(await b.cli([]), 0);
  assert.equal(await b.cli(['--check']), 0);
});

test('requests refused outright stop the run with the answer and leave no fallback files', async (t) => {
  const b = await bench(t);
  const locked = service(() => reply({ error: { message: 'No auth credentials found', code: 401 } }, 401));
  assert.equal(await b.cli([], locked), 1);
  assert.match(
    b.text(),
    /관계 추출 요청이 처음 3건 모두 거부되어 중단했습니다: Inference HTTP 401: .*No auth credentials found/,
  );
  assert.deepEqual(await b.files(), {});
  // Three refusals stop the stage; calls already in flight (concurrency 4) may still land.
  assert.ok(locked.chat().length >= 3 && locked.chat().length <= 5);

  // A schema the provider rejects for the rerank only.
  const schema = service((kind, input) =>
    kind === 'facts' ? readerApplies(input) : reply({ error: { message: 'Invalid schema: anyOf' } }, 400),
  );
  assert.equal(await b.cli([], schema), 1);
  assert.match(b.text(), /재정렬 요청이 처음 3건 모두 거부.*Invalid schema: anyOf/);
  assert.ok(Object.values(await b.files()).every((text) => JSON.parse(text).entry === null));
});

test('only reports the articles it left behind when new facts changed their pairs', async (t) => {
  const b = await bench(t);
  // Before anything else exists a single article can be tried on its own.
  const trial = service();
  assert.equal(await b.cli(['--only', 'tone'], trial), 0);
  assert.deepEqual(trial.chat(), ['facts:' + TITLES.tone, 'rank:tone']);
  assert.deepEqual(Object.keys(await b.files()), ['tone.json']);
  assert.match(b.text(), /글 6편 \(대상 1편\)/);
  assert.equal(await b.cli([]), 0);

  // reader is extracted again and its OCR fact is gone. scan, index and voice were ranked with
  // that fact; they are outside the scope and the build would now reject their records.
  const narrow = service((kind, input) => (kind === 'facts' ? { facts: [] } : pick(input)));
  assert.equal(await b.cli(['--only', 'reader', '--reextract'], narrow), 1);
  assert.deepEqual(narrow.chat(), ['facts:' + TITLES.reader, 'rank:reader']);
  assert.match(b.text(), /대상 밖 3편의 추천이 현재 입력과 맞지 않습니다: scan, index, voice/);
  assert.match(b.text(), /--only 없이 npm run recommendations:sync/);
  const full = service((kind, input) => (kind === 'facts' ? { facts: [] } : first3(input)));
  assert.equal(await b.cli([], full), 0);
  assert.deepEqual(full.chat().toSorted(), ['rank:index', 'rank:scan', 'rank:voice']);
});
// scan can no longer call reader an application of OCR.
const first3 = (input) => ({
  recommendations: input.candidates.slice(0, 1).map((c) => ({
    id: c.id,
    relation: 'related_method',
    reason: c.id + ' 추천',
    sourceBlock: 0,
    targetBlock: 0,
  })),
});

test('missing keys and bad arguments end with 1 before any request or write', async (t) => {
  const b = await bench(t);
  const idle = service();
  assert.equal(await b.cli([], idle, {}), 1);
  assert.match(b.text(), /Set OPENROUTER_API_KEY to sync recommendations/);
  // A plan needs no chat key, but blocks that were never embedded need the embedding key.
  assert.equal(await b.cli(['--dry-run'], idle, {}), 1);
  assert.match(b.text(), /Set OPENROUTER_API_KEY to embed new blocks/);
  for (const [args, message] of [
    [['--offline'], /--offline is only valid with --check/],
    [['--check', '--only', 'scan'], /--check verifies every article/],
    [['--only'], /--only needs a value/],
    [['--only', ' , '], /--only needs at least one article ID/],
    [['--only', 'scan,nope'], /Not a public article: nope/],
    [['--bogus'], /Unknown option --bogus/],
  ]) {
    assert.equal(await b.cli(args, idle), 1, args.join(' '));
    assert.match(b.text(), message);
  }
  assert.equal(
    await b.cli([], idle, { OPENROUTER_API_KEY: KEY, SUPABASE_URL: 'https://project.supabase.co' }),
    1,
  );
  assert.match(b.text(), /SUPABASE_SECRET_KEY is not set/);
  // A key pasted with a line break is named, never printed.
  assert.equal(await b.cli([], idle, { OPENROUTER_API_KEY: 'sk-or-SEC\nRET' }), 1);
  assert.match(b.text(), /^OPENROUTER_API_KEY contains/);
  assert.ok(!/SEC|RET/.test(b.text()));
  assert.deepEqual(idle.calls, []);
  assert.deepEqual(await b.files(), {});
});

test('the build projects valid records and keeps the rules for every other article', async (t) => {
  const b = await bench(t);
  const raw = () => ({ articles: library(b.state.change), concepts, series: [], overrides: {} });
  const ruled = (corpus) =>
    generateRelated(corpus.articles, corpus.concepts, corpus.series, corpus.overrides);
  const related = (articles, id) => articles.find((a) => a.id === id).related;

  // No directory: exactly the rule-based articles, nothing to report.
  const baseline = ruled(raw());
  const none = await applyRelated(raw(), baseline, b.dir);
  assert.equal(none.articles, baseline);
  assert.deepEqual(
    [none.report.directory, none.report.llm, none.report.rules, none.report.missing, none.report.invalid],
    [false, 0, 6, 6, 0],
  );
  // An empty directory: every record is missing.
  await mkdir(b.dir);
  const empty = await applyRelated(raw(), baseline, b.dir);
  assert.deepEqual(empty.articles, baseline);
  assert.deepEqual([empty.report.directory, empty.report.missing], [true, 6]);

  assert.equal(await b.cli([]), 0);
  const all = await applyRelated(raw(), baseline, b.dir);
  assert.deepEqual([all.report.llm, all.report.rules, all.report.noEvidenceIds, all.report.invalidIds], [5, 1, ['blank'], []]);
  assert.deepEqual(related(all.articles, 'scan'), [{ id: 'reader', reason: 'OCR 적용 사례' }]);
  assert.deepEqual(related(all.articles, 'tone'), []);
  // Only the ID and the reason reach the registry.
  assert.ok(all.articles.every((a) => a.related.every((row) => Object.keys(row).join() === 'id,reason')));
  assert.ok(!/inputHash|candidates|sourceEvidence|blockId/.test(JSON.stringify(all.articles)));

  // One record tampered, one deleted, one orphan, one stray file.
  assert.ok(related(baseline, 'voice').length > 0 && related(baseline, 'index').length > 0);
  await b.rewrite('voice', (record) => (record.entry.recommendations[0].sourceEvidence.quote = 'altered'));
  await rm(join(b.dir, 'index.json'));
  await writeFile(join(b.dir, 'retired.json'), await readFile(join(b.dir, 'tone.json'), 'utf8'));
  await writeFile(join(b.dir, '.DS_Store'), 'x');
  const mixed = await applyRelated(raw(), baseline, b.dir);
  assert.deepEqual(related(mixed.articles, 'voice'), related(baseline, 'voice'));
  assert.deepEqual(related(mixed.articles, 'index'), related(baseline, 'index'));
  assert.deepEqual(related(mixed.articles, 'scan'), [{ id: 'reader', reason: 'OCR 적용 사례' }]);
  assert.deepEqual(related(mixed.articles, 'reader'), related(all.articles, 'reader'));
  assert.deepEqual(
    {
      llm: mixed.report.llm,
      rules: mixed.report.rules,
      missing: mixed.report.missing,
      invalid: mixed.report.invalid,
      invalidIds: mixed.report.invalidIds,
      orphans: mixed.report.orphans,
    },
    {
      llm: 3,
      rules: 3,
      missing: 1,
      invalid: 1,
      invalidIds: [{ id: 'voice', reason: '추천 근거 불일치' }],
      orphans: ['retired'],
    },
  );
});

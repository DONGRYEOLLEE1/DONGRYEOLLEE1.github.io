import assert from 'node:assert/strict';
import test from 'node:test';
import { appendFile, mkdtemp, readFile, rm, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  embedProvider,
  embeddingInput,
  embeddingProfile,
  fileStore,
  layered,
  loadVectors,
  memoryStore,
  pairScore,
  similarityIndex,
  supabaseStore,
  unitVector,
  vectorStore,
} from '../tools/knowledge/embeddings.mjs';
import { hash } from '../tools/knowledge/evidence.mjs';
import { chatProvider } from '../tools/knowledge/model.mjs';
import { prepareCorpus } from '../tools/knowledge/phase2a.mjs';
import { removeOrphans, writeRecord } from '../tools/knowledge/store.mjs';
import { article, embedder } from './fixtures.mjs';

const reply = (payload, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => payload,
  text: async () => (typeof payload === 'string' ? payload : JSON.stringify(payload)),
});
function recorder(respond) {
  const calls = [];
  const fetch = async (url, init = {}) => {
    calls.push({ url, ...init, body: init.body && JSON.parse(init.body) });
    return respond(url, init, calls.length);
  };
  return { calls, fetch };
}
const hashes = (n, prefix = 'h') => Array.from({ length: n }, (_, i) => prefix + String(i).padStart(4, '0'));
const length = (vector) => Math.hypot(...vector);

test('longer embeddings are truncated and renormalised; shorter ones are an error', async () => {
  const vector = unitVector([3, 4, 12, 99], 2);
  assert.equal(vector.length, 2);
  assert.ok(Math.abs(vector[0] - 0.6) < 1e-6 && Math.abs(vector[1] - 0.8) < 1e-6);
  assert.throws(() => unitVector([1, 2], 3), /expected 3/);
  // Values that would poison every later comparison are refused before they can be cached.
  assert.throws(() => unitVector([0, 0], 2), /zero vector/);
  assert.throws(() => unitVector([NaN, 1, 2], 3), /finite number/);
  assert.throws(() => unitVector(['a', 1, 2], 3), /finite number/);
  assert.throws(() => unitVector([1, null, 2], 3), /finite number/);
  assert.throws(() => unitVector('AAAAAAAAAAAAAAAA', 4), /not an array/);
  assert.throws(() => unitVector(undefined, 4), /not an array/);
  assert.deepEqual([...unitVector(Float32Array.of(0, 2), 2)], [0, 1]);
  // A bad value past the cut does not matter.
  assert.deepEqual([...unitVector([0, 2, NaN], 2)], [0, 1]);

  const env = { OPENROUTER_API_KEY: 'k', EMBEDDINGS_DIMENSIONS: '3' };
  const long = recorder((_, init) => {
    const { input } = JSON.parse(init.body);
    // Returned out of order on purpose: `index` decides the position.
    const data = input.map((text, index) => ({ index, embedding: [text.length, 1, 1, 50, 50] })).reverse();
    return reply({ data, usage: { prompt_tokens: input.length } });
  });
  const provider = embedProvider(env, long);
  const texts = Array.from({ length: 130 }, (_, i) => 'x'.repeat(i + 1));
  const vectors = await provider.embed(texts);
  assert.equal(vectors.length, 130);
  assert.ok(
    vectors.every((v) => v instanceof Float32Array && v.length === 3 && Math.abs(length(v) - 1) < 1e-6),
  );
  // Every vector belongs to its own text, whatever order the batch came back in.
  texts.forEach((text, i) => {
    const expected = unitVector([text.length, 1, 1], 3);
    assert.deepEqual([...vectors[i]], [...expected], 'text ' + i);
  });
  assert.deepEqual(
    long.calls.map((call) => call.body.input.length),
    [64, 64, 2],
  );
  assert.deepEqual([provider.usage.requests, provider.usage.tokens], [3, 130]);
  const call = long.calls[0];
  assert.equal(call.url, 'https://openrouter.ai/api/v1/embeddings');
  assert.equal(call.headers.Authorization, 'Bearer k');
  assert.deepEqual(
    [call.body.model, call.body.dimensions, call.body.encoding_format],
    ['google/gemini-embedding-2', 3, 'float'],
  );

  const short = recorder(() => reply({ data: [{ index: 0, embedding: [1, 2] }] }));
  await assert.rejects(embedProvider(env, short).embed(['a']), /expected 3/);
  const failing = recorder(() => reply('nope', 429));
  await assert.rejects(embedProvider(env, failing).embed(['a']), /Embedding HTTP 429/);
  await assert.rejects(embedProvider({}, long).embed(['a']), /OPENROUTER_API_KEY/);
});

test('the embedding profile names the cache namespace', () => {
  const base = embeddingProfile({});
  assert.deepEqual(base.profile, {
    provider: 'openrouter',
    model: 'google/gemini-embedding-2',
    dimensions: 768,
    rule: 'block-v1',
  });
  assert.match(base.profileId, /^[0-9a-f]{16}$/);
  assert.notEqual(embeddingProfile({ EMBEDDINGS_DIMENSIONS: '256' }).profileId, base.profileId);
  assert.notEqual(embeddingProfile({ EMBEDDINGS_MODEL: 'other/model' }).profileId, base.profileId);
});

test('pair score is the mean of the three best block matches, shared by both directions', () => {
  const unit = (...values) => unitVector(values, 3);
  const S = { vectors: [unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1), unit(1, 1, 0)] };
  const T = { vectors: [unit(1, 0, 0), unit(0, 1, 0)] };
  // Best matches of S: 1, 1, 0, 0.7071 -> top three average 0.9024.
  assert.equal(pairScore(S, T), 0.9024);
  assert.equal(pairScore(T, S), 1);
  assert.equal(pairScore({ vectors: [unit(0, 0, 1)] }, T), 0);
  assert.equal(pairScore({ vectors: [] }, T), 0);

  const index = similarityIndex({
    byArticle: new Map([
      ['s', S],
      ['t', T],
      ['empty', { vectors: [] }],
      ['a', T],
    ]),
  });
  assert.equal(index.score('s', 't'), 0.9024);
  assert.equal(index.score('t', 's'), 1);
  assert.deepEqual(
    [...index.blockMax('s', 't')].map((v) => Math.round(v * 1e4) / 1e4),
    [1, 1, 0, 0.7071],
  );
  // Equal scores fall back to article ID; an article without blocks is never ranked.
  assert.deepEqual(index.ranked('s'), ['a', 't']);
  assert.deepEqual(index.ranked('empty'), []);
});

test('the Supabase store pages reads by 80 and writes by 200 with PostgREST upsert headers', async () => {
  const env = { SUPABASE_URL: 'https://project.supabase.co/', SUPABASE_SERVICE_ROLE_KEY: 'service' };
  const remote = recorder((url, init) => {
    if (init.method === 'POST') return reply('', 201);
    const wanted = /input_hash=in\.\(([^)]*)\)/.exec(url)[1].split(',');
    return reply(
      wanted.filter((h) => h !== 'h0001').map((h) => ({ input_hash: h, embedding: '[0.5,-0.25]' })),
    );
  });
  const store = supabaseStore(env, remote);
  const found = await store.get('profile1', hashes(170));
  assert.equal(found.size, 169);
  assert.ok(!found.has('h0001'));
  assert.deepEqual([...found.get('h0000')], [0.5, -0.25]);
  assert.ok(found.get('h0000') instanceof Float32Array);
  assert.equal(remote.calls.length, 3);
  assert.equal(
    remote.calls[2].url,
    'https://project.supabase.co/rest/v1/block_embeddings?profile=eq.profile1&input_hash=in.(' +
      hashes(170).slice(160).join(',') +
      ')&select=input_hash,embedding',
  );
  assert.deepEqual(
    remote.calls.map((call) => /in\.\(([^)]*)\)/.exec(call.url)[1].split(',').length),
    [80, 80, 10],
  );
  assert.deepEqual(
    [remote.calls[0].headers.apikey, remote.calls[0].headers.Authorization],
    ['service', 'Bearer service'],
  );

  remote.calls.length = 0;
  const vectors = new Map(hashes(450).map((h) => [h, Float32Array.of(0.5, -0.25)]));
  await store.put('profile1', vectors);
  assert.deepEqual(
    remote.calls.map((call) => call.body.length),
    [200, 200, 50],
  );
  const post = remote.calls[0];
  assert.equal(
    post.url,
    'https://project.supabase.co/rest/v1/block_embeddings?on_conflict=profile,input_hash',
  );
  assert.equal(post.method, 'POST');
  assert.equal(post.headers.Prefer, 'resolution=merge-duplicates,return=minimal');
  assert.equal(post.headers.apikey, 'service');
  assert.equal(post.headers.Authorization, 'Bearer service');
  assert.deepEqual(post.body[0], { profile: 'profile1', input_hash: 'h0000', embedding: '[0.5,-0.25]' });

  const down = recorder(() => reply('x'.repeat(500), 503));
  await assert.rejects(supabaseStore(env, down).get('profile1', ['h']), (error) => {
    assert.match(error.message, /503/);
    assert.ok(error.message.includes('x'.repeat(200)) && !error.message.includes('x'.repeat(201)));
    return true;
  });
  await assert.rejects(supabaseStore(env, down).put('profile1', vectors), /503/);
});

test('the layered store asks Supabase only for local misses and writes to both', async () => {
  const local = memoryStore();
  const shared = memoryStore();
  const asked = [];
  const remote = {
    get: async (profile, wanted) => (asked.push(wanted), shared.get(profile, wanted)),
    put: (profile, vectors) => shared.put(profile, vectors),
  };
  const store = layered(local, remote);
  await local.put('p', new Map([['a', Float32Array.of(1)]]));
  await shared.put('p', new Map([['b', Float32Array.of(2)]]));

  assert.deepEqual([...(await store.get('p', ['a'])).keys()], ['a']);
  assert.deepEqual(asked, []);
  assert.deepEqual([...(await store.get('p', ['a', 'b', 'c'])).keys()].sort(), ['a', 'b']);
  assert.deepEqual(asked, [['b', 'c']]);
  // The remote hit was copied down, so the next read stays local.
  await store.get('p', ['a', 'b']);
  assert.equal(asked.length, 1);

  await store.put('p', new Map([['d', Float32Array.of(4)]]));
  assert.ok((await local.get('p', ['d'])).has('d') && (await shared.get('p', ['d'])).has('d'));
  const failing = layered(local, {
    get: remote.get,
    put: async () => Promise.reject(Error('Supabase HTTP 500')),
  });
  await assert.rejects(failing.put('p', new Map([['e', Float32Array.of(5)]])), /500/);
  assert.ok(!(await local.get('p', ['e'])).has('e'));
});

test('the file store appends base64 float32 lines and a new instance reads them back', async (t) => {
  const dir = await mkdtemp(join(tmpdir(), 'vectors-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const vector = unitVector([1, 2, 3], 3);
  const store = fileStore(dir);
  assert.equal((await store.get('p', ['a'])).size, 0);
  await store.put('p', new Map([['a', vector]]));
  await store.put('p', new Map([['b', Float32Array.of(0.5, 0, 0)]]));
  const lines = (await readFile(join(dir, 'p.jsonl'), 'utf8'))
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  assert.deepEqual(
    lines.map((line) => line.h),
    ['a', 'b'],
  );
  assert.equal(Buffer.from(lines[0].v, 'base64').length, 12);
  const reopened = await fileStore(dir).get('p', ['a', 'b', 'c']);
  assert.deepEqual([...reopened.keys()], ['a', 'b']);
  assert.deepEqual([...reopened.get('a')], [...vector]);
  assert.equal((await fileStore(dir).get('other', ['a'])).size, 0);

  const warnings = [];
  const plain = vectorStore({}, { dir, warn: (message) => warnings.push(message) });
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /Supabase가 설정되지 않아 로컬 캐시만 사용/);
  assert.ok((await plain.get('p', ['a'])).has('a'));
  const remote = recorder(() => reply([]));
  const both = vectorStore(
    { SUPABASE_URL: 'https://project.supabase.co', SUPABASE_SERVICE_ROLE_KEY: 'service' },
    { dir, fetch: remote.fetch, warn: (message) => warnings.push(message) },
  );
  await both.get('p', ['a']);
  assert.equal(remote.calls.length, 0);
  await both.get('p', ['zz']);
  assert.equal(remote.calls.length, 1);
  assert.equal(warnings.length, 1);
});

test('the chat provider sends seed and reasoning to OpenRouter and temperature elsewhere', async () => {
  const input = { source: { id: 'a', title: 'a' }, candidates: [] };
  const good = {
    choices: [{ finish_reason: 'stop', message: { content: '{"recommendations":[]}' } }],
    usage: { prompt_tokens: 11, completion_tokens: 5 },
  };
  const remote = recorder(() => reply(good));
  const provider = chatProvider({ OPENROUTER_API_KEY: 'k' }, remote);
  assert.equal(provider.model, 'openai/gpt-6-luna');
  assert.deepEqual(provider.settings, { seed: 7, reasoning: { effort: 'low' } });
  assert.deepEqual(await provider.complete('instruction', input), { recommendations: [] });
  await provider.complete('instruction', input);
  assert.deepEqual(provider.usage, { requests: 2, promptTokens: 22, completionTokens: 10 });
  const { url, headers, body } = remote.calls[0];
  assert.equal(url, 'https://openrouter.ai/api/v1/chat/completions');
  assert.equal(headers.Authorization, 'Bearer k');
  assert.ok(!('temperature' in body));
  assert.deepEqual(
    [body.model, body.seed, body.reasoning, body.max_tokens, body.provider],
    ['openai/gpt-6-luna', 7, { effort: 'low' }, 8000, { require_parameters: true }],
  );
  assert.equal(body.response_format.json_schema.strict, true);

  const tuned = chatProvider(
    {
      OPENROUTER_API_KEY: 'k',
      RECOMMENDATIONS_MODEL: 'm',
      RECOMMENDATIONS_REASONING_EFFORT: 'high',
      RECOMMENDATIONS_MAX_TOKENS: '1234',
    },
    remote,
  );
  await tuned.complete('instruction', input);
  assert.deepEqual(
    [remote.calls.at(-1).body.model, remote.calls.at(-1).body.reasoning, remote.calls.at(-1).body.max_tokens],
    ['m', { effort: 'high' }, 1234],
  );
  assert.deepEqual(tuned.settings, { seed: 7, reasoning: { effort: 'high' } });

  const local = chatProvider(
    { RECOMMENDATIONS_API_BASE: 'http://127.0.0.1:8080/v1/', OPENROUTER_API_KEY: 'k' },
    remote,
  );
  await local.complete('instruction', input);
  const custom = remote.calls.at(-1);
  assert.equal(custom.url, 'http://127.0.0.1:8080/v1/chat/completions');
  assert.equal(custom.body.temperature, 0);
  assert.ok(!('reasoning' in custom.body) && !('provider' in custom.body) && !('seed' in custom.body));
  // The OpenRouter key is never sent to another host.
  assert.ok(!('Authorization' in custom.headers));
  assert.deepEqual(local.settings, { temperature: 0 });

  assert.throws(() => chatProvider({}), /OPENROUTER_API_KEY/);
  assert.equal(chatProvider({}, { requireKey: false }).model, 'openai/gpt-6-luna');
});

test('truncated output, HTTP errors and unparsable output are distinct errors', async () => {
  const input = { source: { id: 'a', title: 'a' }, candidates: [] };
  const env = { OPENROUTER_API_KEY: 'k' };
  const complete = (response) =>
    chatProvider(
      env,
      recorder(() => response),
    ).complete('instruction', input);
  const cut = {
    choices: [{ finish_reason: 'length', message: { content: '{"recommendations":[' } }],
    usage: { prompt_tokens: 3, completion_tokens: 8000 },
  };
  await assert.rejects(complete(reply(cut)), { message: 'Inference output truncated' });
  await assert.rejects(complete(reply('upstream', 502)), { message: 'Inference HTTP 502' });
  await assert.rejects(
    complete(reply({ choices: [{ finish_reason: 'stop', message: { content: 'not json' } }] })),
    { message: 'Inference output is not valid JSON' },
  );
  await assert.rejects(
    complete(reply({ choices: [{ finish_reason: 'stop', message: { content: null } }] })),
    { message: 'Inference returned no content' },
  );
  // A failed call is still billed, so its tokens are counted.
  const provider = chatProvider(
    env,
    recorder(() => reply(cut)),
  );
  await assert.rejects(provider.complete('instruction', input));
  assert.equal(provider.usage.completionTokens, 8000);
});

test('an embedding response that does not cover every input is an error', async () => {
  const env = { OPENROUTER_API_KEY: 'k', EMBEDDINGS_DIMENSIONS: '2' };
  const embed = (respond) => embedProvider(env, recorder(respond)).embed(['a', 'b']);
  const row = (index) => ({ index, embedding: [1, index + 1] });
  await assert.rejects(
    embed(() => reply({ data: [row(0)] })),
    /does not match the request size/,
  );
  await assert.rejects(
    embed(() => reply({ data: [row(0), row(1), row(2)] })),
    /does not match the request size/,
  );
  // Two rows for the same input leave the other one without a vector.
  await assert.rejects(
    embed(() => reply({ data: [row(0), row(0)] })),
    /does not cover every input/,
  );
  await assert.rejects(
    embed(() => reply({ data: [row(0), row(5)] })),
    /does not cover every input/,
  );
  // Rows without an index are taken in order.
  const ordered = await embed(() => reply({ data: [{ embedding: [1, 0] }, { embedding: [0, 1] }] }));
  assert.deepEqual(
    ordered.map((v) => [...v]),
    [
      [1, 0],
      [0, 1],
    ],
  );
  // A provider that ignores encoding_format and answers base64 must not be cached as NaN.
  await assert.rejects(
    embed(() => reply({ data: [{ index: 0, embedding: 'AACAPwAAAEA=' }, row(1)] })),
    /not an array of numbers/,
  );
  await assert.rejects(
    embed(() => reply({ data: [{ index: 0, embedding: [1, null] }, row(1)] })),
    /finite number/,
  );
  await assert.rejects(
    embed(() => reply({ data: [{ index: 0, embedding: [0, 0] }, row(1)] })),
    /zero vector/,
  );
});

test('embedding failures say what the provider answered and never print the key', async () => {
  const env = { OPENROUTER_API_KEY: 'sk-or-secret', EMBEDDINGS_DIMENSIONS: '2' };
  const embed = (respond, other = env) => embedProvider(other, recorder(respond)).embed(['a']);
  await assert.rejects(
    embed(() => reply({ error: { message: 'Insufficient credits for sk-or-secret' } }, 402)),
    {
      message: 'Embedding HTTP 402: {"error":{"message":"Insufficient credits for ***"}}',
    },
  );
  await assert.rejects(
    embed(() => reply('x'.repeat(500), 500)),
    (error) => error.message.length < 240,
  );
  await assert.rejects(
    embed(() => reply({ error: { message: 'No endpoints found', code: 404 } })),
    {
      message: 'Embedding error response: No endpoints found',
    },
  );
  const html = { ok: true, status: 200, json: async () => JSON.parse('<html>'), text: async () => '<html>' };
  await assert.rejects(
    embed(() => html),
    { message: 'Embedding response is not JSON' },
  );
  // undici quotes the whole header in its message; only the service and the code are passed on.
  const refused = Object.assign(new TypeError('fetch failed: Bearer sk-or-secret'), {
    cause: { code: 'ECONNREFUSED' },
  });
  await assert.rejects(
    embed(() => Promise.reject(refused)),
    { message: 'Embedding request failed (ECONNREFUSED)' },
  );

  // A key with a line break, or a URL with credentials, is refused by name, not by value.
  for (const [bad, name] of [
    [{ OPENROUTER_API_KEY: 'sk-or-SEC\nRET' }, 'OPENROUTER_API_KEY'],
    [{ EMBEDDINGS_API_BASE: 'http://127.0.0.1:1/v1', EMBEDDINGS_API_KEY: 'two words' }, 'EMBEDDINGS_API_KEY'],
    [{ EMBEDDINGS_API_BASE: 'http://user:PASSWORD@127.0.0.1:1/v1' }, 'EMBEDDINGS_API_BASE'],
    [{ EMBEDDINGS_API_BASE: 'sk-or-secret' }, 'EMBEDDINGS_API_BASE'],
  ])
    assert.throws(
      () => embedProvider(bad),
      (error) => error.message.startsWith(name) && !/SEC|RET|two words|PASSWORD|sk-or/.test(error.message),
    );
});

test('a custom embedding endpoint has its own profile and never receives the OpenRouter key', async () => {
  const remote = recorder((_, init) =>
    reply({ data: JSON.parse(init.body).input.map((_, index) => ({ index, embedding: [1, 2] })) }),
  );
  const base = { EMBEDDINGS_API_BASE: 'http://127.0.0.1:9000/v1/', EMBEDDINGS_DIMENSIONS: '2' };
  const borrowed = embedProvider({ ...base, OPENROUTER_API_KEY: 'k' }, remote);
  await borrowed.embed(['a']);
  assert.equal(remote.calls[0].url, 'http://127.0.0.1:9000/v1/embeddings');
  assert.ok(!('Authorization' in remote.calls[0].headers));
  await embedProvider({ ...base, OPENROUTER_API_KEY: 'k', EMBEDDINGS_API_KEY: 'own' }, remote).embed(['a']);
  assert.equal(remote.calls[1].headers.Authorization, 'Bearer own');

  const standard = embeddingProfile({ EMBEDDINGS_DIMENSIONS: '2' });
  assert.equal(borrowed.profile.provider, 'http://127.0.0.1:9000/v1');
  assert.notEqual(borrowed.profileId, standard.profileId);
  assert.equal(
    embedProvider({ EMBEDDINGS_DIMENSIONS: '2', OPENROUTER_API_KEY: 'k' }).profileId,
    standard.profileId,
  );
});

test('both directions of the block-by-block maximum are kept apart', () => {
  const unit = (...values) => unitVector(values, 3);
  const S = { vectors: [unit(1, 0, 0), unit(1, 1, 0), unit(0, 0, 1)] };
  const T = { vectors: [unit(1, 0, 0), unit(0, 1, 0), unit(1, 0, 0), unit(0, 1, 1)] };
  const index = similarityIndex({
    byArticle: new Map([
      ['s', S],
      ['t', T],
    ]),
  });
  const rounded = (values) => [...values].map((v) => Math.round(v * 1e4) / 1e4);
  assert.deepEqual(rounded(index.blockMax('s', 't')), [1, 0.7071, 0.7071]);
  assert.deepEqual(rounded(index.blockMax('t', 's')), [1, 0.7071, 1, 0.7071]);
  assert.deepEqual([index.score('s', 't'), index.score('t', 's')], [0.8047, 0.9024]);
  assert.equal(index.score('s', 't'), pairScore(S, T));
  assert.equal(index.score('t', 's'), pairScore(T, S));
  assert.notEqual(index.score('s', 't'), index.score('t', 's'));
});

test('loadVectors stores what it embeds, fails loudly and never embeds for a check', async () => {
  const ctx = prepareCorpus({
    concepts: [],
    series: [],
    overrides: {},
    articles: [article('a', '<p>alpha one</p><p>beta two</p>'), article('b', '<p>gamma three</p>')],
  });
  const embed = embedder();
  const store = memoryStore();
  const loaded = await loadVectors(ctx, embed, store);
  assert.deepEqual([loaded.embedded, loaded.cached], [3, 0]);
  assert.equal(loaded.byArticle.get('a').vectors.length, 2);
  const key = hash(embeddingInput(ctx.byId.get('a'), ctx.usable.get('a')[1]));
  assert.deepEqual(
    [...(await store.get(embed.profileId, [key])).get(key)],
    [...loaded.byArticle.get('a').vectors[1]],
  );

  // A cache that cannot be written is an error, not a silent re-embedding on every run.
  const readOnly = {
    get: async () => new Map(),
    put: async () => Promise.reject(Error('Supabase HTTP 500: down')),
  };
  await assert.rejects(loadVectors(ctx, embedder(), readOnly), /Supabase HTTP 500/);

  // --check: a block without a vector is reported, and nothing is embedded or stored.
  const idle = embedder();
  const empty = memoryStore();
  await assert.rejects(loadVectors(ctx, idle, empty, { embedMissing: false }), (error) => {
    assert.equal(error.code, 'EMBEDDINGS_MISSING');
    assert.match(error.message, /3 blocks/);
    return true;
  });
  assert.deepEqual(idle.batches, []);
  assert.equal((await empty.get(idle.profileId, [key])).size, 0);
  const warm = await loadVectors(ctx, idle, store, { embedMissing: false });
  assert.deepEqual([warm.embedded, warm.cached, idle.batches.length], [0, 3, 0]);

  // A cached vector with a NaN is embedded again and replaced; a provider's NaN is never stored.
  await store.put(embed.profileId, new Map([[key, Float32Array.of(NaN, 0, 0, 0, 0, 0, 0, 0)]]));
  const healed = await loadVectors(ctx, embed, store);
  assert.deepEqual([healed.embedded, healed.cached], [1, 2]);
  assert.ok((await store.get(embed.profileId, [key])).get(key).every(Number.isFinite));
  const poisoned = {
    ...embedder(),
    embed: async (texts) => texts.map(() => Float32Array.of(NaN, 0, 0, 0, 0, 0, 0, 0)),
  };
  const untouched = memoryStore();
  await assert.rejects(loadVectors(ctx, poisoned, untouched), /unusable vector/);
  assert.equal((await untouched.get(poisoned.profileId, [key])).size, 0);
  await assert.rejects(
    loadVectors(ctx, embedder(), {
      get: async () => new Map([[key, Float32Array.of(1, 0)]]),
      put: async () => {},
    }),
    /wrong size/,
  );
});

test('a vector appended after an interrupted write is not glued to the partial line', async (t) => {
  const dir = await mkdtemp(join(tmpdir(), 'vectors-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  await fileStore(dir).put('p', new Map([['h1', Float32Array.of(1, 0)]]));
  // The process died in the middle of the next line.
  await appendFile(join(dir, 'p.jsonl'), '{"h":"h2","v":"AAC');
  const resumed = fileStore(dir);
  assert.deepEqual([...(await resumed.get('p', ['h1', 'h2'])).keys()], ['h1']);
  await resumed.put(
    'p',
    new Map([
      ['h3', Float32Array.of(0, 1)],
      ['h4', Float32Array.of(1, 1)],
    ]),
  );
  const reopened = await fileStore(dir).get('p', ['h1', 'h2', 'h3', 'h4']);
  assert.deepEqual([...reopened.keys()], ['h1', 'h3', 'h4']);
  assert.deepEqual([...reopened.get('h3')], [0, 1]);
  // An intact file gets no blank lines.
  await fileStore(dir).put('p', new Map([['h5', Float32Array.of(1, 0)]]));
  assert.ok(!(await readFile(join(dir, 'p.jsonl'), 'utf8')).includes('\n\n'));
});

test('one Supabase variable without the other stops instead of running on a private cache', () => {
  const warnings = [];
  const options = { dir: '/nonexistent', warn: (message) => warnings.push(message) };
  assert.throws(() => vectorStore({ SUPABASE_URL: 'https://project.supabase.co' }, options), {
    message: /^SUPABASE_SECRET_KEY is not set/,
  });
  assert.throws(
    () => vectorStore({ SUPABASE_SERVICE_ROLE_KEY: 'srk-secret' }, options),
    (error) => {
      assert.match(error.message, /^SUPABASE_URL is not set/);
      return !error.message.includes('srk-secret');
    },
  );
  assert.deepEqual(warnings, []);
  // The two values swapped, a key with a line break: named, not printed.
  const swapped = { SUPABASE_URL: 'sb_secret_abc', SUPABASE_SERVICE_ROLE_KEY: 'https://project.supabase.co' };
  assert.throws(() => vectorStore(swapped, options), { message: 'SUPABASE_URL is not an http(s) URL.' });
  assert.throws(
    () =>
      vectorStore(
        { SUPABASE_URL: 'https://project.supabase.co', SUPABASE_SERVICE_ROLE_KEY: 'srk-SEC\nRET' },
        options,
      ),
    (error) => error.message.startsWith('SUPABASE_SERVICE_ROLE_KEY') && !/SEC|RET/.test(error.message),
  );
});

test('a Supabase transport failure names the service and nothing else', async () => {
  const env = { SUPABASE_URL: 'https://project.supabase.co', SUPABASE_SERVICE_ROLE_KEY: 'srk-secret' };
  const down = recorder(() =>
    Promise.reject(Object.assign(new TypeError('fetch failed'), { cause: { code: 'ENOTFOUND' } })),
  );
  await assert.rejects(supabaseStore(env, down).get('p', ['h']), {
    message: 'Supabase request failed (ENOTFOUND)',
  });
  const echo = recorder(() => reply('permission denied for srk-secret', 403));
  await assert.rejects(supabaseStore(env, echo).put('p', new Map([['h', Float32Array.of(1)]])), {
    message: 'Supabase HTTP 403: permission denied for ***',
  });
});

test('inference failures carry the provider answer, and refusals are told apart from hiccups', async () => {
  const input = { source: { id: 'a', title: 'a' }, candidates: [] };
  const env = { OPENROUTER_API_KEY: 'sk-or-secret' };
  const failure = (respond) =>
    chatProvider(env, recorder(respond))
      .complete('instruction', input)
      .then(
        () => assert.fail('expected a failure'),
        (error) => error,
      );
  const schema = await failure(() =>
    reply({ error: { message: "Invalid schema for response_format 'judgment'", code: 400 } }, 400),
  );
  assert.equal(schema.message, 'Inference HTTP 400');
  assert.equal(schema.status, 400);
  assert.match(schema.detail, /^\{"error":\{"message":"Invalid schema for response_format 'judgment'"/);
  assert.equal(schema.rejected, true);
  for (const status of [401, 402, 403, 404, 422])
    assert.equal((await failure(() => reply('no', status))).rejected, true);
  for (const status of [408, 429, 500, 502, 503])
    assert.equal((await failure(() => reply('no', status))).rejected, false);
  assert.ok((await failure(() => reply('y'.repeat(900) + ' sk-or-secret', 500))).detail.length <= 200);
  assert.ok(
    !(await failure(() => reply('key sk-or-secret is invalid', 401))).detail.includes('sk-or-secret'),
  );

  // OpenRouter answers 200 with an error object when the upstream provider refused.
  const upstream = await failure(() =>
    reply({ error: { message: 'No endpoints found that can handle the requested parameters', code: 404 } }),
  );
  assert.deepEqual(
    [upstream.message, upstream.detail, upstream.rejected],
    ['Inference error response', 'No endpoints found that can handle the requested parameters', true],
  );
  const choice = await failure(() =>
    reply({ choices: [{ error: { message: 'Provider overloaded', code: 503 }, message: { content: '' } }] }),
  );
  assert.deepEqual(
    [choice.message, choice.detail, choice.rejected],
    ['Inference error response', 'Provider overloaded', false],
  );

  const refused = await failure(() =>
    Promise.reject(
      Object.assign(new TypeError('Headers.append: "Bearer sk-or-secret" is invalid'), {
        cause: { code: 'ECONNREFUSED' },
      }),
    ),
  );
  assert.deepEqual([refused.message, refused.unreachable], ['Inference request failed (ECONNREFUSED)', true]);
  const slow = await failure(() =>
    Promise.reject(Object.assign(new Error('The operation timed out'), { name: 'TimeoutError' })),
  );
  assert.deepEqual([slow.message, slow.unreachable], ['Inference request failed (TimeoutError)', false]);

  for (const [bad, name] of [
    [{ OPENROUTER_API_KEY: 'sk-or-SEC\nRET' }, 'OPENROUTER_API_KEY'],
    [{ RECOMMENDATIONS_API_BASE: 'http://user:PASSWORD@127.0.0.1:1/v1' }, 'RECOMMENDATIONS_API_BASE'],
    [
      { RECOMMENDATIONS_API_BASE: 'http://127.0.0.1:1/v1', RECOMMENDATIONS_API_KEY: 'k\u0000' },
      'RECOMMENDATIONS_API_KEY',
    ],
  ])
    assert.throws(
      () => chatProvider(bad),
      (error) => error.message.startsWith(name) && !/SEC|RET|PASSWORD/.test(error.message),
    );
});

test('orphan cleanup leaves a temp file that another sync may still be writing', async (t) => {
  const dir = await mkdtemp(join(tmpdir(), 'records-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const record = { id: 'kept', model: 'm', settings: {}, facts: null, entry: null };
  await writeRecord(dir, record);
  await writeRecord(dir, { ...record, id: 'gone' });
  const inFlight = join(dir, 'kept.json.4242.1.tmp');
  const abandoned = join(dir, 'kept.json.4242.2.tmp');
  await writeFile(inFlight, '{');
  await writeFile(abandoned, '{');
  const hourAgo = new Date(Date.now() - 60 * 60 * 1000);
  await utimes(abandoned, hourAgo, hourAgo);

  assert.deepEqual(await removeOrphans(dir, ['kept']), ['gone']);
  // The other process can still rename its file into place.
  assert.equal(await readFile(inFlight, 'utf8'), '{');
  await assert.rejects(readFile(abandoned), { code: 'ENOENT' });
  await assert.rejects(readFile(join(dir, 'gone.json')), { code: 'ENOENT' });
  assert.ok((await readFile(join(dir, 'kept.json'), 'utf8')).endsWith('}\n'));
  assert.deepEqual(await removeOrphans(join(dir, 'missing'), []), []);
});

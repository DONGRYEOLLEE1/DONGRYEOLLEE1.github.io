import { appendFile, mkdir, open, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { hash } from './evidence.mjs';
import { baseUrl, bodyExcerpt, excerpt, secret, send } from './http.mjs';

const OPENROUTER = 'https://openrouter.ai/api/v1';
const round4 = (value) => Math.round(value * 10000) / 10000;
const byId = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
function* chunks(list, size) {
  for (let start = 0; start < list.length; start += size) yield list.slice(start, start + size);
}

/** The profile is the vector cache namespace: vectors from different profiles are never compared. */
export function embeddingProfile(env = process.env) {
  const base = (env.EMBEDDINGS_API_BASE || OPENROUTER).replace(/\/+$/, '');
  const profile = {
    provider: base === OPENROUTER ? 'openrouter' : base,
    model: env.EMBEDDINGS_MODEL || 'google/gemini-embedding-2',
    dimensions: Number(env.EMBEDDINGS_DIMENSIONS) || 768,
    rule: 'block-v1',
  };
  return { profile, profileId: hash(profile).slice(0, 16) };
}

export const embeddingInput = (article, block) => article.title + '\n' + block.heading + '\n' + block.quote;

const finite = (vector) => vector.every(Number.isFinite);

/**
 * Matryoshka truncation: keep the leading dimensions, then restore unit length. Anything that is
 * not a list of numbers with a direction is refused here, before it can be cached for good.
 */
export function unitVector(values, dimensions) {
  if (!Array.isArray(values) && !ArrayBuffer.isView(values))
    throw new Error('Embedding is not an array of numbers');
  if (values.length < dimensions)
    throw new Error(`Embedding has ${values.length} dimensions; expected ${dimensions}`);
  const vector = new Float32Array(dimensions);
  let norm = 0;
  for (let i = 0; i < dimensions; i++) {
    if (typeof values[i] !== 'number' || !Number.isFinite(values[i]))
      throw new Error('Embedding holds a value that is not a finite number');
    vector[i] = values[i];
    norm += vector[i] * vector[i];
  }
  norm = Math.sqrt(norm);
  if (!(norm > 0) || !Number.isFinite(norm)) throw new Error('Embedding is a zero vector');
  for (let i = 0; i < dimensions; i++) vector[i] /= norm;
  return vector;
}

export function embedProvider(env = process.env, { fetch: request = fetch } = {}) {
  const base = baseUrl('EMBEDDINGS_API_BASE', env.EMBEDDINGS_API_BASE || OPENROUTER);
  // The OpenRouter key is never sent to another host.
  const apiKey =
    secret('EMBEDDINGS_API_KEY', env.EMBEDDINGS_API_KEY) ||
    (base === OPENROUTER ? secret('OPENROUTER_API_KEY', env.OPENROUTER_API_KEY) : undefined);
  const { profile, profileId } = embeddingProfile(env);
  const usage = { requests: 0, tokens: 0 };
  return {
    profile,
    profileId,
    usage,
    async embed(texts) {
      // Checked here rather than at construction: a fully cached corpus needs no key.
      if (!apiKey && base === OPENROUTER) throw new Error('Set OPENROUTER_API_KEY to embed new blocks.');
      const vectors = [];
      for (const input of chunks(texts, 64)) {
        const headers = { 'Content-Type': 'application/json' };
        if (apiKey) headers.Authorization = 'Bearer ' + apiKey;
        usage.requests++;
        const response = await send('Embedding', request, base + '/embeddings', {
          method: 'POST',
          headers,
          signal: AbortSignal.timeout(120000),
          body: JSON.stringify({
            model: profile.model,
            input,
            dimensions: profile.dimensions,
            encoding_format: 'float',
          }),
        });
        if (!response.ok) {
          const detail = await bodyExcerpt(response, [apiKey]);
          throw new Error('Embedding HTTP ' + response.status + (detail ? ': ' + detail : ''));
        }
        let payload;
        try {
          payload = await response.json();
        } catch {
          throw new Error('Embedding response is not JSON');
        }
        if (payload?.error) {
          const reported = payload.error;
          const text =
            typeof reported === 'string' ? reported : (reported.message ?? JSON.stringify(reported));
          throw new Error('Embedding error response: ' + excerpt(text, [apiKey]));
        }
        if (!Array.isArray(payload?.data) || payload.data.length !== input.length)
          throw new Error('Embedding response does not match the request size');
        usage.tokens += payload.usage?.prompt_tokens ?? payload.usage?.total_tokens ?? 0;
        // `index`, not arrival order, says which input a vector belongs to; every input needs one.
        const rows = new Array(input.length);
        payload.data.forEach((row, position) => (rows[row?.index ?? position] = row?.embedding));
        if (rows.length !== input.length || rows.includes(undefined))
          throw new Error('Embedding response does not cover every input');
        for (const row of rows) vectors.push(unitVector(row, profile.dimensions));
      }
      return vectors;
    },
  };
}

export function memoryStore() {
  const profiles = new Map();
  const table = (profileId) => profiles.get(profileId) || profiles.set(profileId, new Map()).get(profileId);
  return {
    async get(profileId, hashes) {
      const rows = table(profileId);
      return new Map(hashes.filter((h) => rows.has(h)).map((h) => [h, rows.get(h)]));
    },
    async put(profileId, vectors) {
      for (const [h, vector] of vectors) table(profileId).set(h, vector);
    },
  };
}

// False when an interrupted append left the file ending in a partial line.
async function endsWithNewline(path) {
  let file;
  try {
    file = await open(path, 'r');
  } catch (error) {
    if (error.code === 'ENOENT') return true;
    throw error;
  }
  try {
    const { size } = await file.stat();
    if (!size) return true;
    const { buffer } = await file.read(Buffer.alloc(1), 0, 1, size - 1);
    return buffer[0] === 0x0a;
  } finally {
    await file.close();
  }
}

/** Append-only local cache: `<dir>/<profileId>.jsonl`, one `{"h": hash, "v": base64 float32}` per line. */
export function fileStore(dir) {
  const loaded = new Map();
  async function load(profileId) {
    if (loaded.has(profileId)) return loaded.get(profileId);
    const rows = new Map();
    let text = '';
    try {
      text = await readFile(join(dir, profileId + '.jsonl'), 'utf8');
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    for (const line of text.split('\n')) {
      if (!line) continue;
      try {
        const { h, v } = JSON.parse(line);
        const bytes = Buffer.from(v, 'base64');
        // Copy: a pooled Buffer is not guaranteed to be 4-byte aligned.
        rows.set(h, new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length)));
      } catch {
        // An interrupted append leaves a partial last line; that block is simply embedded again.
      }
    }
    loaded.set(profileId, rows);
    return rows;
  }
  return {
    async get(profileId, hashes) {
      const rows = await load(profileId);
      return new Map(hashes.filter((h) => rows.has(h)).map((h) => [h, rows.get(h)]));
    },
    async put(profileId, vectors) {
      if (!vectors.size) return;
      const rows = await load(profileId);
      await mkdir(dir, { recursive: true });
      const lines = [...vectors].map(([h, vector]) =>
        JSON.stringify({
          h,
          v: Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength).toString('base64'),
        }),
      );
      const path = join(dir, profileId + '.jsonl');
      // Start on a fresh line, or the first new vector is glued to the partial one and lost too.
      const lead = (await endsWithNewline(path)) ? '' : '\n';
      await appendFile(path, lead + lines.join('\n') + '\n');
      for (const [h, vector] of vectors) rows.set(h, vector);
    },
  };
}

// The dashboard labels the project URL either way.
const supabaseUrl = (env) => env.SUPABASE_URL || env.SUPABASE_PROJECT_URL;

// Supabase names the same server-side key two ways: the newer `sb_secret_…` secret key and the
// legacy service_role JWT. Either bypasses RLS and is sent the same way.
function supabaseKey(env) {
  return env.SUPABASE_SECRET_KEY
    ? ['SUPABASE_SECRET_KEY', env.SUPABASE_SECRET_KEY]
    : ['SUPABASE_SERVICE_ROLE_KEY', env.SUPABASE_SERVICE_ROLE_KEY];
}

/** Shared cache in Supabase pgvector, reached through PostgREST with the secret (service role) key. */
export function supabaseStore(env = process.env, { fetch: request = fetch } = {}) {
  const table = baseUrl('SUPABASE_URL', supabaseUrl(env) || '') + '/rest/v1/block_embeddings';
  const key = secret(...supabaseKey(env));
  if (!key) throw new Error('Set SUPABASE_SECRET_KEY to use the shared embedding cache.');
  const auth = { apikey: key, Authorization: 'Bearer ' + key };
  async function call(url, init) {
    const response = await send('Supabase', request, url, { ...init, signal: AbortSignal.timeout(60000) });
    if (!response.ok)
      throw new Error(
        `Supabase HTTP ${response.status}: ${(await response.text()).replaceAll(key, '***').slice(0, 200)}`,
      );
    return response;
  }
  return {
    async get(profileId, hashes) {
      const found = new Map();
      for (const part of chunks(hashes, 80)) {
        const url = `${table}?profile=eq.${profileId}&input_hash=in.(${part.join(',')})&select=input_hash,embedding`;
        const rows = await (await call(url, { headers: auth })).json();
        for (const row of rows) {
          const values = typeof row.embedding === 'string' ? JSON.parse(row.embedding) : row.embedding;
          found.set(row.input_hash, Float32Array.from(values));
        }
      }
      return found;
    },
    async put(profileId, vectors) {
      for (const part of chunks([...vectors], 200)) {
        await call(table + '?on_conflict=profile,input_hash', {
          method: 'POST',
          headers: {
            ...auth,
            'Content-Type': 'application/json',
            Prefer: 'resolution=merge-duplicates,return=minimal',
          },
          body: JSON.stringify(
            part.map(([h, vector]) => ({
              profile: profileId,
              input_hash: h,
              embedding: '[' + Array.from(vector).join(',') + ']',
            })),
          ),
        });
      }
    },
  };
}

export function layered(local, remote) {
  return {
    async get(profileId, hashes) {
      const found = await local.get(profileId, hashes);
      const absent = hashes.filter((h) => !found.has(h));
      if (!absent.length) return found;
      const shared = await remote.get(profileId, absent);
      await local.put(profileId, shared);
      for (const [h, vector] of shared) found.set(h, vector);
      return found;
    },
    async put(profileId, vectors) {
      // Remote first: a vector that exists only locally would be embedded again, and possibly
      // differently, by the other publishing server.
      await remote.put(profileId, vectors);
      await local.put(profileId, vectors);
    },
  };
}

export function vectorStore(
  env = process.env,
  { dir = '.cache/embeddings', fetch: request, warn = console.warn } = {},
) {
  const local = fileStore(dir);
  const key = supabaseKey(env)[1];
  const url = supabaseUrl(env);
  if (url && key)
    return layered(local, supabaseStore(env, request ? { fetch: request } : {}));
  // One of the two alone is a typo, not a choice: running on would fill a cache nobody shares.
  if (url || key)
    throw new Error(
      (url ? 'SUPABASE_SECRET_KEY' : 'SUPABASE_URL') +
        ' is not set. Set both Supabase variables, or neither to use the local cache only.',
    );
  warn('Supabase가 설정되지 않아 로컬 캐시만 사용합니다. 다른 발행 서버와 벡터를 공유하지 않습니다.');
  return local;
}

/**
 * Vectors for every usable block of every public article. Only blocks absent from the cache are
 * embedded. `embedMissing: false` turns a cache miss into an error (used by --check).
 */
export async function loadVectors(ctx, provider, store, { embedMissing = true } = {}) {
  const { profileId } = provider;
  const dimensions = provider.profile.dimensions;
  const inputs = new Map();
  const articles = ctx.articles.map((article) => {
    const blocks = ctx.usable.get(article.id);
    const hashes = blocks.map((block) => {
      const input = embeddingInput(article, block);
      const inputHash = hash(input);
      inputs.set(inputHash, input);
      return inputHash;
    });
    return { id: article.id, blocks, hashes };
  });
  const known = await store.get(profileId, [...inputs.keys()]);
  for (const [h, vector] of known) {
    if (vector.length !== dimensions) throw new Error('Embedding cache holds a vector of the wrong size');
    // A NaN never wins a comparison, so the block would silently drop out of every ranking.
    if (!finite(vector)) known.delete(h);
  }
  const absent = [...inputs.keys()].filter((h) => !known.has(h));
  if (absent.length && !embedMissing) {
    const error = new Error(`${absent.length} blocks are missing from the embedding cache`);
    error.code = 'EMBEDDINGS_MISSING';
    throw error;
  }
  // Stored in slices so an interrupted initial run keeps what it already paid for.
  for (const part of chunks(absent, 256)) {
    const vectors = await provider.embed(part.map((h) => inputs.get(h)));
    if (
      vectors.length !== part.length ||
      !vectors.every((v) => v instanceof Float32Array && v.length === dimensions && finite(v))
    )
      throw new Error('Embedding provider returned an unusable vector');
    const fresh = new Map(part.map((h, i) => [h, vectors[i]]));
    await store.put(profileId, fresh);
    for (const [h, vector] of fresh) known.set(h, vector);
  }
  const byArticle = new Map(
    articles.map(({ id, blocks, hashes }) => {
      const matrix = new Float32Array(blocks.length * dimensions);
      hashes.forEach((h, i) => matrix.set(known.get(h), i * dimensions));
      const vectors = blocks.map((_, i) => matrix.subarray(i * dimensions, (i + 1) * dimensions));
      return [id, { blocks, vectors, matrix }];
    }),
  );
  return { profileId, dimensions, byArticle, embedded: absent.length, cached: inputs.size - absent.length };
}

function flat(entry) {
  if (entry.matrix) return entry.matrix;
  const dimensions = entry.vectors[0]?.length || 0;
  const matrix = new Float32Array(entry.vectors.length * dimensions);
  entry.vectors.forEach((vector, i) => matrix.set(vector, i * dimensions));
  return matrix;
}

/**
 * One pass over the block-by-block cosine matrix of two articles. Returns, for each S block, its
 * best match in T, and the reverse, so both directions share the work.
 */
export function maxCosine(S, T) {
  const n = S.vectors.length;
  const m = T.vectors.length;
  const rows = new Float32Array(n).fill(-1);
  const columns = new Float32Array(m).fill(-1);
  if (!n || !m) return [rows, columns];
  const a = flat(S);
  const b = flat(T);
  const d = S.vectors[0].length;
  for (let i = 0; i < n; i++) {
    const p = i * d;
    let best = -1;
    for (let j = 0; j < m; j++) {
      const q = j * d;
      let s0 = 0;
      let s1 = 0;
      let s2 = 0;
      let s3 = 0;
      let k = 0;
      for (; k + 3 < d; k += 4) {
        s0 += a[p + k] * b[q + k];
        s1 += a[p + k + 1] * b[q + k + 1];
        s2 += a[p + k + 2] * b[q + k + 2];
        s3 += a[p + k + 3] * b[q + k + 3];
      }
      for (; k < d; k++) s0 += a[p + k] * b[q + k];
      const dot = s0 + s1 + s2 + s3;
      if (dot > best) best = dot;
      if (dot > columns[j]) columns[j] = dot;
    }
    rows[i] = best;
  }
  return [rows, columns];
}

function topMean(best) {
  if (!best.length) return 0;
  const top = Array.from(best)
    .sort((a, b) => b - a)
    .slice(0, 3);
  return round4(top.reduce((sum, value) => sum + value, 0) / top.length);
}

/** Mean of the three best "S block against its closest T block" cosines; fewer when S is short. */
export function pairScore(S, T) {
  return S.vectors.length && T.vectors.length ? topMean(maxCosine(S, T)[0]) : 0;
}

/** All-pairs similarity, computed once and shared by candidate search and excerpt selection. */
export function similarityIndex(vectors) {
  const ids = [...vectors.byArticle.keys()];
  const position = new Map(ids.map((id, index) => [id, index]));
  const entries = ids.map((id) => vectors.byArticle.get(id));
  const best = ids.map(() => new Array(ids.length));
  const scores = ids.map(() => new Float64Array(ids.length));
  for (let s = 0; s < ids.length; s++) {
    for (let t = s + 1; t < ids.length; t++) {
      const [forward, backward] = maxCosine(entries[s], entries[t]);
      best[s][t] = forward;
      best[t][s] = backward;
      if (forward.length && backward.length) {
        scores[s][t] = topMean(forward);
        scores[t][s] = topMean(backward);
      }
    }
  }
  return {
    score: (source, target) => scores[position.get(source)]?.[position.get(target)] ?? 0,
    /** Best cosine of each usable `source` block against any `target` block, in block order. */
    blockMax: (source, target) => best[position.get(source)]?.[position.get(target)],
    /** Other articles with a positive score, best first; ties by article ID. */
    ranked(source) {
      const row = scores[position.get(source)];
      return ids
        .filter((id, index) => id !== source && row[index] > 0)
        .sort((a, b) => row[position.get(b)] - row[position.get(a)] || byId(a, b));
    },
  };
}

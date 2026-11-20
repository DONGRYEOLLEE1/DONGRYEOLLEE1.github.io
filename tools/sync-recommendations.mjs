import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadPublicCorpus } from './content/build.mjs';
import { embedProvider, loadVectors, vectorStore } from './knowledge/embeddings.mjs';
import { chatProvider } from './knowledge/model.mjs';
import { auditAll, auditSummary, prepareCorpus, syncAll } from './knowledge/phase2a.mjs';
import { readRecords } from './knowledge/store.mjs';

// Usage: node tools/sync-recommendations.mjs [--only id1,id2] [--reextract] [--dry-run] [--json path]
//        node tools/sync-recommendations.mjs --check [--offline] [--json path]
const DIR = fileURLToPath(new URL('../content/related/', import.meta.url));
const CACHE = fileURLToPath(new URL('../.cache/embeddings/', import.meta.url));
const list = (ids) => (ids.length ? ids.join(', ') : '없음');

function options(args) {
  const valued = ['--only', '--json'];
  const known = ['--check', '--offline', '--dry-run', '--reextract', ...valued];
  const parsed = {};
  for (let index = 0; index < args.length; index++) {
    const name = args[index];
    if (!known.includes(name)) throw new Error('Unknown option ' + name);
    if (!valued.includes(name)) parsed[name] = true;
    else if (!args[index + 1] || args[index + 1].startsWith('--')) throw new Error(name + ' needs a value');
    else parsed[name] = args[++index];
  }
  if (parsed['--offline'] && !parsed['--check']) throw new Error('--offline is only valid with --check');
  if (parsed['--check'] && (parsed['--only'] || parsed['--reextract'] || parsed['--dry-run']))
    throw new Error('--check verifies every article and takes no other mode');
  const only = parsed['--only']
    ?.split(',')
    .map((id) => id.trim())
    .filter(Boolean);
  if (only && !only.length) throw new Error('--only needs at least one article ID');
  return {
    check: Boolean(parsed['--check']),
    offline: Boolean(parsed['--offline']),
    dryRun: Boolean(parsed['--dry-run']),
    reextract: Boolean(parsed['--reextract']),
    jsonPath: parsed['--json'],
    only,
  };
}

/**
 * Runs one command and returns its exit code: 0 done, 1 stale or unusable, 2 some calls failed.
 * `deps` replaces the environment, directories, corpus loader, fetch and output for tests.
 */
export async function run(args, deps = {}) {
  const {
    env = process.env,
    dir = DIR,
    cache = CACHE,
    loadCorpus = loadPublicCorpus,
    fetch: request,
    out = console.log,
    warn = console.warn,
    err = console.error,
  } = deps;
  let exitCode = 0;

  /** The verification the build runs: stored records against the current text, no vectors or keys. */
  async function verifyOffline(ctx) {
    const audit = auditAll(ctx, await readRecords(dir));
    const summary = auditSummary(audit);
    out(
      `연관 글: LLM ${summary.llm}편, 규칙 ${summary.rules}편(파일 없음 ${summary.missing}, 무효 ${summary.invalid}` +
        (summary.fallback ? `, 추론 실패 ${summary.fallback}` : '') +
        (summary.noEvidence ? `, 근거 없음 ${summary.noEvidence}` : '') +
        ')',
    );
    for (const { id, reason } of summary.invalidIds) warn(`무효: ${id} (${reason})`);
    if (summary.orphans.length) warn('공개 글이 아닌 추천 파일: ' + list(summary.orphans));
    return { audit, summary };
  }

  async function main() {
    const { check, offline, dryRun, reextract, jsonPath, only } = options(args);
    const report = async (summary) => {
      if (jsonPath) await writeFile(jsonPath, JSON.stringify(summary, null, 2) + '\n');
    };

    if (offline) {
      const { summary } = await verifyOffline(prepareCorpus(await loadCorpus()));
      await report(summary);
      // A missing file is the normal state before the first sync; only a wrong file fails.
      if (summary.invalid) exitCode = 1;
      return;
    }

    const plan = check || dryRun;
    // Fails on a missing key before the corpus is rendered, not after.
    const provider = chatProvider(env, { fetch: request, requireKey: !plan });
    const embedder = embedProvider(env, { fetch: request });
    const store = vectorStore(env, { dir: cache, fetch: request, warn });
    const ctx = prepareCorpus(await loadCorpus());
    // Before any block is embedded for a run that cannot start.
    for (const id of only || []) if (!ctx.byId.has(id)) throw new Error('Not a public article: ' + id);
    let vectors;
    try {
      vectors = await loadVectors(ctx, embedder, store, { embedMissing: !check });
    } catch (error) {
      if (error.code !== 'EMBEDDINGS_MISSING') throw error;
      throw new Error(
        `임베딩이 필요합니다: ${error.message}. npm run recommendations:sync 를 실행하거나, ` +
          '벡터 없이 검사하려면 --check --offline 을 사용하세요.',
      );
    }
    const result = await syncAll(ctx, provider, {
      dir,
      vectors,
      only,
      reextract,
      dryRun: plan,
      concurrency: Number(env.RECOMMENDATIONS_CONCURRENCY) || 4,
      log: warn,
    });
    const summary = {
      ...result,
      embeddings: { ...result.embeddings, requests: embedder.usage.requests, tokens: embedder.usage.tokens },
      // The re-ranked articles and why: this is k when a new article is published.
      reranked: result.rank.items.filter((item) => item.call).map(({ id, reason }) => ({ id, reason })),
    };

    out(`글 ${summary.articles}편` + (only ? ` (대상 ${summary.scope}편)` : '') + `, 모델 ${summary.model}`);
    out(
      `임베딩: 새로 ${summary.embeddings.embedded}블록 / 캐시 ${summary.embeddings.cached}블록 / 요청 ${summary.embeddings.requests}회`,
    );
    if (plan) {
      const calls = (items) => items.filter((item) => item.call).length;
      const lines = (items) => items.map((item) => `  ${item.id} (${item.reason})`).join('\n');
      out(`facts 호출 대상 ${calls(summary.facts.items)}편 / 재사용 ${summary.facts.reused}편`);
      if (summary.facts.items.length) out(lines(summary.facts.items));
      out(`rank 호출 대상 ${calls(summary.rank.items)}편 / 재사용 ${summary.rank.reused}편`);
      if (summary.rank.items.length) out(lines(summary.rank.items));
      if (summary.facts.items.length) out('facts를 새로 추출하면 관계가 달라져 rank 대상이 늘 수 있습니다.');
      if (summary.orphans.length) out('삭제 대상 추천 파일: ' + list(summary.orphans));
      if (!check) {
        await report(summary);
        return;
      }
      // The plan compares hashes only. The build also checks each record's header and rows, so the
      // gate runs that verification too: what passes here is what the build will publish.
      const { audit } = await verifyOffline(ctx);
      // A stored fallback has a current hash and is a valid published state; sync retries it anyway.
      const stale = new Set(
        [...summary.facts.items, ...summary.rank.items]
          .filter((item) => item.reason !== 'fallback-retry')
          .map((item) => item.id),
      );
      for (const id of audit.invalid.keys()) stale.add(id);
      await report({ ...summary, stale: [...stale] });
      if (stale.size || summary.orphans.length) {
        if (stale.size) err(`낡은 글 ${stale.size}편: ${list([...stale])}`);
        if (summary.orphans.length)
          err(`공개 글이 아닌 추천 파일 ${summary.orphans.length}개가 남아 있습니다.`);
        err('npm run recommendations:sync 를 실행하세요.');
        exitCode = 1;
      } else {
        out('모든 글의 저장 해시가 현재 입력과 같습니다.');
      }
      return;
    }
    await report(summary);

    out(
      `facts: 호출 ${summary.facts.called} / 재사용 ${summary.facts.reused} / 실패 ${summary.facts.failed}`,
    );
    out(
      `rank: 호출 ${summary.rank.called} / 재사용 ${summary.rank.reused} / 보류 ${summary.rank.abstained} / 실패 ${summary.rank.failed}`,
    );
    out(
      `토큰: LLM 입력 ${summary.usage.promptTokens} · 출력 ${summary.usage.completionTokens} (요청 ${summary.usage.requests}회), 임베딩 ${summary.embeddings.tokens}`,
    );
    if (summary.orphans.length) out('삭제한 추천 파일: ' + list(summary.orphans));
    if (summary.facts.failed || summary.rank.failed) exitCode = 2;

    // A publisher may pull or edit posts while model calls are in flight.
    const latest = prepareCorpus(await loadCorpus());
    const { audit } = await verifyOffline(latest);
    const scope = new Set(only || latest.articles.map((article) => article.id));
    const broken = [...audit.invalid.keys(), ...audit.missing].filter((id) => scope.has(id));
    if (broken.length) {
      err(`동기화 중 원문이 바뀌었습니다. 다시 실행하세요: ${list(broken)}`);
      exitCode = 1;
    }
    // New facts of an --only article change the relations of every pair it is part of; the other
    // side of those pairs was not re-ranked and the build would send it back to the rules.
    const outside = [...audit.invalid.keys()].filter((id) => !scope.has(id));
    if (outside.length) {
      err(
        `대상 밖 ${outside.length}편의 추천이 현재 입력과 맞지 않습니다: ${list(outside)}\n` +
          '--only 없이 npm run recommendations:sync 를 실행하세요.',
      );
      exitCode = 1;
    }
  }

  try {
    await main();
  } catch (error) {
    err(env.DEBUG ? error : error.message);
    exitCode = 1;
  }
  return exitCode;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  process.exitCode = await run(process.argv.slice(2));

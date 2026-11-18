import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const directory = path.dirname(fileURLToPath(import.meta.url));
const roles = new Map([['about', 1], ['applies', 0.8]]);
const allowedStates = new Set(['published', 'draft']);
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const unique = (items, label) => {
  assert(items.every(x => typeof x === 'string' && x.length > 0), `${label}: invalid ID`);
  assert(new Set(items).size === items.length, `${label}: duplicate ID`);
};

export function buildGraph(dataset, vocabulary, registry) {
  assert(vocabulary.schemaVersion === 1 && registry.schemaVersion === 1, 'Unsupported schema version');
  assert(Array.isArray(dataset.posts) && Array.isArray(dataset.series), 'Invalid dataset');
  assert(Array.isArray(vocabulary.concepts) && Array.isArray(registry.articles), 'Invalid knowledge inputs');
  unique(dataset.posts.map(p => p.id), 'Article dataset');
  unique(vocabulary.concepts.map(c => c.id), 'Concept registry');
  unique(registry.articles.map(a => a.id), 'Article knowledge registry');
  const concepts = new Map(vocabulary.concepts.map(c => [c.id, c]));
  const posts = new Map(dataset.posts.map(p => [p.id, p]));
  const articles = new Map(registry.articles.map(a => [a.id, a]));
  assert(posts.size === articles.size && [...posts.keys()].every(id => articles.has(id)), 'Knowledge registry must cover exactly the input dataset');

  for (const c of concepts.values()) {
    assert(typeof c.label === 'string' && c.label.trim(), `Missing concept label: ${c.id}`);
    assert(Array.isArray(c.aliases) && c.aliases.every(x => typeof x === 'string'), `Invalid aliases: ${c.id}`);
    assert(Array.isArray(c.broaderIds), `Invalid broader IDs: ${c.id}`);
    unique(c.broaderIds, `Broader IDs for ${c.id}`);
    assert(c.recommendationWeight === 0 || c.recommendationWeight === 1, `Invalid recommendation weight: ${c.id}`);
    for (const parent of c.broaderIds) assert(parent !== c.id && concepts.has(parent), `Unknown or self broader concept: ${c.id} -> ${parent}`);
  }
  const visited = new Set(), visiting = new Set();
  const visit = id => {
    assert(!visiting.has(id), `Concept hierarchy cycle at ${id}`);
    if (visited.has(id)) return;
    visiting.add(id);
    for (const parent of concepts.get(id).broaderIds) visit(parent);
    visiting.delete(id); visited.add(id);
  };
  for (const id of concepts.keys()) visit(id);
  for (const article of articles.values()) {
    assert(allowedStates.has(article.publicationState), `Invalid publication state: ${article.id}`);
    assert(Array.isArray(article.concepts), `Invalid concept references: ${article.id}`);
    unique(article.concepts.map(r => r.id), `Concept references for ${article.id}`);
    assert(article.concepts.length > 0, `No concepts: ${article.id}`);
    for (const ref of article.concepts) {
      assert(concepts.has(ref.id), `Unknown concept ${ref.id} in ${article.id}`);
      assert(roles.has(ref.role), `Invalid concept role ${ref.role} in ${article.id}`);
    }
    const original = posts.get(article.id);
    assert(!(article.publicationState === 'published' && (original.published === false || original.publicationState === 'draft')), `Conflicting publication state: ${article.id}`);
  }
  const seriesNeighbors = new Map();
  unique(dataset.series.map(s => s.id), 'Series registry');
  for (const series of dataset.series) {
    assert(Array.isArray(series.articleIds), `Invalid series: ${series.id}`);
    unique(series.articleIds, `Series ${series.id}`);
    for (const id of series.articleIds) assert(posts.has(id), `Unknown article in series ${series.id}: ${id}`);
    for (let i = 0; i + 1 < series.articleIds.length; i++) {
      const from = series.articleIds[i], to = series.articleIds[i + 1];
      seriesNeighbors.set(`${from}:${to}`, {type:'series-neighbor',seriesId:series.id,direction:'next'});
      seriesNeighbors.set(`${to}:${from}`, {type:'series-neighbor',seriesId:series.id,direction:'previous'});
    }
  }

  const published = [...articles.values()].filter(a => a.publicationState === 'published').sort((a,b) => a.id.localeCompare(b.id, 'en'));
  const N = published.length;
  const idf = new Map([...concepts.keys()].map(id => {
    const df = published.filter(a => a.concepts.some(c => c.id === id)).length;
    return [id, Math.min(3, 1 + Math.log((N + 1) / (df + 1)))];
  }));
  const describe = (concept, sourceRole, targetRole) => {
    if (sourceRole === 'about' && targetRole === 'applies') return {code:'application-of',text:`적용 사례 · ${concept.label}`};
    if (sourceRole === 'applies' && targetRole === 'about') return {code:'core-topic',text:`핵심 주제로 다룬 글 · ${concept.label}`};
    if (sourceRole === 'applies' && targetRole === 'applies') return {code:'shared-application',text:`같은 기술을 적용한 사례 · ${concept.label}`};
    return {code:'shared-concept',text:`함께 다룬 주제 · ${concept.label}`};
  };
  const perArticle = {};
  for (const source of published) {
    const candidates = [];
    const a = new Map(source.concepts.map(c => [c.id, c.role]));
    for (const target of published) {
      if (source.id === target.id) continue;
      const b = new Map(target.concepts.map(c => [c.id, c.role]));
      const specificShared = [...a.keys()].filter(id => b.has(id) && concepts.get(id).recommendationWeight > 0);
      const series = seriesNeighbors.get(`${source.id}:${target.id}`);
      if (!specificShared.length && !series) continue;
      let numerator = 0, denominator = 0;
      for (const id of new Set([...a.keys(), ...b.keys()])) {
        if (!concepts.get(id).recommendationWeight) continue;
        const wA = roles.get(a.get(id)) || 0, wB = roles.get(b.get(id)) || 0;
        numerator += idf.get(id) * Math.min(wA, wB);
        denominator += idf.get(id) * Math.max(wA, wB);
      }
      const conceptScore = denominator ? numerator / denominator : 0;
      const seriesScore = series ? 1 : 0;
      const score = 0.85 * conceptScore + 0.15 * seriesScore;
      if (score <= 0) continue;
      specificShared.sort((x,y) => idf.get(y)*Math.min(roles.get(a.get(y)),roles.get(b.get(y))) - idf.get(x)*Math.min(roles.get(a.get(x)),roles.get(b.get(x))) || x.localeCompare(y,'en'));
      const evidence = specificShared.map(id => ({type:'shared-concept',conceptId:id,sourceRole:a.get(id),targetRole:b.get(id)}));
      if (series) evidence.push(series);
      const reason = specificShared.length ? describe(concepts.get(specificShared[0]), a.get(specificShared[0]), b.get(specificShared[0])) : {code:'series-neighbor',text:series.direction === 'next' ? '같은 시리즈의 다음 글' : '같은 시리즈의 이전 글'};
      candidates.push({id:target.id,score:Number(score.toFixed(6)),signals:{concept:Number(conceptScore.toFixed(6)),series:seriesScore},reason:reason.text,reasonCode:reason.code,evidence});
    }
    candidates.sort((a,b) => b.score - a.score || a.id.localeCompare(b.id,'en'));
    perArticle[source.id] = {targets:candidates.slice(0,4)};
  }
  const input = {vocabulary,registry,posts:dataset.posts.map(({related,...post})=>post),series:dataset.series};
  const sourceHash = createHash('sha256').update(JSON.stringify(input)).digest('hex');
  const result = {manifest:{schemaVersion:1,mode:'concept',scorerVersion:'concept-jaccard-series-v1',sourceHash,articleCount:N,conceptCount:concepts.size,candidatePairCount:N*(N-1),algorithm:'0.85 × role/IDF-weighted concept Jaccard + 0.15 × explicit series adjacency',scope:'10-post UI prototype; reviewed concept metadata; no LLM extraction or graph database'},perArticle};
  for (const [id, entry] of Object.entries(perArticle)) {
    unique(entry.targets.map(t=>t.id), `Recommendation targets for ${id}`);
    assert(entry.targets.every(t => t.id !== id && perArticle[t.id]), `Invalid or nonpublic target in ${id}`);
  }
  return result;
}

export async function loadInputs() {
  // This is a fixed, trusted preview data file in this repository, not user-supplied JavaScript.
  const source = await readFile(path.join(directory,'../data.js'),'utf8');
  const context = {window:{}};
  vm.runInNewContext(source,context,{timeout:1000,filename:'trusted-preview-data.js',contextCodeGeneration:{strings:false,wasm:false}});
  const dataset = JSON.parse(JSON.stringify(context.window.BLOG_DATA));
  const vocabulary = JSON.parse(await readFile(path.join(directory,'concepts.json'),'utf8'));
  const registry = JSON.parse(await readFile(path.join(directory,'article-concepts.json'),'utf8'));
  return {dataset,vocabulary,registry};
}

async function main() {
  const {dataset,vocabulary,registry} = await loadInputs();
  const graph = buildGraph(dataset,vocabulary,registry);
  const rolesByArticle = Object.fromEntries(registry.articles.map(a => [a.id,Object.fromEntries(a.concepts.map(c => [c.id,[c.role]]))]));
  const json = JSON.stringify(graph,null,2)+'\n';
  const js = `window.ONTOLOGY_RELATED = ${JSON.stringify(graph)};\n(() => {\n  const roles = ${JSON.stringify(rolesByArticle)};\n  for (const p of window.BLOG_DATA.posts) {\n    p.related = (window.ONTOLOGY_RELATED.perArticle[p.id]?.targets || []).map(t => ({id:t.id,reason:t.reason,score:t.score,reasonCode:t.reasonCode,evidence:t.evidence}));\n    p.topicRoles = roles[p.id] || {};\n  }\n  window.BLOG_DATA.meta.relatedAreEditorialExamples = false;\n  window.BLOG_DATA.meta.relatedGeneratedFromReviewedConcepts = true;\n  window.BLOG_DATA.meta.note = '실제 글 10편의 검토된 개념·역할과 시리즈 순서를 이용해 빌드할 때 관련 글을 계산한 시안입니다.';\n})();\n`;
  await writeFile(path.join(directory,'related.json'),json);
  await writeFile(path.join(directory,'related.generated.js'),js);
  console.log(JSON.stringify({articles:graph.manifest.articleCount,concepts:graph.manifest.conceptCount,sourceHash:graph.manifest.sourceHash,recommendations:Object.fromEntries(Object.entries(graph.perArticle).map(([id,r])=>[id,r.targets.map(t=>t.id)]))}));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}

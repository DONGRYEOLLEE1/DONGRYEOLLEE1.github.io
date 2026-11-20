/** Build-time recommendations. Approved relationships are facts; scores are derived navigation. */
const BROAD = new Set(['rag', 'peft', 'finetuning', 'pretraining', 'python-language', 'model-serving', 'remote-development']);
const ROLES = new Set(['about', 'applies']);
const STOP = new Set(('the and with from this that these those for are was were will can could not but into have has had use used using when then than their our its some all also more most how what why which where model models learning training language large neural network networks paper review introduction background method results conclusion reference references example examples python import print return self none true false def class torch tensor transformers http https github com arxiv org article new task tasks data set dataset error problem solution developing daily via on in to of a an is as at by or be it we you i they만드는').split(/\s+/));
const KO_STOP = new Set(['있습니다','합니다','정리합니다','살펴봅니다','기록합니다','사용하는','발생하는','사용할','경우','통해','위한','대한','모델의','모델을','학습을','학습','모델','이러한','그리고','하지만','이것은','합니다','있음','수행','방식','방법','결과','때문에']);
const fail = message => { throw new Error(`Knowledge validation: ${message}`); };
const publicOnly = articles => articles.filter(a => (a.publicationState ?? (a.published === false ? 'draft' : 'published')) === 'published');
const mapUnique = (values, name) => {
  const map = new Map();
  for (const value of values) {
    if (!value || typeof value.id !== 'string' || !value.id.trim()) fail(`${name} needs an id`);
    if (map.has(value.id)) fail(`duplicate ${name} id ${value.id}`);
    map.set(value.id, value);
  }
  return map;
};

/** Check both ontology structure and references before emitting any public recommendation. */
export function validateKnowledge(articles, concepts, series, overrides = {}) {
  const articleMap = mapUnique(articles, 'article');
  const publicIds = new Set(publicOnly(articles).map(article => article.id));
  const conceptMap = mapUnique(concepts, 'concept');
  const seriesMap = mapUnique(series, 'series');
  for (const concept of concepts) {
    if (!concept.label || !concept.description || !Array.isArray(concept.aliases) || !Array.isArray(concept.broaderIds)) fail(`invalid concept ${concept.id}`);
    if (new Set(concept.broaderIds).size !== concept.broaderIds.length) fail(`duplicate broader relation ${concept.id}`);
    for (const parent of concept.broaderIds) {
      if (!conceptMap.has(parent)) fail(`unknown broader concept ${parent}`);
      if (parent === concept.id) fail(`self hierarchy ${concept.id}`);
    }
  }
  const visiting = new Set(), visited = new Set();
  function visit(id) {
    if (visiting.has(id)) fail(`hierarchy cycle at ${id}`);
    if (visited.has(id)) return;
    visiting.add(id);
    for (const parent of conceptMap.get(id).broaderIds) visit(parent);
    visiting.delete(id); visited.add(id);
  }
  for (const id of conceptMap.keys()) visit(id);
  for (const article of articles) {
    if (!Array.isArray(article.conceptRefs)) fail(`article ${article.id} needs a conceptRefs array`);
    const refs = new Set();
    for (const ref of article.conceptRefs) {
      if (!conceptMap.has(ref.id)) fail(`unknown concept ${ref.id} in ${article.id}`);
      if (!ROLES.has(ref.role)) fail(`invalid concept role ${ref.role} in ${article.id}`);
      const key = `${ref.id}:${ref.role}`;
      if (refs.has(key)) fail(`duplicate concept role ${key} in ${article.id}`);
      refs.add(key);
    }
  }
  const membership = new Map();
  for (const item of seriesMap.values()) {
    if (!Array.isArray(item.orderedArticleIds) || !item.orderedArticleIds.length) fail(`empty series ${item.id}`);
    const seen = new Set();
    for (const id of item.orderedArticleIds) {
      if (!articleMap.has(id)) fail(`unknown series article ${id}`);
      if (seen.has(id)) fail(`duplicate series entry ${id}`);
      if (membership.has(id)) fail(`article ${id} belongs to multiple series`);
      seen.add(id); membership.set(id, item.id);
    }
  }
  for (const kind of ['pins', 'excludes', 'buildsOn']) {
    const table = overrides[kind] || {};
    if (!table || Array.isArray(table) || typeof table !== 'object') fail(`invalid override table ${kind}`);
    for (const [source, targets] of Object.entries(table)) {
      if (!articleMap.has(source)) fail(`unknown override source ${source}`);
      if (!Array.isArray(targets)) fail(`override ${kind}.${source} must be an array`);
      if (new Set(targets).size !== targets.length) fail(`duplicate ${kind} override for ${source}`);
      for (const target of targets) {
        if (!articleMap.has(target)) fail(`unknown override target ${target}`);
        if (source === target) fail(`self ${kind} override ${source}`);
        if (!publicIds.has(source) || !publicIds.has(target)) fail(`non-public override ${source} -> ${target}`);
      }
    }
  }
  for (const [source, targets] of Object.entries(overrides.pins || {})) {
    for (const target of targets) if ((overrides.excludes?.[source] || []).includes(target)) fail(`pin/exclude conflict ${source} -> ${target}`);
  }
  return true;
}

export function tokenWeights(article, bodyLimit = 18000) {
  const title = `${article.title || ''} ${article.description || ''}`.toLowerCase();
  const body = String(article.plainText || '').slice(0, bodyLimit).toLowerCase();
  const map = new Map();
  for (const [text, weight] of [[body, 1], [title, 3]]) {
    const unique = new Set(text.match(/[\p{L}\p{N}][\p{L}\p{N}_+.-]{2,}/gu) || []);
    for (let word of unique) {
      word = word.replace(/[.]+$/, '');
      if (word.length < 3 || word.length > 36 || STOP.has(word) || KO_STOP.has(word) || /^\d+$/.test(word)) continue;
      map.set(word, (map.get(word) || 0) + weight);
    }
  }
  return map;
}
export function weightedJaccard(a, b, idf = () => 1) {
  let intersection = 0, union = 0;
  for (const term of new Set([...a.keys(), ...b.keys()])) {
    const av = a.get(term) || 0, bv = b.get(term) || 0, weight = idf(term);
    intersection += Math.min(av, bv) * weight;
    union += Math.max(av, bv) * weight;
  }
  return union ? intersection / union : 0;
}
function weightedCoverage(source, target, idf) {
  let intersection = 0, total = 0;
  for (const [id, weight] of source) {
    total += weight * idf(id);
    intersection += (target.has(id) ? weight : 0) * idf(id);
  }
  return total ? intersection / total : 0;
}
function conceptWeights(article) {
  const map = new Map();
  for (const ref of article.conceptRefs) map.set(ref.id, Math.max(map.get(ref.id) || 0, ref.role === 'about' ? 1 : .65));
  return map;
}
function normalizedPath(path) {
  try { return decodeURI(path).replace(/\/+$/, '') || '/'; } catch { return path.replace(/\/+$/, '') || '/'; }
}
function articleCitations(article, pathToId, ids) {
  const cites = new Set((article.cites || []).filter(id => ids.has(id)));
  const ownHosts = new Set(['dongryeollee1.github.io']);
  try { if (process.env.SITE_URL) ownHosts.add(new URL(process.env.SITE_URL).hostname); } catch { /* Invalid origins fail site configuration separately. */ }
  for (const match of String(article.html || '').matchAll(/href\s*=\s*["']([^"']+)["']/g)) {
    const value = match[1].replaceAll('&amp;', '&');
    if (value.startsWith('#')) continue;
    try {
      const url = new URL(value, `https://dongryeollee1.github.io${article.path || '/'}`);
      if (!ownHosts.has(url.hostname)) continue;
      const id = pathToId.get(normalizedPath(url.pathname));
      if (id && id !== article.id) cites.add(id);
    } catch { /* Non-HTTP and malformed URLs do not establish an internal citation. */ }
  }
  return cites;
}

/** Return a new public Article[]; no input mutation, model calls, or runtime database. */
export function generateRelated(inputArticles, concepts, series, overrides = {}, limit = 4) {
  validateKnowledge(inputArticles, concepts, series, overrides);
  const articles = publicOnly(inputArticles);
  const ids = new Set(articles.map(a => a.id));
  const conceptMap = new Map(concepts.map(c => [c.id, c]));
  const conceptDf = new Map(), textDf = new Map();
  const conceptVectors = new Map(), textVectors = new Map();
  for (const article of articles) {
    const cv = conceptWeights(article), tv = tokenWeights(article);
    conceptVectors.set(article.id, cv); textVectors.set(article.id, tv);
    for (const id of cv.keys()) conceptDf.set(id, (conceptDf.get(id) || 0) + 1);
    for (const word of tv.keys()) textDf.set(word, (textDf.get(word) || 0) + 1);
  }
  const conceptIdf = id => (Math.log(1 + articles.length / (1 + (conceptDf.get(id) || 0))) + 1) * (BROAD.has(id) ? .12 : 1);
  const textIdf = word => Math.log(1 + articles.length / (1 + (textDf.get(word) || 0)));
  const pathToId = new Map(articles.map(a => [normalizedPath(a.path || `/${a.id}/`), a.id]));
  const citations = new Map(articles.map(a => [a.id, articleCitations(a, pathToId, ids)]));
  const seriesInfo = new Map();
  for (const item of series) {
    const publishedIds = item.orderedArticleIds.filter(id => ids.has(id));
    publishedIds.forEach((id, index) => seriesInfo.set(id, { id: item.id, index, ordered: publishedIds }));
  }
  return articles.map(source => {
    const ownSeries = seriesInfo.get(source.id);
    const excluded = new Set(overrides.excludes?.[source.id] || []);
    const pins = (overrides.pins?.[source.id] || []).filter(id => ids.has(id));
    const ranked = [];
    for (const target of articles) {
      if (target.id === source.id || excluded.has(target.id)) continue;
      const sameSeries = ownSeries && seriesInfo.get(target.id)?.id === ownSeries.id;
      const seriesDistance = sameSeries ? seriesInfo.get(target.id).index - ownSeries.index : 0;
      const cited = citations.get(source.id).has(target.id), citedBy = citations.get(target.id).has(source.id);
      const buildsOn = (overrides.buildsOn?.[source.id] || []).includes(target.id);
      const builtBy = (overrides.buildsOn?.[target.id] || []).includes(source.id);
      const shared = [...conceptVectors.get(source.id).keys()].filter(id => conceptVectors.get(target.id).has(id));
      const specific = shared.filter(id => !BROAD.has(id)).sort((a, b) => conceptIdf(b) - conceptIdf(a) || a.localeCompare(b));
      const pinned = pins.includes(target.id);
      // Shared generic categories or incidental words alone never create a relation.
      if (!specific.length && !sameSeries && !cited && !citedBy && !buildsOn && !builtBy && !pinned) continue;
      // Directed coverage avoids penalizing a useful implementation for its additional topics.
      const conceptual = .65 * weightedCoverage(conceptVectors.get(source.id), conceptVectors.get(target.id), conceptIdf)
        + .35 * weightedJaccard(conceptVectors.get(source.id), conceptVectors.get(target.id), conceptIdf);
      const lexical = weightedJaccard(textVectors.get(source.id), textVectors.get(target.id), textIdf);
      let score = .72 * conceptual + .12 * lexical;
      if (sameSeries) score += (Math.abs(seriesDistance) === 1 ? .58 : .26 / Math.abs(seriesDistance));
      if (cited) score += .46;
      if (citedBy) score += .28;
      if (buildsOn || builtBy) score += .6;
      if (pinned) score += 2 + (pins.length - pins.indexOf(target.id)) / 100;
      const concept = specific[0], label = conceptMap.get(concept)?.label;
      let reason;
      if (buildsOn) reason = '명시한 선행 작업';
      else if (builtBy) reason = '이 글을 바탕으로 한 작업';
      else if (sameSeries) reason = seriesDistance === 1 ? '같은 시리즈의 다음 글' : seriesDistance === -1 ? '같은 시리즈의 이전 글' : '같은 시리즈의 다른 글';
      else if (cited) reason = '본문에서 연결한 글';
      else if (citedBy) reason = '이 글을 본문에서 연결한 글';
      else if (label && target.conceptRefs.some(r => r.id === concept && r.role === 'applies') && source.conceptRefs.some(r => r.id === concept && r.role === 'about')) reason = `적용 사례 · ${label}`;
      else if (label && source.conceptRefs.some(r => r.id === concept && r.role === 'applies') && target.conceptRefs.some(r => r.id === concept && r.role === 'about')) reason = `개념 설명 · ${label}`;
      else if (label) reason = `공통 개념 · ${label}`;
      else reason = '작성자가 선택한 이어 읽기';
      if (score >= .055 || pinned) ranked.push({ id: target.id, reason, score: Number(score.toFixed(6)), date: target.publishedAt || '' });
    }
    ranked.sort((a, b) => b.score - a.score || b.date.localeCompare(a.date) || a.id.localeCompare(b.id));
    const { seriesId: _oldSeries, seriesOrder: _oldOrder, ...cleanSource } = source;
    return { ...cleanSource, ...(ownSeries ? { seriesId: ownSeries.id, seriesOrder: ownSeries.index + 1 } : {}), related: ranked.slice(0, limit).map(({date, ...result}) => result) };
  });
}

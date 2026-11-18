import { getArticles, getConcepts } from './content';
import { absoluteUrl } from './site';

type Kind = 'tags' | 'categories' | 'archives' | 'about';
const escape = (value: unknown) => String(value ?? '').replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]!));
const normalize = (value: string) => value.toLocaleLowerCase().replace(/[\s_-]+/g, '');

export function legacyDocument(kind: Kind) {
  const articles = getArticles(), concepts = getConcepts();
  const target = kind === 'about' ? '/about/' : kind === 'tags' ? '/topics/' : '/articles/';
  const title = { tags: '이전 태그', categories: '이전 카테고리', archives: '이전 글 보관함', about: '소개 페이지 이동' }[kind];
  const values = kind === 'archives' ? [...new Set(articles.map(article => article.publishedAt.slice(0,4)))].sort().reverse()
    : kind === 'about' ? [] : [...new Set(articles.flatMap(article => kind === 'tags' ? article.legacyTags : article.legacyCategories))].sort((a,b) => a.localeCompare(b));
  const groups = values.map(value => ({ value,
    articles: articles.filter(article => kind === 'archives' ? article.publishedAt.startsWith(value) : (kind === 'tags' ? article.legacyTags : article.legacyCategories).includes(value)),
    concept: concepts.find(concept => [concept.id, concept.label, ...concept.aliases].some(alias => normalize(alias) === normalize(value))),
  }));
  const map = Object.fromEntries(groups.filter(group => group.concept).map(group => [group.value, `/topics/${group.concept!.id}/`]));
  const mapJson = JSON.stringify(map).replaceAll('<', '\\u003c');
  return `<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escape(title)} · 동렬</title><meta name="description" content="저장해 둔 이전 주소에서 글과 주제를 이어서 찾을 수 있습니다."><meta name="robots" content="noindex,follow"><link rel="canonical" href="${escape(absoluteUrl(target))}"><link rel="icon" href="/favicon.svg" type="image/svg+xml"><link rel="alternate" type="application/rss+xml" href="/feed.xml" title="동렬 기술 블로그"><style>
@font-face{font-family:Pretendard;src:url('/fonts/PretendardVariable.woff2') format('woff2');font-weight:45 920;font-display:swap}*{box-sizing:border-box}html{scroll-padding-top:35px}body{margin:0;background:#faf9f5;color:#141413;font:16px/1.75 Pretendard,system-ui,sans-serif;font-synthesis:none}a{color:inherit;text-underline-offset:4px}header{max-width:1280px;margin:auto;padding:30px 0;display:flex;justify-content:space-between;gap:30px}nav{display:flex;gap:24px;flex-wrap:wrap;font-size:14px;font-weight:550}.brand{font-size:24px;font-weight:700;text-decoration:none;letter-spacing:-.02em}main{max-width:1000px;margin:48px auto 100px}h1{font-size:40px;font-weight:650;line-height:1.2;letter-spacing:-.025em;margin:0 0 24px}h2{font-size:24px;font-weight:600;line-height:1.4;letter-spacing:-.015em}p{color:#67665f}.groups{display:flex;flex-wrap:wrap;gap:12px 20px;margin:40px 0;font-size:13px}.group{border-top:1px solid #d4d1c8;margin-top:40px;padding-top:16px}.section-heading{display:flex;justify-content:space-between;gap:24px;align-items:center}.section-heading>a{font-size:13px}ul{padding:0;list-style:none}li{margin:18px 0}::selection{background:#d6eae4;color:#122f29}:focus-visible{outline:2px solid #2c6659;outline-offset:5px}@media(max-width:1400px){header,main{margin-left:36px;margin-right:36px}}@media(max-width:767px){header,main{margin-left:20px;margin-right:20px}header{padding:24px 0}nav{gap:14px}main{margin-top:32px}h1{font-size:30px}.section-heading{align-items:start;flex-direction:column;gap:0}}@media(prefers-color-scheme:dark){body{background:#1d1d1b;color:#f0efe8}p{color:#b2b0a7}.group{border-color:#4e4c45}::selection{background:#315e53;color:#f4faf7}}
</style></head><body><header><a class="brand" href="/">동렬</a><nav aria-label="기본 메뉴"><a href="/articles/">글</a><a href="/topics/">주제</a><a href="/about/">소개</a></nav></header><main><h1>${escape(title)}</h1><p>글은 기존 주소에서 계속 읽을 수 있습니다. 새 메뉴에서 글과 주제를 찾아보세요.</p><a href="${target}">새 ${kind === 'about' ? '소개' : kind === 'tags' ? '주제 목록' : '글 목록'}으로 이동 ↗</a>${groups.length ? `<nav class="groups" aria-label="이전 분류 바로가기">${groups.map(group => `<a href="#${encodeURIComponent(group.value)}">${escape(group.value)} (${group.articles.length})</a>`).join('')}</nav>` : ''}${groups.map(group => `<section class="group" id="${escape(group.value)}"><div class="section-heading"><h2>${escape(group.value)}</h2>${group.concept ? `<a href="/topics/${group.concept.id}/">${escape(group.concept.label)} 주제로 읽기 ↗</a>` : ''}</div><ul>${group.articles.map(article => `<li><a href="${escape(article.path)}">${escape(article.title)}</a></li>`).join('')}</ul></section>`).join('')}</main><script>
const knownTargets=${mapJson};let fragment='';try{fragment=decodeURIComponent(location.hash.replace(/^#(?:h-)?/,''));}catch{}const destination=fragment?(Object.hasOwn(knownTargets,fragment)?knownTargets[fragment]:undefined):${JSON.stringify(target)};if(typeof destination==='string'&&destination.startsWith('/')&&!destination.startsWith('//'))location.replace(destination);else if(fragment)document.getElementById(fragment)?.scrollIntoView();
</script></body></html>`;
}

export const legacyResponse = (kind: Kind) => new Response(legacyDocument(kind), { headers: { 'Content-Type': 'text/html; charset=utf-8' } });

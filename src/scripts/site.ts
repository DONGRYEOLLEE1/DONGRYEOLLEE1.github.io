import { renderRelatedGraph } from '../lib/related-graph';
import { measureRelatedDialogHeight } from './graph-preview';
import { ARCHIVE_SORT_LABELS, ARTICLE_TYPE_LABELS, CONCEPT_ROLE_LABELS, QUERY_LENGTH_MESSAGE, articleCountLabel, topicCountLabel } from '../lib/archive-copy';
import type { Article, Registry } from '../lib/types';

// The HTML remains the source of navigation and reading when JavaScript is unavailable.
type PublicArticle = Omit<Article, 'html' | 'plainText'> & { plainText?: string };
type SiteData = Omit<Registry, 'articles'> & { articles: PublicArticle[] };
type SearchHit = { article: PublicArticle; excerpt: string; rank: number };
type PagefindResult = { url: string; excerpt: string; meta: Record<string, string> };
type Pagefind = { search: (term: string) => Promise<{ results: { data: () => Promise<PagefindResult> }[] }> };
type ListState = { q: string; topics: string[]; types: string[]; sort: 'latest' | 'relevance'; page: number; view: 'grid' | 'list'; role: 'all' | 'about' | 'applies'; notice: string };

const $ = <T extends Element = HTMLElement>(selector: string, root: ParentNode = document) => root.querySelector<T>(selector);
const $$ = <T extends Element = HTMLElement>(selector: string, root: ParentNode = document) => [...root.querySelectorAll<T>(selector)];
const escape = (value: unknown) => String(value ?? '').replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]!));
const labels: Record<string, string> = ARTICLE_TYPE_LABELS;
let data: SiteData = { articles: [], concepts: [], series: [] };
try { data = JSON.parse($('#site-data')?.textContent || '{}'); } catch { /* Static HTML is still usable. */ }
data.articles ||= []; data.concepts ||= []; data.series ||= [];
const articles = new Map(data.articles.map(article => [article.id, article]));
const concepts = new Map(data.concepts.map(concept => [concept.id, concept]));
const normalizedPath = (path: string) => {
  try { return decodeURIComponent(new URL(path, location.origin).pathname).replace(/\/index\.html$/, '').replace(/\/+$/, '') || '/'; }
  catch { return ''; }
};
const articlesByPath = new Map(data.articles.map(article => [normalizedPath(article.path), article]));
const textOnlyExcerpt = (html: string) => html.split(/(<\/?mark>)/i).map(part => /^<\/?mark>$/i.test(part) ? part.toLowerCase() : escape(part)).join('');
const currentUrl = () => new URL(location.href);

let pagefindPromise: Promise<Pagefind> | undefined;
const searchCache = new Map<string, Promise<SearchHit[]>>();
async function pagefind(): Promise<Pagefind> {
  if (!pagefindPromise) {
    const moduleUrl = '/pagefind/pagefind.js';
    pagefindPromise = import(/* @vite-ignore */ moduleUrl).catch(error => { pagefindPromise = undefined; throw error; });
  }
  return pagefindPromise;
}
async function searchArticles(query: string): Promise<SearchHit[]> {
  const key = query.trim();
  if (!key) return data.articles.map((article, rank) => ({ article, rank, excerpt: '' }));
  if (!searchCache.has(key)) {
    const pending = (async () => {
      const api = await pagefind();
      const result = await api.search(key);
      const loaded = await Promise.all(result.results.map(async (hit, rank) => {
        const item = await hit.data();
        const article = articles.get(item.meta?.article_id || '') || articlesByPath.get(normalizedPath(item.url));
        return article ? { article, excerpt: item.excerpt || '', rank } : undefined;
      }));
      const seen = new Set<string>();
      return loaded.filter((hit): hit is SearchHit => !!hit && !seen.has(hit.article.id) && !!seen.add(hit.article.id));
    })();
    searchCache.set(key, pending);
    pending.catch(() => searchCache.delete(key));
  }
  return searchCache.get(key)!;
}
function retrySearch() { pagefindPromise = undefined; searchCache.clear(); }
function validQuery(input: HTMLInputElement | null, query: string, message = '검색어는 200자 이내로 입력해 주세요.'): boolean {
  const tooLong = [...query].length > 200;
  input?.setCustomValidity(tooLong ? message : '');
  const error = input?.form?.querySelector<HTMLElement>('[data-input-error]');
  if (error) { error.hidden = !tooLong; error.textContent = tooLong ? message : ''; }
  if (tooLong) input?.reportValidity();
  return !tooLong;
}
function relatedButton(article: PublicArticle) {
  return article.related?.length ? `<button type="button" class="card-related-trigger" data-related-open="${escape(article.id)}" aria-label="${escape(article.title)} 관련 글 보기" aria-haspopup="dialog" aria-expanded="false" aria-controls="related-dialog">···</button>` : '';
}
function card(article: PublicArticle, view: 'list' | 'grid', excerpt = ''): string {
  const grid = view === 'grid';
  const href = escape(article.path);
  const heading = `<header class="post-card-heading"><h2 class="${grid ? 'archive-card-title' : 'post-title'}"><a href="${href}" lang="${article.language}">${escape(article.title)}</a></h2>${relatedButton(article)}</header>`;
  if (grid) {
    const media = article.hero ? `<a class="archive-card-media ${article.hero.kind === 'photo' ? 'is-photo' : ''}" href="${href}" tabindex="-1" aria-hidden="true"><img src="${escape(article.hero.src)}" alt="" loading="lazy" decoding="async"></a>` : '';
    const description = article.hero ? '' : `<p class="archive-card-description" lang="${article.language}">${escape(article.description)}</p>`;
    return `<article class="archive-card ${article.hero ? '' : 'archive-card--text'}">${media}${heading}${description}</article>`;
  }
  const description = excerpt ? textOnlyExcerpt(excerpt) : escape(article.description);
  const chips = article.conceptRefs.slice(0, 3).map(ref => `<a class="chip" href="/topics/${escape(ref.id)}/">${escape(concepts.get(ref.id)?.label || ref.id)}</a>`).join('');
  return `<article class="post-row">${heading}<p class="post-description${excerpt ? ' search-result-excerpt' : ''}" lang="${article.language}">${description}</p><div class="topic-chips">${chips}</div></article>`;
}

// Theme preferences remain independent of the system's current light/dark appearance.
const themeToggle = $<HTMLButtonElement>('[data-theme-toggle]');
const themeQuery = matchMedia('(prefers-color-scheme: dark)');
let themePreference = 'system';
try { themePreference = localStorage.getItem('blog-theme') || 'system'; } catch { /* Storage may be disabled. */ }
if (!['system', 'light', 'dark'].includes(themePreference)) themePreference = 'system';
function applyTheme(value: string, persist = false) {
  themePreference = ['system', 'light', 'dark'].includes(value) ? value : 'system';
  const resolved = themePreference === 'system' ? (themeQuery.matches ? 'dark' : 'light') : themePreference;
  document.documentElement.dataset.theme = resolved;
  document.body.dataset.theme = resolved;
  if (themeToggle) {
    const label = resolved === 'dark' ? '라이트 모드로 전환' : '다크 모드로 전환';
    themeToggle.setAttribute('aria-label', label);
    themeToggle.title = label;
  }
  if (persist) try { localStorage.setItem('blog-theme', themePreference); } catch { /* Preference still applies for this page. */ }
}
applyTheme(themePreference);
themeToggle?.addEventListener('click', () => {
  applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark', true);
});
themeQuery.addEventListener('change', () => { if (themePreference === 'system') applyTheme('system'); });
const mobileMenu = $('#mobile-menu');
const mobileToggle = $<HTMLButtonElement>('[data-menu]');
function closeMobileMenu() { if (mobileMenu) mobileMenu.hidden = true; mobileToggle?.setAttribute('aria-expanded', 'false'); }
mobileToggle?.addEventListener('click', () => { if (mobileMenu) { mobileMenu.hidden = !mobileMenu.hidden; mobileToggle.setAttribute('aria-expanded', String(!mobileMenu.hidden)); } });
window.addEventListener('resize', () => { if (innerWidth > 767) closeMobileMenu(); });

// Quick search: a missing index is an error with recovery, never an empty result set.
const searchDialog = $<HTMLDialogElement>('#search-dialog');
const searchInput = $<HTMLInputElement>('[data-search-input]');
const searchResults = $('[data-search-results]');
const searchStatus = $('[data-search-status]');
let searchFocus: HTMLElement | null = null;
let quickVersion = 0, quickTimer: ReturnType<typeof setTimeout>;
let composing = false;
function lockScroll() { document.documentElement.style.overflow = 'hidden'; }
function unlockScroll() { if (!$$<HTMLDialogElement>('dialog').some(dialog => dialog.open && dialog.matches(':modal'))) document.documentElement.style.overflow = ''; }
function closeSearch(restore = true) { quickVersion++; searchDialog?.close(); unlockScroll(); if (restore && searchFocus?.isConnected) searchFocus.focus(); }
function openSearch(trigger?: HTMLElement) {
  if (!searchDialog || !searchInput) return;
  searchFocus = trigger || (document.activeElement instanceof HTMLElement ? document.activeElement : null);
  closeRelated(false); closeMobileMenu();
  if (!searchDialog.open) searchDialog.showModal();
  lockScroll(); searchInput.focus(); void quickSearch();
}
async function quickSearch() {
  if (!searchResults || !searchInput) return;
  const version = ++quickVersion;
  const query = searchInput.value.trim();
  if (!validQuery(searchInput, query)) { if (searchStatus) searchStatus.textContent = '검색어는 200자 이내로 입력해 주세요.'; return; }
  if (!query) {
    if (searchStatus) searchStatus.textContent = '';
    searchResults.innerHTML = '<div class="search-empty"><p>제목이나 기술 개념을 입력하세요.</p><div class="search-suggestions">' + ['QLoRA', 'Reranking', 'ThinkingCap'].map(term => `<button type="button" data-search-suggestion="${term}">${term} ↗</button>`).join('') + '</div></div>';
    return;
  }
  searchResults.setAttribute('aria-busy', 'true');
  if (searchStatus) searchStatus.textContent = '글을 찾고 있어요.';
  try {
    const hits = await searchArticles(query);
    if (version !== quickVersion || !searchDialog?.open) return;
    if (searchStatus) searchStatus.textContent = `${hits.length}개의 글을 찾았습니다.`;
    searchResults.innerHTML = hits.length ? hits.slice(0, 6).map(({ article, excerpt }) => `<a class="search-result" href="${escape(article.path)}"><div class="search-result-title" lang="${article.language}">${escape(article.title)} <span aria-hidden="true">↗</span></div><p class="search-result-excerpt" lang="${article.language}">${textOnlyExcerpt(excerpt) || escape(article.description)}</p></a>`).join('') + `<a class="all-search-results" href="/search/?q=${encodeURIComponent(query)}">전체 결과 보기 ↗</a>` : '<div class="empty-state"><p>조건에 맞는 글이 없어요.</p><a href="/topics/">주제로 찾아보기 ↗</a></div>';
  } catch {
    if (version !== quickVersion || !searchDialog?.open) return;
    if (searchStatus) searchStatus.textContent = '검색을 불러오지 못했어요.';
    searchResults.innerHTML = '<div class="empty-state error"><p>잠시 후 다시 시도하거나 글 목록에서 찾아보세요.</p><button type="button" data-search-retry>다시 시도</button> <a href="/articles/">전체 글 보기 ↗</a></div>';
  } finally { if (version === quickVersion) searchResults.removeAttribute('aria-busy'); }
}
searchInput?.addEventListener('input', event => { searchInput.setCustomValidity(''); clearTimeout(quickTimer); if (!(event as InputEvent).isComposing && !composing) quickTimer = setTimeout(() => void quickSearch(), 180); });
searchInput?.addEventListener('compositionstart', () => { composing = true; clearTimeout(quickTimer); });
searchInput?.addEventListener('compositionend', () => { composing = false; quickTimer = setTimeout(() => void quickSearch(), 180); });
$('[data-search-form]')?.addEventListener('submit', event => { event.preventDefault(); if (composing || !searchInput) return; const query = searchInput.value.trim(); if (query && validQuery(searchInput, query)) { closeSearch(false); location.href = '/search/?q=' + encodeURIComponent(query); } });
searchDialog?.addEventListener('cancel', event => { event.preventDefault(); closeSearch(); });
searchDialog?.addEventListener('click', event => { if (event.target === searchDialog) { const rect = searchDialog.getBoundingClientRect(); const e = event as MouseEvent; if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) closeSearch(); } });

// Listing controls keep their committed state in the URL. Editing the search field is a draft.
const listing = $('[data-article-list], [data-archive]');
const listForm = $<HTMLFormElement>('[data-list-form], [data-archive-form]');
const listResults = $('[data-list-results], [data-archive-results]');
const listCount = $('[data-list-count], [data-result-count]');
const listNotice = $('[data-list-notice], [data-archive-status]');
const listPagination = $('[data-list-pagination], [data-archive-pagination]');
const fullSearch = $('[data-search-page]');
const fullForm = $<HTMLFormElement>('[data-full-search-form]');
const fullResults = $('[data-full-search-results]');
const fullStatus = $('[data-full-search-status]');
const fullPagination = $('[data-full-search-pagination]');
let listVersion = 0, fullVersion = 0;
const composingForms = new WeakSet<HTMLFormElement>();
function stateFromUrl(_pageSize: number, initialPage = 1): ListState {
  const params = currentUrl().searchParams;
  const rawTopics = [...new Set(params.getAll('topic').filter(Boolean))], rawTypes = [...new Set(params.getAll('type').filter(Boolean))];
  const topics = rawTopics.filter(id => concepts.has(id)), types = rawTypes.filter(type => Object.hasOwn(labels, type));
  const query = (params.get('q') || '').trim();
  const rawRole = params.get('role') || 'all';
  const pageString = params.get('page');
  const parsedPage = pageString ? Number(pageString) : initialPage;
  const defaultView = types.length === 1 && types[0] === 'research' ? 'grid' : 'list';
  return { q: query, topics, types, sort: params.get('sort') === 'latest' || !query ? 'latest' : 'relevance', page: Number.isInteger(parsedPage) && parsedPage > 0 ? parsedPage : 1, view: params.get('view') === 'grid' || params.get('view') === 'list' ? params.get('view') as 'grid' | 'list' : defaultView, role: ['about', 'applies'].includes(rawRole) ? rawRole as 'about' | 'applies' : 'all', notice: topics.length !== rawTopics.length || types.length !== rawTypes.length || (params.has('view') && !['grid', 'list'].includes(params.get('view')!)) || !['all', 'about', 'applies'].includes(rawRole) ? 'Some filters were unavailable and have been cleared.' : '' };
}
function stateParams(state: ListState, preserveView = false) {
  const params = new URLSearchParams();
  if (state.q) params.set('q', state.q);
  state.topics.forEach(id => params.append('topic', id)); state.types.forEach(type => params.append('type', type));
  if (state.q && state.sort === 'latest') params.set('sort', 'latest');
  if (state.page > 1) params.set('page', String(state.page));
  if (preserveView || currentUrl().searchParams.has('view')) params.set('view', state.view);
  if (state.role !== 'all') params.set('role', state.role);
  return params;
}
function syncForm(form: HTMLFormElement | null, state: ListState) {
  if (!form) return;
  for (const input of $$<HTMLInputElement | HTMLSelectElement>('[name]', form)) {
    if (input.name === 'q') input.value = state.q;
    else if (input.name === 'sort') {
      input.value = state.sort;
      if (input instanceof HTMLSelectElement) {
        let relevance = [...input.options].find(option => option.value === 'relevance');
        if (!relevance && state.q) { relevance = new Option(ARCHIVE_SORT_LABELS.relevance, 'relevance'); input.prepend(relevance); }
        if (relevance) relevance.disabled = !state.q;
        input.value = state.sort;
      }
    } else if (input.name === 'topic' || input.name === 'type') {
      const values = input.name === 'topic' ? state.topics : state.types;
      if (input instanceof HTMLInputElement && ['checkbox', 'radio'].includes(input.type)) input.checked = values.includes(input.value);
      else if (input instanceof HTMLSelectElement && input.multiple) [...input.options].forEach(option => option.selected = values.includes(option.value));
      else input.value = values[0] || '';
    } else if (input.name === 'view') input.value = state.view;
  }
  $$<HTMLElement>('[data-view]').filter(element => element.matches('button,a')).forEach(element => { const on = element.dataset.view === state.view; element.classList.toggle('active', on); element.setAttribute('aria-pressed', String(on)); });
  const selectedCount = $('[data-selected-filter-count]', form); if (selectedCount) selectedCount.textContent = state.topics.length + state.types.length ? String(state.topics.length + state.types.length) : '';
  $$<HTMLElement>('[data-role]').forEach(element => { const on = element.dataset.role === state.role; element.classList.toggle('active', on); element.setAttribute('aria-pressed', String(on)); });
}
function inputState(form: HTMLFormElement, previous: ListState): ListState {
  const values = new FormData(form);
  const query = String(values.get('q') || '').trim();
  const preserveSort = currentUrl().searchParams.has('sort') || query === previous.q;
  return { ...previous, q: query, topics: values.getAll('topic').map(String).filter(Boolean), types: values.getAll('type').map(String).filter(Boolean), sort: query ? preserveSort && values.get('sort') === 'latest' ? 'latest' : 'relevance' : 'latest', page: 1, notice: '' };
}
function descendantIds(id: string): Set<string> {
  const ids = new Set([id]); let changed = true;
  while (changed) { changed = false; for (const concept of data.concepts) if (!ids.has(concept.id) && concept.broaderIds.some(parent => ids.has(parent))) { ids.add(concept.id); changed = true; } }
  return ids;
}
function filterHits(hits: SearchHit[], state: ListState, topicId?: string) {
  const topicIds = topicId ? descendantIds(topicId) : undefined;
  const selectedIds = new Set(state.topics.flatMap(id => [...descendantIds(id)]));
  return hits.filter(({ article }) => (!state.topics.length || article.conceptRefs.some(ref => selectedIds.has(ref.id))) && (!state.types.length || state.types.includes(article.contentType)) && (!topicId || article.conceptRefs.some(ref => topicIds!.has(ref.id) && (state.role === 'all' || ref.role === state.role))));
}
function listUrl(state: ListState, basePath: string, allowStatic = true, preserveView = false) {
  const params = stateParams(state, preserveView);
  const noFilters = !state.q && !state.topics.length && !state.types.length && state.role === 'all' && !params.has('view');
  if (allowStatic && noFilters && basePath === '/articles/') { params.delete('page'); return state.page > 1 ? `/articles/page/${state.page}/` : '/articles/'; }
  return basePath + (params.size ? '?' + params : '');
}
function rememberScroll() { history.replaceState({ ...history.state, scrollY: scrollY }, '', location.href); }
const initialRobots = $<HTMLMetaElement>('meta[name=robots]')?.content || 'index,follow';
const canonicalOrigin = (() => { try { return new URL($<HTMLLinkElement>('link[rel=canonical]')?.href || location.origin).origin; } catch { return location.origin; } })();
function updateSeo(url: string) {
  const target = new URL(url, location.origin);
  const canonicalPath = fullSearch ? '/search/' : listing ? (target.search ? listing.dataset.basePath || '/articles/' : target.pathname) : topicForm ? '/topics/' : target.pathname;
  const canonical = $<HTMLLinkElement>('link[rel=canonical]');
  if (canonical) canonical.href = new URL(canonicalPath, canonicalOrigin).href;
  const ogUrl = $<HTMLMetaElement>('meta[property="og:url"]'); if (ogUrl && canonical) ogUrl.content = canonical.href;
  const robots = $<HTMLMetaElement>('meta[name=robots]');
  if (robots) robots.content = initialRobots.includes('noindex') ? initialRobots : target.search ? 'noindex,follow' : initialRobots;
}
function commitUrl(url: string, push: boolean) { updateSeo(url); if (url === location.pathname + location.search) return; if (push) { rememberScroll(); history.pushState({ scrollY }, '', url); } else history.replaceState({ ...history.state }, '', url); }
function paginationHtml(state: ListState, pageCount: number, basePath: string, allowStatic = true) {
  if (pageCount <= 1) return '';
  const visible = [...new Set([1, state.page - 1, state.page, state.page + 1, pageCount].filter(page => page >= 1 && page <= pageCount))].sort((a, b) => a - b);
  let previous = 0;
  const previousLink = state.page > 1 ? `<a href="${escape(listUrl({ ...state, page: state.page - 1 }, basePath, allowStatic))}" data-result-page="${state.page - 1}" rel="prev" aria-label="Previous page">← Previous</a>` : '';
  const nextLink = state.page < pageCount ? `<a href="${escape(listUrl({ ...state, page: state.page + 1 }, basePath, allowStatic))}" data-result-page="${state.page + 1}" rel="next" aria-label="Next page">Next →</a>` : '';
  return '<nav class="pagination" aria-label="Article pages" lang="en">' + previousLink + visible.map(page => { const gap = page > previous + 1 ? '<span aria-hidden="true">…</span>' : ''; previous = page; return gap + `<a href="${escape(listUrl({ ...state, page }, basePath, allowStatic))}" data-result-page="${page}" ${page === state.page ? 'aria-current="page"' : ''} aria-label="Page ${page}">${page}</a>`; }).join('') + nextLink + '</nav>';
}
function activeFilters(state: ListState, root: HTMLElement) {
  const container = $('[data-active-filters]', root); if (!container) return;
  container.innerHTML = (state.q ? `<button type="button" class="filter-chip" data-remove-filter="q">Search “${escape(state.q)}” <span aria-hidden="true">×</span></button>` : '') + state.topics.map(id => `<button type="button" class="filter-chip" data-remove-filter="topic" data-filter-value="${escape(id)}">${escape(concepts.get(id)?.label || id)} <span aria-hidden="true">×</span></button>`).join('') + state.types.map(type => `<button type="button" class="filter-chip" data-remove-filter="type" data-filter-value="${escape(type)}">${labels[type]} <span aria-hidden="true">×</span></button>`).join('');
}
async function renderListing(push = false, next?: ListState) {
  if (!listing || !listResults || !listForm) return;
  const version = ++listVersion;
  const pageSize = Number(listing.dataset.pageSize) || 12;
  const basePath = listing.dataset.basePath || (listing.dataset.topicId ? `/topics/${listing.dataset.topicId}/` : '/articles/');
  const initialPage = Number(location.pathname.match(/\/page\/(\d+)\/?$/)?.[1] || 1);
  const state = next || stateFromUrl(pageSize, initialPage);
  const input = $<HTMLInputElement>('[name=q]', listForm);
  if (!validQuery(input, state.q, QUERY_LENGTH_MESSAGE)) { if (listNotice) listNotice.textContent = QUERY_LENGTH_MESSAGE; return; }
  syncForm(listForm, state); activeFilters(state, listing);
  commitUrl(listUrl(state, basePath), push);
  if (listNotice) listNotice.textContent = state.notice;
  listResults.setAttribute('aria-busy', 'true');
  if (state.q && listNotice) listNotice.textContent = 'Searching articles…';
  try {
    let hits = filterHits(await searchArticles(state.q), state, listing.dataset.topicId);
    if (version !== listVersion) return;
    if (state.sort === 'latest') hits.sort((a, b) => b.article.publishedAt.localeCompare(a.article.publishedAt) || a.article.id.localeCompare(b.article.id));
    const pageCount = Math.max(1, Math.ceil(hits.length / pageSize));
    state.page = Math.min(state.page, pageCount);
    commitUrl(listUrl(state, basePath), false);
    listResults.classList.toggle('archive-grid', state.view === 'grid'); listResults.classList.toggle('post-list', state.view === 'list');
    const roleIds = listing.dataset.topicId ? descendantIds(listing.dataset.topicId) : undefined;
    listResults.innerHTML = hits.length ? hits.slice((state.page - 1) * pageSize, state.page * pageSize).map(hit => {
      const rendered = card(hit.article, state.view, state.q ? hit.excerpt : '');
      if (!roleIds) return rendered;
      const roles = new Set(hit.article.conceptRefs.filter(ref => roleIds.has(ref.id)).map(ref => ref.role));
      const roleLabels = (['about', 'applies'] as const).filter(role => roles.has(role)).map(role => `<span>${CONCEPT_ROLE_LABELS[role]}</span>`).join('');
      return `<div class="topic-role-row"><div class="topic-role-labels" lang="en">${roleLabels}</div>${rendered}</div>`;
    }).join('') : '<div class="empty-state" lang="en"><h2>No matching articles</h2><p>Try another search or clear the filters.</p><button type="button" data-filter-reset>Clear filters</button> <a href="/topics/">Browse by topic ↗</a></div>';
    if (listCount) listCount.textContent = articleCountLabel(hits.length);
    if (listNotice) listNotice.textContent = state.notice;
    if (listPagination) listPagination.innerHTML = paginationHtml(state, pageCount, basePath);
    listing.dataset.view = state.view;
    syncForm(listForm, state);
  } catch {
    if (version !== listVersion) return;
    if (listNotice) listNotice.innerHTML = 'Search is unavailable. <button type="button" data-list-retry>Try again</button> <a href="/articles/">View all articles</a>';
  } finally { if (version === listVersion) listResults.removeAttribute('aria-busy'); }
}
async function renderFullSearch(push = false, next?: ListState) {
  if (!fullSearch || !fullForm || !fullResults) return;
  const version = ++fullVersion;
  const state = next || stateFromUrl(10);
  const input = $<HTMLInputElement>('[name=q]', fullForm);
  syncForm(fullForm, state); activeFilters(state, fullSearch);
  if (!validQuery(input, state.q, QUERY_LENGTH_MESSAGE)) { if (fullStatus) fullStatus.textContent = QUERY_LENGTH_MESSAGE; return; }
  if (!state.q) { commitUrl(listUrl(state, '/search/', false), push); fullResults.innerHTML = '<div class="empty-state" lang="en"><h2>Search the archive</h2><p>Enter a title or topic to find articles.</p><a href="/topics/">Browse by topic ↗</a></div>'; if (fullStatus) fullStatus.textContent = ''; if (fullPagination) fullPagination.innerHTML = ''; return; }
  commitUrl(listUrl(state, '/search/', false), push);
  fullResults.setAttribute('aria-busy', 'true');
  if (fullStatus) fullStatus.textContent = 'Searching articles…';
  try {
    let hits = filterHits(await searchArticles(state.q), state);
    if (version !== fullVersion) return;
    if (state.sort === 'latest') hits.sort((a, b) => b.article.publishedAt.localeCompare(a.article.publishedAt));
    const pageCount = Math.max(1, Math.ceil(hits.length / 10)); state.page = Math.min(state.page, pageCount);
    commitUrl(listUrl(state, '/search/', false), false);
    fullResults.innerHTML = hits.length ? hits.slice((state.page - 1) * 10, state.page * 10).map(hit => card(hit.article, 'list', hit.excerpt)).join('') : '<div class="empty-state" lang="en"><h2>No matching articles</h2><p>Try another search or clear the filters.</p><a href="/topics/">Browse by topic ↗</a></div>';
    if (fullStatus) fullStatus.textContent = `${articleCountLabel(hits.length)} found.${state.notice ? ' ' + state.notice : ''}`;
    if (fullPagination) fullPagination.innerHTML = paginationHtml(state, pageCount, '/search/', false);
    syncForm(fullForm, state);
  } catch {
    if (version !== fullVersion) return;
    if (fullStatus) fullStatus.textContent = 'Search is unavailable.';
    fullResults.innerHTML = '<div class="empty-state error" lang="en"><p>Try again or browse the full archive.</p><button type="button" data-full-search-retry>Try again</button> <a href="/articles/">View all articles ↗</a></div>';
  } finally { if (version === fullVersion) fullResults.removeAttribute('aria-busy'); }
}
for (const form of [listForm, fullForm]) if (form) {
  form.addEventListener('compositionstart', () => composingForms.add(form));
  form.addEventListener('compositionend', () => composingForms.delete(form));
  form.addEventListener('submit', event => { event.preventDefault(); if (composingForms.has(form)) return; const state = inputState(form, stateFromUrl(form === fullForm ? 10 : 12)); if (form === fullForm) void renderFullSearch(true, state); else void renderListing(true, state); });
  form.addEventListener('change', event => { const target = event.target as HTMLInputElement | HTMLSelectElement; if (target.name === 'q') return; const state = inputState(form, stateFromUrl(form === fullForm ? 10 : 12)); // A filter change preserves the committed query, not an unfinished draft.
    state.q = currentUrl().searchParams.get('q') || ''; if (!state.q) state.sort = 'latest';
    if (form === fullForm) void renderFullSearch(true, state); else void renderListing(true, state);
  });
  form.addEventListener('input', event => { const target = event.target; if (target instanceof HTMLInputElement) target.setCustomValidity(''); });
}

// Topic name/alias search also commits on submission and remains shareable.
const topicForm = $<HTMLFormElement>('[data-topic-form], [data-topic-search]');
function filterTopics(push = false) {
  if (!topicForm) return;
  const params = new URLSearchParams(location.search), q = (params.get('q') || '').trim(), area = params.get('area') || '';
  const input = $<HTMLInputElement>('[name=q]', topicForm); if (input) input.value = q;
  if (!validQuery(input, q, QUERY_LENGTH_MESSAGE)) return;
  const areaInput = $<HTMLSelectElement>('[name=area]', topicForm); if (areaInput) areaInput.value = area;
  let count = 0;
  $$<HTMLElement>('[data-topic-tile]').forEach(tile => {
    const concept = concepts.get(tile.dataset.topicId || '');
    const text = concept ? [concept.label, ...concept.aliases].join(' ') : tile.dataset.search || tile.textContent || '';
    const visible = text.toLocaleLowerCase().includes(q.toLocaleLowerCase()) && (!area || (concept?.area || tile.dataset.area) === area);
    tile.hidden = !visible; if (visible) count++;
  });
  $$<HTMLElement>('[data-topic-area]').forEach(section => { section.hidden = !$$<HTMLElement>('[data-topic-tile]', section).some(tile => !tile.hidden); });
  const countElement = $('[data-topic-results-count], [data-topic-count]'); if (countElement) countElement.textContent = topicCountLabel(count);
  const empty = $('[data-topic-empty]'); if (empty) empty.hidden = count !== 0;
  const canonical = new URLSearchParams(); if (q) canonical.set('q', q); if (area) canonical.set('area', area);
  commitUrl('/topics/' + (canonical.size ? '?' + canonical : ''), push);
}
topicForm?.addEventListener('submit', event => { event.preventDefault(); if (composingForms.has(topicForm)) return; const fd = new FormData(topicForm); const params = new URLSearchParams(); const q = String(fd.get('q') || '').trim(); if (!validQuery($<HTMLInputElement>('[name=q]', topicForm), q, QUERY_LENGTH_MESSAGE)) return; if (q) params.set('q', q); if (fd.get('area')) params.set('area', String(fd.get('area'))); rememberScroll(); history.pushState({ scrollY }, '', '/topics/' + (params.size ? '?' + params : '')); filterTopics(); });
topicForm?.addEventListener('compositionstart', () => composingForms.add(topicForm));
topicForm?.addEventListener('compositionend', () => composingForms.delete(topicForm));
topicForm?.addEventListener('input', event => { if (event.target instanceof HTMLInputElement) event.target.setCustomValidity(''); });
topicForm?.addEventListener('change', event => { if ((event.target as HTMLInputElement).name !== 'area') return; const params = new URLSearchParams(location.search), area = String(new FormData(topicForm).get('area') || ''); if (area) params.set('area', area); else params.delete('area'); rememberScroll(); history.pushState({ scrollY }, '', '/topics/' + (params.size ? '?' + params : '')); filterTopics(); });

const recoveryForm = $<HTMLFormElement>('[data-recovery-search]');
recoveryForm?.addEventListener('compositionstart', () => composingForms.add(recoveryForm));
recoveryForm?.addEventListener('compositionend', () => composingForms.delete(recoveryForm));
recoveryForm?.addEventListener('submit', event => {
  const input = $<HTMLInputElement>('[name=q]', recoveryForm);
  if (composingForms.has(recoveryForm) || !validQuery(input, input?.value.trim() || '')) event.preventDefault();
});
document.addEventListener('input', event => {
  if (!(event.target instanceof HTMLInputElement) || event.target.name !== 'q') return;
  event.target.setCustomValidity('');
  const error = event.target.form?.querySelector<HTMLElement>('[data-input-error]');
  if (error) { error.hidden = true; error.textContent = ''; }
});

const relatedDialog = $<HTMLDialogElement>('#related-dialog');
let relatedFocus: HTMLElement | null = null;
function closeRelated(restore = true) {
  if (!relatedDialog?.open) return;
  relatedDialog.close();
  if (relatedFocus?.isConnected) { relatedFocus.setAttribute('aria-expanded', 'false'); const source = articles.get(relatedFocus.dataset.relatedOpen || ''); relatedFocus.setAttribute('aria-label', `${source?.title || ''} 관련 글 보기`); if (restore) relatedFocus.focus({ preventScroll: true }); }
  const content = $('[data-related-content], .related-dialog-content', relatedDialog); if (content) content.replaceChildren();
}
function openRelated(id: string, trigger: HTMLElement) {
  if (!relatedDialog) return;
  if (relatedDialog.open && relatedFocus === trigger) { closeRelated(); return; }
  closeRelated(false);
  const source = articles.get(id), content = $('[data-related-content], .related-dialog-content', relatedDialog); if (!source?.related.length || !content) return;
  const graph = renderRelatedGraph(source, articles);
  if (!graph) return;
  relatedFocus = trigger;
  content.innerHTML = `<div class="graph-wrap">${graph}</div>`;
  trigger.setAttribute('aria-expanded', 'true'); trigger.setAttribute('aria-label', `${source.title} 관련 글 닫기`);
  relatedDialog.show();
  const anchor = trigger.getBoundingClientRect(), box = relatedDialog.getBoundingClientRect();
  relatedDialog.style.left = Math.max(16, Math.min(anchor.left, innerWidth - box.width - 16)) + 'px';
  relatedDialog.style.top = Math.max(12, Math.min(anchor.bottom + 8, innerHeight - measureRelatedDialogHeight(relatedDialog) - 12)) + 'px';
  $<HTMLButtonElement>('[data-close-related]', relatedDialog)?.focus({ preventScroll: true });
}
relatedDialog?.addEventListener('cancel', event => { event.preventDefault(); closeRelated(); });
window.addEventListener('resize', () => closeRelated(false));

// Reading tools enhance the original code and image elements without hiding the source.
const body = $('[data-article-body], .article-body');
body?.querySelectorAll('pre').forEach(pre => {
  if (!pre.querySelector('code')) return;
  const wrapper = document.createElement('div'); wrapper.className = 'code-block';
  pre.before(wrapper); wrapper.append(pre);
  const button = document.createElement('button'); button.type = 'button'; button.className = 'code-copy'; button.dataset.codeCopy = ''; button.textContent = '코드 복사'; button.setAttribute('aria-label', '코드 전체 복사');
  const status = document.createElement('span'); status.className = 'copy-status'; status.setAttribute('role', 'status');
  wrapper.prepend(button); wrapper.append(status);
});
body?.querySelectorAll<HTMLImageElement>('img').forEach(img => {
  if (img.closest('button, pre') || img.dataset.noZoom !== undefined) return;
  const source = img.currentSrc || img.src;
  const alt = img.alt || '본문 이미지';
  const caption = img.closest('figure')?.querySelector('figcaption')?.textContent || '';
  const anchor = img.closest<HTMLAnchorElement>('a');
  let imageLink = false;
  if (anchor) {
    try {
      const target = new URL(anchor.href), image = new URL(source);
      imageLink = (target.origin === image.origin && target.pathname === image.pathname) || /\.(?:png|jpe?g|webp|gif|svg|avif|bmp|tiff?)$/i.test(target.pathname);
    } catch { /* Keep an unrelated or invalid link's native behavior. */ }
  }
  if (anchor && imageLink) {
    // Retain href so no-JS, modified clicks, and the context menu can open the original.
    anchor.dataset.imageOpen = ''; anchor.dataset.src = anchor.href;
    anchor.dataset.alt = alt; anchor.dataset.caption = caption;
    anchor.setAttribute('aria-haspopup', 'dialog'); anchor.setAttribute('aria-label', `${alt} 확대`);
    anchor.style.cursor = 'zoom-in';
    return;
  }
  const button = document.createElement('button'); button.type = 'button'; button.className = 'image-zoom-trigger'; button.dataset.imageZoom = ''; button.dataset.src = source; button.dataset.alt = alt; button.dataset.caption = caption; button.textContent = '이미지 확대 ↗'; button.setAttribute('aria-label', `${alt} 확대`);
  (anchor || img).after(button);
});
body?.querySelectorAll<HTMLHeadingElement>('h2[id], h3[id], h4[id]').forEach(heading => { if (heading.querySelector('.heading-anchor')) return; const link = document.createElement('a'); link.href = '#' + encodeURIComponent(heading.id); link.className = 'heading-anchor'; link.setAttribute('aria-label', `${heading.textContent || ''} 절 바로가기 링크`); link.textContent = '#'; heading.append(link); });
$$<HTMLDetailsElement>('[data-filter-panel]').forEach(panel => { panel.open = innerWidth > 767; });
const imageDialog = $<HTMLDialogElement>('#image-dialog');
const zoomImage = $<HTMLImageElement>('[data-zoom-image], [data-image-full]');
let imageFocus: HTMLElement | null = null;
function setImageMode(actual: boolean) {
  if (!imageDialog || !zoomImage) return;
  imageDialog.dataset.zoomMode = actual ? 'actual' : 'fit';
  zoomImage.style.width = actual && zoomImage.naturalWidth ? zoomImage.naturalWidth + 'px' : '';
  zoomImage.style.maxWidth = actual ? 'none' : '';
  zoomImage.style.maxHeight = actual ? 'none' : '';
  $$<HTMLButtonElement>('[data-image-fit], [data-image-actual]', imageDialog).forEach(button => button.setAttribute('aria-pressed', String(button.hasAttribute(actual ? 'data-image-actual' : 'data-image-fit'))));
}
function closeImage() { imageDialog?.close(); unlockScroll(); imageFocus?.focus(); }
function openImage(trigger: HTMLElement) {
  if (!imageDialog || !zoomImage) return;
  imageFocus = trigger; const linkedImage = trigger.querySelector('img'); zoomImage.src = trigger.dataset.src || (trigger instanceof HTMLAnchorElement ? trigger.href : linkedImage?.src) || ''; zoomImage.alt = trigger.dataset.alt || linkedImage?.alt || '본문 이미지';
  const caption = $('[data-image-caption]', imageDialog); if (caption) caption.textContent = trigger.dataset.caption || trigger.dataset.alt || '';
  const original = $<HTMLAnchorElement>('a[data-image-original]', imageDialog); if (original) { original.href = zoomImage.src; original.target = '_blank'; original.rel = 'noopener'; }
  imageDialog.showModal(); lockScroll(); setImageMode(false); $<HTMLButtonElement>('[data-close-image]', imageDialog)?.focus();
}
imageDialog?.addEventListener('cancel', event => { event.preventDefault(); closeImage(); });
imageDialog?.addEventListener('click', event => { if (event.target === imageDialog) { const r = imageDialog.getBoundingClientRect(), e = event as MouseEvent; if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) closeImage(); } });
const progress = $('#reading-progress');
let readingFrame = 0;
function updateReading() {
  readingFrame = 0; if (!body) return;
  const start = body.getBoundingClientRect().top + scrollY - 110, end = start + body.scrollHeight - innerHeight + 160;
  const percentage = Math.max(0, Math.min(1, (scrollY - start) / Math.max(1, end - start)));
  if (progress) progress.style.transform = `scaleX(${percentage})`;
  const headings = $$<HTMLElement>('h2[id],h3[id]', body); let active = headings[0]?.id;
  headings.forEach(heading => { if (heading.getBoundingClientRect().top <= 155) active = heading.id; });
  $$<HTMLAnchorElement>('[data-toc] a[href^="#"], .article-toc a[href^="#"]').forEach(link => { const on = decodeURIComponent(link.hash.slice(1)) === active; link.classList.toggle('active', on); if (on) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current'); });
}
if (body) { addEventListener('scroll', () => { if (!readingFrame) readingFrame = requestAnimationFrame(updateReading); }, { passive: true }); addEventListener('resize', updateReading); updateReading(); }

// Delegation also covers cards and result lists inserted after filtering.
document.addEventListener('click', event => {
  const target = event.target instanceof Element ? event.target : null; if (!target) return;
  const relatedTrigger = target.closest<HTMLElement>('[data-related-open]'); if (relatedTrigger) { openRelated(relatedTrigger.dataset.relatedOpen!, relatedTrigger); return; }
  if (target.closest('[data-close-related]')) { closeRelated(); return; }
  if (relatedDialog?.open && !target.closest('#related-dialog')) closeRelated(false);
  const searchTrigger = target.closest<HTMLElement>('[data-open-search]'); if (searchTrigger) { openSearch(searchTrigger); return; }
  if (target.closest('[data-close-search]')) { closeSearch(); return; }
  if (target.closest('[data-search-retry]')) { retrySearch(); void quickSearch(); return; }
  const suggestion = target.closest<HTMLElement>('[data-search-suggestion]'); if (suggestion && searchInput) { searchInput.value = suggestion.dataset.searchSuggestion || ''; void quickSearch(); searchInput.focus(); return; }
  if (target.closest('[data-list-retry]')) { retrySearch(); void renderListing(); return; }
  if (target.closest('[data-full-search-retry]')) { retrySearch(); void renderFullSearch(); return; }
  const zoom = target.closest<HTMLElement>('[data-image-zoom], [data-image-open]'); if (zoom) { const mouse = event as MouseEvent; if (zoom instanceof HTMLAnchorElement && (mouse.metaKey || mouse.ctrlKey || mouse.shiftKey || mouse.altKey || mouse.button !== 0)) return; event.preventDefault(); openImage(zoom); return; }
  if (target.closest('[data-close-image]')) { closeImage(); return; }
  if (target.closest('button[data-image-fit]')) { setImageMode(false); return; }
  if (target.closest('button[data-image-actual]')) { setImageMode(true); return; }
  const copy = target.closest<HTMLButtonElement>('[data-code-copy]'); if (copy) { const wrapper = copy.closest('.code-block'), content = wrapper?.querySelector('code')?.textContent || '', status = wrapper?.querySelector('.copy-status'); void (async () => { try { await navigator.clipboard.writeText(content); if (status) status.textContent = '복사했어요.'; } catch { if (status) status.textContent = '복사할 수 없어요. 코드를 직접 선택해 주세요.'; } })(); return; }
  const view = target.closest<HTMLElement>('button[data-view],a[data-view]'); if (view && listing) { event.preventDefault(); const state = stateFromUrl(12, Number(location.pathname.match(/\/page\/(\d+)/)?.[1]) || 1); state.view = view.dataset.view === 'grid' ? 'grid' : 'list'; const params = stateParams(state, true); rememberScroll(); history.pushState({ scrollY }, '', (listing.dataset.basePath || '/articles/') + '?' + params); void renderListing(); return; }
  const role = target.closest<HTMLElement>('[data-role]'); if (role && listing) { event.preventDefault(); const state = stateFromUrl(12); state.role = ['about','applies'].includes(role.dataset.role!) ? role.dataset.role as 'about' | 'applies' : 'all'; state.page = 1; void renderListing(true, state); return; }
  const reset = target.closest('[data-filter-reset]'); if (reset) { event.preventDefault(); const state = stateFromUrl(12); const next: ListState = { ...state, q: fullSearch ? state.q : '', topics: [], types: [], sort: fullSearch && state.q ? 'relevance' : 'latest', page: 1, role: 'all', notice: '' }; const base = fullSearch ? '/search/' : listing?.dataset.basePath || '/articles/'; commitUrl(listUrl(next, base, false, true), true); if (fullSearch) void renderFullSearch(); else void renderListing(); return; }
  const remove = target.closest<HTMLElement>('[data-remove-filter]'); if (remove) { const state = stateFromUrl(12); const kind = remove.dataset.removeFilter; if (kind === 'q') state.q = ''; else if (kind === 'topic') state.topics = state.topics.filter(id => id !== remove.dataset.filterValue); else state.types = state.types.filter(type => type !== remove.dataset.filterValue); state.page = 1; if (fullSearch) void renderFullSearch(true, state); else void renderListing(true, state); return; }
  const page = target.closest<HTMLElement>('[data-result-page]'); if (page && (listing || fullSearch)) { if ((event as MouseEvent).metaKey || (event as MouseEvent).ctrlKey || (event as MouseEvent).shiftKey || (event as MouseEvent).altKey) return; event.preventDefault(); const state = stateFromUrl(fullSearch ? 10 : 12); state.page = Number(page.dataset.resultPage) || 1; const work = fullSearch ? renderFullSearch(true, state) : renderListing(true, state); void work.then(() => { const focus = fullSearch ? fullStatus : listCount; focus?.setAttribute('tabindex', '-1'); focus?.focus({ preventScroll: true }); (fullResults || listResults)?.scrollIntoView({ block: 'start', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' }); }); return; }
  const link = target.closest<HTMLAnchorElement>('a[href]'); if (link) { rememberScroll(); if (searchDialog?.open) closeSearch(false); closeMobileMenu(); }
});
document.addEventListener('keydown', event => {
  if (event.isComposing) return;
  if (event.key === 'Escape') { if (relatedDialog?.open) { event.preventDefault(); closeRelated(); return; } if (searchDialog?.open) { event.preventDefault(); closeSearch(); return; } if (imageDialog?.open) { event.preventDefault(); closeImage(); return; } if (mobileMenu && !mobileMenu.hidden) { closeMobileMenu(); mobileToggle?.focus(); } }
  const element = event.target instanceof Element ? event.target : null;
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k' && !element?.closest('input,textarea,select,[contenteditable=true]')) { event.preventDefault(); openSearch(); }
});
window.addEventListener('popstate', () => { const work = fullSearch ? renderFullSearch() : listing ? renderListing() : Promise.resolve(); filterTopics(); void work.then(() => { if (typeof history.state?.scrollY === 'number') requestAnimationFrame(() => scrollTo(0, history.state.scrollY)); }); });
window.addEventListener('pagehide', rememberScroll);
void renderListing(); void renderFullSearch(); filterTopics();

import { getArticles, getConcepts, getSeries } from './content';
import { SITE, absoluteUrl } from './site';

export const xml = (value: unknown) => String(value ?? '').replace(/[<>&"']/g, character => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&apos;' }[character]!));

export function rssDocument() {
  const articles = [...getArticles()].sort((a, b) => b.publishedAt.localeCompare(a.publishedAt));
  return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom"><channel>
<title>${xml(SITE.title)}</title><link>${xml(absoluteUrl('/'))}</link><description>${xml(SITE.description)}</description><language>ko</language>
<atom:link href="${xml(absoluteUrl('/feed.xml'))}" rel="self" type="application/rss+xml"/>
${articles.map(article => `<item><title>${xml(article.title)}</title><link>${xml(absoluteUrl(article.path))}</link><guid isPermaLink="true">${xml(absoluteUrl(article.path))}</guid><pubDate>${new Date(article.publishedAt).toUTCString()}</pubDate><description>${xml(article.description)}</description>${article.conceptRefs.map(ref => `<category>${xml(ref.id)}</category>`).join('')}</item>`).join('\n')}
</channel></rss>`;
}

export function sitemapDocument() {
  const routes: { path: string; lastmod?: string }[] = [
    ...['/', '/articles/', '/topics/', '/about/'].map(path => ({ path })),
    ...getArticles().map(article => ({ path: article.path, lastmod: (article.updatedAt || article.publishedAt).slice(0, 10) })),
    ...getConcepts().map(concept => ({ path: `/topics/${concept.id}/` })),
    ...getSeries().map(series => ({ path: `/series/${series.id}/` })),
  ];
  const seen = new Set<string>();
  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${routes.filter(route => !seen.has(route.path) && seen.add(route.path)).map(route => `<url><loc>${xml(absoluteUrl(route.path))}</loc>${route.lastmod ? `<lastmod>${xml(route.lastmod)}</lastmod>` : ''}</url>`).join('\n')}\n</urlset>`;
}

export const xmlResponse = (document: string, contentType = 'application/xml; charset=utf-8') => new Response(document, { headers: { 'Content-Type': contentType } });

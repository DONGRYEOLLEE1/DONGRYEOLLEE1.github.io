import { readFile, writeFile, mkdir, cp, rm } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const configuredOrigin = process.env.SITE_URL;
if (!configuredOrigin) throw new Error('최종 운영 주소를 SITE_URL로 지정해야 합니다. 먼저 같은 SITE_URL로 npm run build를 실행하세요.');
const origin = new URL(configuredOrigin);
if (origin.protocol !== 'https:' || origin.pathname !== '/' || origin.search || origin.hash || origin.username || origin.password || origin.hostname.toLowerCase() === 'dongryeollee1.github.io') throw new Error('SITE_URL은 새 운영 사이트의 https origin이어야 합니다. 경로·쿼리·fragment·인증 정보는 사용할 수 없습니다.');
const registry = JSON.parse(await readFile(resolve(root, 'generated/registry.json'), 'utf8'));
const feed = await readFile(resolve(root, 'dist/feed.xml'), 'utf8');
if (!feed.includes(origin.origin + '/')) throw new Error('dist의 feed.xml이 지정한 SITE_URL로 빌드되지 않았습니다. 먼저 같은 SITE_URL로 npm run build를 실행하세요.');
const output = resolve(root, 'dist-legacy');
// This tool owns only the reproducible output directory. Source and deployed Pages are untouched.
await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
await cp(resolve(root, 'public'), output, { recursive: true });
const escape = value => String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
const bridges = [
  { path: '/', target: '/', title: '동렬 · 기술 블로그' },
  { path: '/archives.html', target: '/articles/', title: '글 목록' },
  { path: '/categories.html', target: '/categories.html', title: '이전 카테고리' },
  { path: '/tags.html', target: '/tags.html', title: '이전 태그' },
  { path: '/about.html', target: '/about/', title: '소개' },
  ...registry.articles.map(article => ({ path: article.path, target: article.path, title: article.title })),
];
for (const bridge of bridges) {
  const target = new URL(bridge.target, origin);
  if (target.origin !== origin.origin || target.search || target.hash) throw new Error(`안전하지 않은 이전 대상: ${bridge.target}`);
  const pathname = decodeURIComponent(bridge.path).replace(/^\/+/, '');
  const file = resolve(output, pathname.endsWith('.html') ? pathname : pathname + (pathname.endsWith('/') || !pathname ? '' : '/') + 'index.html');
  if (!file.startsWith(output + '/')) throw new Error(`안전하지 않은 이전 경로: ${bridge.path}`);
  const href = target.href;
  const document = `<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escape(bridge.title)}</title><link rel="canonical" href="${escape(href)}"><meta name="robots" content="noindex,follow"><meta http-equiv="refresh" content="5;url=${escape(href)}"><script>location.replace(${JSON.stringify(href).replaceAll('<', '\\u003c')}+location.hash);</script><style>body{max-width:42rem;margin:12vh auto;padding:24px;background:#faf9f5;color:#141413;font:17px/1.8 system-ui,sans-serif}a{color:inherit;text-underline-offset:4px}h1{font-size:28px;line-height:1.4}</style></head><body><h1>${escape(bridge.title)}</h1><p>블로그 주소가 변경되었습니다. 같은 글의 새 주소로 이동합니다.</p><p><a href="${escape(href)}">이 글을 새 주소에서 읽기 ↗</a></p></body></html>`;
  await mkdir(dirname(file), { recursive: true });
  await writeFile(file, document);
}
await writeFile(resolve(output, '.nojekyll'), '');
for (const name of ['feed.xml', 'rss.xml', 'sitemap.xml']) await cp(resolve(root, 'dist', name), resolve(output, name));
await writeFile(resolve(output, 'robots.txt'), `User-agent: *\nAllow: /\nSitemap: ${origin.origin}/sitemap.xml\n`);
console.log(JSON.stringify({ output: 'dist-legacy', origin: origin.origin, bridges: bridges.length, deployed: false }));

import { readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { resolve, relative, extname, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { parse } from 'parse5';
import { SaxesParser } from 'saxes';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dist = resolve(root, 'dist');
const registry = JSON.parse(await readFile(resolve(root, 'generated/registry.json'), 'utf8'));
const manifest = JSON.parse(await readFile(resolve(root, 'generated/url-manifest.json'), 'utf8'));
const vercel = JSON.parse(await readFile(resolve(root, 'vercel.json'), 'utf8'));
const errors = [], warnings = [];
const htmlCache = new Map();
const outputFiles = [];
async function walk(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) await walk(path);
    else outputFiles.push(path);
  }
}
await walk(dist);
const outputLookup = new Map(outputFiles.map(path => [path.normalize('NFC'), path]));
const exists = async path => { try { return (await stat(path)).isFile(); } catch { return false; } };
function decodedPath(path) { try { return decodeURIComponent(path); } catch { return path; } }
async function outputFor(path) {
  const pathname = decodedPath(path).replace(/^\/+/, '');
  const file = resolve(dist, pathname || 'index.html');
  if (file !== dist && !file.startsWith(dist + '/')) return null;
  for (const candidate of [file, resolve(file, 'index.html'), `${file}.html`]) {
    const actual = outputLookup.get(candidate.normalize('NFC'));
    if (actual) return actual;
  }
  return null;
}
function collect(document) {
  const ids = new Set(), duplicates = [], references = [];
  let canonical, description, noindex = false, articleBody, h1Count = 0;
  const visit = (node, inBody = false, inCode = false) => {
    const attributes = Object.fromEntries((node.attrs || []).map(attribute => [attribute.name, attribute.value]));
    const body = inBody || Object.hasOwn(attributes, 'data-article-body') || (attributes.class || '').split(/\s+/).includes('article-body');
    const code = inCode || ['pre', 'code'].includes(node.tagName);
    if (attributes.id) { if (ids.has(attributes.id)) duplicates.push(attributes.id); ids.add(attributes.id); }
    if (node.tagName === 'h1') h1Count++;
    if (node.tagName === 'link' && attributes.rel === 'canonical') canonical = attributes.href;
    if (node.tagName === 'meta' && attributes.name === 'description') description = attributes.content;
    if (node.tagName === 'meta' && attributes.name === 'robots' && attributes.content?.includes('noindex')) noindex = true;
    if (body && ['script', 'style', 'iframe', 'object', 'embed'].includes(node.tagName)) errors.push({ type: 'unsafe-article-element', tag: node.tagName });
    if (body && Object.keys(attributes).some(name => /^on/i.test(name))) errors.push({ type: 'inline-article-handler', tag: node.tagName });
    if (body && !code && node.nodeName === '#text' && /{%\s*(?:post_url|include|raw|endraw)|{{\s*[^}]*\|\s*relative_url/.test(node.value || '')) errors.push({ type: 'unconverted-liquid', text: node.value.slice(0, 100) });
    for (const attribute of ['href', 'src', 'poster']) if (attributes[attribute]) references.push({ value: attributes[attribute], attribute, tag: node.tagName, body });
    if (attributes.srcset) for (const candidate of attributes.srcset.split(',')) references.push({ value: candidate.trim().split(/\s+/)[0], attribute: 'srcset', tag: node.tagName, body });
    if (body && !inBody) articleBody = node;
    for (const child of node.childNodes || []) visit(child, body, code);
    if (node.content) visit(node.content, body, code);
  };
  visit(document);
  return { ids, duplicates, references, canonical, description, noindex, articleBody, h1Count };
}
async function htmlFor(file) {
  if (!htmlCache.has(file)) htmlCache.set(file, collect(parse(await readFile(file, 'utf8'))));
  return htmlCache.get(file);
}
const sourceFiles = (await readdir(resolve(root, '_posts'))).filter(name => /\.md$/i.test(name));
const articleByPath = new Map(registry.articles.map(article => [article.path, article]));
const idSet = new Set();
const pathSet = new Set();
for (const article of registry.articles) {
  if (idSet.has(article.id)) errors.push({ type: 'duplicate-article-id', id: article.id });
  if (pathSet.has(article.path)) errors.push({ type: 'duplicate-article-path', path: article.path });
  idSet.add(article.id); pathSet.add(article.path);
  if (!article.title?.trim() || !article.description?.trim()) errors.push({ type: 'missing-article-metadata', id: article.id });
  const file = await outputFor(article.path);
  if (!file) { errors.push({ type: 'missing-article-output', id: article.id, path: article.path }); continue; }
  if (!extname(article.path) && !article.path.endsWith('/') && !vercel.rewrites?.some(rule => rule.source === article.path && rule.destination === '/' + relative(dist, file))) errors.push({ type: 'missing-extensionless-rewrite', path: article.path });
  const document = await htmlFor(file);
  if (!document.canonical || new URL(document.canonical).pathname !== new URL(article.path, 'https://example.test').pathname) errors.push({ type: 'canonical-path-mismatch', id: article.id, expected: article.path, actual: document.canonical });
  if (!document.description?.trim()) errors.push({ type: 'missing-description', path: article.path });
  if (document.h1Count !== 1) errors.push({ type: 'heading-one-count', path: article.path, count: document.h1Count });
  for (const heading of article.headings || []) if (!document.ids.has(heading.id)) errors.push({ type: 'missing-heading-anchor', path: article.path, id: heading.id });
  for (const alias of Object.keys(article.legacyHeadingIds || {})) if (!document.ids.has(alias)) errors.push({ type: 'missing-legacy-anchor', path: article.path, id: alias });
}
const registeredSources = new Set(registry.articles.map(article => article.sourcePath));
for (const entry of manifest.articles) {
  const source = await readFile(resolve(root, entry.sourcePath));
  if (createHash('sha256').update(source).digest('hex') !== entry.sourceDigest) errors.push({ type: 'stale-source-manifest', sourcePath: entry.sourcePath });
  const article = registry.articles.find(article => article.id === entry.id);
  if (!article || article.path !== entry.canonicalPath || article.sourcePath !== entry.sourcePath) errors.push({ type: 'registry-manifest-mismatch', id: entry.id });
  if (!await exists(resolve(dist, entry.outputPath.replace(/^\//, '')))) errors.push({ type: 'missing-manifest-output', path: entry.outputPath });
}
if (manifest.articles.length !== registry.articles.length) errors.push({ type: 'registry-manifest-count-mismatch' });
for (const name of sourceFiles) {
  const sourcePath = `_posts/${name}`;
  const source = await readFile(resolve(root, sourcePath), 'utf8');
  const frontmatter = /^---\r?\n([\s\S]*?)\r?\n---/.exec(source)?.[1] || '';
  const draft = /^published:\s*false\s*$/m.test(frontmatter) || /^publicationState:\s*draft\s*$/m.test(frontmatter);
  if (!draft && !registeredSources.has(sourcePath)) errors.push({ type: 'unmigrated-source', sourcePath });
  if (draft && registeredSources.has(sourcePath)) errors.push({ type: 'draft-in-public-registry', sourcePath });
}
const sourceCache = new Map();
let checkedLinks = 0, checkedAssets = 0;
for (const file of outputFiles.filter(path => extname(path) === '.html')) {
  const document = await htmlFor(file);
  const article = [...articleByPath.values()].find(item => outputPathMatches(file, item.path));
  const pagePath = article?.path || '/' + relative(dist, file).replace(/index\.html$/, '');
  const origin = document.canonical ? new URL(document.canonical).origin : 'https://dongryeollee1.github.io';
  if (document.duplicates.length) errors.push({ type: 'duplicate-dom-id', path: pagePath, ids: document.duplicates });
  for (const reference of document.references) {
    const value = reference.value;
    if (/^\s*javascript:/i.test(value)) { errors.push({ type: 'unsafe-url', path: pagePath, target: value }); continue; }
    if (!value || /^(?:data:|mailto:|tel:|blob:)/i.test(value)) continue;
    let url;
    try { url = new URL(value, origin + pagePath); } catch { errors.push({ type: 'invalid-url', path: pagePath, value }); continue; }
    if (!['http:', 'https:'].includes(url.protocol)) continue;
    if (![origin, 'https://dongryeollee1.github.io', 'https://DONGRYEOLLEE1.github.io'.toLowerCase()].includes(url.origin.toLowerCase())) continue;
    const target = await outputFor(url.pathname);
    const asset = reference.attribute !== 'href' || Boolean(extname(url.pathname) && !/\.html?$/.test(url.pathname));
    if (asset) checkedAssets++; else checkedLinks++;
    if (!target) {
      const detail = { type: asset ? 'missing-local-asset' : 'broken-internal-link', path: pagePath, target: value };
      if (!asset && reference.body && article && await preexistingReference(article, value, url)) warnings.push({ ...detail, origin: 'source-content' });
      else errors.push(detail);
      continue;
    }
    if (!asset && url.hash && /\.html$/.test(target)) {
      const fragment = decodedPath(url.hash.slice(1));
      if (!(await htmlFor(target)).ids.has(fragment)) {
        const detail = { type: 'missing-fragment', path: pagePath, target: value };
        if (reference.body && article && await preexistingReference(article, value, url)) warnings.push({ ...detail, origin: 'source-content' });
        else errors.push(detail);
      }
    }
  }
}
function outputPathMatches(file, path) {
  const relativeFile = '/' + relative(dist, file);
  return relativeFile === decodedPath(path) || relativeFile === decodedPath(path).replace(/\/$/, '') + '/index.html' || relativeFile === decodedPath(path).replace(/\/$/, '') + '.html';
}
async function preexistingReference(article, value, url) {
  if (!sourceCache.has(article.sourcePath)) sourceCache.set(article.sourcePath, await readFile(resolve(root, article.sourcePath), 'utf8'));
  const source = sourceCache.get(article.sourcePath);
  return source.includes(value) || source.includes(decodedPath(value)) || (url.hash && source.includes(url.hash));
}
function parseXml(document, name) {
  const result = { locs: [], links: [], itemCount: 0 };
  const parser = new SaxesParser({ xmlns: false });
  const stack = [];
  let textContent = '';
  parser.on('opentag', tag => { stack.push(tag.name); if (tag.name === 'item') result.itemCount++; if (['loc', 'link'].includes(tag.name)) textContent = ''; });
  parser.on('text', text => { if (['loc', 'link'].includes(stack.at(-1))) textContent += text; });
  parser.on('closetag', () => { const tag = stack.pop(); if (tag === 'loc') result.locs.push(textContent); if (tag === 'link') result.links.push(textContent); });
  parser.on('error', error => errors.push({ type: 'invalid-xml', file: name, message: error.message }));
  parser.write(document).close();
  return result;
}
for (const name of ['sitemap.xml', 'feed.xml', 'rss.xml']) {
  const file = resolve(dist, name);
  if (!await exists(file)) { errors.push({ type: 'missing-public-file', path: name }); continue; }
  const parsed = parseXml(await readFile(file, 'utf8'), name);
  const locations = new Set((name === 'sitemap.xml' ? parsed.locs : parsed.links).map(value => { try { return new URL(value).pathname; } catch { return value; } }));
  for (const article of registry.articles) if (!locations.has(new URL(article.path, 'https://example.test').pathname)) errors.push({ type: 'missing-feed-or-sitemap-article', file: name, article: article.id });
}
for (const name of ['robots.txt', '404.html', 'googlee56e195b7180d2c5.html', 'naver19d7558f1b7416cc256e7401c11fcb64.html', 'fonts/PretendardVariable.woff2', 'fonts/OFL.txt', 'pagefind/pagefind.js']) if (!await exists(resolve(dist, name))) errors.push({ type: 'missing-public-file', path: name });
const preview = Boolean(process.env.VERCEL_ENV && process.env.VERCEL_ENV !== 'production');
if (preview) {
  for (const file of outputFiles.filter(path => extname(path) === '.html' && !/google|naver/.test(path))) if (!(await htmlFor(file)).noindex) errors.push({ type: 'indexable-preview', path: relative(dist, file) });
  if (!(await readFile(resolve(dist, 'robots.txt'), 'utf8')).includes('Disallow: /\n')) errors.push({ type: 'indexable-preview-robots' });
}
const unique = array => [...new Map(array.map(item => [JSON.stringify(item), item])).values()];
const report = { status: errors.length ? 'failed' : 'passed', articles: registry.articles.length, sourceMarkdownFiles: sourceFiles.length, htmlFiles: htmlCache.size, checkedLinks, checkedAssets, errors: unique(errors), warnings: unique(warnings), externalResources: 'not fetched by this check' };
await writeFile(resolve(root, 'generated/build-verification.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ ...report, errors: report.errors.slice(0, 30), warnings: report.warnings.slice(0, 15) }, null, 2));
if (errors.length) process.exitCode = 1;

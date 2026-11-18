import { readFile, writeFile, readdir, mkdir, stat } from 'node:fs/promises';
import { resolve, dirname, extname, basename, relative, posix } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import matter from 'gray-matter';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';
import GithubSlugger from 'github-slugger';
import { createHighlighter } from 'shiki';
import { generateRelated, validateKnowledge } from '../knowledge/related.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const readJson = async path => JSON.parse(await readFile(resolve(ROOT, path), 'utf8'));
const legacyLinks = await readJson('content/legacy-links.json');
const exists = async path => { try { return (await stat(path)).isFile(); } catch { return false; } };
const digest = value => createHash('sha256').update(value).digest('hex');
const strings = value => value == null ? [] : Array.isArray(value) ? value.map(String) : [String(value)];
const plain = node => node.type === 'text' || node.type === 'inlineCode' || node.type === 'code' || node.type === 'math' || node.type === 'inlineMath' ? node.value || '' : (node.children || []).map(plain).join('');
function walk(node, fn, parent) { fn(node, parent); for (const child of [...(node.children || [])]) walk(child, fn, node); }
const normalizeText = value => String(value).normalize('NFKC').replace(/[\s\p{P}\p{S}]/gu, '').toLowerCase();
const decoded = value => { try { return decodeURIComponent(value); } catch { return value; } };
const escapeHtml = text => String(text).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

/** Jekyll's filename slug keeps case; its category placeholders are lower-case. */
export function legacyPath(filename, data = {}) {
  if (data.permalink) return String(data.permalink).replace(/^https?:\/\/[^/]+/, '').replace(/^\/{2,}/, '/');
  const match = /^(\d{4})-(\d{2})-(\d{2})-(.+)\.md$/i.exec(filename);
  if (!match) throw new Error(`Post filename must include a publication date: ${filename}`);
  const [, year, month, day, title] = match;
  const categories = strings(data.categories ?? data.category).map(c => c.toLowerCase().replace(/\s+/g, '-')).join('/');
  const slug = String(data.slug || title).replace(/\s+/g, '-');
  return `/${categories ? categories + '/' : ''}${year}/${month}/${day}/${slug}.html`;
}

/** Interpret only the two Liquid forms used by the original articles. */
export function convertLiquid(source, sourceMap) {
  return source.replace(/\{\{\s*(['"])(.*?)\1\s*\|\s*(?:relative_url|absolute_url)\s*\}\}/g, (_, _quote, path) => path.startsWith('/') ? path : '/' + path)
    .replace(/\{%\s*post_url\s+([^%]+?)\s*%\}/g, (_, key) => {
      const item = sourceMap.get(key.trim().replace(/^['"]|['"]$/g, ''));
      if (!item) throw new Error(`Unknown post_url target ${key}`);
      return item.path;
    }).replace(/\{%\s*(?:raw|endraw)\s*%\}/g, '');
}

const schema = {
  ...defaultSchema,
  clobberPrefix: '',
  tagNames: [...defaultSchema.tagNames, 'figure', 'figcaption', 'details', 'summary', 'section', 'div', 'kbd', 'samp', 'mark', 'sub', 'sup'],
  attributes: {
    ...defaultSchema.attributes,
    '*': [...(defaultSchema.attributes['*'] || []), 'className', 'id', 'title', 'ariaHidden', 'role'],
    code: [...(defaultSchema.attributes.code || []), ['className', /^language-/, 'math-inline', 'math-display']],
    div: [...(defaultSchema.attributes.div || []), ['className', 'math', 'math-display', 'math-inline']],
    span: [...(defaultSchema.attributes.span || []), ['className', 'math', 'math-inline', 'math-display']],
    img: [...(defaultSchema.attributes.img || []), 'width', 'height', 'loading', 'decoding'],
    th: [...(defaultSchema.attributes.th || []), 'colSpan', 'rowSpan', 'scope'],
    td: [...(defaultSchema.attributes.td || []), 'colSpan', 'rowSpan'],
    details: ['open'],
  },
};

let highlighter;
async function prepareHighlighter() {
  highlighter ||= await createHighlighter({themes:['github-light','github-dark'],langs:['python','javascript','typescript','json','bash','yaml','markdown','html','css','sql','dockerfile','xml','cpp','c','java','diff','toml']});
}
const sourceResources = new Map();
const report = { version: 1, articles: [], warnings: [], extractedImages: [], removedEmbeddedCode: [], repairedLinks: [] };
const assetNfc = new Map();
async function indexAssets(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) await indexAssets(path);
    else assetNfc.set('/' + relative(ROOT, path).split('\\').join('/').normalize('NFC'), path);
  }
}

function remarkHeadings(article) {
  return tree => {
    const slugger = new GithubSlugger();
    const seen = new Set();
    const headings = [], aliases = {};
    let bodyUnderH1 = false;
    tree.children = tree.children.flatMap(node => {
      if (node.type !== 'heading') return [node];
      const before = plain(node);
      const explicit = /\s*\{#([^\s}]+)\}\s*$/.exec(before);
      if (explicit) {
        const last = node.children.at(-1);
        if (last?.type === 'text') last.value = last.value.replace(/\s*\{#[^\s}]+\}\s*$/, '');
      }
      const text = plain(node).trim();
      const id = explicit?.[1] || slugger.slug(text) || `section-${headings.length + 1}`;
      if (seen.has(id)) throw new Error(`Duplicate heading id ${id} in ${article.sourcePath}`);
      seen.add(id);
      const candidates = new Set([`h-${id}`]);
      const kramdown = text.toLowerCase().replace(/^[^\p{L}]+/u, '').replace(/[^\p{L}\p{N}_ -]/gu, '').replace(/ /g, '-');
      if (kramdown && kramdown !== id) { candidates.add(kramdown); candidates.add(`h-${kramdown}`); }
      const aliasNodes = [];
      for (const alias of candidates) if (alias !== id && !seen.has(alias)) {
        seen.add(alias); aliases[alias] = id;
        aliasNodes.push({ type: 'html', value: `<span id="${escapeHtml(alias)}" class="anchor-alias" aria-hidden="true"></span>` });
      }
      if (node.depth === 1 && normalizeText(text) === normalizeText(article.title)) {
        aliases[id] = 'article-title';
        return [...aliasNodes, { type: 'html', value: `<span id="${escapeHtml(id)}" class="anchor-alias" aria-hidden="true"></span>` }];
      }
      if (node.depth === 1) { node.depth = 2; bodyUnderH1 = true; }
      else if (bodyUnderH1) node.depth = Math.min(node.depth + 1, 6);
      node.data = { ...node.data, hProperties: { id } };
      headings.push({ depth: node.depth, id, text });
      return [...aliasNodes, node];
    });
    article.headings = headings;
    article.legacyHeadingIds = aliases;
  };
}

function transformHtml(article, sourceMap) {
  return async tree => {
    const resources = [];
    const codeBlocks = [];
    const pending = [];
    const idSet = new Set(article.headings.map(h => h.id));
    walk(tree, (node, parent) => {
      if (node.type !== 'element') return;
      if (['script', 'style', 'iframe', 'object', 'embed'].includes(node.tagName)) {
        report.removedEmbeddedCode.push({ sourcePath: article.sourcePath, element: node.tagName });
        if (node.tagName === 'iframe' && /^https?:/.test(node.properties?.src || '')) {
          node.tagName = 'p'; node.children = [{type:'element',tagName:'a',properties:{href:node.properties.src,rel:['noopener','noreferrer'],target:'_blank'},children:[{type:'text',value:node.properties.title || '원문에 첨부한 자료 열기'}]}]; node.properties = {};
        } else { node.type = 'text'; node.value = ''; delete node.children; delete node.tagName; delete node.properties; }
        return;
      }
      if (/^h[1-6]$/.test(node.tagName)) {
        if (node.tagName === 'h1') node.tagName = 'h2';
        if (!node.properties?.id) {
          const text = plain(node); const slugger = new GithubSlugger(); let id = slugger.slug(text) || 'section';
          while (idSet.has(id)) id = slugger.slug(text);
          node.properties ||= {}; node.properties.id = id; idSet.add(id);
          article.headings.push({depth:Number(node.tagName[1]),id,text});
        }
      }
      if (node.tagName === 'pre' && node.children?.[0]?.tagName === 'code') codeBlocks.push({node,parent});
      for (const attribute of ['src','href','poster']) {
        const value = node.properties?.[attribute];
        if (typeof value !== 'string' || !value) continue;
        resources.push({ attribute, value, tag: node.tagName });
        if (/^data:image\/(png|jpeg|gif|webp|svg\+xml);base64,/i.test(value)) {
          pending.push((async()=>{
            const match = /^data:image\/([^;]+);base64,([\s\S]+)$/.exec(value);
            const bytes = Buffer.from(match[2].replace(/\s/g,''),'base64');
            const extension = {jpeg:'jpg','svg+xml':'svg'}[match[1]] || match[1];
            const path = `/assets/content/${digest(bytes).slice(0,20)}.${extension}`;
            await mkdir(resolve(ROOT,'public/assets/content'),{recursive:true});
            await writeFile(resolve(ROOT,'public'+path),bytes);
            node.properties[attribute] = path;
            report.extractedImages.push({ sourcePath: article.sourcePath, path, bytes: bytes.length });
          })());
          continue;
        }
        if (value.startsWith('#') || /^(?:mailto:|tel:|data:|https?:)/i.test(value) && !/^https?:\/\/dongryeollee1\.github\.io\//i.test(value)) continue;
        let target = value.replace(/^https?:\/\/dongryeollee1\.github\.io/i,'').replace(/^\/{2,}/,'/');
        const [withoutHash, hash] = target.split('#');
        const repair = legacyLinks[withoutHash];
        if (repair) {
          const linked = [...sourceMap.values()].find(item=>item.id===repair.articleId);
          if (!linked) throw new Error(`Invalid legacy link repair target ${repair.articleId}`);
          target = linked.path + (hash ? '#'+hash : '');
          report.repairedLinks.push({ sourcePath:article.sourcePath, original:value, target, reason:repair.reason });
          node.properties[attribute] = target;
          continue;
        }
        const decodedPath = decoded(withoutHash);
        if (decodedPath.endsWith('.md')) {
          const key = basename(decodedPath,'.md'); const linked = sourceMap.get(key);
          if (linked) target = linked.path + (hash ? '#'+hash : '');
        } else if (!target.startsWith('/')) {
          const rooted = decodedPath.replace(/^(?:\.\.\/|\.\/)+/g,'');
          if (/^(img|assets)\//.test(rooted)) target = '/' + rooted + (hash ? '#'+hash : '');
          else if (assetNfc.has('/_posts/'+rooted.normalize('NFC'))) target = '/' + rooted + (hash ? '#'+hash : '');
          else if (attribute !== 'href') report.warnings.push({ type:'unmapped-relative-asset', sourcePath:article.sourcePath, value });
        }
        node.properties[attribute] = target;
      }
      if (node.tagName === 'img') {
        node.properties.loading = 'lazy'; node.properties.decoding = 'async';
        if (!node.properties.alt) node.properties.alt = '';
      }
      if (node.tagName === 'a' && /^https?:\/\//.test(node.properties.href || '')) node.properties.rel = ['noopener','noreferrer'];
    });
    await Promise.all(pending);
    sourceResources.set(article.id, resources);
  };
}

function highlightCode() {
  return tree => {
    walk(tree, (node, parent) => {
      if (node.type !== 'element' || node.tagName !== 'pre' || node.children?.[0]?.tagName !== 'code' || !parent) return;
      const code = node.children[0];
      const label = (code.properties?.className || []).find(c => c.startsWith('language-'))?.slice(9) || 'text';
      const aliases = {py:'python',shell:'bash',sh:'bash',zsh:'bash',console:'text',plaintext:'text',output:'text',none:'text',txt:'text',docker:'dockerfile'};
      let lang = aliases[label.toLowerCase()] || label.toLowerCase();
      if (!highlighter.getLoadedLanguages().includes(lang)) lang = 'text';
      const result = highlighter.codeToHast(plain(code).replace(/\n$/,''),{ lang, themes:{light:'github-light',dark:'github-dark'} });
      const highlighted = result.children.find(n=>n.type==='element');
      highlighted.properties.tabIndex = 0;
      const index = parent.children.indexOf(node); if (index >= 0) parent.children[index] = highlighted;
    });
  };
}

/** Render untrusted legacy Markdown through one shared allowlisted renderer. */
export async function renderMarkdown(source, article, sourceMap = new Map()) {
  await prepareHighlighter();
  const normalized = convertLiquid(source,sourceMap);
  if (/\{\{[^}]*\|\s*(?:relative_url|absolute_url)|\{%\s*post_url/.test(normalized)) throw new Error(`Unconverted Liquid in ${article.sourcePath}`);
  const processor = unified().use(remarkParse).use(remarkGfm).use(remarkMath,{singleDollarTextMath:false})
    .use(()=>remarkHeadings(article)).use(remarkRehype,{allowDangerousHtml:true}).use(rehypeRaw)
    .use(()=>transformHtml(article,sourceMap)).use(rehypeSanitize,schema).use(rehypeKatex,{throwOnError:false,strict:false})
    .use(highlightCode).use(rehypeStringify);
  return processor.process(normalized);
}

export async function buildContent() {
  await mkdir(resolve(ROOT,'generated'),{recursive:true});
  const metadata = await readJson('content/article-metadata.json');
  const concepts = await readJson('content/knowledge/concepts.json');
  const series = await readJson('content/knowledge/series.json');
  const projects = await readJson('content/knowledge/projects.json');
  const overrides = await readJson('content/knowledge/overrides.json');
  const filenames = (await readdir(resolve(ROOT,'_posts'))).filter(f=>/\.md$/i.test(f)).sort();
  const parsed = [];
  for (const filename of filenames) {
    const sourcePath = '_posts/'+filename, source = await readFile(resolve(ROOT,sourcePath),'utf8');
    const {data,content} = matter(source); const meta = metadata[sourcePath];
    if (!meta) throw new Error(`Missing reviewed metadata: ${sourcePath}`);
    if (data.published === false || (meta.publicationState || data.publicationState || 'published') !== 'published') continue;
    const title = String(meta.title || data.title || '').trim();
    if (!title || !meta.description?.trim() || !meta.id || !Array.isArray(meta.conceptRefs) || !meta.conceptRefs.length) throw new Error(`Incomplete article metadata: ${sourcePath}`);
    const explicitDate = /^date:\s*["']?(\d{4}-\d{2}-\d{2})/m.exec(source)?.[1];
    const publishedAt = explicitDate || filename.slice(0,10);
    const article = { id:meta.id, sourcePath, path:legacyPath(filename,data), title, description:meta.description,
      publishedAt, language:meta.language || data.lang || 'ko', contentType:meta.contentType,
      conceptRefs:meta.conceptRefs, projectIds:meta.projectIds || [], featured:Boolean(meta.featured), ...(meta.hero?{hero:meta.hero}:{}),
      ...(meta.updatedAt?{updatedAt:meta.updatedAt}:{}), legacyCategories:strings(data.categories ?? data.category),legacyTags:strings(data.tags),
      readingMinutes:0, html:'', headings:[], plainText:'',related:[], publicationState:'published' };
    parsed.push({article,source,content,filename});
  }
  const sourceMap = new Map(parsed.map(({article,filename})=>[basename(filename,'.md'),article]));
  const allPaths = new Set(), allIds = new Set();
  for (const {article} of parsed) {
    if (allPaths.has(article.path) || allIds.has(article.id)) throw new Error(`Duplicate article path/id ${article.path} / ${article.id}`);
    if (!article.path.startsWith('/') || /[?#]|\/\.\.?\//.test(article.path)) throw new Error(`Invalid canonical path: ${article.path}`);
    allPaths.add(article.path);allIds.add(article.id);
  }
  for (const directory of ['img','assets','_posts']) await indexAssets(resolve(ROOT,directory));
  await prepareHighlighter();
  for (const {article,source,content,filename} of parsed) {
    const output = await renderMarkdown(content,article,sourceMap);
    article.html = String(output);
    article.plainText = article.html.replace(/<[^>]+>/g,' ').replace(/&(?:[a-z]+|#\d+);/gi,' ').replace(/\s+/g,' ').trim();
    const koChars=(article.plainText.match(/[가-힣]/g)||[]).length; const latinWords=(article.plainText.match(/[A-Za-z0-9_]+/g)||[]).length;
    article.readingMinutes = Math.max(1,Math.ceil(koChars/450 + latinWords/220));
    if (/<(?:script|style|iframe|object|embed)[\s>]/i.test(article.html)) throw new Error(`Unsafe active HTML remains in ${article.sourcePath}`);
    if (article.html.includes('katex-error')) report.warnings.push({type:'math-render-error',sourcePath:article.sourcePath});
    for (const message of output.messages) report.warnings.push({type:'markdown-message',sourcePath:article.sourcePath,message:message.reason});
    const outputPath = article.path.startsWith('/research/') ? article.path.replace(/\/$/,'')+'/index.html' : article.path.endsWith('/') ? article.path+'index.html' : article.path.endsWith('.html') ? article.path : article.path+'.html';
    report.articles.push({id:article.id,sourcePath:article.sourcePath,canonicalPath:article.path,outputPath,sourceDigest:digest(source),headingIds:article.headings.map(h=>h.id),legacyHeadingIds:article.legacyHeadingIds,sourceResources:sourceResources.get(article.id),pathEvidence:source.includes('permalink:')?'explicit front matter permalink':'Jekyll default date/category/filename URL; live verification pending'});
  }
  const rawArticles=parsed.map(p=>p.article);
  validateKnowledge(rawArticles,concepts,series,overrides,projects);
  const articles=generateRelated(rawArticles,concepts,series,overrides).sort((a,b)=>b.publishedAt.localeCompare(a.publishedAt)||a.id.localeCompare(b.id));
  const registry={articles,concepts,series,projects};
  await writeFile(resolve(ROOT,'generated/registry.json'),JSON.stringify(registry));
  await writeFile(resolve(ROOT,'generated/url-manifest.json'),JSON.stringify({version:1,articles:report.articles.map(({sourceResources,...rest})=>rest)},null,2)+'\n');
  await writeFile(resolve(ROOT,'generated/migration-report.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({articles:articles.length,concepts:concepts.length,series:series.length,projects:projects.length,headings:articles.reduce((n,a)=>n+a.headings.length,0),warnings:report.warnings.length,extractedImages:report.extractedImages.length}));
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await buildContent();

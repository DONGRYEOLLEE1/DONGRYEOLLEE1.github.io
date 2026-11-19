import { createHash } from 'node:crypto';
import { parseFragment } from 'parse5';

export const hash = (value) => createHash('sha256').update(JSON.stringify(value)).digest('hex');

function textOf(node) {
  if (node.nodeName === '#text') return node.value;
  if (node.attrs?.some((attr) => attr.name === 'aria-hidden' && attr.value === 'true')) return '';
  return (node.childNodes || []).map(textOf).join(node.tagName === 'tr' ? ' | ' : '');
}

// Resolve evidence against the same sanitized HTML and heading IDs that Astro publishes.
function articleLinks(node, base) {
  const paths = [];
  function visit(child) {
    if (child.tagName === 'a') {
      const href = child.attrs.find((a) => a.name === 'href')?.value;
      if (href && !href.startsWith('#') && URL.canParse(href, base)) {
        const url = new URL(href, base);
        if (url.origin === base.origin || url.hostname === 'dongryeollee1.github.io') {
          paths.push(url.pathname.replace(/\/+$/, ''));
        }
      }
    }
    for (const item of child.childNodes || []) visit(item);
  }
  visit(node);
  return paths;
}

export function evidenceBlocks(article) {
  const base = new URL(article.path, process.env.SITE_URL || 'https://dongryeol-tech-blog.vercel.app');
  const blocks = [];
  let anchor = null;
  let heading = '';
  let sequence = 0;
  function visit(node) {
    if (/^h[1-6]$/.test(node.tagName)) {
      anchor = node.attrs.find((attr) => attr.name === 'id')?.value ?? null;
      heading = textOf(node).trim();
      sequence = 0;
      return;
    }
    if (
      ['p', 'pre', 'tr', 'figcaption'].includes(node.tagName) ||
      (node.tagName === 'li' &&
        !node.childNodes.some((child) => ['p', 'pre', 'ol', 'ul'].includes(child.tagName)))
    ) {
      const text = textOf(node).replace(/\s+/gu, ' ').trim();
      if (!/[\p{L}\p{N}]/u.test(text)) return;
      // A block is a bounded, verbatim excerpt. Root blocks cover preambles and headingless code.
      const links = articleLinks(node, base);
      for (let start = 0; start < text.length;) {
        let end = Math.min(start + 480, text.length);
        if (end < text.length) {
          const sentence = text.lastIndexOf('. ', end - 1);
          const space = sentence > start + 160 ? sentence + 1 : text.lastIndexOf(' ', end);
          if (space > start + 160) end = space;
        }
        const quote = text.slice(start, end).trim();
        const id = JSON.stringify([anchor, ++sequence]);
        blocks.push({ id, anchor, heading, quote, links, hash: hash(quote) });
        start = end;
      }
      return;
    }
    for (const child of node.childNodes || []) visit(child);
  }
  visit(parseFragment(article.html));
  return blocks;
}

export function citation(articleId, block) {
  return { articleId, blockId: block.id, anchor: block.anchor, quote: block.quote, hash: block.hash };
}

export function validCitation(evidence, articleId, blockMap) {
  if (!evidence || evidence.articleId !== articleId) return false;
  const block = blockMap?.get(evidence.blockId);
  return Boolean(
    block &&
    block.quote === evidence.quote &&
    block.anchor === evidence.anchor &&
    block.hash === evidence.hash,
  );
}

// The article outline (h2 → h3 tree) and the reading-time estimate shown in the sidebar.
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import { READING_LINE, buildToc, currentEntry, proseText, readingMinutes } from '../src/lib/toc.ts';

const heading = (depth, id) => ({ depth, id, text: `${id} 제목` });
/** The tree as nested ids, e.g. `a[a1 a2] b`. */
const shape = items => items.map(item => item.id + (item.children.length ? `[${shape(item.children)}]` : '')).join(' ');
const flatten = items => items.flatMap(item => [item, ...flatten(item.children)]);

test('an h2 owns the h3s that follow it until the next h2', () => {
  const tree = buildToc([heading(2, 'a'), heading(3, 'a1'), heading(3, 'a2'), heading(2, 'b'), heading(3, 'b1')]);
  assert.equal(shape(tree), 'a[a1 a2] b[b1]');
});

test('an h3 before the first h2 is a top-level item, not the child of an invented parent', () => {
  assert.equal(shape(buildToc([heading(3, 'x'), heading(3, 'y'), heading(2, 'a'), heading(3, 'a1')])), 'x y a[a1]');
  assert.equal(shape(buildToc([heading(3, 'x'), heading(3, 'y')])), 'x y');
  assert.deepEqual(buildToc([heading(3, 'x')])[0].children, []);
});

test('consecutive h2s stay siblings without children', () => {
  const tree = buildToc([heading(2, 'a'), heading(2, 'b'), heading(2, 'c')]);
  assert.equal(shape(tree), 'a b c');
  assert.ok(tree.every(item => Array.isArray(item.children) && item.children.length === 0));
});

test('no headings give an empty tree', () => {
  assert.deepEqual(buildToc([]), []);
});

test('document order, ids and texts are carried through', () => {
  const headings = [heading(2, 'b'), heading(3, 'z'), heading(3, 'a'), heading(2, 'a-2'), heading(3, 'm')];
  const tree = buildToc(headings);
  assert.deepEqual(flatten(tree).map(({ id, text }) => ({ id, text })), headings.map(({ id, text }) => ({ id, text })));
  assert.deepEqual(tree[0], { id: 'b', text: 'b 제목', children: [{ id: 'z', text: 'z 제목', children: [] }, { id: 'a', text: 'a 제목', children: [] }] });
});

test('headings deeper than h3 are left out and do not break a branch', () => {
  assert.equal(shape(buildToc([heading(2, 'a'), heading(4, 'deep'), heading(3, 'a1'), heading(5, 'deeper'), heading(3, 'a2')])), 'a[a1 a2]');
});

test('the input is not mutated', () => {
  const headings = [heading(2, 'a'), heading(3, 'a1'), heading(3, 'a2')].map(Object.freeze);
  const before = JSON.stringify(headings);
  buildToc(Object.freeze(headings));
  assert.equal(JSON.stringify(headings), before);
});

test('Korean reading time counts characters without whitespace, 500 a minute', () => {
  assert.equal(readingMinutes('가'.repeat(500), 'ko'), 1);
  assert.equal(readingMinutes('가'.repeat(501), 'ko'), 2);
  assert.equal(readingMinutes('가'.repeat(5000), 'ko'), 10);
  assert.equal(readingMinutes(Array(500).fill('가').join(' \n\t'), 'ko'), 1, 'whitespace is not counted');
  // Korean articles mix Hangul with Latin letters, digits and punctuation, and every one of those characters counts.
  assert.equal(readingMinutes('가a1'.repeat(334), 'ko'), 3, '1,002 characters, a third of them Hangul');
  assert.equal(readingMinutes('vLLM 0.5.3, (GPU)!'.repeat(62), 'ko'), 2, '992 characters without a single Hangul one');
});

test('English reading time counts words, 230 a minute', () => {
  const words = count => Array(count).fill('word').join(' ');
  assert.equal(readingMinutes(words(230), 'en'), 1);
  assert.equal(readingMinutes(words(231), 'en'), 2);
  assert.equal(readingMinutes(`  ${words(460).replaceAll(' ', '\n  ')}  `, 'en'), 2, 'runs of whitespace separate words once');
  assert.equal(readingMinutes(Array(231).fill('word').join('\n'), 'en'), 2, 'a newline alone separates words');
  assert.equal(readingMinutes(Array(231).fill('word').join('\t'), 'en'), 2, 'a tab alone separates words');
  assert.equal(readingMinutes(Array(230).fill('word').join(' \n\t '), 'en'), 1, 'a run of mixed whitespace is one separator');
  assert.equal(readingMinutes(words(2301), 'en'), 11);
});

test('an empty or tiny text is one minute, and the result is always a whole number', () => {
  for (const language of ['ko', 'en']) {
    assert.equal(readingMinutes('', language), 1);
    assert.equal(readingMinutes(' \n ', language), 1);
    assert.equal(readingMinutes('짧은 글 short', language), 1);
  }
  assert.ok(Number.isInteger(readingMinutes('가'.repeat(1234), 'ko')));
  assert.equal(readingMinutes('가'.repeat(1234), 'ko'), 3);
});

test('the text for the estimate leaves out <pre> blocks, tags and entities', () => {
  const html = '<p>본문 &amp; <code>inline</code> 글</p>\n<pre class="shiki" tabindex="0"><code><span class="line">log line 1</span>\n<span class="line">log line 2</span></code></pre><p>끝</p><PRE>X</PRE>';
  assert.equal(proseText(html).replace(/\s+/g, ' ').trim(), '본문 inline 글 끝');
  assert.equal(proseText('<p>a</p><pre>one</pre><p>b</p><pre>two</pre><p>c</p>').replace(/\s+/g, ''), 'abc', 'each block ends at its own closing tag');
  assert.equal(proseText('<p>no code here</p>').trim(), 'no code here');
  assert.equal(proseText('<p>&lt;pre&gt; as text and a <code>&#60;pre&#62;</code> stay</p>').replace(/\s+/g, ' ').trim(), 'pre as text and a pre stay');
  assert.equal(proseText('<prefix-el>kept</prefix-el>').trim(), 'kept', 'only the pre element is a block');
  // A short fix-note with a long pasted log is a one-minute read, not a half-hour one.
  const note = `<p>${'가'.repeat(400)}</p><pre><code>${'ERROR 0x1f '.repeat(1500)}</code></pre>`;
  assert.equal(readingMinutes(proseText(note), 'ko'), 1);
  assert.equal(readingMinutes(note.replace(/<[^>]+>/g, ' '), 'ko'), 28, 'the same note with the log read as prose');
});

test('the current entry is the last heading at or above the reading line', () => {
  const offsets = [0, 400, 900]; // heading tops relative to the article body
  const at = (bodyTop, relatedTop = 5000) => currentEntry(offsets, bodyTop, relatedTop, false);
  assert.equal(at(300), 0, 'the first heading is current before the reader reaches it');
  assert.equal(at(READING_LINE - 400), 1, 'a heading exactly on the line is current');
  assert.equal(at(READING_LINE - 400 + 1), 0, 'one pixel below the line it is not yet');
  assert.equal(at(-5000), 2);
  assert.equal(currentEntry([], 0, 5000, false), -1, 'no headings and not at the related section: nothing is marked');
});

test('"이어 읽기" is current once its section reaches the line, or at the end of the page', () => {
  const offsets = [0, 400, 900];
  assert.equal(currentEntry(offsets, -2000, READING_LINE, false), 3);
  assert.equal(currentEntry(offsets, -2000, READING_LINE + 1, false), 2);
  assert.equal(currentEntry(offsets, -700, 373, true), 3, 'the section cannot rise to the line, the page end stands in for it');
  assert.equal(currentEntry([], 0, 600, true), 0, 'without headings it is the only entry');
  assert.equal(currentEntry([], 0, READING_LINE, false), 0);
});

test('a heading the reader just jumped to keeps its place at the end of the page while it is in the window', () => {
  const offsets = [0, 400, 900], height = 900;
  // The last heading sits at 113px, right under the site header, and the page cannot scroll further (related section at 373px).
  assert.equal(currentEntry(offsets, 113 - 900, 373, true, 2, height), 2);
  // A jumped-to heading further down the window, which the end of the page keeps below the line, is still the one the reader chose.
  assert.equal(currentEntry(offsets, 300 - 400, 1400, true, 1, height), 1);
  // It never beats the related section once that has reached the line.
  assert.equal(currentEntry(offsets, 113 - 900, READING_LINE, true, 2, height), 3);
  // Away from the end the line decides as usual, whatever was jumped to.
  assert.equal(currentEntry(offsets, 113 - 900, 373, false, 0, height), 2);
  // Out of the window (above it, or below it) the jumped-to heading gives way to the end of the page.
  assert.equal(currentEntry(offsets, -2000, 373, true, 0, height), 3);
  assert.equal(currentEntry(offsets, 950 - 900, 373, true, 2, height), 3);
  // No jump: the end of the page marks "이어 읽기".
  assert.equal(currentEntry(offsets, 113 - 900, 373, true, -1, height), 3);
  assert.equal(currentEntry(offsets, 113 - 900, 373, true), 3);
});

const registryUrl = new URL('../generated/registry.json', import.meta.url);
test('every h2 and h3 of every article is in its tree exactly once, in document order', { skip: existsSync(registryUrl) ? false : 'generated/registry.json is absent; run `npm run prepare:content`' }, () => {
  const { articles } = JSON.parse(readFileSync(registryUrl, 'utf8'));
  assert.ok(articles.length > 0);
  for (const article of articles) {
    const expected = article.headings.filter(item => item.depth <= 3), tree = buildToc(article.headings);
    assert.deepEqual(flatten(tree).map(item => item.id), expected.map(item => item.id), article.path);
    assert.deepEqual(flatten(tree).map(item => item.text), expected.map(item => item.text), article.path);
    assert.ok(tree.every(item => item.children.every(child => child.children.length === 0)), `${article.path}: two levels at most`);
    // A child is always an h3, and its parent an h2.
    const depthOf = new Map(expected.map(item => [item.id, item.depth]));
    for (const item of tree) for (const child of item.children) { assert.equal(depthOf.get(child.id), 3, article.path); assert.equal(depthOf.get(item.id), 2, article.path); }
    assert.ok(Number.isInteger(readingMinutes(article.plainText, article.language)) && readingMinutes(article.plainText, article.language) >= 1, article.path);
  }
});

test('the estimate of every article counts its prose and not its code blocks', { skip: existsSync(registryUrl) ? false : 'generated/registry.json is absent; run `npm run prepare:content`' }, () => {
  const { articles } = JSON.parse(readFileSync(registryUrl, 'utf8'));
  let withCode = 0, shorter = 0;
  for (const article of articles) {
    const minutes = readingMinutes(proseText(article.html), article.language), whole = readingMinutes(article.plainText, article.language);
    assert.ok(Number.isInteger(minutes) && minutes >= 1 && minutes <= whole, `${article.path}: ${minutes} of ${whole}`);
    // Without a code block the estimate is the one for the whole plain text.
    if (!/<pre[\s>]/i.test(article.html)) assert.equal(minutes, whole, article.path);
    else { withCode++; if (minutes < whole) shorter++; }
    // The same count as the plain text of the article with its code blocks cut out.
    const stripped = article.html.replace(/<pre[\s>][\s\S]*?<\/pre>/gi, ' ').replace(/<[^>]+>/g, ' ').replace(/&(?:[a-z]+|#\d+);/gi, ' ').replace(/\s+/g, ' ').trim();
    assert.equal(minutes, readingMinutes(stripped, article.language), article.path);
  }
  assert.ok(withCode === 0 || shorter > 0, 'articles with code blocks exist, so some estimate must be shorter than the whole-text one');
});

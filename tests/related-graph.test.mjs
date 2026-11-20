// The related-articles graph: rank → edge length, the deterministic layout, second-hop selection, the
// server-rendered markup, where its popup opens and the styles the figure depends on. The geometry checks here are
// written independently of the layout's own search.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { placePopup } from '../src/lib/popup-placement.ts';
import { EDGE_LENGTHS, SECOND_HOP_LIMIT, edgeLength, layoutRelatedGraph, pickSecondHops, renderRelatedGraph } from '../src/lib/related-graph.ts';

const MARGIN = 20, NODE_GAP = 40, EDGE_GAP = 14, LENGTH_TOLERANCE = .5;
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
function pointToSegment(point, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const t = Math.max(0, Math.min(1, ((point.x - a.x) * dx + (point.y - a.y) * dy) / (dx * dx + dy * dy)));
  return distance(point, { x: a.x + t * dx, y: a.y + t * dy });
}
const orientation = (a, b, c) => Math.sign((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
const onSegment = (a, b, point) => orientation(a, b, point) === 0 && Math.min(a.x, b.x) <= point.x && point.x <= Math.max(a.x, b.x) && Math.min(a.y, b.y) <= point.y && point.y <= Math.max(a.y, b.y);
function segmentsIntersect(a, b, c, d) {
  if (orientation(a, b, c) !== orientation(a, b, d) && orientation(c, d, a) !== orientation(c, d, b)) return true;
  return onSegment(a, b, c) || onSegment(a, b, d) || onSegment(c, d, a) || onSegment(c, d, b);
}

/** Every R4 constraint for one layout; returns the list of violations. */
function violations(layout) {
  const found = [];
  const marks = [{ id: null, ...layout.origin }, ...layout.nodes];
  const position = id => marks.find(mark => mark.id === id);
  if (layout.width !== 284) found.push(`width ${layout.width}`);
  for (const mark of marks) {
    if (mark.x < MARGIN || mark.x > layout.width - MARGIN || mark.y < MARGIN || mark.y > layout.height - MARGIN) found.push(`${mark.id} outside the margin`);
  }
  marks.forEach((a, index) => marks.slice(index + 1).forEach(b => {
    if (distance(a, b) < NODE_GAP) found.push(`${a.id} and ${b.id} are ${distance(a, b).toFixed(1)} apart`);
  }));
  if (layout.edges.length !== layout.nodes.length) found.push('one edge per node expected');
  layout.edges.forEach((edge, index) => {
    const from = position(edge.from), to = position(edge.to), node = layout.nodes.find(item => item.id === edge.to);
    const drawn = distance({ x: edge.x1, y: edge.y1 }, { x: edge.x2, y: edge.y2 });
    if (edge.from !== node.parentId || edge.hop !== node.hop) found.push(`edge to ${edge.to} does not start at its parent`);
    if (from.x !== edge.x1 || from.y !== edge.y1 || to.x !== edge.x2 || to.y !== edge.y2) found.push(`edge to ${edge.to} misses its endpoints`);
    if (edge.length !== edgeLength(node.rank) || Math.abs(drawn - edge.length) > LENGTH_TOLERANCE) found.push(`edge to ${edge.to} is ${drawn.toFixed(2)}, expected ${edgeLength(node.rank)}`);
    for (const mark of marks) {
      if (mark.id !== edge.from && mark.id !== edge.to && pointToSegment(mark, from, to) < EDGE_GAP) found.push(`edge to ${edge.to} passes ${pointToSegment(mark, from, to).toFixed(1)} from ${mark.id}`);
    }
    for (const other of layout.edges.slice(index + 1)) {
      if ([other.from, other.to].includes(edge.from) || [other.from, other.to].includes(edge.to)) continue;
      if (segmentsIntersect(from, to, position(other.from), position(other.to))) found.push(`edges to ${edge.to} and ${other.to} cross`);
    }
  });
  return found;
}

const ranks = [0, 1, 2, 3];
const firstIds = count => ranks.slice(0, count).map(rank => 'first-' + rank);
/** Every way to hang up to `limit` second-hop nodes (one per parent, any rank) on `count` first-hop nodes. */
function secondHopCases(count, limit) {
  let cases = [[]];
  for (let size = 1; size <= Math.min(limit, count); size++) {
    cases = cases.concat(cases.filter(hops => hops.length === size - 1).flatMap(hops => firstIds(count)
      .filter((_, parent) => parent > (hops.length ? firstIds(count).indexOf(hops.at(-1).parentId) : -1))
      .flatMap(parentId => ranks.map(rank => [...hops, { id: 'second-' + hops.length, parentId, rank }]))));
  }
  return cases;
}

test('a better rank gives a shorter edge, in four clearly separated steps', () => {
  assert.equal(EDGE_LENGTHS.length, 4);
  for (const rank of [1, 2, 3]) assert.ok(edgeLength(rank) - edgeLength(rank - 1) >= 16, `rank ${rank} must be at least 16 longer than rank ${rank - 1}`);
  assert.deepEqual(ranks.map(edgeLength), [...EDGE_LENGTHS]);
  assert.equal(edgeLength(9), EDGE_LENGTHS[3]);
  // Rank is absolute: a single recommendation still gets the shortest edge.
  const [only] = layoutRelatedGraph(['only']).edges;
  assert.equal(only.length, EDGE_LENGTHS[0]);
  assert.ok(Math.abs(distance({ x: only.x1, y: only.y1 }, { x: only.x2, y: only.y2 }) - EDGE_LENGTHS[0]) <= LENGTH_TOLERANCE);
});

test('every layout with at most one second-hop node satisfies the geometric constraints', () => {
  let cases = 0;
  for (const count of [1, 2, 3, 4]) for (const second of secondHopCases(count, 1)) {
    const layout = layoutRelatedGraph(firstIds(count), second);
    assert.deepEqual(violations(layout), [], `${count} first-hop, second-hop ${JSON.stringify(second)}`);
    assert.deepEqual(layoutRelatedGraph(firstIds(count), second), layout, 'the layout is deterministic');
    assert.deepEqual(layout.nodes.map(({ id, hop, parentId, rank }) => ({ id, hop, parentId, rank })), [
      ...firstIds(count).map((id, rank) => ({ id, hop: 1, parentId: null, rank })),
      ...second.map(({ id, parentId, rank }) => ({ id, hop: 2, parentId, rank })),
    ]);
    cases++;
  }
  assert.equal(cases, 44);
});

test('the shipped second-hop limit is one, and the layout also holds for the limit it would be raised to', () => {
  assert.equal(SECOND_HOP_LIMIT, 1);
  for (const count of [2, 3, 4]) for (const second of secondHopCases(count, Math.max(2, SECOND_HOP_LIMIT))) {
    assert.deepEqual(violations(layoutRelatedGraph(firstIds(count), second)), [], `${count} first-hop, second-hop ${JSON.stringify(second)}`);
  }
});

test('first-hop nodes surround the current article instead of gathering on one side', () => {
  for (const count of [2, 3, 4]) {
    const { origin, nodes } = layoutRelatedGraph(firstIds(count));
    const pull = nodes.reduce((sum, node) => ({ x: sum.x + (node.x - origin.x) / distance(node, origin), y: sum.y + (node.y - origin.y) / distance(node, origin) }), { x: 0, y: 0 });
    assert.ok(Math.hypot(pull.x, pull.y) < .05, `${count} first-hop nodes pull the figure to one side`);
  }
});

test('the layout rejects input it cannot draw', () => {
  assert.throws(() => layoutRelatedGraph([]), /first-hop/);
  assert.throws(() => layoutRelatedGraph(['a', 'b', 'c', 'd', 'e']), /first-hop/);
  assert.throws(() => layoutRelatedGraph(['a'], [{ id: 'x', parentId: 'missing', rank: 0 }]), /parent/);
});

function article(id, related = [], extra = {}) {
  return { id, path: `/${id}/`, title: `Title ${id}`, language: 'ko', publishedAt: '2025-03-0' + (id.length % 9 + 1), contentType: 'paper-review', related: related.map(ref => typeof ref === 'string' ? { id: ref, reason: `${id} → ${ref}` } : ref), ...extra };
}
const lookup = (...articles) => new Map(articles.map(item => [item.id, item]));
const count = (html, pattern) => html.split(pattern).length - 1;

test('the second hop is the first new, known article recommended by the best first-hop article', () => {
  const source = article('source', ['a', 'b']);
  const pick = (...others) => {
    const articles = lookup(source, ...others);
    const first = source.related.map(({ id }) => articles.get(id)).filter(Boolean);
    return pickSecondHops(source.id, first, articles).map(({ article: target, via, rank, reason }) => ({ id: target.id, via: via.id, rank, reason }));
  };

  // Skips the current article, a first-hop member and an id missing from the map; the rank is the list position.
  assert.deepEqual(pick(article('a', ['source', 'b', 'ghost', 'x', 'y']), article('b', ['z']), article('x'), article('y'), article('z')),
    [{ id: 'x', via: 'a', rank: 3, reason: 'a → x' }]);
  // Only one, even though both first-hop articles could add one.
  assert.deepEqual(pick(article('a', ['x']), article('b', ['y']), article('x'), article('y')), [{ id: 'x', via: 'a', rank: 0, reason: 'a → x' }]);
  // Falls through to the next first-hop article when the best one has nothing new.
  assert.deepEqual(pick(article('a', ['source', 'b']), article('b', ['a', 'y']), article('y')), [{ id: 'y', via: 'b', rank: 1, reason: 'b → y' }]);
  // None available.
  assert.deepEqual(pick(article('a', ['source', 'b', 'ghost']), article('b', ['a'])), []);

  const articles = lookup(source, article('a', ['x']), article('b', ['x', 'y']), article('x'), article('y'));
  const picked = pickSecondHops(source.id, [articles.get('a'), articles.get('b')], articles, 2);
  assert.deepEqual(picked.map(({ article: target, via }) => [target.id, via.id]), [['x', 'a'], ['y', 'b']], 'a raised limit never draws the same article twice');
});

test('the markup draws marks only, selects the best recommendation and ships its complete card', () => {
  const source = article('source', [{ id: 'a', reason: '같은 기법을 다룹니다.' }, 'b', 'c', 'd', 'e']);
  const articles = lookup(source, article('a', ['source', 'far'], { title: 'Best pick', language: 'en', publishedAt: '2024-11-02' }), article('b'), article('c'), article('d'), article('e'),
    article('far', [], { title: 'Far away', contentType: 'note' }));
  const html = renderRelatedGraph(source, articles);

  assert.match(html, /<svg viewBox="0 0 284 \d+" role="group" aria-label="관련 글 연결">/);
  assert.doesNotMatch(html, /<text|<image|<img|<foreignObject/);
  assert.equal(count(html, '<a class="graph-node'), 5, 'four first-hop nodes and one second-hop node');
  assert.equal(count(html, '<line class="graph-line'), 5);
  assert.equal(count(html, 'graph-line-far'), 1);
  assert.equal(count(html, 'aria-current="true"'), 1);
  assert.match(html, /<a class="graph-node" href="\/a\/" lang="en" aria-label="Best pick" aria-current="true"/);
  assert.doesNotMatch(html, /href="\/e\/"/, 'at most four first-hop nodes');
  assert.ok(count(html, 'class="graph-node-hit" r="20"') === 5, 'pointer targets have a radius of at least 20');

  const card = /<a class="graph-card" href="\/a\/">([\s\S]*?)<\/a>/.exec(html)[1];
  assert.match(card, /<span class="graph-card-type" lang="en">Paper Review<\/span> · <time class="graph-card-date" datetime="2024-11-02">2024-11-02<\/time>/);
  assert.match(card, /<strong class="graph-card-title" lang="en">Best pick<\/strong>/);
  assert.match(card, /<span class="graph-card-via" hidden><\/span>/);
  assert.match(card, /<span class="graph-card-reason">같은 기법을 다룹니다\.<\/span>/);

  // The second-hop node carries the intermediate article's reason and names that article.
  const far = /<a class="graph-node graph-node-far"[^>]*>/.exec(html)[0];
  assert.match(far, /href="\/far\/"/);
  assert.match(far, /aria-label="Far away \(한 단계 건너 연결된 글\)"/);
  assert.match(far, /data-title="Far away" data-type="Learning Note"/);
  assert.match(far, /data-reason="a → far"/);
  assert.match(far, /data-via="「Best pick」의 관련 글"/);
  assert.match(html, /<p class="graph-legend"><span>선이 짧을수록 추천 순위가 높습니다\.<\/span> <span>점선은 한 단계 건너 연결된 글입니다\.<\/span><\/p>/);
  assert.match(html, /<a class="graph-more" href="#related-reading">관련 글 모두 보기 →<\/a>/);
});

test('without a second hop there is no dashed edge and no legend line about it', () => {
  const source = article('source', ['a']);
  const html = renderRelatedGraph(source, lookup(source, article('a', ['source'])));
  assert.equal(count(html, '<a class="graph-node'), 1);
  assert.doesNotMatch(html, /graph-line-far|graph-node-far|점선/);
  assert.match(html, /<p class="graph-legend"><span>선이 짧을수록 추천 순위가 높습니다\.<\/span><\/p>/);
});

test('each article\'s link is drawn on its own layout node, at the distance its rank gives', () => {
  // The second hop is the intermediate article's rank-1 entry, so its dashed edge must not get the shortest length.
  const source = article('source', ['a', 'b', 'c', 'd']);
  const html = renderRelatedGraph(source, lookup(source, article('a', ['source', 'x']), article('b'), article('c'), article('d'), article('x')));
  const layout = layoutRelatedGraph(['a', 'b', 'c', 'd'], [{ id: 'x', parentId: 'a', rank: 1 }]);
  assert.match(html, new RegExp(`viewBox="0 0 ${layout.width} ${layout.height}"`));
  assert.ok(html.includes(`<g class="graph-origin" aria-hidden="true" transform="translate(${layout.origin.x} ${layout.origin.y})">`));

  // Each link together with the position of the mark inside it.
  const drawn = new Map([...html.matchAll(/<a class="graph-node[^"]*" href="\/(\w+)\/"[^>]*>(?:(?!<\/a>)[\s\S])*?<g aria-hidden="true" transform="translate\(([\d.]+) ([\d.]+)\)">/g)]
    .map(([, id, x, y]) => [id, { x: Number(x), y: Number(y) }]));
  assert.deepEqual([...drawn.keys()], layout.nodes.map(node => node.id));
  for (const node of layout.nodes) {
    assert.deepEqual(drawn.get(node.id), { x: node.x, y: node.y }, `the link to ${node.id} is not on its own node`);
    const from = node.parentId ? drawn.get(node.parentId) : layout.origin;
    assert.ok(Math.abs(distance(drawn.get(node.id), from) - edgeLength(node.rank)) <= LENGTH_TOLERANCE, `the link to ${node.id} is ${distance(drawn.get(node.id), from).toFixed(1)} away, expected ${edgeLength(node.rank)}`);
  }
  assert.deepEqual(['a', 'b', 'c', 'd'].map(id => Math.round(distance(drawn.get(id), layout.origin))), [...EDGE_LENGTHS], 'the best pick is nearest');
  for (const edge of layout.edges) {
    assert.ok(html.includes(`<line class="graph-line${edge.hop === 2 ? ' graph-line-far' : ''}" x1="${edge.x1}" y1="${edge.y1}" x2="${edge.x2}" y2="${edge.y2}"/>`), `edge to ${edge.to}`);
  }
});

test('titles, reasons, paths and the more link are escaped', () => {
  const hostile = '<script>alert("x")</script> & co';
  const source = article('source', [{ id: 'a', reason: hostile }], { title: hostile });
  const articles = lookup(source, article('a', [{ id: 'b', reason: hostile }], { title: hostile, path: '/a/?q="1"&r=<2>' }), article('b', [], { title: hostile }));
  const html = renderRelatedGraph(source, articles, { moreHref: '/source/?a="1"&b=<2>#related-reading' });
  const escaped = '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; co';

  assert.doesNotMatch(html, /<script|alert\("x"\)|& co|="1"/);
  assert.ok(html.includes(`<title>현재 글: ${escaped}</title>`));
  assert.ok(html.includes(`aria-label="${escaped}"`) && html.includes(`data-title="${escaped}"`) && html.includes(`data-reason="${escaped}"`));
  assert.ok(html.includes(`data-via="「${escaped}」의 관련 글"`));
  assert.ok(html.includes(`<strong class="graph-card-title" lang="ko">${escaped}</strong>`) && html.includes(`<span class="graph-card-reason">${escaped}</span>`));
  assert.ok(html.includes('href="/a/?q=&quot;1&quot;&amp;r=&lt;2&gt;"'));
  assert.ok(html.includes('<a class="graph-more" href="/source/?a=&quot;1&quot;&amp;b=&lt;2&gt;#related-reading">'));
});

test('the more link defaults to the reading list on the same page and accepts another target', () => {
  const source = article('source', ['a']), articles = lookup(source, article('a'));
  assert.match(renderRelatedGraph(source, articles), /<a class="graph-more" href="#related-reading">/);
  assert.match(renderRelatedGraph(source, articles, { moreHref: '/source/#related-reading' }), /<a class="graph-more" href="\/source\/#related-reading">/);
});

test('nothing is rendered when no related article can be resolved', () => {
  assert.equal(renderRelatedGraph(article('source'), lookup()), '');
  assert.equal(renderRelatedGraph(article('source', ['ghost']), lookup(article('other'))), '');
});

test('the popup opens clear of its trigger whenever a side has room, and reports when it has to cover it', () => {
  const GAP = 8, EDGE_X = 16, EDGE_Y = 12, TRIGGER = 32;
  const trigger = (left, top) => ({ left, top, right: left + TRIGGER, bottom: top + TRIGGER });

  // The reported case: a list page at 1280×800, the trigger low on the right edge. The popup used to be pulled up over it.
  const low = placePopup(trigger(1185, 584), 336, 570, { width: 1280, height: 800 });
  assert.deepEqual(low, { left: 1185 - GAP - 336, top: 800 - EDGE_Y - 570, coversAnchor: false });
  // Room underneath: straight under the trigger, as before.
  assert.deepEqual(placePopup(trigger(600, 100), 336, 570, { width: 1280, height: 800 }), { left: 600, top: 100 + TRIGGER + GAP, coversAnchor: false });
  // A phone is too narrow to open beside the trigger and too short to open under it: above it.
  assert.deepEqual(placePopup(trigger(323, 700), 336, 520, { width: 375, height: 812 }), { left: 375 - EDGE_X - 336, top: 700 - GAP - 520, coversAnchor: false });
  // No side has room, so the overlap is reported to the caller.
  assert.equal(placePopup(trigger(323, 400), 336, 520, { width: 375, height: 812 }).coversAnchor, true);

  let cases = 0, covered = 0;
  for (const [vw, vh] of [[1280, 800], [1366, 657], [1024, 768], [768, 1024], [375, 812], [320, 568]]) {
    const width = Math.min(336, vw - 32);
    for (const height of [300, 453, 570].map(value => Math.min(value, vh - 24))) {
      for (let left = EDGE_X; left + TRIGGER <= vw - EDGE_X; left += 23) for (let top = EDGE_Y; top + TRIGGER <= vh; top += 19) {
        const anchor = trigger(left, top), spot = placePopup(anchor, width, height, { width: vw, height: vh }), where = `${vw}×${vh}, popup ${width}×${height}, trigger at ${left},${top}`;
        assert.ok(spot.left >= EDGE_X && spot.left + width <= vw - EDGE_X && spot.top >= EDGE_Y && spot.top + height <= vh - EDGE_Y, `outside the viewport: ${where}`);
        const overlaps = spot.left < anchor.right && spot.left + width > anchor.left && spot.top < anchor.bottom && spot.top + height > anchor.top;
        const room = [vh - EDGE_Y - anchor.bottom - GAP >= height, anchor.top - GAP - EDGE_Y >= height, vw - EDGE_X - anchor.right - GAP >= width, anchor.left - GAP - EDGE_X >= width];
        if (room.includes(true)) assert.ok(!overlaps, `covers the trigger although a side has room: ${where}`);
        assert.equal(spot.coversAnchor, overlaps, where);
        if (room[0]) assert.equal(spot.top, anchor.bottom + GAP, `not under the trigger: ${where}`);
        cases++; covered += overlaps;
      }
    }
  }
  assert.ok(cases > 10000 && covered > 0 && covered < cases / 10, `${covered} of ${cases} positions cover the trigger`);
});

const css = readFileSync(new URL('../src/styles/global.css', import.meta.url), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
/** Declarations of every rule for `selector`, grouped by the @media block they are in (the first group is outside any). */
const declarations = selector => css.split('@media').map(block => [...block.matchAll(/([^{}]+)\{([^{}]*)\}/g)]
  .filter(([, selectors]) => selectors.split(',').some(item => item.trim() === selector)).map(([, , body]) => body).join(';'));

test('the edges of the figure reach the 3:1 non-text contrast minimum in both themes', () => {
  const tokens = block => Object.fromEntries([...block.matchAll(/--([\w-]+):\s*(#[0-9a-f]{6})\b/gi)].map(([, name, value]) => [name, value]));
  const light = tokens(/:root \{([^}]*)\}/.exec(css)[1]), dark = { ...light, ...tokens(/:root\[data-theme="dark"\][^{]*\{([^}]*)\}/.exec(css)[1]) };
  const channels = (value, theme) => {
    const mix = /^color-mix\(in srgb, (.+), (.+) (\d+)%\)$/.exec(value), name = /^var\(--([\w-]+)\)$/.exec(value);
    if (mix) return channels(mix[1], theme).map((channel, index) => channel + (channels(mix[2], theme)[index] - channel) * mix[3] / 100);
    return [1, 3, 5].map(at => parseInt((name ? theme[name[1]] : value).slice(at, at + 2), 16));
  };
  const luminance = rgb => rgb.map(value => value / 255).map(value => value <= .04045 ? value / 12.92 : ((value + .055) / 1.055) ** 2.4).reduce((sum, value, index) => sum + value * [.2126, .7152, .0722][index], 0);
  const contrast = (a, b) => (Math.max(luminance(a), luminance(b)) + .05) / (Math.min(luminance(a), luminance(b)) + .05);
  const property = (selector, name) => new RegExp(`(?:^|;)\\s*${name}:\\s*([^;]+)`).exec(declarations(selector)[0])[1].trim();

  const stroke = property('.graph-line', 'stroke'), background = property('.graph-wrap svg', 'background');
  for (const [name, theme] of [['light', light], ['dark', dark]]) {
    const ratio = contrast(channels(stroke, theme), channels(background, theme));
    assert.ok(ratio >= 3, `${name} theme: edges are ${ratio.toFixed(2)}:1 against the figure's background`);
  }
});

test('the sticky rail is capped to the viewport, and every rule that unsticks it lifts the cap again', () => {
  const [base, ...media] = declarations('.related-rail');
  assert.match(base, /position:\s*sticky/);
  assert.match(base, /max-height:\s*calc\(100dvh - \d+px\)/);
  assert.match(base, /overflow-y:\s*auto/);
  const unstuck = media.filter(block => /position:\s*static/.test(block));
  assert.equal(unstuck.length, 2, 'the narrow layout and the short viewport');
  for (const block of unstuck) assert.match(block, /max-height:\s*none.*overflow:\s*visible/s);
});

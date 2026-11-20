// The explicit `.ts` extension lets `node --test` import this module without a bundler.
import { ARTICLE_TYPE_LABELS } from './archive-copy.ts';
import type { Article } from './types';

type GraphArticle = Pick<Article, 'id' | 'path' | 'title' | 'language' | 'publishedAt' | 'contentType' | 'related'>;
type Point = { x: number; y: number };
export type SecondHop = { id: string; parentId: string; rank: number };
export type GraphNode = Point & { id: string; hop: 1 | 2; parentId: string | null; rank: number };
export type GraphEdge = { from: string | null; to: string; hop: 1 | 2; length: number; x1: number; y1: number; x2: number; y2: number };
export type GraphLayout = { width: number; height: number; origin: Point; nodes: GraphNode[]; edges: GraphEdge[] };

// Rank is the only strength signal the recommendations carry: the best pick gets the shortest edge.
export const EDGE_LENGTHS = [46, 64, 82, 100] as const;
export const SECOND_HOP_LIMIT = 1;
export const edgeLength = (rank: number) => EDGE_LENGTHS[Math.max(0, Math.min(Math.trunc(rank) || 0, EDGE_LENGTHS.length - 1))];

const WIDTH = 284, PADDING = 28, MIN_HEIGHT = 132;
// One unit stricter than the published limits (40 between nodes, 14 beside an edge) so rounding cannot break them.
const NODE_GAP = 41, EDGE_GAP = 15;
// Candidate directions. First-hop nodes are spread evenly and the whole figure is turned by a rotation;
// a second-hop node leaves its parent along the parent's own direction, bent by one of BENDS.
const ROTATIONS = Array.from({ length: 24 }, (_, index) => index * 15 - 35);
const BENDS = [0, 30, -30, 60, -60, 90, -90];
const BEND_COST = .2;
// Ranks alternate around the circle so the two longest edges point in opposite directions.
const SLOTS = [0, 2, 1, 3];

const round = (value: number) => Math.round(value * 10) / 10;
const polar = (from: Point, length: number, degrees: number): Point => ({ x: from.x + length * Math.cos(degrees * Math.PI / 180), y: from.y + length * Math.sin(degrees * Math.PI / 180) });
const turn = (a: Point, b: Point, c: Point) => Math.sign((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
function segmentDistance(point: Point, a: Point, b: Point) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const t = Math.max(0, Math.min(1, ((point.x - a.x) * dx + (point.y - a.y) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(point.x - a.x - t * dx, point.y - a.y - t * dy);
}
const crosses = (a: Point, b: Point, c: Point, d: Point) => turn(a, b, c) !== turn(a, b, d) && turn(c, d, a) !== turn(c, d, b);
// points[0] is the current article; an edge is a pair of indexes into points.
function isClear(points: readonly Point[], edges: readonly (readonly [number, number])[]) {
  return points.every((point, index) => points.slice(index + 1).every(other => Math.hypot(point.x - other.x, point.y - other.y) >= NODE_GAP))
    && edges.every(([a, b], index) => points.every((point, node) => node === a || node === b || segmentDistance(point, points[a], points[b]) >= EDGE_GAP)
      && edges.slice(index + 1).every(([c, d]) => a === c || a === d || b === c || b === d || !crosses(points[a], points[b], points[c], points[d])));
}
const bendChoices = (count: number): number[][] => count ? bendChoices(count - 1).flatMap(rest => BENDS.map(bend => [...rest, bend])) : [[]];

/**
 * Places the current article, its first-hop articles (best first) and any second-hop articles. Pure and deterministic:
 * it tries every rotation with every bend and keeps the flattest figure that fits the width and breaks no spacing rule.
 */
export function layoutRelatedGraph(first: readonly string[], second: readonly SecondHop[] = []): GraphLayout {
  if (!first.length || first.length > EDGE_LENGTHS.length) throw new Error(`related graph: cannot lay out ${first.length} first-hop nodes`);
  const parents = second.map(({ parentId }) => first.indexOf(parentId));
  if (parents.includes(-1)) throw new Error('related graph: second-hop parent is not a first-hop node');
  const slots = SLOTS.filter(slot => slot < first.length), choices = bendChoices(second.length);
  const edges = [...first.map((_, index) => [0, index + 1] as const), ...parents.map((parent, index) => [parent + 1, first.length + index + 1] as const)];
  let best: { cost: number; points: Point[]; left: number; top: number; width: number; height: number } | undefined;

  for (const rotation of ROTATIONS) for (const bends of choices) {
    const angles = first.map((_, rank) => rotation + slots[rank] * 360 / first.length);
    const points = [{ x: 0, y: 0 }, ...angles.map((angle, rank) => polar({ x: 0, y: 0 }, edgeLength(rank), angle))];
    second.forEach(({ rank }, index) => points.push(polar(points[parents[index] + 1], edgeLength(rank), angles[parents[index]] + bends[index])));
    const xs = points.map(point => point.x), ys = points.map(point => point.y);
    const left = Math.min(...xs), top = Math.min(...ys), width = Math.max(...xs) - left, height = Math.max(...ys) - top;
    // Anything flatter than the minimum box costs the same, so short chains keep the first (tilted) rotation.
    const cost = Math.max(height, MIN_HEIGHT - 2 * PADDING) + BEND_COST * bends.reduce((sum, bend) => sum + Math.abs(bend), 0);
    if (width <= WIDTH - 2 * PADDING && (!best || cost < best.cost) && isClear(points, edges)) best = { cost, points, left, top, width, height };
  }
  if (!best) throw new Error(`related graph: no layout for ${first.length} first-hop and ${second.length} second-hop nodes`);

  // Centre the figure in the box.
  const height = Math.max(MIN_HEIGHT, Math.ceil(best.height + 2 * PADDING));
  const dx = (WIDTH - best.width) / 2 - best.left, dy = (height - best.height) / 2 - best.top;
  const [origin, ...placed] = best.points.map(point => ({ x: round(point.x + dx), y: round(point.y + dy) }));
  const nodes: GraphNode[] = [
    ...first.map((id, rank) => ({ ...placed[rank], id, hop: 1 as const, parentId: null, rank })),
    ...second.map(({ id, parentId, rank }, index) => ({ ...placed[first.length + index], id, hop: 2 as const, parentId, rank })),
  ];
  return {
    width: WIDTH, height, origin, nodes,
    edges: edges.map(([from, to]) => {
      const start = from ? nodes[from - 1] : origin, end = nodes[to - 1];
      return { from: end.parentId, to: end.id, hop: end.hop, length: edgeLength(end.rank), x1: start.x, y1: start.y, x2: end.x, y2: end.y };
    }),
  };
}

const escapes: Record<string, string> = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' };
const escapeMarkup = (value: string) => value.replace(/[&<>"]/g, character => escapes[character]);

type Entry = { article: GraphArticle; reason: string; rank: number; via?: GraphArticle };

/** Walks the first-hop articles best first; each may add its own best recommendation that is not drawn yet. */
export function pickSecondHops(sourceId: string, first: readonly GraphArticle[], articles: ReadonlyMap<string, GraphArticle>, limit = SECOND_HOP_LIMIT) {
  const drawn = new Set([sourceId, ...first.map(article => article.id)]);
  const picked: Entry[] = [];
  for (const via of first) {
    if (picked.length >= limit) break;
    const rank = via.related.findIndex(({ id }) => !drawn.has(id) && articles.has(id));
    if (rank < 0) continue;
    const { id, reason } = via.related[rank];
    drawn.add(id);
    picked.push({ article: articles.get(id)!, reason, rank, via });
  }
  return picked;
}

// Everything the card shows, already escaped: nodes carry it as data attributes so the script can swap the card.
function cardFields({ article, reason, via }: Entry) {
  return {
    path: escapeMarkup(article.path), language: escapeMarkup(article.language), title: escapeMarkup(article.title),
    type: escapeMarkup(ARTICLE_TYPE_LABELS[article.contentType] || article.contentType || ''), date: escapeMarkup(article.publishedAt),
    reason: escapeMarkup(reason), via: via ? escapeMarkup(`「${via.title}」의 관련 글`) : '',
  };
}

export function renderRelatedGraph(
  source: Pick<Article, 'id' | 'title' | 'related'>,
  articles: ReadonlyMap<string, GraphArticle>,
  { moreHref = '#related-reading' }: { moreHref?: string } = {},
) {
  const first: Entry[] = source.related.flatMap(({ id, reason }) => {
    const article = articles.get(id);
    return article ? [{ article, reason }] : [];
  }).slice(0, EDGE_LENGTHS.length).map((entry, rank) => ({ ...entry, rank }));
  if (!first.length) return '';

  const second = pickSecondHops(source.id, first.map(entry => entry.article), articles);
  const entries = [...first, ...second];
  const layout = layoutRelatedGraph(first.map(entry => entry.article.id), second.map(({ article, rank, via }) => ({ id: article.id, parentId: via!.id, rank })));
  const lines = layout.edges.map(edge => `<line class="graph-line${edge.hop === 2 ? ' graph-line-far' : ''}" x1="${edge.x1}" y1="${edge.y1}" x2="${edge.x2}" y2="${edge.y2}"/>`);
  const links = entries.map((entry, index) => {
    const { x, y } = layout.nodes[index], far = !!entry.via, fields = cardFields(entry);
    return `<a class="graph-node${far ? ' graph-node-far' : ''}" href="${fields.path}" lang="${fields.language}" aria-label="${fields.title}${far ? ' (한 단계 건너 연결된 글)' : ''}"${index ? '' : ' aria-current="true"'} data-title="${fields.title}" data-type="${fields.type}" data-published-at="${fields.date}" data-reason="${fields.reason}"${far ? ` data-via="${fields.via}"` : ''}>
      <desc lang="ko">${far ? fields.via + '. ' : ''}${fields.reason} 게시일 ${fields.date}</desc>
      <g aria-hidden="true" transform="translate(${x} ${y})">
        <circle class="graph-node-hit" r="20"/>
        <circle class="graph-node-halo" r="${far ? 13 : 17}"/>
        <circle class="graph-node-ring" r="${far ? 7 : 11}"/>${far ? '' : `
        <circle class="graph-node-dot" r="3"/>`}
      </g>
    </a>`;
  });
  const card = cardFields(first[0]);

  return `<svg viewBox="0 0 ${layout.width} ${layout.height}" role="group" aria-label="관련 글 연결">
    <g aria-hidden="true">${lines.join('')}</g>
    <g class="graph-origin" aria-hidden="true" transform="translate(${layout.origin.x} ${layout.origin.y})">
      <title>현재 글: ${escapeMarkup(source.title)}</title>
      <circle class="graph-origin-wash" r="21"/>
      <circle class="graph-origin-dot" r="7"/>
    </g>
    ${links.join('')}
  </svg>
  <p class="graph-legend"><span>선이 짧을수록 추천 순위가 높습니다.</span>${second.length ? ' <span>점선은 한 단계 건너 연결된 글입니다.</span>' : ''}</p>
  <a class="graph-card" href="${card.path}">
    <span class="graph-card-meta"><span class="graph-card-type" lang="en">${card.type}</span> · <time class="graph-card-date" datetime="${card.date}">${card.date}</time></span>
    <strong class="graph-card-title" lang="${card.language}">${card.title}</strong>
    <span class="graph-card-via" hidden></span>
    <span class="graph-card-reason">${card.reason}</span>
  </a>
  <a class="graph-more" href="${escapeMarkup(moreHref)}">관련 글 모두 보기 →</a>`;
}

import type { Article } from './types';

type GraphArticle = Pick<Article, 'id' | 'path' | 'title' | 'language'>;
type Point = readonly [x: number, y: number];

const origin: Point = [142, 99];
const positions: readonly Point[] = [[64, 48], [220, 50], [218, 158], [66, 162]];
const escapes: Record<string, string> = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' };
const escapeMarkup = (value: string) => value.replace(/[&<>"]/g, character => escapes[character]);

function shortLabel(title: string) {
  const text = title.replace(/^\([^)]{1,40}\)\s*\d+(?:[-.]\d+)*[.\s]+/, '');
  let label = '';
  let width = 0;

  for (const character of text) {
    width += /[\u0000-\u00ff]/.test(character) ? 0.55 : 1;
    if (width > 9.5) return label.trimEnd() + '…';
    label += character;
  }

  return label;
}

export function renderRelatedGraph(
  source: Pick<Article, 'title' | 'related'>,
  articles: ReadonlyMap<string, GraphArticle>,
) {
  const related = source.related.flatMap(({ id }) => {
    const article = articles.get(id);
    return article ? [article] : [];
  }).slice(0, positions.length);
  if (!related.length) return '';

  const isPair = related.length === 1;
  const [originX, originY] = isPair ? [72, 50] : origin;
  const layout = isPair ? [[212, 50]] : positions;
  const points = related.length === 2 ? [positions[0], positions[2]] : layout;
  const links = related.map((article, index) => {
    const [x, y] = points[index];
    return `<a class="graph-node" href="${escapeMarkup(article.path)}" lang="${escapeMarkup(article.language)}" aria-label="${escapeMarkup(article.title)}">
      <title>${escapeMarkup(article.title)}</title>
      <g aria-hidden="true" transform="translate(${x} ${y})">
        <circle class="graph-node-hit" r="26"/>
        <circle class="graph-ripple" r="12"/>
        <circle class="graph-ripple graph-ripple-echo" r="12"/>
        <circle class="graph-node-ring" r="11"/>
        <circle class="graph-node-dot" r="3"/>
        <text class="graph-label" y="31">${escapeMarkup(shortLabel(article.title))}</text>
      </g>
    </a>`;
  });
  const connections = related.map((_, index) => {
    const [x, y] = points[index];
    return `<line class="graph-line" x1="${originX}" y1="${originY}" x2="${x}" y2="${y}"/>`;
  });

  return `<svg viewBox="0 0 284 ${isPair ? 102 : 208}" role="group" aria-label="관련 글 연결">
    <g aria-hidden="true">${connections.join('')}</g>
    <g class="graph-origin" aria-hidden="true" transform="translate(${originX} ${originY})">
      <title>${escapeMarkup(source.title)}</title>
      <circle class="graph-origin-wash" r="21"/>
      <circle class="graph-origin-dot" r="7"/>
      <text class="graph-label" y="36">현재 글</text>
    </g>
    ${links.join('')}
  </svg>`;
}

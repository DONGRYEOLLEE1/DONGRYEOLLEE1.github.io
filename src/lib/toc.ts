// The outline and reading time of the article sidebar. Pure, so `node --test` imports it without a bundler.
import type { Heading } from './types';

export type TocItem = { id: string; text: string; children: TocItem[] };

/** An h2 is a top-level item and owns the h3s up to the next h2. An h3 before the first h2 stays at the top level
    instead of getting an invented parent. Headings deeper than h3 are left out. */
export function buildToc(headings: readonly Heading[]): TocItem[] {
  const items: TocItem[] = [];
  let parent: TocItem | undefined;
  for (const { depth, id, text } of headings) {
    if (depth > 3) continue;
    const item: TocItem = { id, text, children: [] };
    if (depth === 3 && parent) parent.children.push(item);
    else { items.push(item); if (depth < 3) parent = item; }
  }
  return items;
}

/** Whole minutes, never less than one: about 500 characters a minute in Korean (whitespace not counted), about 230 words a minute in English. */
export function readingMinutes(plainText: string, language: 'ko' | 'en'): number {
  const minutes = language === 'en' ? plainText.split(/\s+/).filter(Boolean).length / 230 : plainText.replace(/\s/g, '').length / 500;
  return Math.max(1, Math.ceil(minutes));
}

/** The article HTML as the text a reader goes through at those rates: tags and entities are dropped the way `plainText` drops them,
    and `<pre>` blocks go first, because a pasted log or a code listing is skimmed, not read as prose. */
export function proseText(html: string): string {
  return html.replace(/<pre[\s>][\s\S]*?<\/pre>/gi, ' ').replace(/<[^>]+>/g, ' ').replace(/&(?:[a-z]+|#\d+);/gi, ' ');
}

/** A heading is current once its top is this far down the window, or higher. */
export const READING_LINE = 155;

/** The current entry of the outline: an index into `offsets` (heading tops relative to the article body, in document order),
    `offsets.length` for the "이어 읽기" entry that follows the last heading, or -1 when there is nothing to mark.
    "이어 읽기" is outside the body: it is current once its section reaches the line, or at the end of the page when the section cannot rise that far.
    `pinned` is the heading the reader just jumped to. At the end of the page it keeps its place while it is in the window,
    so a click on one of the last headings marks that heading and not "이어 읽기". */
export function currentEntry(offsets: readonly number[], bodyTop: number, relatedTop: number, atEnd: boolean, pinned = -1, windowHeight = 0): number {
  if (relatedTop <= READING_LINE) return offsets.length;
  if (atEnd) { const y = pinned < 0 ? -1 : offsets[pinned] + bodyTop; return y > 0 && y < windowHeight ? pinned : offsets.length; }
  let index = 0; while (index + 1 < offsets.length && offsets[index + 1] + bodyTop <= READING_LINE) index++;
  return offsets.length ? index : -1;
}

import type { Article, Concept } from './types';

export const latestArticles = (articles: Article[]) => [...articles].sort((a,b) => b.publishedAt.localeCompare(a.publishedAt) || a.id.localeCompare(b.id));
export function conceptIds(conceptId: string, concepts: Concept[]): Set<string> {
  const ids = new Set([conceptId]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const concept of concepts) if (!ids.has(concept.id) && concept.broaderIds.some(id => ids.has(id))) {
      ids.add(concept.id); changed = true;
    }
  }
  return ids;
}
export function articlesForConcept(id: string, articles: Article[], concepts: Concept[], role?: 'about'|'applies'): Article[] {
  const ids = conceptIds(id, concepts);
  return latestArticles(articles.filter(article => article.conceptRefs.some(ref => ids.has(ref.id) && (!role || ref.role === role))));
}
export function usedConcepts(articles: Article[], concepts: Concept[]): Concept[] {
  return concepts.filter(concept => articlesForConcept(concept.id, articles, concepts).length > 0);
}
export const areaId = (area: string) => `area-${Array.from(area).map(char => char.codePointAt(0)!.toString(36)).join('-')}`;

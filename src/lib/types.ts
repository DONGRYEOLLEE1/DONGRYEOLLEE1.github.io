export type ContentType = 'research' | 'paper-review' | 'implementation' | 'troubleshooting' | 'note';
export type ConceptRef = { id: string; role: 'about' | 'applies' };
export type Hero = { src: string; alt: string; kind?: 'photo' | 'figure' };
export type Heading = { depth: number; id: string; text: string };
export type RelatedArticle = { id: string; reason: string; score?: number };
export interface Article {
  id: string; sourcePath: string; path: string; title: string; description: string;
  publishedAt: string; updatedAt?: string; language: 'ko' | 'en'; contentType: ContentType;
  conceptRefs: ConceptRef[]; featured?: boolean; hero?: Hero;
  legacyCategories: string[]; legacyTags: string[];
  html: string; headings: Heading[]; legacyHeadingIds?: Record<string,string>;
  plainText: string; related: RelatedArticle[]; seriesId?: string; seriesOrder?: number;
}
export interface Concept { id: string; label: string; aliases: string[]; description: string; broaderIds: string[]; area: string }
export interface Series { id: string; title: string; description: string; orderedArticleIds: string[] }
export interface Registry { articles: Article[]; concepts: Concept[]; series: Series[] }

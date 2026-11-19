import type { ConceptRef, ContentType } from './types';

export const ARTICLE_TYPE_LABELS: Record<ContentType, string> = {
  research: 'Research',
  'paper-review': 'Paper Review',
  implementation: 'Implementation',
  troubleshooting: 'Troubleshooting',
  note: 'Learning Note',
};

export const CONCEPT_ROLE_LABELS: Record<ConceptRef['role'], string> = {
  about: 'About this topic',
  applies: 'Applications',
};

export const ARCHIVE_SORT_LABELS = { relevance: 'Relevance', latest: 'Newest first' };
export const QUERY_LENGTH_MESSAGE = 'Enter a search term of 200 characters or fewer.';

export const articleCountLabel = (count: number) => `${count} ${count === 1 ? 'article' : 'articles'}`;
export const topicCountLabel = (count: number) => `${count} ${count === 1 ? 'topic' : 'topics'}`;

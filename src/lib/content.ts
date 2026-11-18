import { readFileSync } from 'node:fs';
import type { Registry } from './types';

let registry: Registry | undefined;
export function getRegistry(): Registry {
  if (!registry) registry = JSON.parse(readFileSync(process.cwd() + '/generated/registry.json', 'utf8')) as Registry;
  return registry;
}
export const getArticles = () => getRegistry().articles;
export const getConcepts = () => getRegistry().concepts;
export const getSeries = () => getRegistry().series;
export const getProjects = () => getRegistry().projects;
export const getArticleById = (id: string) => getArticles().find(article => article.id === id);

import { sitemapDocument, xmlResponse } from '../lib/feeds';
export const prerender = true;
export const GET = () => xmlResponse(sitemapDocument());

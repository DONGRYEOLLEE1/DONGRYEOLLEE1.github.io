import { rssDocument, xmlResponse } from '../lib/feeds';
export const prerender = true;
export const GET = () => xmlResponse(rssDocument(), 'application/rss+xml; charset=utf-8');

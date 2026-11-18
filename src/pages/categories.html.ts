import { legacyResponse } from '../lib/legacy';
export const prerender = true;
export const GET = () => legacyResponse('categories');

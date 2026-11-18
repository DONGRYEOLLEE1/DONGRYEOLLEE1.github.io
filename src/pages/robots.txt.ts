import { absoluteUrl } from '../lib/site';
export const prerender = true;
export const GET = () => {
  const preview = Boolean(process.env.VERCEL_ENV && process.env.VERCEL_ENV !== 'production');
  const body = preview ? 'User-agent: *\nDisallow: /\n' : `User-agent: *\nAllow: /\nDisallow: /search/\nSitemap: ${absoluteUrl('/sitemap.xml')}\n`;
  return new Response(body, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
};

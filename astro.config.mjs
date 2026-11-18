import { defineConfig } from 'astro/config';
import { SITE } from './src/lib/site.ts';

export default defineConfig({
  site: SITE.origin,
  output: 'static',
  trailingSlash: 'ignore',
  build: { format: 'preserve' },
  devToolbar: { enabled: false },
});

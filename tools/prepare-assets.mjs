import { cp, mkdir, readdir } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
await mkdir(resolve(root, 'public/fonts'), { recursive: true });
for (const name of ['PretendardVariable.woff2', 'OFL.txt']) {
  await cp(resolve(root, 'docs/preview/fonts', name), resolve(root, 'public/fonts', name));
}
for (const dir of ['img', 'assets']) {
  await cp(resolve(root, dir), resolve(root, 'public', dir), {
    recursive: true, force: true,
    filter: source => !source.endsWith('.DS_Store') && !source.includes('/.git'),
  });
}
for (const name of await readdir(root)) {
  if (/^(?:google|naver)[\da-z]+\.html$/i.test(name)) await cp(resolve(root, name), resolve(root, 'public', name));
}
// A legacy Markdown file references its adjacent _posts image by a relative path.
for (const name of await readdir(resolve(root, '_posts'))) {
  if (/\.(?:png|jpe?g|webp|gif|svg)$/i.test(name)) {
    await cp(resolve(root, '_posts', name), resolve(root, 'public', name));
  }
}
console.log('Copied original public assets, verification files and licensed local font.');

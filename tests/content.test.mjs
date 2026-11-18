import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { parseFragment } from 'parse5';
import { legacyPath, convertLiquid, renderMarkdown } from '../tools/content/build.mjs';

const root=resolve(import.meta.dirname,'..');
const read=path=>readFile(resolve(root,path),'utf8');
const registry=JSON.parse(await read('generated/registry.json'));
const migration=JSON.parse(await read('generated/migration-report.json'));
const byId=id=>{const item=registry.articles.find(a=>a.id===id);assert.ok(item,`missing article ${id}`);return item;};
function elements(html) {
  const result=[]; const walk=node=>{if(node.tagName)result.push(node);for(const child of node.childNodes||[])walk(child);};
  walk(parseFragment(html)); return result;
}
const props=node=>Object.fromEntries((node.attrs||[]).map(a=>[a.name,a.value]));
const fixture=()=>({id:'fixture',sourcePath:'_posts/fixture.md',title:'Fixture',html:'',headings:[],plainText:'',related:[]});

test('every current public original has one unchanged source digest and registry entry',async()=>{
  const filenames=(await readdir(resolve(root,'_posts'))).filter(f=>f.endsWith('.md'));
  assert.equal(registry.articles.length,113);
  assert.equal(new Set(registry.articles.map(a=>a.path)).size,registry.articles.length);
  for(const name of filenames){
    const source=await read('_posts/'+name);
    if(/^published:\s*false\s*$/m.test(source))continue;
    const row=migration.articles.find(a=>a.sourcePath==='_posts/'+name);
    assert.ok(row,name);assert.equal(row.sourceDigest,createHash('sha256').update(source).digest('hex'));
    assert.ok(registry.articles.some(a=>a.sourcePath===row.sourcePath));
  }
});

test('canonical URLs preserve filename case, punctuation and explicit slash policy',()=>{
  assert.equal(legacyPath('2023-01-02-BERT.md',{categories:'PaperReview'}),'/paperreview/2023/01/02/BERT.html');
  assert.equal(legacyPath('2024-01-11-args&kwargs.md',{categories:'python'}),'/python/2024/01/11/args&kwargs.html');
  assert.equal(byId('book-spine').path,'/research/in-browser-spine-reader');
  assert.equal(byId('thinkingcap').path,'/research/thinkingcap-2b/');
});

test('Liquid conversion resolves post identities and fails unknown references',()=>{
  const targets=new Map([['2026-01-02-test',{path:'/research/test/'}]]);
  assert.equal(convertLiquid("[figure]({{ '/assets/a.png' | relative_url }}) [post]({% post_url 2026-01-02-test %})",targets),'[figure](/assets/a.png) [post](/research/test/)');
  assert.throws(()=>convertLiquid('{% post_url missing %}',targets),/Unknown post_url/);
  const server=byId('inference-server');
  assert.ok(!server.html.includes('{% post_url'));
  const links=elements(server.html).filter(n=>n.tagName==='a').map(n=>props(n).href);
  assert.ok(links.some(href=>href?.includes('fast-inference-via-speculative-decoding')));
  assert.ok(links.some(href=>href?.includes('dflash-block-diffusion-speculative-decoding')));
});

test('known stale category links are repaired through the explicit mapping, with the NMS anchor intact',()=>{
  const overfeat=byId('overfeat');
  const links=elements(overfeat.html).filter(n=>n.tagName==='a').map(n=>props(n).href);
  assert.ok(links.includes(byId('object-detection-basics').path));
  assert.ok(links.includes(byId('r-cnn').path+'#h-nmsnon-maximum-suppression'));
  assert.ok(elements(byId('r-cnn').html).some(n=>props(n).id==='h-nmsnon-maximum-suppression'));
  assert.equal(migration.repairedLinks.length,2);
});

test('explicit section IDs, old h-prefixed links, body contents and unique IDs survive',()=>{
  for(const article of registry.articles){
    const ids=elements(article.html).map(n=>props(n).id).filter(Boolean);
    assert.equal(ids.length,new Set(ids).size,article.sourcePath);
    for(const heading of article.headings)assert.ok(ids.includes(heading.id),heading.id);
    for(const alias of Object.keys(article.legacyHeadingIds||{}))assert.ok(ids.includes(alias),alias);
    assert.ok(!elements(article.html).some(n=>n.tagName==='h1'),article.sourcePath);
  }
  const thinking=byId('thinkingcap');
  assert.ok(thinking.html.includes('id="motivation"'));
  assert.ok(thinking.html.includes('id="h-motivation"'));
  assert.ok(thinking.html.includes('1. Small reasoners overthink'));
});

test('active HTML is removed while mathematical formulas and research tables remain',()=>{
  const thinking=byId('thinkingcap');
  const tags=elements(thinking.html).map(n=>n.tagName);
  assert.ok(!tags.some(t=>['script','style','iframe','object','embed'].includes(t)));
  assert.ok(tags.includes('math'),'KaTeX MathML is missing');
  assert.ok(tags.includes('table'),'research tables are missing');
  assert.ok(thinking.html.includes('class="katex"'));
  assert.ok(!thinking.html.includes('katex-error'));
  assert.ok(byId('kodistill').html.includes('/assets/research/kodistill-vl-2b/data/benchmarks.csv'));
});

test('code is escaped, highlighted once, and carries both light and dark token colors',()=>{
  const codeArticles=registry.articles.filter(a=>a.html.includes('class="shiki'));
  assert.equal(codeArticles.length,63);
  assert.ok(codeArticles.some(a=>a.html.includes('--shiki-dark')));
  for(const article of codeArticles)assert.ok(article.html.includes('tabindex="0"'));
});

test('base64 image is extracted byte-for-byte to a public hashed asset',async()=>{
  assert.equal(migration.extractedImages.length,1);
  const image=migration.extractedImages[0];
  const source=await read(image.sourcePath);
  const base64=/data:image\/[^;]+;base64,([A-Za-z0-9+/=]+)/.exec(source)?.[1];
  assert.ok(base64);
  const original=Buffer.from(base64,'base64');
  const copied=await readFile(resolve(root,'public'+image.path));
  assert.deepEqual(copied,original);
  const article=registry.articles.find(a=>a.sourcePath===image.sourcePath);
  assert.ok(article.html.includes(image.path));assert.ok(!article.html.includes('data:image/'));
});

test('renderer rejects executable attributes and preserves safe raw details, tables and literal code',async()=>{
  const article=fixture();
  const output=String(await renderMarkdown('## Test {#explicit}\n\n<script>window.hacked=true</script>\n\n<style>body{display:none}</style>\n\n<a href="javascript:alert(1)" onclick="alert(1)">unsafe</a>\n\n<details><summary>More</summary><table><tr><td>42</td></tr></table></details>\n\n```html\n<script>literal example</script>\n```',article));
  const nodes=elements(output);
  assert.ok(!nodes.some(n=>['script','style'].includes(n.tagName)));
  assert.ok(nodes.some(n=>n.tagName==='details'));assert.ok(nodes.some(n=>n.tagName==='table'));
  assert.ok(!nodes.some(n=>Object.entries(props(n)).some(([key,value])=>key.startsWith('on')||value.startsWith('javascript:'))));
  const text=node=>node.nodeName==='#text'?node.value:(node.childNodes||[]).map(text).join('');
  assert.ok(nodes.filter(n=>n.tagName==='code').some(n=>text(n).includes('<script>literal example</script>')));
  assert.ok(article.headings.some(h=>h.id==='explicit'));
  assert.ok(output.includes('id="h-explicit"'));
});

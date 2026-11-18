import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { generateRelated, validateKnowledge } from '../tools/knowledge/related.mjs';

const concepts = [
  {id:'ocr',label:'OCR',description:'문자 인식',aliases:[],broaderIds:[]},
  {id:'retrieval',label:'검색',description:'문서 검색',aliases:[],broaderIds:[]},
  {id:'rag',label:'RAG',description:'검색 증강',aliases:[],broaderIds:[]},
];
const article = (id, refs=[{id:'ocr',role:'about'}], extra={}) => ({id,path:`/${id}/`,title:id,description:'',plainText:'',html:'',publishedAt:'2026-11-01',projectIds:[],conceptRefs:refs,...extra});
const about = id => ({id,role:'about'});
const applies = id => ({id,role:'applies'});

test('generation is deterministic and does not mutate source data', () => {
  const source = [article('first'),article('second')], before=JSON.stringify(source);
  assert.deepEqual(generateRelated(source,concepts,[]),generateRelated(source,concepts,[]));
  assert.equal(JSON.stringify(source),before);
});

test('unknown concepts, roles, duplicate ids, and hierarchy cycles fail the build', () => {
  assert.throws(()=>generateRelated([article('a',[about('missing')])],concepts,[]),/unknown concept/);
  assert.throws(()=>generateRelated([article('a',[{id:'ocr',role:'mentions'}])],concepts,[]),/invalid concept role/);
  assert.throws(()=>generateRelated([article('a'),article('a')],concepts,[]),/duplicate article/);
  const cyclic=structuredClone(concepts);cyclic[0].broaderIds=['retrieval'];cyclic[1].broaderIds=['ocr'];
  assert.throws(()=>generateRelated([],cyclic,[]),/hierarchy cycle/);
});

test('same concept can have both approved roles without duplicating a role', () => {
  assert.equal(validateKnowledge([article('a',[about('ocr'),applies('ocr')])],concepts,[]),true);
  assert.throws(()=>validateKnowledge([article('a',[about('ocr'),about('ocr')])],concepts,[]),/duplicate concept role/);
});

test('draft, archived and unlisted pages are absent from all public recommendations', () => {
  const source=[article('a'),article('b'),article('hidden',undefined,{publicationState:'draft'}),article('archived',undefined,{publicationState:'archived'}),article('unlisted',undefined,{publicationState:'unlisted'})];
  const result=generateRelated(source,concepts,[]);
  assert.deepEqual(result.map(a=>a.id),['a','b']);
  assert.deepEqual(result[0].related.map(r=>r.id),['b']);
});

test('a shared broad category or incidental vocabulary never produces a fake relation', () => {
  const result=generateRelated([article('a',[about('rag')],{plainText:'same common words'}),article('b',[about('rag')],{plainText:'same common words'})],concepts,[]);
  assert.equal(result[0].related.length,0);
});

test('applied and explanatory articles receive directional reasons, not research lineage', () => {
  const result=generateRelated([article('explanation'),article('implementation',[applies('ocr')])],concepts,[]);
  assert.equal(result[0].related[0].reason,'적용 사례 · OCR');
  assert.equal(result[1].related[0].reason,'개념 설명 · OCR');
  assert.doesNotMatch(result[0].related[0].reason,/선행|후속/);
});

test('only explicit series order establishes next and previous reading', () => {
  const result=generateRelated([article('a'),article('b')],concepts,[{id:'series',orderedArticleIds:['b','a']}]);
  assert.equal(result[0].seriesOrder,2);
  assert.equal(result[0].related[0].reason,'같은 시리즈의 이전 글');
  assert.equal(result[1].related[0].reason,'같은 시리즈의 다음 글');
  assert.throws(()=>generateRelated([article('a')],concepts,[{id:'s1',orderedArticleIds:['a']},{id:'s2',orderedArticleIds:['a']}]),/multiple series/);
});

test('internal links establish citations and preserve the direction of their evidence', () => {
  const result=generateRelated([article('a',[about('retrieval')],{html:'<a href="https://dongryeollee1.github.io/b/#section">원문</a>'}),article('b')],concepts,[]);
  assert.equal(result[0].related[0].reason,'본문에서 연결한 글');
  assert.equal(result[1].related[0].reason,'이 글을 본문에서 연결한 글');
});

test('adding a new article refreshes old recommendations and removal clears stale edges', () => {
  const first=generateRelated([article('a')],concepts,[]);
  assert.equal(first[0].related.length,0);
  const second=generateRelated([article('a'),article('new',[applies('ocr')])],concepts,[]);
  assert.equal(second[0].related[0].id,'new');
  const third=generateRelated([second[0]],concepts,[]);
  assert.equal(third[0].related.length,0);
});

test('pin and exclude are directional, valid, and cannot conflict', () => {
  const source=[article('a'),article('b'),article('c',[about('retrieval')])];
  const result=generateRelated(source,concepts,[],{pins:{a:['c']},excludes:{a:['b']}});
  assert.deepEqual(result[0].related.map(r=>r.id),['c']);
  assert.ok(result[1].related.some(r=>r.id==='a'));
  assert.throws(()=>generateRelated(source,concepts,[],{pins:{a:['missing']}}),/unknown override target/);
  assert.throws(()=>generateRelated(source,concepts,[],{pins:{a:['b']},excludes:{a:['b']}}),/pin\/exclude conflict/);
  assert.throws(()=>generateRelated(source,concepts,[],{pins:{a:['a']}}),/self pins/);
  assert.throws(()=>generateRelated([...source,article('private',undefined,{publicationState:'draft'})],concepts,[],{pins:{a:['private']}}),/non-public override/);
});

test('real migration metadata covers all 113 articles with valid project and series references', () => {
  const json = path => JSON.parse(readFileSync(new URL(path,import.meta.url),'utf8'));
  const metadata=json('../content/article-metadata.json');
  const articles=Object.values(metadata);
  assert.equal(articles.length,113);
  assert.ok(articles.every(a=>a.description.trim() && a.language && a.contentType && a.conceptRefs.length));
  assert.equal(validateKnowledge(articles,json('../content/knowledge/concepts.json'),json('../content/knowledge/series.json'),json('../content/knowledge/overrides.json'),json('../content/knowledge/projects.json')),true);
  const result=generateRelated(articles,json('../content/knowledge/concepts.json'),json('../content/knowledge/series.json'),json('../content/knowledge/overrides.json'));
  assert.ok(result.every(a=>a.related.length<=4&&!a.related.some(r=>r.id===a.id)));
  assert.ok(result.find(a=>a.id==='reranking').related.some(r=>r.id==='book-spine'));
  assert.ok(result.find(a=>a.id==='qlora').related.some(r=>r.id==='qlora-troubleshooting'));
  assert.equal(result.find(a=>a.id==='kodistill').related[0].id,'book-spine');
});

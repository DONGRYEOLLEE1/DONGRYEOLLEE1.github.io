import assert from 'node:assert/strict';
import { buildGraph, loadInputs } from './build-related.mjs';

const {dataset,vocabulary,registry} = await loadInputs();
const clone = value => JSON.parse(JSON.stringify(value));
const graph = buildGraph(dataset,vocabulary,registry);
assert.deepEqual(graph,buildGraph(dataset,vocabulary,registry),'Same inputs must yield the same output');

const unknown = clone(registry);
unknown.articles[0].concepts[0].id = 'not-a-real-concept';
assert.throws(()=>buildGraph(dataset,vocabulary,unknown),/Unknown concept/);
const duplicate = clone(registry);
duplicate.articles[0].concepts.push(duplicate.articles[0].concepts[0]);
assert.throws(()=>buildGraph(dataset,vocabulary,duplicate),/duplicate ID/);
const cycle = clone(vocabulary);
cycle.concepts[0].broaderIds = [cycle.concepts[1].id];
cycle.concepts[1].broaderIds = [cycle.concepts[0].id];
assert.throws(()=>buildGraph(dataset,cycle,registry),/cycle/);
const self = clone(vocabulary);
self.concepts[0].broaderIds = [self.concepts[0].id];
assert.throws(()=>buildGraph(dataset,self,registry),/self broader/);

const draft = clone(registry);
draft.articles.find(a=>a.id==='book-spine').publicationState = 'draft';
const draftGraph = buildGraph(dataset,vocabulary,draft);
assert.equal(draftGraph.perArticle['book-spine'],undefined,'Draft article must not have public output');
assert(!Object.values(draftGraph.perArticle).some(r=>r.targets.some(t=>t.id==='book-spine')),'Draft must not be recommended by other articles');

const application = graph.perArticle.reranking.targets.find(t=>t.id==='book-spine');
assert.equal(application.reasonCode,'application-of');
assert(application.evidence.some(e=>e.conceptId==='reranking'&&e.sourceRole==='about'&&e.targetRole==='applies'));
assert.equal(graph.perArticle.thinkingcap.targets.length,0,'No shared specific concept or explicit series must yield no recommendations');
assert(Object.entries(graph.perArticle).every(([id,r])=>r.targets.every(t=>t.id!==id)),'Self recommendations must not occur');

const expandedDataset = clone(dataset);
const expandedRegistry = clone(registry);
expandedDataset.posts.push({id:'test-new-ocr',title:'검증용 OCR 글',type:'note',language:'ko',date:'2026-11-18',topics:['korean-ocr'],description:'새 글 추가 검증용 데이터입니다.',sections:[]});
expandedRegistry.articles.push({id:'test-new-ocr',publicationState:'published',concepts:[{id:'korean-ocr',role:'applies'}]});
const expanded = buildGraph(expandedDataset,vocabulary,expandedRegistry);
assert(!graph.perArticle.kodistill.targets.some(t=>t.id==='test-new-ocr'));
assert(expanded.perArticle.kodistill.targets.some(t=>t.id==='test-new-ocr'),'Adding an article must also update existing article recommendations');

console.log(JSON.stringify({status:'passed',checks:['deterministic output','unknown concept','duplicate concept','hierarchy cycle','self hierarchy','draft removal','evidence-based application reason','empty recommendations','no self recommendation','new article updates existing recommendations'],articles:graph.manifest.articleCount}));

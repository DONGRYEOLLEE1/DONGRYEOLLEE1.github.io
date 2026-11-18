import {readFile,writeFile} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';

const here=dirname(fileURLToPath(import.meta.url));
const repo=resolve(here,'../..');
await promisify(execFile)(process.execPath,[resolve(here,'knowledge/build-related.mjs')],{cwd:repo});
const read=name=>readFile(resolve(here,name),'utf8');
const assets={};
for(const [key,path] of [
  ['thinkingcap-figure','assets/research/thinkingcap-2b/figures/fig01_tradeoff.png'],
  ['kodistill-figure','assets/research/kodistill-vl-2b/figures/benchmark_comparison.png'],
  ['book-spine-photo','assets/research/in-browser-spine-reader/figures/fig02_sample_result.png']
]) assets[key]='data:image/png;base64,'+(await readFile(resolve(repo,path))).toString('base64');

const searchIcon='<svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.8" stroke="currentColor" stroke-width="1.6"/><path d="m16 16 5 5" stroke="currentColor" stroke-width="1.6"/></svg>';
const font=(await readFile(resolve(here,'fonts/PretendardVariable.woff2'))).toString('base64');
const license=await read('fonts/OFL.txt');
const fontFace=`/* Pretendard Variable 1.3.9\n${license.replaceAll('*/','* /')}\n*/\n@font-face{font-family:"Pretendard";font-style:normal;font-weight:45 920;font-display:swap;src:url(data:font/woff2;base64,${font}) format("woff2");}`;
const css=fontFace+await read('styles.css')+'\n'+await read('typography.css');
const chrome=`<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>동렬 · 기술블로그 디자인 시안</title><style>${css}</style></head><body data-theme="light"><a class="skip-link" href="#app">본문 바로가기</a><div class="preview-note">디자인 시안 · 실제 글 10편의 발췌·요약으로 구성했습니다.</div><header class="site-header"><div class="header-inner"><a class="wordmark" href="#/home" aria-label="동렬 홈">동렬</a><nav class="nav-links" aria-label="기본 메뉴"><a href="#/articles" data-nav="articles">글</a><a href="#/topics" data-nav="topics">주제</a><a href="#/projects" data-nav="projects">프로젝트</a><a href="#/about" data-nav="about">소개</a></nav><div class="header-actions"><button class="icon-button" data-open-search aria-label="검색 열기">${searchIcon}<kbd class="desktop-hint">⌘K</kbd></button><button class="icon-button" data-theme-toggle aria-label="밝은 테마와 어두운 테마 전환"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M20 15.2A8.6 8.6 0 0 1 8.8 4a8.6 8.6 0 1 0 11.2 11.2Z" stroke="currentColor" stroke-width="1.5"/></svg></button><button class="icon-button mobile-menu-button" data-menu aria-label="메뉴 열기" aria-expanded="false">☰</button></div></div><nav id="mobile-menu" class="mobile-nav" hidden aria-label="모바일 메뉴"><a href="#/articles">글</a><a href="#/topics">주제</a><a href="#/projects">프로젝트</a><a href="#/about">소개</a></nav></header><main id="app"></main><footer class="site-footer"><div class="footer-inner"><div><strong>동렬</strong><span>기술 블로그</span></div><div><a href="https://github.com/DONGRYEOLLEE1" target="_blank" rel="noopener">GitHub ↗</a><a href="https://huggingface.co/drlee1" target="_blank" rel="noopener">Hugging Face ↗</a><a href="#/about">소개</a></div></div></footer><dialog id="search-dialog" class="search-dialog" aria-labelledby="search-title"><div class="dialog-top"><div class="dialog-title"><h2 id="search-title">글 검색</h2><button class="close-dialog" data-close-search aria-label="검색 닫기">닫기 ✕</button></div><label class="search-input-wrap">${searchIcon}<input id="global-search-input" type="search" aria-label="검색어" placeholder="제목이나 기술 개념을 입력하세요" autocomplete="off"></label></div><div id="search-results" class="search-results" aria-live="polite"></div><div class="dialog-footer"><span>Enter로 전체 검색 결과를 엽니다.</span><span class="keyboard-hint"><kbd>Tab</kbd> 링크 이동</span></div></dialog><dialog id="image-dialog" class="image-dialog" aria-label="연구 도표 확대"><div class="image-dialog-toolbar"><span>연구 도표 · 원본 비율</span><button data-close-image>닫기 ✕</button></div><img alt="연구의 실제 결과 도표"><p class="image-caption">전체 평가 조건은 연구 원문을 함께 확인하세요.</p></dialog>`;
const relatedChrome='<dialog id="related-dialog" class="related-dialog" aria-labelledby="related-dialog-title"><header class="related-dialog-header"><h2 id="related-dialog-title">관련 글</h2><button type="button" data-close-related aria-label="관련 글 닫기">닫기 ✕</button></header><div class="related-dialog-content"></div></dialog>';
const sources=['window.PREVIEW_ASSETS = '+JSON.stringify(assets)+';window.PREVIEW_ASSETS.thinkingcap=window.PREVIEW_ASSETS["thinkingcap-figure"];window.PREVIEW_ASSETS.kodistill=window.PREVIEW_ASSETS["kodistill-figure"];',await read('data.js'),await read('knowledge/related.generated.js'),await read('secondary.js'),await read('app.js')];
const html=chrome+relatedChrome+sources.map(s=>'<script>'+s.replaceAll('</script','<\\/script')+'</script>').join('')+'</body></html>';
await writeFile(resolve(here,'index.html'),html);
console.log(JSON.stringify({file:'docs/preview/index.html',bytes:Buffer.byteLength(html),sourceResearchImages:3,embeddedFont:'Pretendard Variable 1.3.9'}));

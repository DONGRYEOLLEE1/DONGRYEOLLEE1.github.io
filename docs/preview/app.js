(() => {
  'use strict';
  const data = window.BLOG_DATA;
  const app = document.getElementById('app');
  const searchDialog = document.getElementById('search-dialog');
  const searchInput = document.getElementById('global-search-input');
  const searchResults = document.getElementById('search-results');
  const typeNames = {research:'연구','paper-review':'논문 리뷰',implementation:'구현기',troubleshooting:'문제 해결',note:'학습 노트'};
  let lastFocus, lastRelatedFocus, relationSourceId, searchTimer, selectedNode, hoverNode, lastRoute='';
  const esc = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const topic = id => data.topics.find(t=>t.id===id);
  const date = value => String(value).replaceAll('-','.');
  const asset = key => window.PREVIEW_ASSETS[key] || '';
  const articleLink = (p, cls='') => `<a class="${cls}" href="#/article/${p.id}">${esc(p.title)}</a>`;
  const topicLink = id => `<a class="chip" href="#/topic/${id}">${esc(topic(id)?.label || id)}</a>`;
  const meta = p => `<span>${typeNames[p.type] || '학습 노트'}</span><span>${date(p.date)}</span>${p.language==='en'?'<span class="language-label" aria-label="영문 글">EN</span>':''}`;
  const renderPostRow = p => `<article class="post-row"><div class="post-main"><div class="post-meta">${meta(p)}${relatedTrigger(p)}</div>${articleLink(p,'post-title')}<p class="post-description">${esc(p.description)}</p><div class="topic-chips">${p.topics.slice(0,3).map(topicLink).join('')}</div></div><a class="row-arrow" href="#/article/${p.id}" aria-label="${esc(p.title)} 읽기">↗</a></article>`;
  const ctx = {data,esc,articleLink,topicLink,renderPostRow,asset};
  const featured = key => data.posts.find(p=>p.id.includes(key));
  function home(){
    const lead=data.posts.find(p=>p.id==='book-spine');
    const stories=[featured('thinkingcap'),featured('kodistill'),data.posts.find(p=>p.id==='inference-server')];
    return `<div class="page-shell home-page"><h1 class="editorial-intro"><strong>동렬의 기술 블로그.</strong> LLM 학습과 추론, 에이전트 개발을 기록합니다. 직접 수행한 실험과 구현 과정을 정리합니다.</h1><section class="editorial-feature" aria-label="최근 연구"><a class="feature-photo" href="#/article/${lead.id}"><img src="${asset('book-spine-photo')}" alt="책등 검출 연구의 실제 실험 이미지. 책마다 검출 상자와 번호가 표시되어 있습니다."></a><div class="feature-copy"><div class="card-meta">${meta(lead)}</div><h2 class="feature-title">${articleLink(lead)}</h2><p class="feature-description">${esc(lead.description)}</p><a class="feature-read" href="#/article/${lead.id}">글 읽기 ↗</a></div></section><section class="editorial-secondary" aria-label="다른 연구와 구현 기록">${stories.map(p=>`<article class="editorial-story"><div class="card-meta">${meta(p)}</div><h2>${articleLink(p)}</h2><p class="description">${esc(p.description)}</p><a class="feature-read" href="#/article/${p.id}">읽기 ↗</a></article>`).join('')}</section><div class="home-lower"><section><div class="section-heading"><h2>최근 글</h2><a href="#/articles">전체 글 보기 ↗</a></div><div class="post-list">${[...data.posts].sort((a,b)=>b.date.localeCompare(a.date)).filter(p=>!stories.includes(p)&&p!==lead).slice(0,4).map(renderPostRow).join('')}</div></section><aside class="home-topics"><div class="section-heading"><h2>주제</h2></div><div class="topic-list">${data.topics.filter(t=>['reranking','qlora','korean-ocr','speculative-decoding','reasoning-efficiency'].includes(t.id)).map(t=>`<a href="#/topic/${t.id}"><span>${esc(t.label)}</span><span>↗</span></a>`).join('')}</div><a class="arrow-link" href="#/topics">모든 주제 보기 ↗</a></aside></div></div>`;
  }
  function matchPosts(q='',posts=data.posts){
    const normalized=q.trim().toLowerCase();
    if(!normalized)return [...posts];
    return posts.filter(p=>[p.title,p.description,...(p.sections||[]).flatMap(s=>s.paragraphs||[]),...p.topics.flatMap(id=>[id,topic(id)?.label,...(topic(id)?.aliases||[])])].join(' ').toLowerCase().includes(normalized));
  }
  const relatedTrigger = p => p.related?.length ? `<button class="card-related-trigger" type="button" data-related-open="${p.id}" aria-label="${esc(p.title)} 관련 글 보기" aria-haspopup="dialog" aria-expanded="false" aria-controls="related-dialog">···</button>` : '';
  function archiveCard(p){
    const imageKey={thinkingcap:'thinkingcap-figure',kodistill:'kodistill-figure','book-spine':'book-spine-photo'}[p.id];
    return `<article class="archive-card ${imageKey?'':'archive-card--text'}"><header class="archive-card-meta">${relatedTrigger(p)}<time datetime="${p.date}">${date(p.date)}</time><span>${typeNames[p.type]}</span></header>${imageKey?`<a class="archive-card-media ${p.id==="book-spine"?"is-photo":""}" href="#/article/${p.id}"><img src="${asset(imageKey)}" alt="${esc(p.shortTitle)}의 실제 연구 이미지"></a>`:''}<h2 class="archive-card-title">${articleLink(p)}</h2>${imageKey?'':`<p class="archive-card-description">${esc(p.description)}</p>`}</article>`;
  }
  function listing(params,isSearch=false){
    const q=params.get('q')||'',t=params.get('topic')||'',type=params.get('type')||'';
    const requestedView=params.get('view');
    const view=isSearch?'list':['grid','list'].includes(requestedView)?requestedView:(type==='research'?'grid':'list');
    let posts=matchPosts(q).filter(p=>(!t||p.topics.includes(t))&&(!type||p.type===type));
    posts.sort((a,b)=>b.date.localeCompare(a.date));
    const emptySearch=isSearch&&!q;
    const viewLink = target=>{const next=new URLSearchParams(params);next.set('view',target);return '#/articles?'+next;};
    return `<div class="page-shell search-page"><header class="page-heading"><h1 class="page-title">${isSearch?'검색':type==='research'?'연구':'전체 글'}</h1><p class="page-description">${isSearch?'제목, 기술 개념과 본문에서 찾습니다.':type==='research'?'직접 수행한 모델 학습·추론·평가 연구를 모았습니다.':'연구, 구현과 문제 해결 과정을 날짜순으로 정리했습니다.'}</p></header><form id="list-search" class="filter-bar"><input type="hidden" name="view" value="${view}"><label class="search-field"><span aria-hidden="true">⌕</span><input id="list-query" name="q" aria-label="글 검색어" placeholder="제목, 기술 개념, 본문에서 검색" value="${esc(q)}"><button class="button" type="submit">검색</button></label><div class="filter-controls"><label>주제 <select name="topic" aria-label="주제 필터"><option value="">모든 주제</option>${data.topics.map(x=>`<option value="${x.id}" ${x.id===t?'selected':''}>${esc(x.label)}</option>`).join('')}</select></label><label>글 유형 <select name="type" aria-label="글 유형 필터"><option value="">모든 유형</option>${Object.entries(typeNames).map(([id,label])=>`<option value="${id}" ${id===type?'selected':''}>${label}</option>`).join('')}</select></label>${q||t||type?`<a class="resource-link" href="${isSearch?'#/search':'#/articles?view='+view}">모두 해제</a>`:''}</div></form><div class="result-line"><span>${emptySearch?'검색어를 입력해 주세요':`${q?`‘${esc(q)}’ · `:''}${posts.length}개의 글`}</span><div class="archive-toolbar"><span class="muted">최신순</span>${isSearch?'':`<nav class="view-switch" aria-label="목록 보기 방식"><a href="${viewLink('grid')}" class="${view==='grid'?'active':''}" aria-current="${view==='grid'?'true':'false'}">그리드</a><a href="${viewLink('list')}" class="${view==='list'?'active':''}" aria-current="${view==='list'?'true':'false'}">목록</a></nav>`}</div></div><div class="${view==='grid'?'archive-grid':'post-list'}">${emptySearch?'<div class="empty-state"><h2>어떤 내용이 궁금한가요?</h2><p>QLoRA, Reranking, ThinkingCap처럼 검색해 보세요.</p><a href="#/topics">주제로 찾아보기 ↗</a></div>':posts.length?posts.map(view==='grid'?archiveCard:renderPostRow).join(''):'<div class="empty-state"><h2>조건에 맞는 글이 없어요.</h2><p>검색어를 바꾸거나 필터를 해제해 보세요.</p><a href="#/articles">전체 글 보기 ↗</a></div>'}</div></div>`;
  }
  const rerankDiagram = `<svg viewBox="0 0 700 230" role="img" aria-label="검색 후보를 찾은 뒤 재정렬하여 질문에 더 관련 있는 문서를 앞에 배치하는 과정"><rect width="700" height="230" rx="6" fill="#eeece4"/><text x="32" y="34" font-family="sans-serif" font-size="11" letter-spacing="1.5" fill="#69665e">검색과 재정렬의 처리 순서</text><rect x="32" y="64" width="150" height="96" rx="6" fill="#fff" stroke="#d0cdc4"/><text x="107" y="104" text-anchor="middle" font-family="sans-serif" font-size="17" fill="#292824">사용자의 질문</text><text x="107" y="133" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#6c6961">query</text><path d="M195 112h49m-9-6 9 6-9 6" stroke="#4d4b43" stroke-width="2" fill="none"/><rect x="260" y="64" width="170" height="96" rx="6" fill="#fff" stroke="#d0cdc4"/><text x="345" y="104" text-anchor="middle" font-family="sans-serif" font-size="17" fill="#292824">후보 문서 검색</text><text x="345" y="133" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#6c6961">keyword / dense retrieval</text><path d="M443 112h49m-9-6 9 6-9 6" stroke="#4d4b43" stroke-width="2" fill="none"/><rect x="507" y="64" width="161" height="96" rx="6" fill="#4d4b43"/><text x="587" y="104" text-anchor="middle" font-family="sans-serif" font-size="17" fill="white">관련성으로 재정렬</text><text x="587" y="133" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#f1eee4">reranking</text><text x="32" y="202" font-family="sans-serif" font-size="12" fill="#69665e">개념 설명을 위한 도식 · 실제 검색 순위나 평가 수치를 나타내지 않습니다.</text></svg>`;
  function graph(p){
    const related=(p.related||[]).map(r=>({...r,post:data.posts.find(x=>x.id===r.id)})).filter(r=>r.post).slice(0,4);
    if(!related.length)return `<div class="related-empty"><p>아직 연결할 관련 글이 없어요.</p><a href="#/topic/${p.topics[0]}">이 주제의 글 보기 ↗</a></div>`;
    const positions=[[52,46],[214,42],[226,154],[60,168]];
    return `<div class="graph-wrap" id="relationship-widget"><svg viewBox="0 0 280 220" role="group" aria-label="현재 글의 관련 글 선택">${related.map((r,i)=>`<line class="graph-line" x1="138" y1="106" x2="${positions[i][0]}" y2="${positions[i][1]}"/>`).join('')}<g class="graph-node selected" data-node="${p.id}" tabindex="0" role="button" aria-label="현재 글 선택" aria-pressed="true"><circle cx="138" cy="106" r="23"/><text x="138" y="143" text-anchor="middle">현재 글</text></g>${related.map((r,i)=>{const[x,y]=positions[i];const label=r.id==='book-spine'?'책등 OCR':r.post.shortTitle||r.post.title;return `<g class="graph-node" data-node="${r.id}" tabindex="0" role="button" aria-label="${esc(r.post.title)} 선택" aria-pressed="false"><circle class="node-hit" cx="${x}" cy="${y}" r="22"/><circle cx="${x}" cy="${y}" r="12"/><text x="${x}" y="${y+30}" text-anchor="middle">${esc(label.length>18?label.slice(0,17)+'…':label)}</text></g>`}).join('')}</svg><div class="graph-caption"><span>● 현재 글</span><span>○ 연결된 글</span></div><div id="related-preview" class="related-preview"></div></div><a class="rail-all-link" href="#related-reading">목록으로 보기 <span>↓</span></a>`;
  }
  function article(p){
    const isRerank=p.id.includes('rerank'),isThinking=p.id.includes('thinkingcap');
    const secs=p.sections||[];
    const related=(p.related||[]).map(r=>({...r,post:data.posts.find(x=>x.id===r.id)})).filter(r=>r.post);
    return `<article class="article-page"><header class="article-heading"><a class="article-breadcrumb" href="#/articles">글 <span>/</span> ${isRerank?'LLM University':'연구와 실험'}</a><div class="article-meta">${meta(p)}</div><h1 class="article-title" lang="${p.language}">${esc(p.title)}</h1><p class="article-description">${esc(p.description)}</p><div class="topic-chips">${p.topics.map(topicLink).join('')}</div></header><div class="article-grid"><aside class="article-toc"><span class="small-label">목차</span><ol>${secs.map((s,i)=>`<li><a href="#section-${i}" data-section="section-${i}">${esc(s.heading||s.title)}</a></li>`).join('')}<li><a href="#related-reading">이어 읽기</a></li></ol><a href="#/articles" class="toc-back">← 전체 글</a></aside><div class="article-body" lang="${p.language}"><div class="mobile-toc"><details><summary>목차 보기</summary>${secs.map((s,i)=>`<a href="#section-${i}">${esc(s.heading||s.title)}</a>`).join('')}</details></div>${isRerank?`<figure class="concept-figure">${rerankDiagram}<figcaption class="figure-caption">검색과 재정렬은 서로 다른 역할을 합니다.</figcaption></figure>`:''}${isThinking?`<figure><button class="figure-button" data-image="thinkingcap-figure" aria-label="ThinkingCap 결과 도표 확대"><img src="${asset('thinkingcap-figure')}" alt="ThinkingCap 정확도와 추론 토큰 비교 도표"></button><figcaption class="figure-caption">원문 연구의 실제 결과 도표. 그림을 눌러 확대할 수 있습니다.</figcaption></figure>`:''}${!isRerank&&!isThinking&&p.id.includes('kodistill')?`<figure><button class="figure-button" data-image="kodistill-figure" aria-label="KoDistill 도표 확대"><img src="${asset('kodistill-figure')}" alt="KoDistill 평가 비교"></button></figure>`:''}${secs.map((s,i)=>`<section class="reading-section" id="section-${i}"><h2>${esc(s.heading||s.title)}</h2>${(s.paragraphs||[s.body||s.text||'']).filter(Boolean).map(text=>`<p>${esc(text)}</p>`).join('')}${isRerank&&i===0?'<blockquote>더 많은 문서를 찾는 단계와,<br>더 적합한 문서를 고르는 단계를 나눠 생각합니다.</blockquote>':''}${isRerank&&i===1?'<div class="inline-note"><strong>읽는 순서</strong><p>Keyword Search와 Dense Retrieval을 먼저 읽었다면, 검색 후보의 품질과 최종 순위를 나눠 이해할 수 있습니다.</p></div>':''}</section>`).join('')}<section class="article-foot" id="related-reading"><div class="section-heading"><h2>이어서 읽어보세요.</h2><span class="small-label">관련 글</span></div><div class="related-bottom">${related.map(r=>`<a class="related-row" href="#/article/${r.id}"><div><span class="small-label">${esc(r.reason)}</span><h3>${esc(r.post.title)}</h3></div><span>↗</span></a>`).join('')||'<p class="muted">아직 연결할 관련 글이 없어요.</p>'}</div><div class="author-note"><strong>동렬</strong><span>언어모델을 연구하고, 서비스로 만듭니다.</span><a href="#/about">작성자 소개 ↗</a></div></section></div><aside class="related-rail"><div class="rail-heading"><span class="small-label">관련 글</span><h2>함께 읽을 글</h2><p>공통 개념과 적용 사례를 확인하세요.</p></div>${graph(p)}</aside></div></article>`;
  }
  function currentPost(){if(document.getElementById('related-dialog')?.open)return data.posts.find(p=>p.id===relationSourceId);const route=location.hash.startsWith('#/')?location.hash.slice(2):lastRoute;const id=route.split('?')[0].split('/')[1];return data.posts.find(p=>p.id===id);}
  function updatePreview(id){
    const p=currentPost(),target=data.posts.find(x=>x.id===id);if(!p||!target)return;
    const root=document.getElementById('related-preview');if(!root)return;
    const reason=id===p.id?'현재 읽고 있는 글입니다.':(p.related||[]).find(r=>r.id===id)?.reason||'연결된 글';
    root.innerHTML=`<div class="related-kicker">${id===p.id?'현재 읽는 글':typeNames[target.type]||'연결된 기록'}</div><h3 class="related-title">${esc(target.shortTitle||target.title)}</h3><p class="related-reason">${esc(reason)}</p>${id!==p.id?`<a class="arrow-link" href="#/article/${id}">글 읽기 ↗</a>`:'<span class="muted preview-current">주변 노드를 선택해 보세요.</span>'}`;
    document.querySelectorAll('.graph-node').forEach(node=>{const on=node.dataset.node===selectedNode;node.classList.toggle('selected',on);node.setAttribute('aria-pressed',String(on));});
  }
  function render(){
    closeRelated(false);
    const raw=location.hash.startsWith('#/')?location.hash.slice(2):lastRoute||'home';
    const [route,query='']=raw.split('?'), params=new URLSearchParams(query), [kind,id]=route.split('/');
    if(location.hash.startsWith('#/')&&route!==lastRoute){lastRoute=route;window.scrollTo(0,0);}
    document.querySelectorAll('.nav-links a').forEach(a=>a.classList.toggle('active',a.dataset.nav===(kind==='article'?'articles':kind==='topic'?'topics':kind)));
    document.getElementById('mobile-menu').hidden=true;
    if(kind==='article'){
      const post=data.posts.find(x=>x.id===id);app.innerHTML=post?article(post):'<div class="page-shell empty-state"><h1>페이지를 찾을 수 없어요.</h1><a href="#/articles">전체 글 보기</a></div>';selectedNode=id;updatePreview(id);
    }else if(kind==='articles'||kind==='search')app.innerHTML=listing(params,kind==='search');
    else if(kind==='topics')app.innerHTML=window.SecondaryViews.topics(ctx);
    else if(kind==='topic')app.innerHTML=window.SecondaryViews.topic(ctx,id);
    else if(kind==='projects')app.innerHTML=window.SecondaryViews.projects(ctx);
    else if(kind==='about')app.innerHTML=window.SecondaryViews.about(ctx);
    else app.innerHTML=home();
    document.title=(kind==='article'?(currentPost()?.title||'글'):({home:'동렬',articles:'전체 글',topics:'주제별로 읽기',topic:topic(id)?.label||'주제',projects:'프로젝트',about:'소개',search:'글 검색'}[kind]||'동렬'))+' · 동렬';
    document.querySelector('.site-footer').hidden=kind==='article'&&false;
  }
  function navigateSearch(form){
    const fd=new FormData(form),q=String(fd.get('q')||'').trim();if([...q].length>200){form.querySelector('#list-query').setCustomValidity('검색어는 200자 이내로 입력해 주세요.');form.reportValidity();return;}
    const params=new URLSearchParams();for(const key of ['q','topic','type','view'])if(fd.get(key))params.set(key,String(fd.get(key)).trim());
    const page=location.hash.startsWith('#/search')?'search':'articles';location.hash=`/${page}${params.size?'?'+params:''}`;
  }
  function openSearch(trigger){const active=trigger||document.activeElement;lastFocus=active?.closest('#related-dialog')?lastRelatedFocus:active;closeRelated(false);if(!searchDialog.open)searchDialog.showModal();document.body.style.overflow='hidden';searchInput.focus();runQuickSearch();}
  function closeSearch(){searchDialog.close();document.body.style.overflow='';lastFocus?.focus();}
  function runQuickSearch(){
    const q=searchInput.value.trim();if(!q){searchResults.innerHTML='<div class="search-empty"><span class="small-label">검색어 예시</span><p>어떤 내용이 궁금한가요?</p><div class="search-suggestions">'+['QLoRA','Reranking','ThinkingCap'].map(x=>`<button type="button" data-suggestion="${x}">${x} ↗</button>`).join('')+'</div></div>';return;}
    if([...q].length>200){searchResults.innerHTML='<p class="empty-state">검색어는 200자 이내로 입력해 주세요.</p>';return;}
    const results=matchPosts(q);searchResults.innerHTML=`<div class="search-count">${results.length}개의 기록을 찾았습니다.</div>`+(results.length?results.slice(0,6).map(p=>`<a class="search-result" href="#/article/${p.id}"><div class="search-result-meta">${typeNames[p.type]} <span>· ${date(p.date)}</span></div><div class="search-result-title">${esc(p.title)} <span>↗</span></div><p class="search-result-excerpt">${esc(p.description)}</p></a>`).join(''):'<div class="empty-state"><p>조건에 맞는 글이 없어요.</p><a href="#/topics">주제로 찾아보기 ↗</a></div>')+`<a class="all-search-results" href="#/search?q=${encodeURIComponent(q)}">전체 결과 보기 <span>↗</span></a>`;
  }
  function openRelated(id,trigger){
    const p=data.posts.find(p=>p.id===id);if(!p)return;
    const dialog=document.getElementById('related-dialog');
    if(dialog.open&&lastRelatedFocus===trigger){closeRelated();return;}
    closeRelated(false);
    relationSourceId=id;lastRelatedFocus=trigger;selectedNode=id;
    const content=graph(p).replace(/<a class="rail-all-link"[\s\S]*?<\/a>$/,'');
    dialog.querySelector('.related-dialog-content').innerHTML=`<p class="related-source-summary">${esc(p.description)}</p>${content}<a class="related-dialog-source" href="#/article/${p.id}">이 글 열기 ↗</a>`;
    trigger.setAttribute('aria-expanded','true');trigger.setAttribute('aria-label',p.title+' 관련 글 닫기');
    dialog.show();updatePreview(id);
    const anchor=trigger.getBoundingClientRect(),box=dialog.getBoundingClientRect();
    dialog.style.left=Math.max(16,Math.min(anchor.left,window.innerWidth-box.width-16))+'px';
    dialog.style.top=Math.max(12,Math.min(anchor.bottom+8,window.innerHeight-box.height-12))+'px';
    dialog.querySelector('[data-close-related]').focus({preventScroll:true});
  }
  function closeRelated(restore=true){const dialog=document.getElementById('related-dialog');if(!dialog?.open)return;dialog.close();if(lastRelatedFocus?.isConnected){lastRelatedFocus.setAttribute('aria-expanded','false');const p=data.posts.find(p=>p.id===relationSourceId);lastRelatedFocus.setAttribute('aria-label',(p?.title||'')+' 관련 글 보기');if(restore)lastRelatedFocus.focus({preventScroll:true});}dialog.querySelector('.related-dialog-content').innerHTML='';relationSourceId=undefined;}
  document.addEventListener('click',e=>{
    const related=e.target.closest('[data-related-open]');if(related){openRelated(related.dataset.relatedOpen,related);return;}
    if(e.target.closest('[data-close-related]')){closeRelated();return;}
    if(document.getElementById('related-dialog').open&&!e.target.closest('#related-dialog'))closeRelated(false);
    const search=e.target.closest('[data-open-search]');if(search){openSearch(search);return;}
    if(e.target.closest('[data-close-search]')){closeSearch();return;}
    const suggest=e.target.closest('[data-suggestion]');if(suggest){searchInput.value=suggest.dataset.suggestion;runQuickSearch();return;}
    const node=e.target.closest('[data-node]');if(node){selectedNode=node.dataset.node;updatePreview(selectedNode);return;}
    const link=e.target.closest('a[href^="#/"]');if(link&&searchDialog.open)closeSearch();if(link&&document.getElementById('related-dialog').open)closeRelated(false);
    const sectionLink=e.target.closest('a[href^="#section-"],a[href="#related-reading"]');if(sectionLink){e.preventDefault();document.querySelector(sectionLink.getAttribute('href'))?.scrollIntoView({behavior:'smooth',block:'start'});}
    const img=e.target.closest('[data-image]');if(img){const d=document.getElementById('image-dialog');d.querySelector('img').src=asset(img.dataset.image);d.showModal();return;}
    if(e.target.closest('[data-close-image]'))document.getElementById('image-dialog').close();
    const role=e.target.closest('[data-role]');if(role){document.querySelectorAll('.role-tabs [data-role]').forEach(b=>{const on=b.dataset.role===role.dataset.role;b.classList.toggle('active',on);b.setAttribute('aria-pressed',String(on));});const rows=[...document.querySelectorAll('.topic-role-row')];rows.forEach(row=>row.hidden=role.dataset.role!=='all'&&!row.dataset.roles.includes(role.dataset.role));const count=rows.filter(row=>!row.hidden).length;const empty=document.getElementById('topic-role-empty');if(empty)empty.hidden=count!==0;const c=document.getElementById('topic-post-count');if(c)c.textContent=count+'편';}
    if(e.target.closest('[data-menu]')){const m=document.getElementById('mobile-menu');m.hidden=!m.hidden;e.target.closest('[data-menu]').setAttribute('aria-expanded',String(!m.hidden));}
    if(e.target.closest('[data-theme-toggle]')){const next=document.body.dataset.theme==='dark'?'light':'dark';document.body.dataset.theme=next;try{localStorage.setItem('lee-preview-theme',next);}catch{};}
  });
  document.addEventListener('submit',e=>{if(e.target.id==='list-search'){e.preventDefault();navigateSearch(e.target);}if(e.target.id==='topic-search-form')e.preventDefault();});
  document.addEventListener('change',e=>{if(e.target.closest('#list-search')&&e.target.tagName==='SELECT')navigateSearch(e.target.form);});
  document.addEventListener('input',e=>{if(e.target.id==='topic-filter'){const q=e.target.value.trim().toLowerCase();const tiles=[...document.querySelectorAll('.topic-tile')];tiles.forEach(t=>t.hidden=!t.dataset.search.toLowerCase().includes(q));const count=tiles.filter(t=>!t.hidden).length;const result=document.getElementById('topic-result-count');if(result)result.textContent=count+'개 주제';const empty=document.getElementById('topic-empty');if(empty)empty.hidden=count!==0;}if(e.target.id==='list-query')e.target.setCustomValidity('');});
  document.addEventListener('keydown',e=>{
    if(e.key==='Escape'&&!e.isComposing&&document.getElementById('related-dialog').open){e.preventDefault();closeRelated();return;}
    if(e.key==='Escape'&&!e.isComposing&&searchDialog.open){e.preventDefault();closeSearch();return;}
    if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'&&!e.isComposing&&!e.target.closest('input,textarea,select,[contenteditable=true]')){e.preventDefault();openSearch();}
    const node=e.target.closest('[data-node]');if(node&&['Enter',' '].includes(e.key)){e.preventDefault();selectedNode=node.dataset.node;updatePreview(selectedNode);}
  });
  searchInput.addEventListener('input',e=>{clearTimeout(searchTimer);if(!e.isComposing)searchTimer=setTimeout(runQuickSearch,180);});
  searchInput.addEventListener('compositionend',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(runQuickSearch,180);});
  searchInput.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.isComposing){e.preventDefault();const q=searchInput.value.trim();if(q&&[...q].length<=200){closeSearch();location.hash='/search?q='+encodeURIComponent(q);}}});
  searchDialog.addEventListener('cancel',e=>{e.preventDefault();closeSearch();});
  document.addEventListener('pointerover',e=>{const n=e.target.closest('[data-node]');if(n&&!document.activeElement?.closest('.related-preview')){hoverNode=n.dataset.node;updatePreview(hoverNode);}});
  document.addEventListener('focusin',e=>{const n=e.target.closest('[data-node]');if(n)updatePreview(n.dataset.node);});
  document.addEventListener('pointerout',e=>{const w=e.target.closest('#relationship-widget');if(w&&!w.contains(e.relatedTarget)&&!w.contains(document.activeElement))updatePreview(selectedNode);});
  document.addEventListener('focusout',e=>{const w=e.target.closest('#relationship-widget');if(w&&!w.contains(e.relatedTarget)&&!w.matches(':hover'))updatePreview(selectedNode);});
  document.getElementById('related-dialog').addEventListener('cancel',e=>{e.preventDefault();closeRelated();});
  window.addEventListener('resize',()=>closeRelated(false));
  window.addEventListener('hashchange',()=>{if(location.hash.startsWith('#/'))render();});
  try{document.body.dataset.theme=localStorage.getItem('lee-preview-theme')||'light';}catch{}
  render();
  window.previewNavigate = route=>{location.hash='/'+route;};
})();

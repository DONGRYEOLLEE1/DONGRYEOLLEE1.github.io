(function () {
  'use strict';

  const asArray = value => Array.isArray(value) ? value : Object.values(value || {});
  const postHref = post => '#/article/' + encodeURIComponent(post.id);
  const topicHref = id => '#/topic/' + encodeURIComponent(id);
  const external = (url, label, esc) => `<a class="resource-link" href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(label)} <span aria-hidden="true">↗</span><span class="sr-only">, 새 탭</span></a>`;
  const searchIcon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="10.8" cy="10.8" r="6.5" stroke="currentColor" stroke-width="1.6"/><path d="m16 16 4.5 4.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';

  function topicRecords(ctx) {
    const posts = asArray(ctx.data.posts);
    return asArray(ctx.data.topics).map(t => ({
      ...t,
      label: t.label || t.name || t.id,
      count: posts.filter(p => (p.topics || []).includes(t.id)).length,
    })).filter(t => t.count > 0);
  }

  function topics(ctx) {
    const { esc } = ctx;
    const topics = topicRecords(ctx);
    return `<section class="page-shell topics-page">
      <header class="page-heading">
        <h1 class="page-title">주제</h1>
        <p class="page-description">기술 개념별로 연구·구현·학습 기록을 모았습니다.</p>
      </header>
      <form class="topic-search search-field" id="topic-search-form" role="search">
        <label class="sr-only" for="topic-filter">주제 이름 찾기</label>
        ${searchIcon}<input id="topic-filter" type="search" placeholder="Reranking, QLoRA, 한국어 OCR" autocomplete="off" maxlength="200">
        <button class="button button-secondary" type="submit">찾기</button>
      </form>
      <div class="section-heading"><h2>전체 주제</h2><span class="muted" id="topic-result-count">${topics.length}개 주제</span></div>
      <div class="topic-grid">
        ${topics.map(t => `<article class="topic-tile" data-search="${esc([t.label,t.id,...(t.aliases||[])].join(' ').toLowerCase())}">
          <h2 class="topic-name"><a href="${topicHref(t.id)}">${esc(t.label)}</a></h2>
          <p>${esc(t.description || '')}</p>
          <div class="topic-tile-footer"><span class="muted">${t.count}편의 글</span><a class="resource-link" href="${topicHref(t.id)}" aria-label="${esc(t.label)} 주제의 글 보기">글 보기 <span aria-hidden="true">↗</span></a></div>
        </article>`).join('')}
      </div>
      <div id="topic-empty" class="empty-state" hidden><h2>일치하는 주제가 없어요.</h2><p class="muted">다른 이름으로 검색하거나 전체 글에서 찾아보세요.</p><a class="resource-link" href="#/articles">전체 글 보기 ↗</a></div>
      <p class="muted topic-search-note">한국어 이름과 영어 별칭으로 같은 주제를 찾을 수 있습니다.</p>
    </section>`;
  }

  function rolesFor(post, slug) {
    const explicit = post.topicRoles && post.topicRoles[slug];
    if (explicit) return Array.isArray(explicit) ? explicit : [explicit];
    if (slug === 'reranking' && /spine|book/i.test(post.id + ' ' + post.title)) return ['applies'];
    return ['about'];
  }

  function topic(ctx, slug) {
    const { esc } = ctx;
    const all = topicRecords(ctx);
    const selected = all.find(t => t.id === slug);
    if (!selected) return `<section class="page-shell"><h1 class="page-title">주제를 찾을 수 없어요.</h1><p class="page-description">주제 목록에서 다른 개념을 살펴보세요.</p><a class="button" href="#/topics">주제 목록 보기</a></section>`;
    const posts = asArray(ctx.data.posts).filter(p => (p.topics || []).includes(slug)).sort((a,b) => String(b.date).localeCompare(String(a.date)));
    const about = posts.filter(p => rolesFor(p,slug).includes('about')).length;
    const applies = posts.filter(p => rolesFor(p,slug).includes('applies')).length;
    const related = all.filter(t => t.id !== slug && posts.some(p => (p.topics || []).includes(t.id))).slice(0,4);
    const aliases = (selected.aliases || []).filter(x => x.toLowerCase() !== selected.label.toLowerCase()).slice(0,3);
    return `<section class="page-shell topic-detail-page">
      <nav class="breadcrumb" aria-label="현재 위치"><a href="#/topics">주제</a><span aria-hidden="true">/</span><span>${esc(selected.label)}</span></nav>
      <header class="topic-header">
        <h1 class="page-title">${esc(selected.label)}</h1>
        ${aliases.length ? `<p class="muted topic-aliases">${aliases.map(esc).join(' · ')}</p>` : ''}
        <p class="page-description">${esc(selected.description || '')}</p>
      </header>
      <div class="topic-detail-divider"></div>
      <div class="section-heading"><h2>연결된 글</h2><span class="muted" id="topic-post-count">${posts.length}편</span></div>
      <div class="role-tabs" role="group" aria-label="연결 역할 선택">
        <button class="active" type="button" data-role="all" aria-pressed="true">전체 <span>${posts.length}</span></button>
        <button type="button" data-role="about" aria-pressed="false">주제로 다룬 글 <span>${about}</span></button>
        <button type="button" data-role="applies" aria-pressed="false">적용 사례 <span>${applies}</span></button>
      </div>
      <div class="post-list topic-post-list">
        ${posts.map(p => `<div class="topic-role-row" data-roles="${rolesFor(p,slug).join(' ')}">${ctx.renderPostRow(p)}</div>`).join('')}
      </div>
      <div id="topic-role-empty" class="empty-state" hidden><h2>이 분류의 글은 아직 없어요.</h2><p class="muted">전체 연결 글에서 이 주제를 살펴보세요.</p><button class="button button-secondary" type="button" data-role="all">전체 연결 글 보기</button></div>
      ${related.length ? `<section class="adjacent-topics"><div class="section-heading"><h2>함께 살펴볼 주제</h2></div><div class="topic-chips">${related.map(t => `<a class="chip" href="${topicHref(t.id)}">${esc(t.label)} <span aria-hidden="true">↗</span></a>`).join('')}</div></section>` : ''}
    </section>`;
  }

  function projects(ctx) {
    const { esc } = ctx;
    const posts = asArray(ctx.data.posts);
    const projects = [
      {
        name: 'ThinkingCap-2B', label: '추론 효율화 · 강화학습', key: 'thinkingcap',
        post: posts.find(p => /thinkingcap/i.test(p.id + ' ' + p.title)),
        purpose: '추론 모델의 종료를 강화학습으로 조정',
        description: '단일 GPU에서 학습 조건을 비교하고, 추론 길이와 정답률을 같은 평가 기준으로 검증했습니다.',
        resources: [['https://github.com/DONGRYEOLLEE1/reducing-think-token','코드'],['https://huggingface.co/drlee1/ThinkingCap-Qwen3.5-2B','공개 모델 B']],
      },
      {
        name: 'KoDistill-VL-2B', label: '한국어 OCR · 비전 언어 모델', key: 'kodistill',
        post: posts.find(p => /kodistill/i.test(p.id + ' ' + p.title)),
        purpose: '한국어 문자 인식을 위한 VLM 후속 학습',
        description: '문자 그대로의 전사와 이중 언어 학습을 연결하고, 한국어·영어 평가에서 나타난 변화와 한계를 정리했습니다.',
        resources: [['https://huggingface.co/drlee1/kodistill-vl-2b','모델']],
      },
    ].filter(p => p.post);
    return `<section class="page-shell projects-page">
      <header class="page-heading">
        <h1 class="page-title">프로젝트</h1>
        <p class="page-description">직접 수행한 연구와 공개한 모델·코드입니다.</p>
      </header>
      <div class="project-grid">
        ${projects.map(project => {
          const image = ctx.asset ? ctx.asset(project.key) : '';
          return `<article class="project-card">
            <a class="project-visual" href="${postHref(project.post)}" aria-label="${esc(project.name)} 연구 글 읽기">
              ${image ? `<img src="${esc(image)}" alt="${esc(project.name)} 연구 결과 도표" loading="lazy">` : `<span>${esc(project.name)}</span>`}
            </a>
            <div class="project-body">
              <h2 class="project-name"><a href="${postHref(project.post)}">${esc(project.name)}</a></h2>
              <h3>${esc(project.purpose)}</h3><p class="muted">${esc(project.description)}</p>
              <div class="project-actions"><a class="resource-link" href="${postHref(project.post)}">연구 글 읽기 <span aria-hidden="true">↗</span></a>${project.resources.map(([url,label]) => external(url,label,esc)).join('')}</div>
            </div>
          </article>`;
        }).join('')}
      </div>
      <section class="project-more"><div class="section-heading"><h2>구현·평가 기록</h2><a class="resource-link" href="#/articles">전체 글 보기 ↗</a></div><p class="muted">각 연구 글에서 실험 조건, 결과 해석과 공개 자료를 확인할 수 있습니다.</p></section>
    </section>`;
  }

  function about(ctx) {
    const { esc } = ctx;
    const posts = asArray(ctx.data.posts);
    const featured = [/thinkingcap/i,/kodistill/i].map(pattern => posts.find(p => pattern.test(p.id + ' ' + p.title))).filter(Boolean);
    return `<section class="page-shell about-page">
      <header class="page-heading"><h1 class="page-title">소개</h1></header>
      <div class="about-layout">
        <aside class="about-aside">
          <h2 class="author-name">동렬</h2><p class="author-role">AI Engineer</p>
          <div class="about-social">${external('https://github.com/DONGRYEOLLEE1','GitHub',esc)}${external('https://huggingface.co/drlee1','Hugging Face',esc)}</div>
        </aside>
        <div class="about-copy">
          <p>언어모델의 학습·추론과 AI 에이전트를 연구하고 개발합니다. 이곳에는 직접 수행한 실험, 구현 과정, 논문을 읽으며 남긴 기록을 정리합니다.</p>
          <section class="about-interests">
            <h3>관심 분야</h3>
            <div class="interest-row"><div><h4>LLM 학습과 추론</h4><p class="muted">소형 모델의 학습, 추론 효율화와 실제 서비스 환경에서의 서빙.</p></div></div>
            <div class="interest-row"><div><h4>AI Agent</h4><p class="muted">도구 사용, 메모리와 맥락 관리, 여러 단계의 작업 수행.</p></div></div>
            <div class="interest-row"><div><h4>평가와 실험</h4><p class="muted">실제 실행 환경을 반영한 성능 측정과 결과 분석.</p></div></div>
          </section>
          <section class="about-selected"><div class="section-heading"><h3>대표 글</h3></div><div class="post-list">${featured.map(p => ctx.renderPostRow(p)).join('')}</div></section>
          <a class="resource-link" href="#/articles">전체 글 보기 ↗</a>
        </div>
      </div>
    </section>`;
  }

  window.SecondaryViews = { topics, topic, projects, about };
})();

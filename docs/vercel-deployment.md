# Vercel 배포와 기존 주소 전환

작성일: 2026-11-18 · Astro 정적 블로그 · 실제 계정 연결과 운영 주소 확정은 소유자가 진행

## 준비된 구성

사이트는 Markdown 원문과 메타데이터를 읽어 글·주제·시리즈 페이지를 정적으로 만든다. 관련 글과 작은 그래프의 데이터도 배포 전에 계산한다. Vercel Functions, 온톨로지 전용 DB, 임베딩 API 키, LLM API 키는 현재 배포에 필요하지 않다. 이는 Vercel 빌드 기준이다. 연관 글 추천을 갱신하는 `npm run recommendations:sync`는 발행 서버에서만 실행하고, 그때 쓰는 OpenRouter·Supabase 키는 Vercel 환경 변수에 넣지 않는다. 본문 검색은 함께 배포하는 Pagefind 색인을 사용한다.

| Vercel 항목 | 설정 |
|---|---|
| Git Repository | DONGRYEOLLEE1/tech-blog |
| Framework Preset | Astro |
| Root Directory | 저장소 루트 . |
| Install Command | npm ci |
| Build Command | npm run build |
| Output Directory | dist |
| Node.js Version | 24.x. package.json과 일치 |
| Production Branch | `main` |
| 필수 외부 서비스 | 없음 |

빌드는 콘텐츠·관계 검증, 자산 준비, HTML 생성, 본문 검색 색인, 전수 검사를 모두 포함한다. astro build만 호출하면 검색 색인과 이전 검증이 빠지므로 위 명령을 유지한다. vercel.json과 lockfile을 함께 Git에 반영해야 한다.

## 사용자가 진행할 연결

Vercel의 New Project에서 이 저장소를 Import한다. GitHub 연결 권한을 요구하면 이 저장소 접근을 허용한다. 위 프로젝트 설정을 확인한 뒤 배포한다. Git 연동 후에는 기본적으로 `main`의 변경은 Production으로, 다른 브랜치의 변경은 Preview로 Vercel에서 빌드한다. [Vercel Git 배포](https://vercel.com/docs/git)

Vercel이 만든 고정 production 도메인을 사용하거나 본인 소유 도메인을 연결한다. 커스텀 도메인은 선택 사항이며 우선 https://프로젝트이름.vercel.app으로 공개할 수 있다. 커밋별 Preview 주소를 최종 운영 주소로 지정하지 않는다.

최종 주소가 정해지면 Settings의 Environment Variables에 SITE_URL을 추가한다. 값은 https://본인도메인 또는 고정 https://프로젝트이름.vercel.app 형식이며 경로·쿼리·fragment는 넣지 않는다. Production과 Preview에 같은 최종 운영 origin을 지정하고 다시 배포한다. SITE_URL이 없으면 Vercel의 production 도메인 시스템 변수를 사용하며 로컬 기본값은 기존 GitHub Pages origin이다. fallback은 실제 운영 주소 확인을 대신하지 않는다.

| 환경 | canonical·feed·sitemap | 검색 노출 |
|---|---|---|
| Production | SITE_URL, 없으면 Vercel production 도메인 | 정상 공개 페이지 index 가능 |
| Preview·Vercel development | 같은 최종 production origin | HTML noindex와 robots 전체 차단 |
| 일반 로컬 실행 | SITE_URL, 없으면 기존 GitHub Pages origin | 실제 공개 배포가 아님 |

도메인을 연결한다면 Vercel Domains 화면이 해당 프로젝트에 제시하는 DNS 값을 사용한다. DNS 값을 예시로 추정해 입력하지 않는다. 인증서와 Domain 상태가 정상인지 확인하고 최종 SITE_URL을 변경한 뒤 다시 배포한다. 별도 비밀키를 브라우저 코드나 PUBLIC_ 환경 변수에 넣을 작업은 없다.

## 배포 후 확인

홈, 전체 글, 주제, 소개를 열고 검색으로 오래된 글과 최신 연구를 각각 찾는다. 연구 그림 확대, 코드 복사, 모바일 목차, 관련 글 선택과 별도 글 이동, 어두운 테마를 확인한다. feed.xml, rss.xml, sitemap.xml의 주소가 최종 origin인지 확인하고 Google·Naver 검증 경로도 연다.

생성된 파일·내부 링크·제목·기존 앵커 검사는 generated/build-verification.json으로 확인한다. 이 검사는 Vercel CDN의 HTTP 상태나 DNS 전환을 대신하지 않는다. 최종 도메인에서 대표 기존 .html 글 주소와 세 연구 permalink의 상태·canonical을 직접 확인한다. q·topic·type·sort·role·view·page 쿼리가 있는 요청은 vercel.json의 별도 조건부 응답 헤더로 X-Robots-Tag: noindex, follow를 제공한다. 이 조건은 로컬 정적 preview의 HTTP 헤더와 다를 수 있으므로 Vercel에서 직접 확인한다. 보고서는 외부 이미지와 출처 링크의 원격 응답을 전부 검사한 결과가 아니다.

## GitHub Pages 주소의 처리

dongryeollee1.github.io는 GitHub 소유 도메인이므로 Vercel의 커스텀 도메인처럼 DNS를 변경할 수 없다. 예를 들어 새 주소가 https://example.com이면 기존 /research/… 글 경로는 새 사이트에서도 유지하지만 기존 origin 방문자는 GitHub Pages에서 별도로 연결해야 한다.

도메인 전환 전까지 GitHub Pages는 `main`의 저장소 루트에서 Jekyll 사이트를 계속 게시한다. 이 브랜치 소스는 GitHub의 내장 Pages 빌드로 처리된다. [GitHub Pages 게시 소스](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

새 운영 주소의 정상 배포를 확인한 후, 같은 SITE_URL로 npm run build를 실행하고 node tools/create-legacy-bridge.mjs를 실행하면 dist-legacy/에 글별 연결 페이지를 만든다. 이 명령에도 같은 SITE_URL 환경 변수가 필요하다. 예를 들어 SITE_URL=https://example.com npm run build 다음 SITE_URL=https://example.com node tools/create-legacy-bridge.mjs 순서다. 대상 주소를 검증하며 원본·현재 배포·Git 브랜치를 변경하지 않는다.

bridge는 각 글을 새 사이트의 같은 경로로 연결하고 기존 fragment를 전달한다. 모든 글을 새 홈으로 보내지 않는다. 태그·카테고리는 새 사이트의 호환 화면에서 대응한다. JS가 꺼져 있어도 안내 링크와 지연된 HTML 이동을 제공하지만 GitHub Pages 정적 HTML 이동은 HTTP 301/308과 같지 않다. RSS는 새 origin의 글 링크를 포함하는 XML로 유지한다.

새 운영 주소에서 기존 글 경로가 정상 응답하는지 확인하고 `dist-legacy/`를 검토한 뒤, 연결 페이지를 게시할 브랜치로 Pages 게시 소스를 변경한다.

## 캐시와 복구

해시가 있는 /_astro/ 자산만 장기 immutable 캐시를 사용한다. 고정 이름의 폰트는 갱신 가능한 캐시를 사용하고 feed·sitemap·robots는 재검증한다. .html URL을 없애는 cleanUrls나 전역 후행 슬래시 변경은 적용하지 않는다. 후행 슬래시 없는 Book-Spine 연구 주소는 기존 canonical을 유지하도록 대응 index.html로 명시적으로 rewrite한다. 원문에서 사용하던 ObjectDetection·R-CNN의 이전 objectdetection 분류 주소 두 개는 각각 확인된 기존 cv·paperreview 글 주소로 영구 이동한다.

문제가 발생하면 Vercel에서 직전 정상 deployment를 production으로 다시 승격한다. 코드로 복구한다면 정상 commit으로 되돌린 뒤 재배포한다. 기존 GitHub Pages는 전환 전 기준 commit과 배포 산출물을 유지한다. DNS 변경 복구는 캐시·전파 때문에 deployment rollback과 즉시성이 다르다.

## 공식 문서

| 자료 | 확인한 내용 |
|---|---|
| [Astro의 Vercel 배포](https://docs.astro.build/en/guides/deploy/vercel/) | 정적 Astro는 adapter 추가 없이 배포 가능 |
| [Astro 설정](https://docs.astro.build/en/reference/configuration-reference/) | 정적 출력·build.format·trailingSlash 동작 |
| [Vercel 프로젝트 설정](https://vercel.com/docs/project-configuration) | build·output·cleanUrls·headers |
| [Vercel 환경 구분](https://vercel.com/docs/deployments/environments) | Preview와 Production의 Git 배포 흐름 |
| [시스템 환경 변수](https://vercel.com/docs/environment-variables/system-environment-variables) | VERCEL_ENV와 production 도메인 변수 |
| [Node.js 버전](https://vercel.com/docs/functions/runtimes/node-js/node-js-versions) | engines.node를 통한 major 버전 고정 |

# 글과 지식 관계 작성 안내

작성일: 2026-11-18 · 원문과 별도 메타데이터를 사용하는 정적 블로그

## 원문과 메타데이터

기존 본문은 _posts/에서 편집한다. 원문을 일괄 MDX로 바꾸지 않는다. content/article-metadata.json은 원본 경로를 키로 사용하는 sidecar이며 영구 ID·설명·언어·유형·개념 관계를 관리한다. front matter와 sidecar가 충돌하면 규칙을 확인하고 원본의 잘못된 값을 수정한다.

새 글은 _posts/YYYY-MM-DD-slug.md로 추가하고 metadata에 같은 sourcePath 항목을 추가한다. ID는 한 번 정해 유지한다. 파일명이나 제목이 바뀌어도 같은 글의 ID를 재발급하지 않는다. 기존 글의 주소를 고칠 때는 generated/url-manifest.json과 외부 링크를 확인하고 이전 정책을 함께 기록한다.

| 필드 | 기준 |
|---|---|
| id | 기존 ID와 겹치지 않는 영구 식별자 |
| title | 원문 제목이 없거나 이전 보완이 필요한 경우 |
| description | 실제 내용을 설명하는 1~2문장. 미확인 결과 수치 생성 금지 |
| language | 실제 본문 기준 ko 또는 en |
| contentType | research / paper-review / implementation / troubleshooting / note |
| conceptRefs | 존재하는 개념 ID와 about / applies 역할 |
| hero | 선택. src·alt·kind(photo/figure) |
| featured | 홈 대표 글로 선정한 경우만 true |

날짜는 front matter 또는 파일명에서 읽고 실제 게시 의미를 유지한다. 수정일은 내용 변경을 확인한 경우만 지정하며 재배포일로 덮어쓰지 않는다. published: false인 글은 공개 글·검색·추천·feed·sitemap에서 제외해야 한다.

## 개념과 관계

content/knowledge/concepts.json에서 이름·별칭·설명·상위 개념·분야를 관리한다. about은 글이 설명하는 핵심 주제이고 applies는 구현이나 실험에 실제 사용한 기술이다. 예를 들어 Reranking 설명 글은 about, 책등 검색에 재정렬을 사용한 글은 applies로 연결할 수 있다. 공통 개념만으로 같은 모델을 사용했다거나 후속 연구라고 기록하지 않는다.

별칭에는 동의어와 표기 차이를 넣는다. 서로 다른 방법인 BM25와 Reranking을 하나의 동의어로 합치지 않는다. 상위 개념은 존재하는 ID만 참조하고 계층 순환을 만들지 않는다. 새 개념을 추가한 뒤 기존 추천도 함께 바뀌는지 확인한다.

series.json의 orderedArticleIds는 연재 순서의 단일 원본이다. 날짜순과 별개로 실제 읽는 순서를 기록한다. overrides.json은 pin·exclude나 명시 관계를 제어한다. 현재 파일의 실제 구조를 따르고 충돌을 수정한다.

## 본문·이미지·코드

일반 Markdown, GFM 표와 fenced code를 사용한다. 코드에는 가능한 언어 이름을 적는다. 이미지·CSV는 기존 공개 경로를 보존한다. 연구 도표는 축·범례가 잘리지 않게 figure로 지정하며 사진만 cover 처리를 허용한다. alt에는 이미지가 보여주는 내용을 적는다.

새 글에는 Liquid include, 본문 script, 전역 style을 추가하지 않는다. 기존의 허용된 relative_url·post_url은 이전 파이프라인이 변환한다. 특별한 동작은 공통 컴포넌트에 구현한다. 명시 heading ID와 이전 앵커는 삭제 전에 대응을 확인한다. 수식은 공통 렌더러를 사용하며 글마다 MathJax 스크립트를 추가하지 않는다.

## 게시 전 확인

npm run check, npm test, npm run build를 실행한다. npm run preview에서 새 글·주제·시리즈·검색 결과와 관련 이유를 확인한다. 원문과 생성 HTML의 제목·표·수식·이미지·코드가 같은 내용을 보여주는지 살핀다. generated/는 산출물이므로 여기에서 수동 수정하지 않는다.

새 글이나 관계를 변경하면 공개 글 전체의 추천을 다시 계산한다. 결과를 다른 글에 수동으로 복사하지 않는다. 자동 탐색 결과와 사람이 확인해야 하는 개념·실제 적용 관계를 구분한다. 빌드와 미리보기에는 임베딩·LLM API 호출이나 그래프 DB 접속이 필요하지 않다.

## 새 글 발행과 연관 글 추천

근거 기반 추천(Phase 2A)은 발행 서버에서 `npm run recommendations:sync`로 갱신한다. `.env`에 `OPENROUTER_API_KEY`, `SUPABASE_URL`, `SUPABASE_SECRET_KEY`가 필요하다(`.env.example` 참고). 이 값은 Vercel에 넣지 않는다.

1. 원문과 metadata를 추가·수정하고 최신 `main`을 반영한다.
2. `npm run recommendations:sync -- --dry-run`으로 호출 대상 글과 이유를 먼저 볼 수 있다. LLM은 호출하지 않는다.
3. `npm run recommendations:sync`를 실행한다. 새 글의 블록만 임베딩하고, 새 글의 관계 추출 1회와 재정렬 1회, 후보나 근거가 바뀐 기존 글의 재정렬만 호출한다. 종료 코드 2는 일부 글이 실패해 `fallback`으로 저장됐다는 뜻이며 다시 실행하면 그 글만 재시도한다. 첫 호출들이 모두 거부되면(키·모델·요청 형식 문제) 파일을 쓰지 않고 공급자 응답과 함께 종료 코드 1로 멈춘다.
4. `npm run recommendations:check`로 낡은 글이 없는지 확인한다. 빌드가 규칙 추천으로 돌릴 글과 공개 글이 아닌 추천 파일도 여기서 걸린다. 키 없이 저장 파일만 검사할 때는 `npm run recommendations:check -- --offline`을 쓴다.
5. `content/related/`의 변경분에서 새 추천과 이유 문장을 읽는다. 무관한 연결은 `content/knowledge/overrides.json`의 `excludes`에 `"기준 글 ID": ["대상 글 ID"]`로 넣고 다시 동기화한다. 연결은 맞지만 문장이 틀리면 `content/knowledge/reasons.json`에 `"기준 글 ID": {"대상 글 ID": "고친 문장"}`을 넣는다(동기화 불필요).
6. 원문, metadata와 함께 `content/related/`의 바뀐 파일(삭제된 파일 포함)을 커밋한다. `.cache/`와 `generated/`의 추천 보고서는 커밋하지 않는다.

동기화를 건너뛰어도 빌드는 성공한다. 추천 파일이 없거나 현재 원문과 맞지 않는 글은 그 글만 규칙 추천을 표시하고, 빌드 로그와 `generated/related-report.json`에 해당 글이 남는다.

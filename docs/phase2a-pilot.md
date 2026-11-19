# 연관 글 추천 Phase 2A 파일럿

2026-11-19 · 기준 커밋 `b5d10e2` · 공개 글 113편에서 3편을 기준 글로 선정

**판정은 전체 적용 보류다.** 그래프·어휘 후보 검색에서 예상한 연결 6개를 모두 찾았지만 모델이 최종 목록에는 3개만 남겼다. 근거 블록이 실제로 존재하는 것을 검증한 후에도 다른 글에 브라우저 실행 환경을 잘못 귀속한 사례가 있었다.

## 실제 추천 비교

각 행은 독자가 왼쪽 글을 열었을 때 보게 되는 순서다. 기준 결과는 운영 사이트에 포함된 113편의 데이터와 대조했다.

| 출발 글 | 현행 규칙 | Phase 2A |
| --- | --- | --- |
| 책등 인식 | mAP → KoDistill → Reranking → Keyword Search | mAP → KoDistill → Reranking |
| Reranking | Dense Retrieval → 책등 인식 → Keyword Search → Semantic Search | Dense Retrieval → 책등 인식 |
| KoDistill | 책등 인식 → ThinkingCap → AlphaLLM → Constitutional AI | PEFT v0.9.0 → TGI 구현 → ThinkingCap |

**책등 인식.** mAP 추천은 원문의 IoU 0.5 평가와 대상 글의 AP 계산 예제를 연결한다. Reranking도 두 글에서 검색 후보를 다루지만 새 설명은 재채점보다 키워드 검색을 강조한다. KoDistill을 추천하면서 쓴 “모두 브라우저 환경에서 OCR 모델을 실행”한다는 설명은 틀렸다. KoDistill의 선택된 발췌는 시각언어모델의 한국어 전사 후속 학습을 설명한다. Keyword Search는 후보에 있었으나 빠졌다.

**Reranking.** 책등 인식에 `application_example`을 붙인 방향은 적절하다. 다만 모델이 선택한 책등 글의 서문에는 n-gram 검색까지만 나온다. 후보 재채점을 보여주는 같은 글의 시스템 절을 인용해야 이유 전체가 뒷받침된다. Dense Retrieval은 함께 읽을 만하지만 생성된 이유에는 대상 글의 밀집 검색 방법에 대한 설명이 빠졌다.

**KoDistill.** ThinkingCap은 두 실험의 소형 모델 후속 학습을 비교하는 데 유용하다. 새로 선정된 PEFT와 TGI 글은 각각 LoRA 어댑터 간 병합과 서빙 전 변환에 초점을 둔다. 이 두 연결은 LoRA 도구를 더 읽는 데에는 유용하나 KoDistill의 핵심인 OCR 전사와 보상 평가에 대한 후속 읽기로는 약하다. 책등 인식 글은 후보 순위 1위였지만 최종 선정에서 빠졌다.

## 측정 조건

- 모델은 로컬 `Qwen3.6-35B-A3B-UD-Q4_K_M`을 사용했다. 공개된 세 글의 관계를 먼저 추출하고, 기준 글마다 15개 후보의 양쪽 발췌를 한 요청에 넣어 0~4편을 골랐다. 본 파일럿은 동일 세 사례로 발췌 선택과 출력 제약을 조정한 개발 평가다.
- 실행 전 정한 기대 연결은 책등 인식 → Reranking·Keyword Search, Reranking → 책등 인식·Keyword Search, KoDistill → 책등 인식·ThinkingCap이다. **후보 포함 6/6**, 기존 최종 포함 **6/6**, 새 최종 포함 **3/6**이다. 이 수치는 여섯 개 기대 연결에 대한 적중 수이며 모든 관련 글을 라벨링한 결과는 아니다.
- 코퍼스 113편 중 시범 적용한 3편을 제외한 110편의 추천과 순서는 기존 결과와 일치했다. 새 추천은 총 8개이며 빈 목록을 반환한 기준 글은 0편이다.
- 출력된 글의 읽을 가치에 대한 정성 판정 기준은 공통된 구체적 처리 단계, 평가 과제 또는 직접적인 배경 설명이다. 이 기준에서 기존은 **10/12**(KoDistill의 일반 RL 배경 두 편을 약한 연결로 판정), 새 결과는 **6/8**(KoDistill의 PEFT 변경 기록과 TGI 서빙 글을 약한 연결로 판정)이다.
- 코드가 확인한 양쪽 발췌·해시 일치는 **8/8**이다. 별도로 이유와 발췌를 읽고 주장 전체를 지지한다고 판정한 것은 **3/8**이다. 책등 인식 → mAP·Reranking과 KoDistill → PEFT가 해당한다. 나머지는 근거 불충분 4개, 실행 환경 사실 오류 1개다. 구조적 검증 성공을 표시하는 `verified`의 의미는 전자이다. 표본이 작고 이번 구현을 진행한 Codex의 판정이므로 독립적인 사람 평가로 해석할 수 없다.
- 다음 품질 판단은 후보 수 확대보다 **결정적인 근거 블록 선택과 추천 이유의 원문 충실성**에 초점을 둔다. 블로그 전체 적용은 별도 결정으로 둔다.

## 실행과 결과 확인

`npm ci` 후 `npm run recommendations:check`로 현재 Git 트리와 스냅샷을 대조한다. `npm run build`는 API 키 없이 정적 페이지를 생성한다. 미리보기에서 다음 경로의 오른쪽 노드와 본문 하단의 연결 이유를 확인할 수 있다.

- 책등 인식 — `/research/in-browser-spine-reader`
- Reranking — `/nlp/2024/11/12/reranking.html`
- KoDistill — `/research/kodistill-vl-2b/`

다시 추론할 때는 발행 서버 환경에 `OPENROUTER_API_KEY`를 설정하고 `npm run recommendations:sync`를 실행한다. 기본 모델은 `qwen/qwen3.7-flash`이고 모델 ID는 `RECOMMENDATIONS_MODEL`로 지정한다. OpenAI 호환 로컬 추론을 사용할 경우에는 `RECOMMENDATIONS_API_BASE`와 해당 서버의 모델 ID를 지정한다. 사용자 지정 엔드포인트의 인증은 `RECOMMENDATIONS_API_KEY`로 설정한다.

`content/related.json`은 실제 출력이다. 글별 사실, 모델·입력 해시, 후보 및 양쪽 블록 목록, 선택한 발췌와 추천 이유를 담는다. 브라우저에는 추천 ID·이유만 전달한다. 실패한 항목은 `fallback`과 종료 코드 2로 보고하며 다음 동기화에서 재시도한다. 입력이 일치하는 성공·보류 결과는 재사용한다. 두 발행 서버에서 변경을 합칠 때 최종 트리에서 `recommendations:check`를 실행하고 불일치하면 동기화한다. 발행본 해시가 달라진 상태에서 빌드하면 경고와 함께 현재 규칙 추천을 제공한다.

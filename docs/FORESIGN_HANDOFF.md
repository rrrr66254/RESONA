# ForeSign 핸드오프 — 새 세션 시작용 전체 맥락

> 작성: 2026-06-10. 이 문서 하나로 새 작업 세션이 프로젝트 전체 맥락을 복원할 수
> 있도록 정리했다. 상세 설계는 `docs/FORESIGN_DESIGN.md` (v0.3)가 단일 진실 소스.

---

## 1. 프로젝트 한 줄 정의

**ForeSign** = 수화 포즈에 대한 최초의 JEPA식 잠재 예측 사전학습. 핵심 기여는
시스템이 아니라 질문: *"수화의 부위 구조는 일반 마스킹보다 나은 SSL prior인가?"*
(C2) + ASL-LEX 음운 프로브(C3) + 교차언어 전이 분석(C4).

## 2. 어떻게 여기까지 왔나 (의사결정 이력)

1. **2024–2026 문헌 전수 조사** (서브에이전트 4종: SLT 번역모델 / 생성·아바타 /
   데이터·언어학·평가 / 인접분야 갭): gloss-free LLM-SLT는 포화 + 재현성 위기
   (arXiv:2603.13240), PHOENIX14T 오염 공인(Frontiers 2026), 환각/근거 부족 문제
   (ICLR 2026, arXiv:2510.18439)가 공식화된 상태.
2. **후보 주제 5개** 중 보유 자산(포즈 전용 895K 클립, RGB·오디오·대화 데이터
   없음)에 맞춰 **JEPA식 예측 사전학습**으로 결정. 탈락: sign-to-speech(오디오
   없음), 풀듀플렉스 대화(대화 코퍼스 없음). 보류: 공간 담화 추적(박사 연계 P4).
3. **S-JEPA(ECCV 2024) 구현 수준 정독**: 공개 코드·arXiv 없음 → MAMP 코드 +
   ECVA 전문으로 재구성. 채택 교훈: EMA 필수(제거 시 완전 붕괴), 타깃은 풀 시퀀스
   타깃 인코더 *출력*에서 선택(+7.2pp), V-JEPA의 시간축 전체 마스킹(>20pt),
   L1 latent 기본 + CE/centering fallback, V-JEPA 2의 상수 LR/anytime checkpoint.
4. **설계 v0.1** 작성 → **3인 모의 심사** (R1 SSL/JEPA, R2 수화언어학/윤리,
   R3 시니어 AC): 전원 4/10 거절. → **v0.2** (전면 개정) → **재심**: R1 5/10,
   R2 6/10 조건부, R3 "설계는 출판 등급" → **v0.3** (재심 잔여 지적 반영, 현재).
5. **git 정리**: 이 브랜치는 ForeSign 전용. RESONA v4 코드(MSM/ForwardPred,
   ST-GCN+Transformer)는 `main`과 히스토리에 보존.

## 3. 모의 심사가 죽인 것들 (다시 주장하면 안 됨)

- ~~"world model"~~, ~~"anticipation capability"~~, ~~"Sign Anticipation Benchmark"
  (헤드라인 기여)~~ → "early recognition under partial observation" 분석 섹션
- ~~"음운론적 마스킹"~~ → "부위 구조(part-structured) 마스킹" + 음운론적 해석은
  ASL-LEX 프로브로 *검증*
- ~~"privacy-by-design" 기여~~ → signer-ID 프로브로 정량화, 서론 1문장
- ~~SLT 평가 (v1)~~ → 제거 (frozen 포즈 + mBART는 SHuBERT/Uni-Sign 표에서 지는 판)
- ~~"최초 멀티링구얼 포즈 사전학습/교차언어 전이"~~ → SignCLIP(44언어),
  OpenHands가 선행. "최초"는 **JEPA식 연속 잠재 예측의 수화 적용**(C1)만
- ~~AAAI 2027 타깃~~ → 산술적으로 불가(26주 일정 종점 12월 vs 마감 8월).
  **1차 타깃 ACL 2027 (ARR ~2027.2)**, 폴백 AAAI 2028

## 4. 핵심 설계 결정 (v0.3 기준, 상세는 설계 문서)

- **토큰화**: 18 공간그룹 × 시간세그먼트(l=4). body를 trunk/L-arm/R-arm으로 분해
  (M1a 손목 누설 제거 — R1 재심의 핵심 수정)
- **마스킹 5조건 게이트**: {M3-only, **M3′(비율·기하 매칭 무작위 — 필수 대조군)**,
  M3+M1, M3+M2, M3+M1+M2} × **사전학습 ≥3 seed**, 판정규칙(Δ > 2×pooled seed-SD)을
  **P1 전 OSF 타임스탬프 등록**. null 시 C4 중심 재조준
- **R-8 리스크**: SLT 제거 후 ISLR 표가 사실상 SOTA 표 — "JEPA ≥ 복원
  (MSM/SignBERT+류)"이 제2 생사 조건. 패배 시 하이브리드 손실로 전환
- **M2는 연속(문장) 클립 전용** + 적용 가능 토큰 비율 P1 전 보고 + 평균 회귀 진단
- **음운 프로브**: ASL Citizen/Sem-Lex ↔ ASL-LEX 매핑 검증 완료(실행 가능).
  raw/random-init floor + 지도 ceiling 병기. C3 결론은 ASL 인용형 한정
- **spotting 프로브 필수** (인용형 박스를 벗어나는 유일한 연속 신호 증거)
- **윤리**: 라이선스 감사(yt_asl/WLASL/AI-Hub) → 가중치 공개 매트릭스 +
  release-clean run의 헤드라인 재현을 본문 보고. 농인 협력자 실명 확보가
  **P0 종료 게이트** (보상 예산 + 해석 주장 승인권)

## 5. 현재 억셉 전망 (재심 후, 결과 양호 가정)

| 학회 | 확률 | 비고 |
|---|---|---|
| ACL/EMNLP main | 25–33% | **1차 타깃** — 음운 프로브+교차언어 구성이 ACL형 |
| AAAI (2028) | 25–40% | 일정상 2028 |
| CVPR/ICCV | 8–14% | 포기 권고 (비전 훅 없음) |
| 2티어 (WACV/Findings/LREC) | 65–85% | null 결과여도 안전망 |

AC 종합: 12개월 내 1티어 ~15%, 24개월 ~30–35%, 2티어 바닥 ~80%.
남은 리스크는 설계가 아니라 **결과 리스크** (P(M1/M2 효과) ≈ 40–50%,
P(JEPA ≥ 복원) ≈ 50–60%) + 실행 리스크(무여유 26주) + 외부 리스크(라이선스,
협력자 섭외, 스쿱 창 3–6개월).

## 6. 데이터 자산 (저장소 외부)

- RESONA-77 H5: 77 joints × (x,y,conf), per-region anchor+scale 정규화 빌드 완료
- 22개 데이터셋, 11개 언어, 895,427클립(≥24프레임). ASL 70%(yt_asl 345K 문장,
  openasl 91K, how2sign 30K, 사전형 다수), **KSL 187K (2개 독립 데이터셋 —
  교차언어 교란 분리의 핵심)**, DGS/LSA 문장 포함, TR/RSL/ISL 등 단어형
- 제외: greek_sl(29), mssl(112)
- 주의: **v4 augment의 horizontal flip은 L/R 스왑 없는 결함** — ForeSign에서는
  기본 OFF, 사용 시 그룹 스왑 동반
- 미확인: SLT용 텍스트 페어가 빌드 파이프라인에 보존됐는지 (v1엔 불필요하나 v2 게이트)

## 7. 다음 할 일 — P0 (6주, 우선순위순)

1. **라이선스 감사** (가장 싸게 가장 큰 리스크 제거): 22개 코퍼스 전수 →
   데이터 스테이트먼트 + 가중치 공개 매트릭스
2. **평가 스위트 구현** (사전학습 코드보다 먼저): ISLR attentive/linear/kNN probe,
   dataset/language/signer-ID 진단, 조기 인식 프로토콜
3. **ASL-LEX 매핑**: asl_citizen/semlex ↔ ASL-LEX 2.0 음운 코딩 + Sign Type +
   iconicity 노름 결합 테이블
4. **농인 자문 협력자 섭외 착수** (P0 종료 게이트)
5. **v4 체크포인트로 MSM 베이스라인 probe 수치 확보** (베이스라인 0호)
6. M2 적용 가능 토큰 비율 산출, OSF 사전등록 문서 초안
7. 이후 P1: ForeSign-S + 5조건×3seed 게이트 → **게이트 결과와 무관하게 arXiv 선점**

## 8. 저장소/브랜치 상태

- 브랜치: `claude/fervent-bardeen-l2yxvh`, **PR #1** (draft):
  https://github.com/rrrr66254/RESONA/pull/1
- 이 브랜치: ForeSign 문서 전용 (README, docs/). **v4 코드는 main에 보존** —
  PR을 main에 머지하면 main에서도 v4 코드가 삭제되므로 머지 전 확인 필요
- 커밋 이력: v0.1(19d498c) → v0.2(006c039) → v0.3(6148d62) → git 정리

## 9. 핵심 참고문헌 (정독/검증 완료)

S-JEPA (ECCV'24, ECVA PDF — 코드 없음, MAMP로 재구성) · MAMP (arXiv:2308.07092) ·
V-JEPA (2404.08471) · V-JEPA 2 (2506.09985) · SignBERT+ (2305.04868) ·
SignCLIP (2407.01264) · OpenHands (2110.05877) · MASA (2405.20666) ·
SignRep (2503.08529) · SHuBERT (2411.16765) · Uni-Sign (2501.15187) ·
Sigma (2509.21223) · SSL-SLR (2509.05188) · SignVerse-2M (2605.01720 — 55언어
2M클립, §4.4 서브셋 실험으로 대응) · ASL-LEX 2.0 · Sem-Lex (2310.00196) ·
ASL Citizen (2304.05934) · 조기행동인식 서베이 (2107.05140) · 평가 비판
(2510.25434) · 재현성 위기 (2603.13240) · "sign+JEPA" arXiv 0건 (2026-06-10 검증)

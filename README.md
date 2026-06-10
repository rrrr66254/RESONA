# ForeSign

**Part-Structured Latent-Predictive Pretraining for Multilingual Lexical Sign Representation**

수화 포즈 시퀀스에 대한 최초의 JEPA식(연속 잠재 예측 + EMA 타깃) 사전학습 연구
프로젝트. RESONA-77 멀티링구얼 포즈 코퍼스(22개 데이터셋, 11개 수화언어,
~895K 클립) 위에서 다음의 과학적 질문을 검증한다:

> **"수화의 부위 구조(수지/비수지 채널 분해, 양손 제약, 시간적 선행)는
> 일반 멀티블록 마스킹보다 나은 자기지도 마스킹 prior인가?"**

## 상태

- **단계: 설계 완료 (v0.3), 구현 전 (P0 착수 대기)**
- 설계 문서: [`docs/FORESIGN_DESIGN.md`](docs/FORESIGN_DESIGN.md) — 3인 모의 심사
  2라운드(v0.1→v0.2→v0.3)를 거친 최종 설계
- 세션 핸드오프: [`docs/FORESIGN_HANDOFF.md`](docs/FORESIGN_HANDOFF.md) — 새 작업
  세션을 위한 전체 맥락 요약 (배경, 결정, 리뷰 결과, 다음 할 일)

## 핵심 설계 (요약)

| 항목 | 내용 |
|---|---|
| 입력 | RESONA-77 포즈: 77 joints × (x,y,conf), per-region 정규화 H5 |
| 토큰화 | part-aware 18 공간그룹(trunk/L-arm/R-arm/hand×12/face×3) × 시간세그먼트(l=4) |
| 목적함수 | L1 latent 예측 (EMA 타깃, 풀 시퀀스 출력에서 타깃 선택), CE+centering fallback |
| 마스킹 | M3 멀티블록(기본) · **M3′ 비율매칭 대조군** · M1 부위 구조 · M2 선행(연속 클립 전용) |
| 스케일 | S(22M, ablation) → B(86M, 본학습) → L(게이트) |
| 평가 | ISLR attentive probe · **ASL-LEX 음운 파라미터 프로브** · 교차언어 전이(iconicity 통제) · 조기 인식 분석 · sign spotting(필수) · dataset/language/signer-ID 진단 |
| 결정 게이트 | S 스케일 5조건 × ≥3 seed, 사전 판정규칙, P1 전 OSF 등록. null 시 교차언어 중심 재조준 |
| 1차 타깃 | **ACL 2027 (ARR ~2027.2)**, 폴백 AAAI 2028, 안전망 WACV/Findings/LREC |

## 이 저장소에 없는 것

- RESONA v4 사전학습 코드(MSM/ForwardPred 파이프라인)는 이 브랜치에서 제거됨 —
  `main` 브랜치와 git 히스토리에 보존되어 있다. ForeSign 구현은 v4와 분리된
  신규 코드로 작성 예정 (설계 문서 §13 골격 참조).
- 데이터(H5 코퍼스)는 저장소 외부 (KISTI/NAS).

## 라이선스

LICENSE 파일 참조. 학습 데이터 코퍼스들은 각자의 라이선스를 따르며,
가중치 공개 가능성은 P0 라이선스 감사 결과에 따름 (설계 문서 §10).

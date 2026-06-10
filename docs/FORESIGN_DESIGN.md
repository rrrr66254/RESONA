# ForeSign — 설계 문서 v0.1

**Predictive Latent Pretraining for Multilingual Sign Language Understanding**

> 프로젝트명 *ForeSign* = **fore**sight + **sign**. 수화 시퀀스의 미래·은닉 부위를
> 잠재공간(latent space)에서 예측하는 사전학습으로, "선행 예측(anticipation)"이라는
> 새 능력을 수화 AI에 부여하는 것이 핵심 정체성이다.
>
> 상태: 설계 단계 (코드 없음). 작성일 2026-06-10.

---

## 0. 한 문단 요약

ForeSign은 11개 수화언어·22개 데이터셋·89.5만 클립의 포즈 시퀀스(RESONA-77 스키마,
77 joints × (x,y,conf)) 위에서 학습하는 **수화 최초의 JEPA(Joint-Embedding Predictive
Architecture) 사전학습 모델**이다. 좌표 복원(MAE/MSM)이 아니라 **EMA 타깃 인코더가 만든
잠재 표현을 예측**하며, 마스킹을 수화 음운론에 맞춰 세 가지로 설계한다 —
(M1) 수지(manual) 부위 전체를 가리고 비수지(non-manual) 신호로부터 추론,
(M2) 미래 구간 전체를 가리는 anticipation 마스크, (M3) V-JEPA식 멀티블록.
기여는 ① 멀티링구얼 sign world-model 사전학습, ② 주석이 필요 없는
**Sign Anticipation Benchmark**, ③ 저자원 수화(KSL 포함)로의 교차언어 전이 분석.

**선행연구 대비 위치**: S-JEPA(ECCV 2024)는 3D 액션 인식(NTU)용 스켈레톤 JEPA,
SHuBERT(ACL 2025)는 ASL 단일·비디오 crop·클러스터 예측, Sigma/SSL-SLR(2025)는
좌표 복원 계열, Uni-Sign(ICLR 2025)은 지도(supervised) 사전학습.
"**잠재 예측 + 11개 언어 + 음운론 기반 마스킹 + anticipation 평가**"의 조합은 전례가 없다.

---

## 1. S-JEPA 정독 결과 요약 (구현 수준)

출처: [ECVA 공개 PDF 전문](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf),
[프로젝트 페이지](https://sjepa.github.io/), [MAMP 코드](https://github.com/maoyunyao/MAMP)
(S-JEPA가 명시적으로 따른다고 밝힘), [V-JEPA](https://arxiv.org/abs/2404.08471) +
[공식 config](https://github.com/facebookresearch/jepa/blob/main/configs/pretrain/vitl16.yaml),
[V-JEPA 2](https://arxiv.org/abs/2506.09985).

⚠️ **S-JEPA는 공개 코드·arXiv 버전이 없다.** 마스킹/전처리 세부는 MAMP 공개 코드에서
재구성한 것이며, 재현 시 이 점을 논문에 명시해야 한다. 역으로, 우리가 코드를 공개하면
"스켈레톤 JEPA의 첫 공개 구현"이라는 부수 기여가 된다.

### 1.1 S-JEPA 핵심 스펙

| 항목 | 값 |
|---|---|
| 입력/토큰화 | NTU 25관절×T=120, 관절별 4프레임 세그먼트 토큰 → 30×25=750 토큰, Ce=256 |
| Context 인코더 | vanilla ViT 블록 8층, d=256, 8헤드, FFN 1024 |
| Target 인코더 | 동일 구조의 EMA 복사본, λ: 0.9999→1.0 cosine, stop-grad |
| Predictor | 5층 MHSA, d=256; 마스크 토큰을 context 인코더 **출력**에 삽입 |
| 마스킹 | motion-aware Gumbel top-k (MAMP 계승), r=0.9 — 고모션 토큰이 타깃이 될 확률↑ |
| 타깃 선택 | **풀 시퀀스를 본 target 인코더의 출력에서 마스크 위치를 선택** (입력 마스킹 대비 +7.2pp — 최대 단일 요인) |
| 손실 | 채널(256차원) softmax CE: 타깃만 centering(EMA β=0.9), τ_pred=0.1, τ_tgt=0.06. MSE 대비 +1.6~+3.4pp |
| 학습 | 1200ep, batch 256, AdamW(0.9,0.95), wd 0.05, lr 1e-3 (floor 5e-4), A100×8 |
| 다운스트림 | **EMA 타깃 인코더를 사용** (view 인코더보다 항상 우수) |
| 붕괴 조건 | EMA 제거 시 완전 붕괴(1.6%) — EMA는 필수 |

### 1.2 V-JEPA / V-JEPA 2에서 가져올 교훈

1. **마스크는 시간축 전체를 덮어야 한다.** V-JEPA: temporally-full 멀티블록 vs 랜덤 튜브
   = K400 frozen 72.9 vs 51.5 (>20pt 차이). S-JEPA의 (관절×4프레임) 독립 마스킹은
   시간 누수가 크다 — **우리는 V-JEPA 기하를 기본으로 채택**.
2. **손실은 단순 L1(latent)로 충분** — centering/softmax/온도 불필요, 대규모에서 붕괴 미관찰.
   S-JEPA의 CE+centering은 좁은 차원(256d)에서의 불안정 대비 **fallback**으로 보존.
3. **V-JEPA 2 단순화 채택**: warmup–constant–decay LR + EMA/wd 고정 →
   상시 체크포인트(anytime checkpoint) 가능. Slurm 48h 체인 잡 운영과 정확히 맞물림.
4. **Progressive length cooldown**: 본학습은 짧은 클립, 마지막 단계만 긴 클립
   (V-JEPA 2: +0.7pt를 1/8.4 비용으로).
5. 평가는 linear probe 단독이 아니라 **attentive probe**(cross-attention pooling) 병행 —
   수백 토큰 mean-pool은 손모양 정보를 희석시킨다.

---

## 2. 데이터 계층

### 2.1 코퍼스 (기존 자산 재사용)

RESONA-77 H5 코퍼스: 22개 데이터셋, 11개 언어, 895,427클립(≥24프레임).
per-region anchor+scale 정규화는 빌드 시 적용 완료 (body/lhand/rhand/face 독립
정규화 — dataset fingerprint 1차 방어). 스키마: (T, 234) = 77×(x,y,conf) + 3 part-validity.

| 언어 | 클립수 | 비고 |
|---|---|---|
| ASL | 626,896 | 문장(yt_asl, openasl, how2sign) + 사전(asl_citizen, semlex, wlasl) + 지문자 |
| KSL | 187,827 | 단어 단위 — **교차언어 전이의 핵심 평가축** |
| DGS / LSA | 18,709 / 11,659 | 문장 포함 |
| TR/RSL/ISL/KArSL/GhSL | 50,195 | 단어 단위 — 저자원 전이 평가 |
| 제외 | greek_sl(29), mssl(112) | 실효성 없음 — 평가·학습 모두 제외 |

### 2.2 샘플링: 언어 균형 temperature sampling

ASL 70% 편중 보정. 언어 ℓ의 샘플링 확률 p(ℓ) ∝ n(ℓ)^α, **α=0.5 기본** (α∈{0.3, 0.5, 0.7, 1.0}
ablation). 언어 내부에서는 dataset-balanced 2차 temperature(α=0.7). 모든 클립에
(language_id, dataset_id) 메타데이터를 인덱스에 보존 — 진단 프로브(§6.5)에 사용.

### 2.3 클립 길이

- **Stage 1 (본학습)**: T=128 (25fps 기준 ~5.1s; 단어 클립은 zero-pad + valid mask)
- **Stage 2 (cooldown)**: T=256 — 문장 데이터(yt_asl/openasl/how2sign/ph14t/lsa_t)만으로
  짧게 이어 학습 (V-JEPA 2 progressive length)

### 2.4 View 증강 (context 인코더 입력에만 적용, target은 클린 입력)

S-JEPA의 "기하 변환으로 두 인코더에 다른 뷰 제공"(+0.7pp)을 2D에 맞게 치환:

- 2D affine: 면내 회전 ±10°, 등방 스케일 0.9–1.1, 평행이동 (per-region 정규화 후
  좌표계 기준 소폭)
- temporal: 속도 리샘플 0.85–1.15, 랜덤 트림 [0.7, 1.0]
- spatial jitter: σ=0.02, conf 채널은 불변
- **horizontal flip은 기본 OFF.** 거울상 손모양은 음운적으로 다른 손모양이며 우세손
  교대는 언어적 변형이다. 사용 시 반드시 lhand↔rhand 그룹 스왑 + body 좌우 인덱스
  스왑을 동반하고, 별도 ablation으로만 평가 (v4의 "x만 뒤집기"는 결함으로 기록)

### 2.5 Confidence 처리 원칙

- conf는 **입력 채널로만** 사용 (Cin=3: x, y, c). 잠재 손실이므로 conf를 회귀할 일 없음
- conf < 0.3 관절은 좌표 0 (빌드 시 처리 완료) — 학습 시 추가 처리 없음
- **타깃 제외 규칙**: 토큰 내 평균 conf < 0.2인 토큰은 prediction target에서 제외
  (쓰레기 검출의 잠재를 예측하는 것은 노이즈 학습)
- motion saliency는 conf 가중: m = min(c_t, c_{t−1}) · |x_t − x_{t−1}|
  (MAMP 원식은 키포인트 지터를 모션으로 오인 — noise-seeking 마스킹 방지)

---

## 3. 모델 아키텍처

### 3.1 토큰화 — part-aware patchify

S-JEPA식 관절별 토큰은 77관절×(128/4)=2,464토큰으로 예산 초과. 손의 정밀도는
보존하되 얼굴·몸은 묶는 **부위 인지 그룹 토큰**:

| 부위 | 관절수 | 공간 토큰 | 구성 |
|---|---|---|---|
| body | 9 | 2 | {어깨·팔꿈치·손목 6} / {코·힙 3} |
| lhand | 21 | 6 | 손바닥(wrist+MCP 5) + 손가락 5 (각 3관절) |
| rhand | 21 | 6 | 동일 |
| face | 26 | 3 | mouth8 / 눈·눈썹 영역 / 윤곽 |
| **계** | 77 | **17** | |

- temporal 세그먼트 **l=4** (25fps에서 160ms — 수화 음소 전이 스케일) → T=128 시
  토큰 수 = 17 × 32 = **544 토큰** (V-JEPA ViT-L의 ~1,568보다 작음, 2,464의 1/4.5)
- 토큰 임베딩: 그룹 관절들의 (x,y,c)×l 프레임 평탄화 → linear projection
  (lhand 손가락 토큰: 3관절×3ch×4f=36차원 → d). **v1은 linear patchify** —
  GCN stem은 ablation (S-JEPA가 vanilla ViT로 충분함을 보였고, 단순한 쪽이 스케일링 분석에 유리)
- 위치 인코딩: **temporal RoPE**(V-JEPA 2 교훈) + **learnable spatial embedding** (17개,
  부위 계층 공유 없음, trunc-normal 0.02)
- part-validity 3채널: 해당 부위 토큰의 임베딩에 learnable "part-missing" 벡터 가산

### 3.2 인코더·Predictor 3종 스케일

| | ForeSign-S (pilot) | ForeSign-B (main) | ForeSign-L (scale) |
|---|---|---|---|
| Context/Target 인코더 | d=384, 12층, 6헤드 (~22M) | d=768, 12층, 12헤드 (~86M) | d=1024, 24층, 16헤드 (~300M) |
| Predictor | d=192, 4층 | d=384, 6층 | d=384, 12층 (V-JEPA 비율) |
| 용도 | 5090 단일 GPU, ablation 전수 | A100×8 본학습 | 스케일링 검증 시에만 |

- 블록: pre-RMSNorm + SDPA attention + SwiGLU FFN (현대 표준; v4와 동일 계열이지만
  코드는 신규 작성)
- Predictor 입력: context 인코더 **출력** + 마스크 토큰(zero-init, V-JEPA) 삽입,
  마스크 토큰은 (spatial embed + RoPE 위치)로 조건화
- **Target은 항상 풀 시퀀스를 본 target 인코더의 출력에서 선택** (S-JEPA +7.2pp 교훈)
- 다운스트림·릴리스는 **EMA target 인코더** 가중치

### 3.3 손실

- **기본: L1 latent loss** (V-JEPA): L = (1/|M|) Σ_{i∈M} ‖ĝ_i − sg(z̄_i)‖₁
- target에 layer-norm 적용 여부는 초기 sweep (V-JEPA는 LN 타깃 사용)
- **Fallback (붕괴/불안정 시)**: S-JEPA 채널-softmax CE — 타깃만 EMA centering(β=0.9),
  τ_pred=0.1, τ_tgt=0.06. d=384(S 모델)는 S-JEPA가 불안정을 보고한 256d와 가까우므로
  S 스케일에서 두 손실을 모두 sweep하고 B부터는 승자만
- 붕괴 모니터링 (필수 로깅): 타깃 표현의 특이값 스펙트럼/유효랭크, 토큰 간 표준편차,
  predictor 출력 분산. EMA λ는 **0.999 고정**으로 시작 (V-JEPA 2식 단순화),
  불안정 시 0.9999→1.0 cosine (S-JEPA식)

---

## 4. 마스킹 설계 — ForeSign의 1차 기여

배치 내 클립마다 세 가족 중 하나를 확률적으로 선택. **모든 마스크는 시간 누수를
차단하는 기하**(V-JEPA 교훈)를 따른다.

### M3 — 멀티블록 (기본 학습 신호, 선택확률 0.6)

V-JEPA 직역: (공간 그룹 × 시간 블록) 직사각 블록의 합집합.

- short-range: 8개 블록, 공간 스케일 0.15 (17토큰 중 ~3그룹), **시간 스케일 1.0 (클립 전체)**
- long-range: 2개 블록, 공간 스케일 0.7, 시간 스케일 1.0
- 합집합 마스킹률 ~85–90%
- conf-가중 motion saliency(§2.5)로 블록 중심 위치를 바이어스 (고모션 구간이 타깃이
  되기 쉽게; Gumbel top-k, MAMP/S-JEPA 계승)

### M1 — 음운론적 부위 마스크 (선택확률 0.25)

수화 특화. 부위 단위를 **클립 전체 시간에 걸쳐** 마스킹:

- M1a (확률 0.6): **양손 12토큰 전부** 마스킹 → body+face만으로 손의 잠재 예측.
  비수지 신호(입모양·표정·몸통)와 수지 신호의 음운론적 상관을 강제 학습
- M1b (0.2): face 3토큰 마스킹 → 손으로부터 비수지 추론 (역방향)
- M1c (0.2): 비우세손(한 손) 6토큰 마스킹 → 양손 수화의 대칭성/지배손 제약
  (Battison's symmetry & dominance conditions) 학습

### M2 — Anticipation 마스크 (선택확률 0.15)

t₀ ~ U[0.4T, 0.8T] 샘플 후 **t₀ 이후의 모든 토큰** 마스킹. predictor는 과거만 보고
미래 구간 전체의 잠재를 예측 — v4 ForwardPredHead(k=16 고정, 동일 인코더 stop-grad,
붕괴 위험)와 달리 가변 지평·EMA 타깃·블록 단위. **Anticipation Benchmark(§6.3)와
직결되는 학습 신호.**

> 선택확률 (0.6/0.25/0.15)은 초기값. M1·M2 각각 0으로 끈 ablation이 "음운론적
> 마스킹이 실제로 기여하는가"라는 논문의 핵심 질문에 답한다.

---

## 5. 학습 레시피

| 항목 | 값 (시작점) |
|---|---|
| Optimizer | AdamW (0.9, 0.95), wd **0.05 고정** (스케줄 없음 — V-JEPA 2) |
| LR | **warmup(5%)–constant–decay(10%)**, peak: S 8e-4 / B 6e-4 / L 4e-4 (eff. batch 1024 기준 선형 스케일) |
| EMA | **0.999 고정** (불안정 시에만 0.9999→1.0 cosine) |
| Batch | effective 1024 클립 (B 모델, A100×8: per-GPU 32 × grad-accum 4) |
| 정밀도 | bf16, grad clip 1.0 |
| Stage 1 | T=128, 전체 코퍼스, ~200K steps (≈ 230M 클립뷰, 코퍼스 ~256 에폭 상당) |
| Stage 2 cooldown | T=256, 문장 데이터셋만, ~20K steps |
| 체크포인트 | constant-LR 구간 덕에 임의 시점 평가 가능 — 10K step마다 probe 자동 실행 |

컴퓨트 플랜: S(22M)는 5090 단일 GPU로 ablation 전수(마스킹 가족, 손실, α, 토큰화).
B(86M)는 A100×8 본학습 1회 + 승자 설정 재학습 1회. L은 B에서 스케일링 신호가
보일 때만.

---

## 6. 평가 스위트 — **구현 1순위** (사전학습 코드보다 먼저)

### 6.1 ISLR 프로브 (frozen encoder)

- **attentive probe**(cross-attention pooling, V-JEPA 프로토콜) + linear probe + kNN 병행
- 데이터셋: asl_citizen(2,731cls), wlasl2000, semlex / **ksl_sign636** / autsl / slovo / include
- 보고: 데이터셋별 top-1/5 + **언어 macro 평균** (ASL 편중 은폐 방지)
- 분할: 공개 signer-independent split 우선, 없으면 signer-disjoint 자체 분할
  (signer-dependent 분할의 점수 인플레는 2025년에 공인된 문제)

### 6.2 교차언어 전이

- zero-shot: 사전학습에서 본 적 없는 언어의 ISLR을 kNN/linear로
- few-shot: 클래스당 1/5/10샷 probe — 저자원 수화 시나리오
- 사전학습 코퍼스 ablation: {ASL-only} vs {ASL+유럽} vs {전체 11개 언어},
  KSL·TR·RSL 전이 성능 비교 → "멀티링구얼 사전학습이 저자원 수화를 돕는가"
- KSL 187k는 (a) 전이 타깃 (사전학습 제외 버전) (b) 사전학습 포함 버전 양쪽 실험

### 6.3 ForeSign-AB: Sign Anticipation Benchmark (신규 기여, 주석 불필요)

- 구성: 사전형 데이터(asl_citizen, semlex, ksl_sign636, autsl)의 기존 테스트 분할에서
  **앞 25% / 50% / 75% / 100% 관찰 조건**의 ISLR 정확도 곡선 (AUC-anticipation)
- 부가 지표: 잠재 예측 오차 vs 지평(horizon) 곡선; 클래스별 "onset 정보량" 분석
  (어느 수화가 초반에 식별 가능한가 — 음운론적으로 흥미로운 부산물)
- 벤치마크 정의(분할·관찰 윈도·지표)를 코드와 함께 공개 → 독립 기여물,
  NeurIPS D&B 또는 본 논문 섹션
- 주의: 일부 수화는 onset만으로 자명할 수 있음 — per-class 분석을 벤치마크
  설계에 포함해 triviality 비판 선제 차단

### 6.4 SLT 파인튜닝 (frozen → partial unfreeze)

- frozen ForeSign 인코더 + mBART-50 (또는 T5) 디코더, 인코더 출력을 length-adapter로 축약
- 주 평가: **How2Sign, OpenASL** (BLEU + BLEURT + chrF)
- 부 평가: PHOENIX14T — train/test 누수(~5%)가 공인됐으므로 **참고치로만**, 본문에 주석
- 텍스트 페어는 H5 빌드 파이프라인의 원본 캡션에서 복원 (선행 확인 필요 — §8 리스크)

### 6.5 진단 프로브 (dataset fingerprint 검증)

- **dataset-ID probe**: frozen 표현에서 22개 데이터셋 분류 — 낮을수록 좋음
  (per-region 정규화 + 표현 학습의 fingerprint 억제 정량화)
- **language-ID probe**: 11개 언어 분류 — 높을수록 좋음 (언어 정보는 보존되어야)
- 두 프로브의 괴리가 "데이터셋이 아닌 언어를 학습했다"는 주장의 증거
- conf-강건성 스트레스: 테스트 시 conf 저하/관절 드롭 시뮬레이션 → 성능 곡선

### 6.6 베이스라인 (모두 동일 토큰화·인코더·데이터에서)

1. **MSM** (좌표 Smooth-L1 복원) — v4 목적함수의 재현 = MAMP/SkeletonMAE 계열 대리
2. **S-JEPA port** — 관절별 토큰 + CE/centering 손실 원형 이식 (첫 공개 재구현)
3. **BYOL temporal contrastive** — two-view 계열 대리
4. from-scratch supervised (probe 데이터셋 직접 학습)
5. 외부 비교: Uni-Sign, SHuBERT 공개 수치 (입력 모달리티 차이를 명시한 참고 비교)

---

## 7. 신규성 방어 논리 (리뷰 대응 선제 정리)

| 예상 공격 | 방어 |
|---|---|
| "S-JEPA가 이미 스켈레톤 JEPA" | S-JEPA는 60~120클래스 액션, 3D Kinect, 단일 도메인. ForeSign은 언어 도메인(수천 어휘·문장), 2D+conf 노이즈, 11개 언어. 마스킹(M1/M2)·평가(anticipation·교차언어)가 전부 다름. S-JEPA를 베이스라인으로 직접 비교 |
| "SHuBERT가 이미 수화 SSL SOTA" | SHuBERT는 비디오 crop 입력·ASL 단일·이산 클러스터 예측(HuBERT 패러다임). ForeSign은 포즈·멀티링구얼·연속 잠재 예측(JEPA 패러다임). 프라이버시(포즈)와 교차언어가 차별축 |
| "포즈는 RGB보다 약함" | 비교는 포즈 트랙(Uni-Sign pose, Sigma, SSL-SLR) 내부에서. 프라이버시-바이-디자인 포지셔닝(동의 없는 생체정보 스크래핑 비판에 대한 구조적 응답). mouthing 손실은 limitation에 명시 |
| "2D라서 안 될 것" | 2D→3D lifting ablation 1개로 차단. S-JEPA의 "노이즈 좌표 복원은 낭비" 논리가 2D 추정 노이즈에서 더 강하게 성립 |
| "anticipation이 자명" | per-class onset 분석 내장, 100% 관찰 대비 상대 곡선으로 보고 |

---

## 8. 리스크 및 완화

| 리스크 | 가능성 | 완화 |
|---|---|---|
| 표현 붕괴 | 중 | EMA 필수(S-JEPA: 제거 시 1.6%로 붕괴), 랭크/분산 모니터링 상시 로깅, CE+centering fallback, S 스케일에서 조기 검증 |
| SLT용 텍스트 페어가 H5 파이프라인에 없음 | **확인 필요** | 빌드 스크립트에서 캡션 보존 여부 즉시 확인 — 없으면 yt_asl/openasl/how2sign 원본에서 재추출 (평가 스위트 1순위인 이유) |
| 토큰 그룹화(17토큰)가 손모양 해상도 부족 | 중 | S 스케일에서 토큰화 ablation (17 vs 손가락 분해 27 vs 관절별) 최우선 수행 |
| 사전형(단어) 클립 편중 → 문장 표현 약화 | 중 | Stage 2 cooldown을 문장 데이터 전용으로; SLT probe를 10K step마다 추적 |
| S-JEPA 재구성 오류 (코드 비공개) | 저 | MAMP 코드 + ECVA 전문 기반 재구성임을 명시; 저자 이메일 문의 병행 |
| ASL 70% 편중 | 저 | temperature sampling + 언어 macro 보고 + 코퍼스 ablation |

---

## 9. 마일스톤 (석사 일정 기준)

| 단계 | 기간 | 산출물 |
|---|---|---|
| P0: 평가 스위트 | 6주 | ISLR/attentive probe, 진단 프로브, ForeSign-AB 정의, 텍스트 페어 확인. **MSM(v4 가중치) probe 수치 = 베이스라인 확보** |
| P1: ForeSign-S + ablation | 8주 | 5090에서 손실(L1 vs CE)·마스킹 가족·토큰화·α sweep → 승자 설정 |
| P2: ForeSign-B 본학습 | 6주 | A100×8 + cooldown, 전체 평가표, 교차언어 실험 |
| P3: 집필 | 6주 | 방법 논문 (CVPR/ICCV 또는 ACL/EMNLP) + ForeSign-AB (D&B 분리 또는 통합) |
| P4 (이후): 공간 담화 추적 | 박사 연계 | frozen ForeSign 위 referent-tracking — 1순위 연구 주제로 복귀 |

## 10. 리포지토리 골격 (신규, v4와 분리)

```
foresign/
├── configs/                  # S/B/L + ablation YAML
├── foresign/
│   ├── data/                 # H5 인덱스, 언어 temperature 샘플러, view 증강(클린/뷰 분리)
│   ├── tokenizer.py          # part-aware patchify (17 spatial groups), part-validity 처리
│   ├── models/               # encoder(RoPE+RMSNorm+SwiGLU), predictor, EMA wrapper
│   ├── masking.py            # M1/M2/M3 가족, conf-가중 saliency, Gumbel top-k
│   ├── losses.py             # L1-latent (기본), CE+centering (fallback)
│   ├── monitors.py           # 붕괴 진단: 유효랭크, 특이값, 분산
│   └── trainer.py            # DDP, warmup-constant-decay, anytime checkpoint
├── evals/                    # ★ 먼저 구현
│   ├── islr_probe.py         # attentive/linear/kNN
│   ├── anticipation.py       # ForeSign-AB
│   ├── crosslingual.py       # zero/few-shot 전이
│   ├── slt_finetune.py       # mBART 디코더
│   └── diagnostics.py        # dataset-ID/language-ID probe, conf 스트레스
└── docs/
```

## 11. 참고 문헌 (핵심)

- S-JEPA: Abdelfattah & Alahi, ECCV 2024 — [ECVA PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf)
- MAMP: [arXiv:2308.07092](https://arxiv.org/abs/2308.07092), [code](https://github.com/maoyunyao/MAMP)
- V-JEPA: [arXiv:2404.08471](https://arxiv.org/abs/2404.08471) / V-JEPA 2: [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
- SHuBERT: [arXiv:2411.16765](https://arxiv.org/abs/2411.16765) (ACL 2025)
- Uni-Sign: [arXiv:2501.15187](https://arxiv.org/abs/2501.15187) (ICLR 2025)
- Sigma: [arXiv:2509.21223](https://arxiv.org/abs/2509.21223) / SSL-SLR: [arXiv:2509.05188](https://arxiv.org/abs/2509.05188)
- 평가 비판: [arXiv:2510.25434](https://arxiv.org/abs/2510.25434), 재현성: [arXiv:2603.13240](https://arxiv.org/abs/2603.13240), PHOENIX 오염: [Frontiers in AI 2026](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1743223/full)

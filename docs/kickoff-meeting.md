# KAMP 2026 팀 킥오프 회의 안건 · 체크리스트 · 회의록

> 목적: 실제 KAMP 문제가 공개되었을 때 3명이 즉시 같은 방식으로 분석·실험을 시작할 수 있도록 팀 운영체계를 확정한다.  
> 팀 구성: 개발자 2인 + 제조업 도메인 전공 1인

---

## 0. 회의 사안

- [ ] 팀의 기본 경쟁 전략
- [ ] 3인의 Primary / Secondary 역할
- [ ] 공통 ML 작업 Pipeline
- [ ] 실제 문제 공개 전 Mock Competition 계획
- [ ] 차별화 전략 후보 2~3개
- [ ] Git / 실험 기록 / 파일 관리 방식

> **팀 기본 원칙 후보 제안**  
> “가장 복잡한 모델을 만드는 것이 아니라, 주어진 제조 문제에서 가장 설득력 있는 분석과 해결책을 만든다.”

---

# 1. 팀 방향 합의 — 우리는 무엇으로 승부할 것인가? 

## 논의할 것

- 단순 모델 성능 극대화가 우선인가?
  - 생각보다 경진대회 기출 기준으로는 중요하진 않았다. (30% 점수)
- 분석 깊이와 제조업 해석을 어느 정도까지 가져갈 것인가?
  - 오히려 이 쪽이 중요하다. (70% 점수)
- 복잡한 최신 모델보다 강한 baseline + 검증 + domain insight를 우선할지에 대한 여부
  - 오히려 이 쪽이 중요하다. 

- 팀 결과 방향성 예시:
```text
좋은 예측 성능
+
신뢰 가능한 Validation
+
왜 그런 결과가 나왔는가
+
어디에서 실패하는가
+
제조 현장에서 어떻게 사용할 것인가
```

### 오늘 결정

- 팀 경쟁 전략:
  - 분석/해석 포커스
- 우리가 피할 전략:
  - 최적화/성능 극대화 
- 실제 문제 공개 후 핵심 주제 확정 시점:
  - 9월 21일에 문제 공개
  - 9월 21~23일 사이에 온라인/오프라인 회의 

---

# 2. 역할 분배 (10분)

역할은 인당 주 역할 + 보조 역할로 정한다.

| 구성원 | Primary | Secondary | 최종 책임 |
|---|---|---|---|
| 뫄뫄 |  |  | 실험 설계의 타당성 |
| 솨솨 |  |  | 실행 가능한 분석 Pipeline |
| 봐봐 |  |  | 제조 관점의 타당성 |

## 분배 해야할 핵심 책임 

### ML / Experiment
- Problem Definition
- Target / Prediction Horizon
- Validation
- Metric
- Baseline 및 모델 비교
- Experiment ladder 관리

### Data / Pipeline
- 데이터 로딩 및 정제
- EDA 코드
- preprocessing
- 모델 실행 pipeline
- HPO / challenger model
- 결과 저장 및 재현성
- 데이터 전처리 파이프라인 트래킹 및 설명 가능해야함. (다이어그램 한 장)

### Domain / Manufacturing
- Feature 의미 파악
- 공정 구조 조사
- Domain leakage 검토
- Domain feature 제안
- 이상치의 제조적 의미 검토
- False Positive / False Negative 비용 해석
- SHAP 및 모델 결과 sanity check
- 제조업 관점 발표 스토리

### 오늘 결정

- @theguardianofthebed: Data/Pipeline
- @junseop1228: ML / experiment 
- @tootoo: Domain / Manufacturing
- 최종 Experiment Decision 담당: 세 명 조율
- 발표 통합 담당: (미정) (당장은 정할 필요없이 이따 결정)

---

# 3. 팀 공통 작업 Pipeline 확정 

실제 데이터가 들어오면 어떠한 순서를 준수할 것인가?
- 초안:
```text
[0] 문제 / 규칙 읽기
 ↓
[1] 제조 문제 정의
 ↓
[2] Target / Prediction Horizon 정의 (모델이 예측하고자 하는 최종 목표 값 및 얼마나 먼 미래까지의 범위를 예측할 것인지 정의)
 ↓
[2-1] Manufacturing Interpretation (제조업 관점에서의 문제 해석)
 ↓
[3] Data Audit (데이터의 품질, 무결성, 보안, 그리고 규정 준수 여부를 체계적으로 평가하고 검증)
 ↓
[4] Leakage Audit (모델 학습 과정에서 데이터 누수(Data Leakage)가 발생했는지 파악하고 검증하는 작업)
 ↓
[5] Validation 설계 (학습된 모델이 새로운 데이터에서도 예측을 잘하는지(일반화 성능)를 평가하기 위해 데이터를 어떻게 나누고 검증할지 계획)
 ↓
[6] Metric 결정 (모델이 예측한 결과가 실제 정답과 비교해 얼마나 우수한지 객관적으로 측정할 기준을 선택)
 ↓
[7] Dummy / Simple Baseline (본격적인 복잡한 모델을 만들기 전에 구축하는 기준점(비교 대상) 역할을 하는 가장 쉽고 간단한 모델 설계)
 ↓
[8] Strong Baselines (현재 가진 데이터와 기술로 달성할 수 있는 현실적이고 강력한 하한선 모델을 설정)
 ↓
[9] Feature Engineering (가공되지 않은 원시 데이터(Raw Data)를 머신러닝 알고리즘이 이해하고 최적의 성능을 낼 수 있도록 특성(Feature)을 선택, 생성, 변환하는 전체 과정과 전략을 기획)
 ↓
[10] Error Analysis (모델이 예측에 실패한 원인을 체계적으로 파악하고, 성능을 효율적으로 개선하기 위한 전략과 기준을 짜는 것)
 ↓
[11] HPO (모델의 성능을 가장 높여주는 최적의 외부 설정값(하이퍼파라미터) 조합을 자동으로 찾아내는 과정과 그 체계를 설계)
 ↓
[12] Ensemble / AutoML Challenger (최적의 모델을 도출하기 위해 자동화된 머신러닝(AutoML) 및 앙상블(Ensemble) 기법을 경쟁(Challenger) 구조로 배치하여 검증)
 ↓
[13] XAI / SHAP (인공지능(AI)이 내린 예측 결과의 이유를 사람이 이해할 수 있도록 설명 가능한 AI(XAI) 기법 중 하나인 SHAP을 활용해 분석·해석하는 구조나 방식을 설계)
 ↓
[14] Final Model / Submission / Report 
```

## 확정된 팀 규칙

- [ ] Validation은 모델 튜닝 전에 결정한다. (Data Leakage, overfitting)
- [ ] Test / Public Leaderboard를 내부 Validation처럼 반복 사용하지 않는다. (Data leakage, overfitting)
- [ ] 한 실험에서는 가능한 한 하나의 주요 변경만 적용한다.
- [ ] 성능 개선 시 “왜 좋아졌는가?”에 대한 가설을 기록한다.
- [ ] Leakage 가능성을 매 주요 실험에서 재검토한다.
- [ ] Feature importance를 인과관계라고 표현하지 않는다.
- [ ] 같은 Validation / Metric 조건에서 모델을 비교한다.
- [ ] 모델 성능뿐 아니라 제조업 관점의 FP/FN 비용을 확인한다.
- [ ] 도메인에 대한 기초 지식 공유한 후 Data audit 3단계 시작.
- [ ] 소통: 모르는거 애매한거 있으면 확실히 질문하고, 비판적인 의견/태클이 있을수록 좋다.  
- [ ] 그 외 더 추가...

### 수정 / 추가할 팀 규칙 후보

---

# 4. Mock Competition 계획 

Mock의 목적은 모델 사용법을 하나씩 배우는 것이 아니라  
3명이 실제 대회처럼 전체 Pipeline을 함께 완주하는 것이다.

## Mock Sprint 1 — KAMP 기출 
### 문제: : 제조기업의 생산성 향상 및 작업환경 개선을 위한 아이디어를 제시하고 인공지능 알고리즘으로 구현
- 데이터 종류
  - 소성가공 품질보증 AI 데이터셋 
> 데이터셋 다운로드 링크: (추후 첨부 예정)
### 스케줄 
```text
[0] 문제 / 규칙 읽기
 ↓
[1] 제조 문제 정의 
 ↓
[2] Target / Prediction Horizon 정의 (Deadline: 8.30)
 ↓
[2-1] Manufacturing Interpretation (제조업 관점에서의 문제 해석) (Deadline: 8.31)  
 ↓
[3] Data Audit
 ↓ (Deadline: 9.01)
1 차 회의 (9.02 19:00, Discord)
 ↓ 
[4] Leakage Audit (모델 학습 과정에서 데이터 누수(Data Leakage)가 발생했는지 파악하고 검증하는 작업) (중간중간 계속 할 필요가 있다.)
 ↓
[5] Validation 설계 (학습된 모델이 새로운 데이터에서도 예측을 잘하는지(일반화 성능)를 평가하기 위해 데이터를 어떻게 나누고 검증할지 계획)
 ↓
[6] Metric 결정 (모델이 예측한 결과가 실제 정답과 비교해 얼마나 우수한지 객관적으로 측정할 기준을 선택)
 ↓
[7] Dummy / Simple Baseline (본격적인 복잡한 모델을 만들기 전에 구축하는 기준점(비교 대상) 역할을 하는 가장 쉽고 간단한 모델 설계)
 ↓
[8] Strong Baselines (현재 가진 데이터와 기술로 달성할 수 있는 현실적이고 강력한 하한선 모델을 설정)
 ↓
[9] Feature Engineering (가공되지 않은 원시 데이터(Raw Data)를 머신러닝 알고리즘이 이해하고 최적의 성능을 낼 수 있도록 특성(Feature)을 선택, 생성, 변환하는 전체 과정과 전략을 기획)
 ↓
[10] Error Analysis (모델이 예측에 실패한 원인을 체계적으로 파악하고, 성능을 효율적으로 개선하기 위한 전략과 기준을 짜는 것)
 ↓
[11] HPO (모델의 성능을 가장 높여주는 최적의 외부 설정값(하이퍼파라미터) 조합을 자동으로 찾아내는 과정과 그 체계를 설계)
 ↓
[12] Ensemble / AutoML Challenger (최적의 모델을 도출하기 위해 자동화된 머신러닝(AutoML) 및 앙상블(Ensemble) 기법을 경쟁(Challenger) 구조로 배치하여 검증)
 ↓
[13] XAI / SHAP (인공지능(AI)이 내린 예측 결과의 이유를 사람이 이해할 수 있도록 설명 가능한 AI(XAI) 기법 중 하나인 SHAP을 활용해 분석·해석하는 구조나 방식을 설계)
 ↓
[14] Final Model / Submission / Report
```
### 완료 조건

- [ ] 문제 정의
- [ ] Target / Feature 확인
- [ ] EDA
- [ ] Leakage audit
- [ ] Validation
- [ ] Metric
- [ ] Simple baseline
- [ ] Strong baseline
- [ ] Feature engineering 1개 이상
- [ ] Error analysis
- [ ] XAI
- [ ] 제조업 관점 해석
- [ ] 5분 결과 공유
- [ ] 그 외...

### Sprint 1에서 검증할 팀 운영 문제

- Git 협업이 불편하지 않은가?
- 두 개발자의 작업이 중복되는가?
- Domain 담당이 너무 늦게 참여하고 있지 않은가?
- Experiment 기록 방식이 충분한가?
- 결과를 다른 팀원이 재현할 수 있는가?
- 그 외...

---

## Mock Sprint 2 — KAMP 또는 더 어려운 제조 데이터

목표:

- 처음 보는 데이터에서 문제를 정의
- 결측 / 다수 feature / 공정 구조 등 더 현실적인 문제 경험
- Domain feature와 validation 설계를 더 적극적으로 연습
- Sprint 1에서 발견된 협업 문제 수정

### 결정사항

- Sprint 1 시작일:
- Sprint 1 종료일:
- Sprint 2 후보 데이터:
- Sprint 2 시작 예정:
- Mock 결과 공유 방식:

---

# 5. 차별화 전략 후보 선정 

추후 문제 및 데이터셋 제시 후 구체적인 주제는 선정하겠나, 차별화 전략의 방향성 후보 정도는 결정해둘 필요가 있다고 판단된다.

## 후보 A — Error Analysis를 깊게

```text
전체 성능
→ 불량/고장 유형별 성능
→ 특정 공정 조건별 성능
→ 반복적으로 틀리는 구간 발견
→ 원인 가설
→ Feature / Model 수정
→ 재실험
```

## 후보 B — 제조 비용 / 운영 의사결정

예:

```text
False Negative
→ 불량 누락 / 고장 미탐지

False Positive
→ 불필요한 검사 / 정비 / 공정 중단
```

Threshold 변화에 따른 성능과 운영 부담을 함께 분석한다.

## 후보 C — Domain Feature + Ablation

```text
Raw Features
→ + Domain Feature A
→ + Domain Feature B
→ + Interaction / Ratio / Difference
```

각 feature가 실제 개선에 기여했는지 ablation으로 확인한다.

## 후보 D — “이 점수를 믿을 수 있는가?” Validation 분석

예:

```text
Random Split 성능
vs
Group / LOT / Time-aware Split 성능
```

점수 차이가 생기는 이유를 공정 구조와 연결해 분석한다.

### 선택할 2~3개 후보

1.
2.
3.

> 최종 차별화 전략은 실제 데이터 EDA + Domain Audit 후 확정한다.

---

# 6. Git / 파일 / 실험 기록 방식 
> 현재 적용된 방식을 정리해뒀으며 어떻게 수정할건지 결정 필요.

## 권장 Repository 구조

```text
kamp-2026/
├─ README.md
├─ docs/
│  ├─ kickoff-meeting.md
│  ├─ competition-brief.md
│  └─ decisions.md
├─ data/
│  ├─ raw/          # Git에 대용량 원본은 올리지 않는 것을 기본값으로 검토
│  └─ processed/
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_baseline.ipynb
│  └─ ...
├─ src/
├─ experiments/
│  └─ experiment_log.csv
├─ outputs/
└─ requirements.txt
```

## Git 규칙 후보

- `main`: 재현 가능한 안정 버전
- 개인 작업: 별도 branch
- 의미 있는 단위로 commit
- 대용량 dataset / model artifact는 `.gitignore`
- 실험 결과는 코드와 함께 기록

## Experiment Log 최소 항목

| ID | 날짜 | 담당 | 변경사항 | Validation | Metric | Score | 가설/해석 | 다음 실험 |
|---|---|---|---|---|---|---|---|---|
| E001 | | | | | | | | |

### 결정 사항

- Repository:
- Branch 규칙:
- Notebook naming:
- Experiment ID 규칙:
- Dataset 공유 방법:
- 결과 공유 위치:
- **구글 드라이브 링크**: 


# KAMP 2026 팀 킥오프 회의 안건 · 체크리스트 · 회의록

> 목적: 실제 KAMP 문제가 공개되었을 때 3명이 즉시 같은 방식으로 분석·실험을 시작할 수 있도록 팀 운영체계를 확정한다.  
> 팀 구성: 개발자 2인 + 제조업 도메인 전공 1인

---

## 0. 오늘 회의의 성공 조건

회의 종료 시 아래 6개가 반드시 확정되어 있어야 한다.

- [ ] 팀의 기본 경쟁 전략
- [ ] 3인의 Primary / Secondary 역할
- [ ] 공통 ML 작업 Pipeline
- [ ] 실제 문제 공개 전 Mock Competition 계획
- [ ] 차별화 전략 후보 2~3개
- [ ] Git / 실험 기록 / 파일 관리 방식

> **팀 기본 원칙 후보**  
> “가장 복잡한 모델을 만드는 것이 아니라, 주어진 제조 문제에서 가장 설득력 있는 분석과 해결책을 만든다.”

---

# 1. 팀 방향 합의 — 우리는 무엇으로 승부할 것인가? (10분)

## 논의할 것

- 단순 Leaderboard 성능 극대화가 우선인가?
- 분석 깊이와 제조업 해석을 어느 정도까지 가져갈 것인가?
- 복잡한 최신 모델보다 강한 baseline + 검증 + domain insight를 우선한다는 데 동의하는가?
- 실제 문제 공개 후 최종 주제를 언제 확정할 것인가?

## 권장 기본 방향

**Strong Model + Domain Feature + Error Analysis + Manufacturing Decision**

즉,

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
  - 
- 우리가 피할 전략:
  - 
- 실제 문제 공개 후 핵심 주제 확정 시점:
  - 

---

# 2. 역할 분배 (10분)

역할은 완전히 분리하지 않고 **Primary + Secondary**로 정한다.

| 구성원 | Primary | Secondary | 최종 책임 |
|---|---|---|---|
| 개발자 A | ML / Validation / Experiment | 분석 해석 / 발표 | 실험 설계의 타당성 |
| 개발자 B | Data / Pipeline / Modeling | HPO / 재현성 | 실행 가능한 분석 Pipeline |
| 제조 도메인 | Domain / Feature / Manufacturing Interpretation | EDA / Error Analysis | 제조 관점의 타당성 |

## 역할별 핵심 책임

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

- 개발자 A:
- 개발자 B:
- 제조 도메인:
- 최종 Experiment Decision 담당:
- 발표 통합 담당:

---

# 3. 팀 공통 작업 Pipeline 확정 (15분)

실제 데이터가 들어오면 아래 순서를 기본으로 한다.

```text
[0] 문제 / 규칙 읽기
 ↓
[1] 제조 문제 정의
 ↓
[2] Target / Prediction Horizon 정의
 ↓
[3] Data Audit
 ↓
[4] Leakage Audit
 ↓
[5] Validation 설계
 ↓
[6] Metric 결정
 ↓
[7] Dummy / Simple Baseline
 ↓
[8] Strong Baselines
 ↓
[9] Feature Engineering
 ↓
[10] Error Analysis
 ↓
[11] HPO
 ↓
[12] Ensemble / AutoML Challenger
 ↓
[13] XAI / SHAP
 ↓
[14] Manufacturing Interpretation
 ↓
[15] Final Model / Submission / Report
```

## 팀 규칙 후보

- [ ] Validation은 모델 튜닝 전에 결정한다.
- [ ] Test / Public Leaderboard를 내부 Validation처럼 반복 사용하지 않는다.
- [ ] 한 실험에서는 가능한 한 하나의 주요 변경만 적용한다.
- [ ] 성능 개선 시 “왜 좋아졌는가?”에 대한 가설을 기록한다.
- [ ] Leakage 가능성을 매 주요 실험에서 재검토한다.
- [ ] Feature importance를 인과관계라고 표현하지 않는다.
- [ ] 같은 Validation / Metric 조건에서 모델을 비교한다.
- [ ] 모델 성능뿐 아니라 제조업 관점의 FP/FN 비용을 확인한다.

### 오늘 수정 / 추가할 팀 규칙

- 
- 
- 

---

# 4. Mock Competition 계획 (15분)

Mock의 목적은 모델 사용법을 하나씩 배우는 것이 아니라  
**3명이 실제 대회처럼 전체 Pipeline을 함께 완주하는 것**이다.

## Mock Sprint 1 — AI4I Predictive Maintenance

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

### Sprint 1에서 검증할 팀 운영 문제

- Git 협업이 불편하지 않은가?
- 두 개발자의 작업이 중복되는가?
- Domain 담당이 너무 늦게 참여하고 있지 않은가?
- Experiment 기록 방식이 충분한가?
- 결과를 다른 팀원이 재현할 수 있는가?

---

## Mock Sprint 2 — KAMP 또는 더 어려운 제조 데이터

목표:

- 처음 보는 데이터에서 문제를 정의
- 결측 / 다수 feature / 공정 구조 등 더 현실적인 문제 경험
- Domain feature와 validation 설계를 더 적극적으로 연습
- Sprint 1에서 발견된 협업 문제 수정

### 오늘 결정

- Sprint 1 시작일:
- Sprint 1 종료일:
- Sprint 2 후보 데이터:
- Sprint 2 시작 예정:
- Mock 결과 공유 방식:

---

# 5. 차별화 전략 후보 선정 (15분)

지금 최종 주제를 확정하지 않는다.  
실제 데이터의 구조를 본 뒤 선택할 수 있도록 **후보 전략을 2~3개 유지**한다.

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

### 오늘 선택할 2~3개 후보

1.
2.
3.

> 최종 차별화 전략은 실제 데이터 EDA + Domain Audit 후 확정한다.

---

# 6. Git / 파일 / 실험 기록 방식 (10분)

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

### 오늘 결정

- Repository:
- Branch 규칙:
- Notebook naming:
- Experiment ID 규칙:
- Dataset 공유 방법:
- 결과 공유 위치:

---

# 7. 실제 문제 공개 직후 24시간 행동 계획 (5분)

## 첫 단계에서 하지 않을 것

- [ ] 곧바로 XGBoost / Transformer부터 돌리지 않는다.
- [ ] Leaderboard 점수만 보고 모델을 선택하지 않는다.
- [ ] EDA 전에 feature를 무작정 제거하지 않는다.
- [ ] Random Split을 자동으로 채택하지 않는다.

## 첫 단계에서 할 것

### 개발자 A
- 문제 / Target / Metric / Validation 후보 정리

### 개발자 B
- 데이터 로딩 / schema / 기본 EDA / missing / duplicate 확인

### Domain 담당
- 데이터 설명서 / 공정 / feature 의미 정리
- 예상 leakage / domain feature / FP-FN 의미 제안

### 3인 공동
- Data Risk Map 작성
- 첫 Validation 결정
- E0 Dummy / E1 Simple Baseline 정의
- 차별화 전략 최종 후보 선정

---

# 8. 회의 종료 체크

- [ ] 팀 전략을 한 문장으로 말할 수 있다.
- [ ] 각자의 Primary / Secondary 역할이 정해졌다.
- [ ] 데이터가 들어오면 첫 5단계가 무엇인지 모두 안다.
- [ ] 첫 Mock의 일정과 완료 조건이 정해졌다.
- [ ] 차별화 후보가 2~3개로 좁혀졌다.
- [ ] Git / Experiment Log 방식을 정했다.
- [ ] 다음 회의 날짜 / 조건을 정했다.

---

# 9. 회의록

## 회의 정보

- 날짜:
- 참석자:
- 회의 시간:

## 결정사항

### 1. 팀 전략
-

### 2. 역할
- 개발자 A:
- 개발자 B:
- 제조 도메인:

### 3. Pipeline 수정사항
-

### 4. Mock Sprint
-

### 5. 차별화 전략 후보
1.
2.
3.

### 6. Git / 실험 관리
-

---

## Action Items

| Action | 담당 | 기한 | 상태 |
|---|---|---|---|
| | | | ⬜ |
| | | | ⬜ |
| | | | ⬜ |
| | | | ⬜ |

---

## 아직 결정하지 않은 사항

- 
- 

## 다음 회의에서 결정할 사항

- 
- 

---

# 팀이 계속 기억할 질문

> **“다음에 어떤 모델을 돌릴까?”가 아니라  
> “지금 어떤 불확실성을 줄이기 위해 어떤 실험을 해야 할까?”**

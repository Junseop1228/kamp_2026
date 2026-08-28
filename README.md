# KAMP 2026 Manufacturing AI Competition

2026 KAMP 제조 AI / 제조 데이터 분석 경진대회를 위한 팀 협업 및 실험 저장소입니다.

현재 단계는 **Competition Readiness / Mock Competition 준비 단계**입니다. 실제 문제와 데이터가 공개되기 전까지 공개 제조 데이터를 이용해 전체 분석 파이프라인과 팀 협업 방식을 검증하고, 실제 대회가 시작되면 같은 구조를 실전 데이터에 적용합니다.

## Team Strategy

우리 팀의 기본 방향은 가장 복잡한 모델 자체를 목표로 하기보다 다음 네 가지를 함께 확보하는 것입니다.

> **Strong Model + Trustworthy Validation + Domain Insight + Manufacturing Decision**

즉, 단순히 높은 validation score를 만드는 데서 끝내지 않고 다음 질문까지 답하는 것을 목표로 합니다.

- 이 validation score를 믿을 수 있는가?
- 어떤 조건에서 모델이 잘못 예측하는가?
- 제조 도메인 지식으로 의미 있는 feature를 만들 수 있는가?
- 개선이 실제인지 leakage / overfitting / 우연인지 어떻게 확인할 것인가?
- False Positive / False Negative가 실제 제조 현장에서 어떤 비용을 만드는가?
- 모델 결과를 품질, 검사, 고장 예방, downtime, yield 등의 의사결정으로 어떻게 연결할 것인가?

## Standard Competition Pipeline

```text
Problem / Rules Review
→ Manufacturing Problem Definition
→ Target / Prediction Horizon
→ Data Audit
→ Leakage Audit
→ Validation Design
→ Metric Selection
→ Dummy / Simple Baseline
→ Strong Baselines
→ Feature Engineering
→ Error Analysis
→ Hyperparameter Optimization
→ Ensemble / AutoML Challenger
→ XAI / SHAP
→ Manufacturing Interpretation
→ Final Model / Submission / Report
```

### Experiment principles

1. Validation design is decided before model tuning.
2. Models are compared under the same validation and metric conditions.
3. Prefer changing one major experimental factor at a time.
4. Every meaningful score improvement should have a hypothesis explaining why it occurred.
5. Leakage is re-checked whenever features, preprocessing, or split logic changes.
6. Feature importance and SHAP describe model dependence, not causal effect.
7. Leaderboard performance does not replace a trustworthy internal validation strategy.

## Team Roles

The detailed ownership will be finalized during the kickoff meeting. The initial structure is:

| Role | Primary responsibility | Secondary responsibility |
|---|---|---|
| Developer A | ML / Validation / Experiment Design | Analysis interpretation / Presentation |
| Developer B | Data / Pipeline / Modeling | HPO / Reproducibility |
| Manufacturing Domain | Domain / Feature / Manufacturing Interpretation | EDA / Error Analysis |

The domain role participates from the beginning of analysis rather than only during final presentation.

## Mock Competition Plan

Before the real competition task is released, mock practice is used to validate the **entire workflow**, not just individual model APIs.

Initial mock dataset:

- **AI4I 2020 Predictive Maintenance**

A mock sprint should ideally complete:

```text
Problem Definition
→ EDA
→ Leakage Audit
→ Validation
→ Baseline
→ Strong Model
→ Feature Engineering
→ Error Analysis
→ XAI
→ Manufacturing Interpretation
```

A later sprint should use a harder or KAMP-origin manufacturing dataset when appropriate.

## Repository Structure

```text
kamp_2026/
├─ README.md
├─ docs/
│  └─ kickoff-meeting.md
├─ data/
│  └─ README.md
├─ notebooks/              # analysis / experiment notebooks when added
├─ src/                    # reusable code when added
├─ experiments/
│  └─ experiment_log.csv
├─ outputs/                # temporary experiment artifacts; ignored by Git
├─ results/                # final reportable artifacts when used
├─ environment.yml
└─ .gitignore
```

The repository may evolve as the real competition structure becomes known.

## Experiment Logging

Meaningful experiments should be recorded in `experiments/experiment_log.csv`.

Minimum questions for each experiment:

1. What changed?
2. What stayed fixed?
3. Was the validation design unchanged?
4. What metric changed and by how much?
5. Why do we think the result changed?
6. Is leakage possible?
7. What experiment should reduce the next uncertainty?

## Data Policy

Raw competition / mock datasets are not committed by default.

- Keep raw data immutable where possible.
- Record the source and acquisition procedure in `data/README.md`.
- Do not commit large model binaries, caches, temporary outputs, or secrets.
- Final lightweight figures / metrics intended for reporting may be tracked separately from temporary outputs.

## Environment

`environment.yml` is the environment specification placeholder for the project. Package versions should be finalized deliberately rather than guessed before the team decides the competition environment.

Target development environment for the preparation curriculum is Python 3.11 unless the actual competition requirements require a change.

## Documents

- [Team kickoff agenda, checklist, and meeting minutes](docs/kickoff-meeting.md)

## Current Status

- [x] Repository created
- [x] Team kickoff template drafted
- [x] Initial experiment log template added
- [ ] Team roles finalized
- [ ] Mock Sprint 1 completed
- [ ] Validation / metric standards finalized for first mock
- [ ] Real competition brief created after official task release

---

### Guiding Question

> **Do not ask only “Which model should we run next?” Ask “Which experiment will reduce the uncertainty that matters most right now?”**

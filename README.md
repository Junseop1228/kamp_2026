# 사출성형 품질불량 사전예측 및 검사 우선순위 프레임워크

제6회 K-인공지능 제조데이터 분석 경진대회(일반 국민/대학(원)생 부문) 문제 ① 제출 코드다. 기계 신호로 잡을 수 있는 불량은 검사 우선순위로 앞당기고, 구조적으로 잡을 수 없는 몫은 숫자로 드러낸다.

7단계: Adapter → Diagnose → Evaluate → Analyze → Prioritize → Infer → Report. 명령 하나로 전처리부터 학습·추론·결과·그림 생성까지 끝난다.

## 실행

```bash
conda env create -f environment.yml
conda run -n kamp2026 python run.py
```

실행 전에 대회 데이터 4개를 `data/raw/injection/`에 둔다.

- `moldset_labeled_cn7.csv`, `moldset_labeled_rg3.csv` (학습용, 라벨 있음)
- `moldset_unlabeled_cn7.csv`, `moldset_unlabeled_rg3.csv` (라벨 없음)

CPU만 쓴다. 소요 시간은 PC에 따라 15~40분이며 대부분 가이드북식 오토인코더 학습이다. 그림의 한글은 Windows의 맑은 고딕을 쓴다.

## 결과 위치 (`results/`)

| 경로 | 내용 |
|---|---|
| `cn7/`, `rg3/` `diagnose_*` | 데이터 진단: 동일입력 쌍 구조, 반올림 민감도, 이상치, 고상관 변수 쌍, 분포 비교 |
| `cv_scores.csv`, `selection.csv` | 후보 6개 + 무작위 기준의 5-fold × 10회 F1·PR-AUC, 사전 등록 선택 규칙 결과 |
| `model_summary.json` | 선택 모델, 확률 보정 품질, 누수 대조, 입력 20개 민감도, 블록 분할 점검 |
| `importance.csv` 외 분석 파일 | 영향 변수, 2×2 상호작용, FN 유형·집중 조건, FP 조건, 도메인 대조, 2D 부분의존도 |
| `budget_table.csv`, `sampling_combo.csv`, `alarm_*.csv` | 검사량별 포착, 무작위 추가 검사 결합, 배치 알람 |
| `figures/` | 보고서 그림 |
| `run_record.json` | 실행 환경, seed, 시각, 커밋 해시 |

## 평가 항목별 근거 파일

경로는 `results/{cn7,rg3}/` 기준이다. 그림은 `results/figures/`에 있다.

| 서면평가 항목 | 근거 파일 | 그림 |
|---|---|---|
| 1. 데이터 이해 및 진단 | `diagnose_summary.json`, `diagnose_rounding.csv`, `diagnose_outliers.csv`, `diagnose_pairs.csv`, `diagnose_shift.csv` | `fig1_*` |
| 2. AI 예측모델 개발 | `cv_scores.csv`, `selection.csv`, `model_summary.json`, `leakage_contrast.csv`, `sensitivity_20.csv`, `block_split.csv` | `fig2_*` |
| 3. 영향요인 및 오류분석 | `importance.csv`, `interaction_2x2.csv`, `pdp_2d.csv`, `fn_types.csv`, `fn_concentration.csv`, `fp_conditions.csv`, `domain_crosscheck.csv`, `leakage_explore.csv` | `fig3_*` |
| 4. 현장 활용방안 | `budget_table.csv`, `sampling_combo.csv`, `alarm_labeled.csv`, `alarm_test.csv`, `prioritize_summary.json`, `predictions.csv`의 `flag_rate` | `fig4_*` |
| 5. 창의성·차별성 | `model_summary.json`의 `calibration`, `fn_types.csv`의 신호 없음, `selection.csv`의 `beats_floor` | — |
| 6. 코드 및 재현성 | `../run_record.json`, `environment.yml`, `tests/` | — |

## 예측 결과 파일

과제 원문의 "테스트데이터"가 어느 파일인지 정의되어 있지 않아 두 해석 모두에 대한 예측을 낸다.

- **unlabeled 파일 예측**: `results/{cn7,rg3}/predictions.csv`
  - `product_id`(원본 첫 열), `proba`(불량 확률, 사전확률 보정), `rank`(배치 내 순위, 1 = 가장 위험), `flag`(1 = 우선 검사 대상), `group`(동일입력 그룹), `flag_rate`(seed 10개 모델 중 표시 비율 = 판정 확신도)
  - 메타: `predictions_meta.json` (표시 개수, 한계, 알람)
- **labeled 교차검증 예측**: `results/{cn7,rg3}/oof_predictions.csv`
  - 모든 labeled 행을 자신이 학습에 쓰이지 않은 fold의 모델로 예측한 값과 실제 라벨

## 설정과 사전 등록

- `configs/injection.json`: 데이터 경로, 진단 기준, 교차검증, 판정, 모델 설정, 선택 규칙, 분석·우선순위 설정. 모델 결과를 보기 전에 커밋했다.
- `configs/variables_injection.csv`: 변수 사전(AI 초안, 도메인 검토 전)
- `config.py`: 경로와 seed(0~9). seed는 여기서만 정한다.
- 결정과 근거: `docs/03_decision_log.md`, 놓쳤다가 고친 오류: `known_issues/`

## 재현성

- 같은 환경에서 다시 실행하면 `run_record.json`을 뺀 모든 결과 파일이 같다(랜덤포레스트는 결정성을 위해 단일 스레드).
- 모든 텍스트 결과는 UTF-8, LF 줄바꿈이다.
- 테스트: `conda run -n kamp2026 python -m pytest`

## 코드 구조

| 경로 | 역할 |
|---|---|
| `run.py` | 실행 진입점(파일 읽기·쓰기 담당) |
| `src/framework/` | 단계별 모듈: `adapter`, `diagnose`, `evaluate`, `analyze`, `prioritize`, `infer`, `report` |
| `src/models/` | 모델 슬롯: 무작위·편차·가이드북식 오토인코더(`baselines`), 로지스틱·랜덤포레스트·부스팅·앙상블(`candidates`) |
| `tests/` | 자동 테스트(기준 수치, 판정 규칙, 모델 계약, 분석) |
| `results_viewer.ipynb` | 결과 열람용 노트북(계산 없음) |

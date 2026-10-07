"""Prioritize 단계(D-012): 검사량 표, 무작위 표본검사 결합표, 배치 알람. 비용 자료가 없으므로 개수만 보고한다."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def budget_table(y: np.ndarray, rank_frac: np.ndarray, budgets: Sequence[float], k: float, aux: float) -> pd.DataFrame:
    """검사량별 포착 수. 순위 비율이 검사량 안이면 검사한다. 무작위 검사의 기대 Recall = 실제 검사 비율."""
    y, rank_frac = np.asarray(y), np.asarray(rank_frac)
    n_def = int(y.sum())
    rows = []
    for label, b in [*((f"{b:.0%}", b) for b in budgets), ("1k", k), ("aux_k", aux)]:
        insp = rank_frac < b
        n, caught = int(insp.sum()), int((insp & (y == 1)).sum())
        share = n / len(y)
        recall = caught / n_def
        rows.append({"budget": label, "budget_frac": float(b), "inspected": n, "caught": caught, "missed": n_def - caught,
                     "recall": recall, "precision": caught / n if n else float("nan"), "random_recall": share,
                     "lift": recall / share if share else float("nan")})
    return pd.DataFrame(rows)


def sampling_combo(budget: pd.DataFrame, n_rows: int, rates: Sequence[float]) -> pd.DataFrame:
    """우선 검사 + 나머지에서 무작위 r% 추가 검사. 기대 추가 포착 = r × 우선 검사가 놓친 불량(경계·모델 차이·신호 없음 전부)."""
    rows = []
    for b in budget.itertuples():
        for r in rates:
            extra_inspect, extra_catch = r * (n_rows - b.inspected), r * b.missed
            rows.append({"budget": b.budget, "random_rate": float(r), "inspected": b.inspected + extra_inspect,
                         "expected_caught": b.caught + extra_catch, "expected_recall": (b.caught + extra_catch) / (b.caught + b.missed)})
    return pd.DataFrame(rows)


def batch_means(proba: np.ndarray, batch_size: int) -> pd.DataFrame:
    """파일 순서로 연속 배치를 만들어 평균 출력을 잰다. 마지막 배치가 절반보다 작으면 앞 배치에 붙인다."""
    proba = np.asarray(proba, dtype=float)
    starts = list(range(0, len(proba), batch_size))
    if len(starts) > 1 and len(proba) - starts[-1] < batch_size / 2:
        starts.pop()
    ends = starts[1:] + [len(proba)]
    return pd.DataFrame([{"batch": i, "start_row": s, "rows": e - s, "mean_output": float(proba[s:e].mean())}
                         for i, (s, e) in enumerate(zip(starts, ends))])


def alarm_scan(proba: np.ndarray, batch_size: int, threshold: float) -> pd.DataFrame:
    """배치 평균이 기준(교차검증 평가 배치 평균의 최댓값)을 넘으면 알람. 뜻은 "평소와 다름"이지 "불량 급증"이 아니다."""
    out = batch_means(proba, batch_size)
    out["alarm"] = out["mean_output"] > threshold
    return out

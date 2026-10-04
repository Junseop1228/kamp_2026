"""Evaluate 단계: 순위 기반 판정 규칙과 교차검증. 확률 임계값은 쓰지 않는다."""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold


def n_flag(k: float, batch_size: int) -> int:
    """배치에서 표시할 명목 개수. half-up 반올림(파이썬 round의 은행가 반올림 아님), 최소 1."""
    return max(1, math.floor(k * batch_size + 0.5))


def flag_top(scores: np.ndarray, k: float) -> np.ndarray:
    """점수 상위 k%를 표시한다. 경계 점수와 같은 제품은 전부 포함하므로 실제 표시 수 ≥ 명목 수."""
    scores = np.asarray(scores, dtype=float)
    m = n_flag(k, len(scores))
    cutoff = np.sort(scores)[::-1][m - 1]
    return scores >= cutoff


def run_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    g: np.ndarray,
    make_model: Callable[..., Any],
    seeds: Sequence[int],
    max_folds: int,
) -> list[dict]:
    """그룹 단위 StratifiedGroupKFold를 seed마다 반복한다. fold마다 학습 fold 불량률 k로 판정한다."""
    y = np.asarray(y)
    g = np.asarray(g)
    n_def_groups = int(pd.Series(g[y == 1]).nunique())
    if n_def_groups < 2:
        raise RuntimeError(f"학습 불가: 불량 그룹 {n_def_groups}개(2개 이상 필요)")
    n_splits = min(max_folds, n_def_groups)

    records: list[dict] = []
    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(X, y, g)):
            model = make_model(random_state=seed).fit(X.iloc[tr], y[tr])
            proba = np.asarray(model.predict_proba(X.iloc[te]), dtype=float)
            k = float(y[tr].mean())
            flags = flag_top(proba, k)
            records.append({
                "seed": seed, "fold": fold, "idx": te, "proba": proba, "flag": flags, "k": k,
                "n_flag_nominal": n_flag(k, len(te)), "n_flag_actual": int(flags.sum()),
            })
    return records


def score_repeats(records: list[dict], y: np.ndarray) -> pd.DataFrame:
    """반복(seed)마다 전 fold 판정을 합쳐 F1·PR-AUC를 한 번씩 계산한다."""
    y = np.asarray(y)
    rows = []
    for seed in sorted({r["seed"] for r in records}):
        recs = [r for r in records if r["seed"] == seed]
        idx = np.concatenate([r["idx"] for r in recs])
        flags = np.concatenate([r["flag"] for r in recs])
        proba = np.concatenate([r["proba"] for r in recs])
        rows.append({
            "seed": seed,
            "f1": float(f1_score(y[idx], flags, zero_division=0)),
            "pr_auc": float(average_precision_score(y[idx], proba)),
            "n_flag_nominal": sum(r["n_flag_nominal"] for r in recs),
            "n_flag_actual": sum(r["n_flag_actual"] for r in recs),
        })
    return pd.DataFrame(rows)


def dup_ceiling_f1(y: np.ndarray, g: np.ndarray) -> float:
    """중복 상한 F1: 불량이 든 그룹을 전부 표시했을 때의 F1. 동일입력 짝을 구분할 수 없으므로 상한으로 쓴다."""
    y = np.asarray(y)
    g = np.asarray(g)
    flagged = np.isin(g, np.unique(g[y == 1]))
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int(y.sum()) - tp
    return 2 * tp / (2 * tp + fp + fn)


def random_floor_f1(y: np.ndarray) -> float:
    """무작위 하한 F1: 무작위로 불량률만큼 표시하면 F1 ≈ 불량률이다."""
    return float(np.mean(y))

"""Evaluate 단계: 순위 기반 판정 규칙, 교차검증, 사전 등록 선택 규칙(D-007, D-009). 확률 임계값은 쓰지 않는다."""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def n_flag(k: float, batch_size: int) -> int:
    """배치에서 표시할 명목 개수. half-up 반올림(파이썬 round의 은행가 반올림 아님), 최소 1."""
    return max(1, math.floor(k * batch_size + 0.5))


def flag_top(scores: np.ndarray, k: float) -> np.ndarray:
    """점수 상위 k%를 표시한다. 경계 점수와 같은 제품은 전부 포함하므로 실제 표시 수 ≥ 명목 수."""
    scores = np.asarray(scores, dtype=float)
    m = n_flag(k, len(scores))
    cutoff = np.sort(scores)[::-1][m - 1]
    return scores >= cutoff


def _fold_record(model_factory, X, y, tr, te, seed, fold) -> dict:
    model = model_factory(random_state=seed).fit(X.iloc[tr], y[tr])
    proba = np.asarray(model.predict_proba(X.iloc[te]), dtype=float)
    k = float(y[tr].mean())
    flags = flag_top(proba, k)
    return {
        "seed": seed, "fold": fold, "idx": te, "proba": proba, "flag": flags, "k": k,
        "n_flag_nominal": n_flag(k, len(te)), "n_flag_actual": int(flags.sum()),
    }


def run_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    g: np.ndarray,
    make_model: Callable[..., Any],
    seeds: Sequence[int],
    max_folds: int,
    ignore_groups: bool = False,
) -> list[dict]:
    """그룹 단위 StratifiedGroupKFold를 seed마다 반복한다. ignore_groups=True는 누수 대조(보고 전용)용 일반 층화 분할."""
    y = np.asarray(y)
    g = np.asarray(g)
    n_def_groups = int(pd.Series(g[y == 1]).nunique())
    if n_def_groups < 2:
        raise RuntimeError(f"학습 불가: 불량 그룹 {n_def_groups}개(2개 이상 필요)")
    n_splits = min(max_folds, n_def_groups)

    records: list[dict] = []
    for seed in seeds:
        if ignore_groups:
            splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y)
        else:
            splits = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y, g)
        for fold, (tr, te) in enumerate(splits):
            records.append(_fold_record(make_model, X, y, tr, te, seed, fold))
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


def oof_summary(records: list[dict], n_rows: int) -> pd.DataFrame:
    """제품별 교차검증 예측 요약: 반복 평균 확률, 표시 비율(판정 확신도), 배치 내 순위 비율의 중앙값(0 = 최상위)."""
    proba, flag, rank = (np.full((n_rows, len({r['seed'] for r in records})), np.nan) for _ in range(3))
    seed_col = {s: i for i, s in enumerate(sorted({r["seed"] for r in records}))}
    for r in records:
        c = seed_col[r["seed"]]
        proba[r["idx"], c] = r["proba"]
        flag[r["idx"], c] = r["flag"]
        rank[r["idx"], c] = (pd.Series(r["proba"]).rank(method="min", ascending=False).to_numpy() - 1) / len(r["idx"])
    return pd.DataFrame({
        "proba_mean": np.nanmean(proba, axis=1),
        "flag_rate": np.nanmean(flag, axis=1),
        "rank_frac_median": np.nanmedian(rank, axis=1),
    })


def _paired(wide: pd.DataFrame, ref: str, ddof: int) -> tuple[pd.Series, pd.Series]:
    diff = wide[ref].to_numpy()[:, None] - wide.to_numpy()
    return pd.Series(diff.mean(axis=0), index=wide.columns), pd.Series(diff.std(axis=0, ddof=ddof), index=wide.columns)


def select_model(scores: pd.DataFrame, order: Sequence[str], floor: str, ddof: int) -> tuple[str, pd.DataFrame]:
    """선택 규칙(D-009): 평균 F1 최고 모델과의 같은 seed 쌍별 차이 평균 ≤ 차이 표준편차면 '차이 없음', 그중 가장 단순한 모델.

    하한 비교 열(beats_floor)은 같은 쌍별 규칙을 무작위 기준모델에 적용한 보고 항목이며 선택에는 쓰지 않는다(D-010).
    """
    wide = scores.pivot(index="seed", columns="model", values="f1")
    cand = wide[list(order)]
    best = cand.mean().idxmax()                                   # 동점이면 복잡도 순서상 앞(단순한) 모델
    d_mean, d_std = _paired(cand, best, ddof)
    f_mean, f_std = _paired(wide[[*order, floor]], floor, ddof)
    table = pd.DataFrame({
        "model": list(order),
        "f1_mean": cand.mean().to_numpy(),
        "f1_std": cand.std(ddof=ddof).to_numpy(),
        "diff_from_best_mean": d_mean.to_numpy(),
        "diff_from_best_std": d_std.to_numpy(),
    })
    table["no_difference"] = table["diff_from_best_mean"] <= table["diff_from_best_std"]
    chosen = table.loc[table["no_difference"], "model"].iloc[0]
    table["selected"] = table["model"] == chosen
    gain_mean, gain_std = -f_mean[list(order)].to_numpy(), f_std[list(order)].to_numpy()
    table["gain_over_floor_mean"], table["gain_over_floor_std"] = gain_mean, gain_std
    table["beats_floor"] = gain_mean > gain_std
    return chosen, table


def block_split(X: pd.DataFrame, y: np.ndarray, g: np.ndarray, make_model, seed: int, n_blocks: int) -> pd.DataFrame:
    """행 순서 블록 분할 점검(보고 전용): 그룹 첫 행 위치로 연속 블록을 정하고, 블록마다 나머지로 학습해 판정한다."""
    y, g = np.asarray(y), np.asarray(g)
    first = pd.Series(np.arange(len(g))).groupby(g).transform("min").to_numpy()
    block = first * n_blocks // len(g)
    rows = []
    for b in range(n_blocks):
        te, tr = np.flatnonzero(block == b), np.flatnonzero(block != b)
        row = {"block": b, "rows": len(te), "eval_defects": int(y[te].sum()), "train_defects": int(y[tr].sum())}
        if y[tr].sum() == 0:
            rows.append({**row, "status": "학습 불가(학습 블록에 불량 없음)", "flagged": 0, "caught": 0})
            continue
        rec = _fold_record(make_model, X, y, tr, te, seed, b)
        rows.append({**row, "status": "ok", "flagged": rec["n_flag_actual"], "caught": int((rec["flag"] & (y[te] == 1)).sum())})
    return pd.DataFrame(rows)


def calibration_summary(proba_mean: np.ndarray, y: np.ndarray) -> dict:
    """확률 보정 품질(보고 전용): 평균 예측확률 vs 실제 불량률, Brier 점수 vs 불량률만 쓰는 기준."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(proba_mean, dtype=float)
    return {
        "mean_predicted": float(p.mean()), "defect_rate": float(y.mean()),
        "brier": float(np.mean((p - y) ** 2)), "brier_prevalence_only": float(np.mean((y.mean() - y) ** 2)),
    }


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

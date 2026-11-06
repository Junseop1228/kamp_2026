"""Analyze 단계(D-011): 선택 모델의 영향요인과 오류 분석. 서술용이며 표본이 작아 통계적 주장은 하지 않는다."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from src.framework.adapter import GROUP_COL, ID_COL, LABEL_COL

FN_TYPES = ("잡힘", "경계", "모델 차이", "신호 없음")


def aux_k(y: np.ndarray, g: np.ndarray) -> float:
    """보조 판정 비율 = 불량률 × 불량이 든 그룹의 평균 크기(사출성형은 약 2배). 그룹이 없으면 1k와 같다."""
    s = pd.DataFrame({"y": np.asarray(y), "g": np.asarray(g)}).groupby("g")["y"].agg(["size", "max"])
    return float(np.mean(y) * s.loc[s["max"] == 1, "size"].mean())


def permutation_importance_cv(X: pd.DataFrame, y: np.ndarray, g: np.ndarray, make_model: Callable, seeds: Sequence[int],
                              max_folds: int, n_repeats: int, top_n: int) -> pd.DataFrame:
    """평가 fold에서 변수 하나를 섞었을 때의 PR-AUC 하락. 신뢰도(top_rate) = 전체 학습 중 상위 top_n에 든 비율."""
    y, g = np.asarray(y), np.asarray(g)
    n_splits = min(max_folds, int(pd.Series(g[y == 1]).nunique()))
    cols, drops, tops = list(X.columns), [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for tr, te in StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y, g):
            model = make_model(random_state=seed).fit(X.iloc[tr], y[tr])
            Xe, ye = X.iloc[te].copy(), y[te]
            base = average_precision_score(ye, model.predict_proba(Xe))
            d = np.zeros(len(cols))
            for j, c in enumerate(cols):
                orig = Xe[c].to_numpy().copy()
                vals = []
                for _ in range(n_repeats):
                    Xe[c] = rng.permutation(orig)
                    vals.append(base - average_precision_score(ye, model.predict_proba(Xe)))
                Xe[c] = orig
                d[j] = np.mean(vals)
            drops.append(d)
            t = np.zeros(len(cols))
            t[np.argsort(-d, kind="stable")[:top_n]] = 1
            tops.append(t)
    drops, tops = np.array(drops), np.array(tops)
    out = pd.DataFrame({"variable": cols, "pr_auc_drop_mean": drops.mean(0), "pr_auc_drop_std": drops.std(0, ddof=1),
                        "top_rate": tops.mean(0)})
    return out.sort_values(["top_rate", "pr_auc_drop_mean"], ascending=False, kind="stable").reset_index(drop=True)


def pick_top_pair(importance: pd.DataFrame, near_pairs: pd.DataFrame) -> tuple[str, str]:
    """상위 변수 2개. 두 번째는 첫 번째와 거의 같은 변수(|r| ≥ 기준)가 아닌 것 중 가장 위 — 같은 정보를 두 번 세지 않는다."""
    first = importance["variable"].iloc[0]
    partners = set(near_pairs.loc[near_pairs["var_a"] == first, "var_b"]) | set(near_pairs.loc[near_pairs["var_b"] == first, "var_a"])
    second = next(v for v in importance["variable"].iloc[1:] if v not in partners)
    return first, second


def interaction_2x2(data: pd.DataFrame, var_a: str, var_b: str) -> pd.DataFrame:
    """두 변수를 각각 중앙값으로 나눈 2×2 표: 행 수, 불량 수, 불량률."""
    hi_a, hi_b = data[var_a] > data[var_a].median(), data[var_b] > data[var_b].median()
    rows = []
    for a in (False, True):
        for b in (False, True):
            m = (hi_a == a) & (hi_b == b)
            rows.append({"var_a": var_a, "level_a": "높음" if a else "낮음", "var_b": var_b, "level_b": "높음" if b else "낮음",
                         "rows": int(m.sum()), "defects": int(data.loc[m, LABEL_COL].sum()),
                         "defect_rate": float(data.loc[m, LABEL_COL].mean()) if m.any() else float("nan")})
    return pd.DataFrame(rows)


def fn_types(oof_by_model: pd.DataFrame, chosen: str, compared: Sequence[str], k: float, aux: float) -> pd.DataFrame:
    """불량 행마다 FN 유형 1개(반복 순위 비율의 중앙값 기준): 잡힘 / 경계 / 모델 차이 / 신호 없음."""
    d = oof_by_model[oof_by_model["label"] == 1]
    wide = d.pivot(index="product_id", columns="model", values="rank_frac_median")
    groups = d.drop_duplicates("product_id").set_index("product_id")["group"]
    others = [m for m in compared if m != chosen]
    final, best_other = wide[chosen], wide[others].min(axis=1)
    kind = np.select([final < k, final < aux, best_other < aux], FN_TYPES[:3], FN_TYPES[3])
    return pd.DataFrame({"product_id": wide.index, "group": groups.loc[wide.index].to_numpy(), "final_rank_frac": final.to_numpy(),
                         "best_other_model": wide[others].idxmin(axis=1).to_numpy(), "best_other_rank_frac": best_other.to_numpy(),
                         "fn_type": kind})


def _quartiles(series: pd.Series) -> pd.Series:
    return pd.qcut(series, 4, labels=False, duplicates="drop")


def fn_concentration(data: pd.DataFrame, rank_frac: np.ndarray, k: float, variables: Sequence[str]) -> pd.DataFrame:
    """불량 중 잡힌 것과 놓친 것을 변수 4분위 구간별로 센다. 놓친 불량이 몰리는 구간 = FN 집중 조건."""
    y = data[LABEL_COL].to_numpy()
    caught = np.asarray(rank_frac) < k
    rows = []
    for v in variables:
        if data[v].nunique() <= 1:
            continue
        q = _quartiles(data[v]).to_numpy()
        for b in np.unique(q):
            m = q == b
            rows.append({"variable": v, "quartile": int(b) + 1, "lo": float(data.loc[m, v].min()), "hi": float(data.loc[m, v].max()),
                         "defects": int(y[m].sum()), "caught": int(((y == 1) & caught & m).sum()),
                         "missed": int(((y == 1) & ~caught & m).sum())})
    return pd.DataFrame(rows)


def fp_analysis(data: pd.DataFrame, rank_frac: np.ndarray, budget: float, variables: Sequence[str]) -> tuple[dict, pd.DataFrame]:
    """검사량 budget에서의 오경보: 구조적 FP(불량이 든 그룹의 양품)와 조건 FP(그 밖의 양품)를 나누고, 조건 FP 비율을 4분위별로 본다."""
    y, g = data[LABEL_COL].to_numpy(), data[GROUP_COL].to_numpy()
    flagged = np.asarray(rank_frac) < budget
    in_defect_group = np.isin(g, np.unique(g[y == 1]))
    clean_good = (y == 0) & ~in_defect_group
    structural, condition = flagged & (y == 0) & in_defect_group, flagged & clean_good
    overall = float(condition.sum() / clean_good.sum())
    summary = {"budget": budget, "flagged": int(flagged.sum()), "caught": int((flagged & (y == 1)).sum()),
               "structural_fp": int(structural.sum()), "condition_fp": int(condition.sum()), "condition_fp_rate": overall}
    rows = []
    for v in variables:
        if data[v].nunique() <= 1:
            continue
        q = _quartiles(data[v]).to_numpy()
        for b in np.unique(q):
            m = (q == b) & clean_good
            rate = float(condition[m].sum() / m.sum()) if m.any() else float("nan")
            rows.append({"variable": v, "quartile": int(b) + 1, "lo": float(data.loc[q == b, v].min()), "hi": float(data.loc[q == b, v].max()),
                         "clean_goods": int(m.sum()), "condition_fp": int(condition[m].sum()), "fp_rate": rate,
                         "ratio_to_overall": rate / overall if overall > 0 else float("nan")})
    return summary, pd.DataFrame(rows)


def domain_crosscheck(data: pd.DataFrame, dictionary: pd.DataFrame, variables: Sequence[str]) -> pd.DataFrame:
    """상위 변수별 예상 관계(변수 사전) vs 불량의 관측 방향. 관측은 양품 기준 z로 잰다. 불일치는 결론이 아니라 검토 과제다."""
    y = data[LABEL_COL].to_numpy()
    good = data[y == 0]
    exp = dictionary.set_index("variable")["expected_relation"]
    rows = []
    for v in variables:
        sd = good[v].std()
        if not sd > 0:
            rows.append({"variable": v, "expected": exp[v], "defect_mean_z": float("nan"), "defect_abs_z": float("nan"),
                         "good_abs_z": float("nan"), "verdict": "관측 불가(양품 상수)"})
            continue
        z = (data[v] - good[v].mean()) / sd
        d_abs, g_abs = float(z[y == 1].abs().mean()), float(z[y == 0].abs().mean())
        kind = exp[v].split(":")[0].strip()
        if kind == "양방향":
            verdict = "일치" if d_abs > g_abs else "불일치(검토 과제)"
        elif kind == "약함":
            verdict = "새 발견(예상은 약함)"
        else:
            verdict = "예상 없음"
        rows.append({"variable": v, "expected": exp[v], "defect_mean_z": float(z[y == 1].mean()), "defect_abs_z": d_abs,
                     "good_abs_z": g_abs, "verdict": verdict})
    return pd.DataFrame(rows)


def pdp_2d(make_model: Callable, X: pd.DataFrame, y: np.ndarray, seed: int, var_a: str, var_b: str, grid: int) -> pd.DataFrame:
    """2D 부분의존도: labeled 전체로 학습한 모델에서 두 변수를 격자값(5~95% 분위)으로 고정했을 때의 평균 출력."""
    model = make_model(random_state=seed).fit(X, y)
    qs = np.linspace(0.05, 0.95, grid)
    ga, gb = X[var_a].quantile(qs).to_numpy(), X[var_b].quantile(qs).to_numpy()
    rows, Xg = [], X.copy()
    for a in ga:
        for b in gb:
            Xg[var_a], Xg[var_b] = a, b
            rows.append({var_a: float(a), var_b: float(b), "mean_output": float(np.mean(model.predict_proba(Xg)))})
    return pd.DataFrame(rows)

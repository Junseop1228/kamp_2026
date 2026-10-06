"""Diagnose 단계: 모델 없이 데이터 구조를 점검한다. 아무것도 제거하거나 고치지 않고 보고만 한다."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.framework.adapter import GROUP_COL, ID_COL, LABEL_COL, ProductTable, assign_groups
from src.framework.evaluate import dup_ceiling_f1, random_floor_f1

EXACT = "exact"
DICT_COLS = ("variable", "meaning", "expected_relation", "confidence", "source")
CONFIDENCE = ("상", "중", "하")


def group_structure(table: ProductTable) -> dict:
    """동일입력 그룹 구조: 그룹 수, 크기 분포, 쌍에 속한 행 비율. train이면 라벨 구성까지."""
    data = table.data
    size = data.groupby(GROUP_COL).size()
    out = {
        "n_rows": int(len(data)),
        "n_groups": int(size.size),
        "size_1": int((size == 1).sum()),
        "size_2": int((size == 2).sum()),
        "size_3plus": int((size >= 3).sum()),
        "rows_in_multi_share": float(size[size >= 2].sum() / len(data)),
    }
    if table.has_label:
        g = data.groupby(GROUP_COL)[LABEL_COL].agg(["size", "sum"])
        pair = g["size"] == 2
        out.update({
            "pairs_0_0": int((pair & (g["sum"] == 0)).sum()),
            "pairs_0_1": int((pair & (g["sum"] == 1)).sum()),
            "pairs_1_1": int((pair & (g["sum"] == 2)).sum()),
            "mixed_groups": int(((g["sum"] > 0) & (g["sum"] < g["size"])).sum()),
            "defect_groups": int((g["sum"] > 0).sum()),
        })
    return out


def rounding_sensitivity(table: ProductTable, decimals: Sequence[int]) -> pd.DataFrame:
    """반올림 자릿수별 그룹 수. 자릿수를 줄였을 때 그룹이 크게 줄면 근사 중복이 있다는 뜻이다."""
    feats = table.data[table.feature_cols]
    rows = []
    for d in [None, *decimals]:
        x = feats if d is None else feats.round(d)
        size = assign_groups(x, table.feature_cols).value_counts()
        rows.append({
            "decimals": EXACT if d is None else str(d),
            "n_groups": int(size.size),
            "rows_in_multi_share": float(size[size >= 2].sum() / len(feats)),
        })
    return pd.DataFrame(rows)


def imbalance(table: ProductTable) -> dict:
    """불균형과 결측: 행 수, 결측 수. train이면 불량 수, 불량률, 불량 그룹 수까지."""
    data = table.data
    out = {"n_rows": int(len(data)), "n_missing": int(data[table.feature_cols].isna().sum().sum())}
    if table.has_label:
        y = data[LABEL_COL]
        out.update({
            "n_defects": int(y.sum()),
            "defect_rate": float(y.mean()),
            "n_defect_groups": int(data.loc[y == 1, GROUP_COL].nunique()),
        })
    return out


def constant_features(table: ProductTable) -> list[str]:
    """값이 하나뿐인 공정변수. 학습 정보가 없고, 표준편차를 쓰는 계산에서 0 나눗셈을 일으킨다."""
    nunique = table.data[table.feature_cols].nunique()
    return nunique[nunique <= 1].index.tolist()


def near_identical_pairs(table: ProductTable, r_min: float) -> pd.DataFrame:
    """|Pearson r| ≥ r_min인 공정변수 쌍. permutation 중요도를 해석할 때 묶어서 본다."""
    constants = set(constant_features(table))
    cols = [c for c in table.feature_cols if c not in constants]
    corr = table.data[cols].corr().to_numpy()
    a, b = np.triu_indices(len(cols), k=1)
    r = corr[a, b]
    keep = np.abs(r) >= r_min
    out = pd.DataFrame({"var_a": np.array(cols)[a[keep]], "var_b": np.array(cols)[b[keep]], "r": r[keep]})
    return out.iloc[np.argsort(-np.abs(out["r"].to_numpy()), kind="stable")].reset_index(drop=True)


def outlier_counts(table: ProductTable, z_max: float) -> pd.DataFrame:
    """변수별 |z| > z_max 행 수. z는 이 표 안에서 다시 계산한다. 이상치는 제거하지 않는다."""
    constants = set(constant_features(table))
    cols = [c for c in table.feature_cols if c not in constants]
    x = table.data[cols]
    mask = ((x - x.mean()) / x.std()).abs() > z_max
    out = pd.DataFrame({"variable": cols, "n_outliers": mask.sum().to_numpy()})
    if table.has_label:
        y = table.data[LABEL_COL].to_numpy()
        out["outlier_defects"] = [int(y[mask[c].to_numpy()].sum()) for c in cols]
        out["outlier_defect_rate"] = out["outlier_defects"] / out["n_outliers"].where(out["n_outliers"] > 0)
    return out.sort_values("n_outliers", ascending=False, kind="stable").reset_index(drop=True)


def twin_adjacency(table: ProductTable, trust_min: float) -> dict:
    """2행 그룹 중 두 행이 파일상 바로 붙어 있는 비율. trust_min 이상이면 행 순서를 생산 순서의 대리로 신뢰한다."""
    g = table.data[GROUP_COL].to_numpy()
    pos = pd.Series(np.arange(len(g)))
    size = pd.Series(g).map(pd.Series(g).value_counts())
    span = pos[size == 2].groupby(g[(size == 2).to_numpy()]).agg(lambda s: s.max() - s.min())
    share = float((span == 1).mean()) if len(span) else 0.0
    return {"n_pairs": int(len(span)), "adjacent_share": share, "row_order_trusted": bool(share >= trust_min)}


def defect_concentration(table: ProductTable) -> dict:
    """불량 행의 파일상 위치 범위와 밀집도. 시간 관련 해석은 행 순서를 신뢰할 때만 한다."""
    y = table.data[LABEL_COL].to_numpy()
    pos = np.flatnonzero(y == 1)
    ids = table.data[ID_COL].to_numpy()[pos]
    span = int(pos.max() - pos.min() + 1)
    return {
        "n_defects": int(len(pos)), "first_row": int(pos.min()), "last_row": int(pos.max()),
        "first_id": int(ids.min()), "last_id": int(ids.max()),
        "span_rows": span, "span_share": float(span / len(y)),
    }


def distribution_shift(train: ProductTable, test: ProductTable, mean_sd: float, std_ratio: Sequence[float]) -> pd.DataFrame:
    """변수별 train·test 평균과 표준편차를 나란히 놓고, 차이가 크면 경고 사유를 단다. 통계 검정·거리 지표는 쓰지 않는다."""
    a, b = train.data[train.feature_cols], test.data[train.feature_cols]
    out = pd.DataFrame({
        "variable": train.feature_cols,
        "train_mean": a.mean().to_numpy(), "train_std": a.std().to_numpy(),
        "test_mean": b.mean().to_numpy(), "test_std": b.std().to_numpy(),
    })
    const = out["train_std"] == 0
    out["mean_gap_sd"] = ((out["test_mean"] - out["train_mean"]).abs() / out["train_std"]).where(~const)
    out["std_ratio"] = (out["test_std"] / out["train_std"]).where(~const)
    lo, hi = std_ratio
    out["warning"] = np.select(
        [const & (out["test_std"] > 0), out["mean_gap_sd"] > mean_sd, (out["std_ratio"] < lo) | (out["std_ratio"] > hi)],
        ["train_constant", "mean_shift", "spread_shift"],
        default="",
    )
    return out


def check_variable_dictionary(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """변수 사전 검증: 5열 구성, 공정변수 전부 수록·중복 없음, 확신도는 상·중·하. 실패하면 ValueError."""
    if tuple(df.columns) != DICT_COLS:
        raise ValueError(f"변수 사전 열 구성 {list(df.columns)} ≠ {list(DICT_COLS)}")
    problems = []
    missing = sorted(set(feature_cols) - set(df["variable"]))
    extra = sorted(set(df["variable"]) - set(feature_cols))
    bad_conf = sorted(set(df["confidence"]) - set(CONFIDENCE))
    if missing:
        problems.append(f"누락 변수 {missing}")
    if extra:
        problems.append(f"공정변수가 아닌 항목 {extra}")
    if df["variable"].duplicated().any():
        problems.append("중복 변수")
    if bad_conf:
        problems.append(f"확신도 값 오류 {bad_conf}")
    if problems:
        raise ValueError("변수 사전 검증 실패: " + "; ".join(problems))


def diagnose_report(train: ProductTable, test: ProductTable, cfg: dict) -> tuple[dict, dict[str, pd.DataFrame]]:
    """진단 전부를 모아 요약(dict)과 표(DataFrame)로 돌려준다. 파일 쓰기는 호출자가 한다."""
    y, g = train.data[LABEL_COL].to_numpy(), train.data[GROUP_COL].to_numpy()
    shift = distribution_shift(train, test, cfg["shift_mean_sd"], cfg["shift_std_ratio"])
    summary = {
        "line": train.line,
        "imbalance": {s.split: imbalance(s) for s in (train, test)},
        "groups": {s.split: group_structure(s) for s in (train, test)},
        "constant_features": {s.split: constant_features(s) for s in (train, test)},
        "twin_adjacency": {s.split: twin_adjacency(s, cfg["adjacency_trust"]) for s in (train, test)},
        "defect_concentration": defect_concentration(train),
        "f1_floor": random_floor_f1(y),
        "f1_ceiling": dup_ceiling_f1(y, g),
        "distribution_warning": [f"{r.variable}:{r.warning}" for r in shift.itertuples() if r.warning],
    }
    tables = {"shift": shift}
    for name, fn, arg in (("rounding", rounding_sensitivity, cfg["rounding_decimals"]),
                          ("outliers", outlier_counts, cfg["outlier_z"]),
                          ("pairs", near_identical_pairs, cfg["near_identical_r"])):
        tables[name] = pd.concat([fn(s, arg).assign(split=s.split) for s in (train, test)], ignore_index=True)
    return summary, tables

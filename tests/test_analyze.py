"""Analyze 검증(D-011): 보조 판정 비율, permutation 신뢰도, 상위 쌍 선택, FN 유형·집중, FP 분리, 도메인 대조, 2D 부분의존도."""
import numpy as np
import pandas as pd
import pytest

from src.framework.analyze import (
    FN_TYPES, aux_k, domain_crosscheck, fn_concentration, fn_types, fp_analysis, interaction_2x2, pdp_2d,
    permutation_importance_cv, pick_top_pair,
)
from src.models import make_factory

PARAMS = {"logistic": {"C": 1.0, "max_iter": 1000}}


def _synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    y = (a + 0.3 * rng.normal(size=n) > 1.6).astype(int)
    data = pd.DataFrame({"product_id": range(n), "a": a, "b": b, "c": c, "label": y, "group": range(n)})
    return data, data[["a", "b", "c"]], y, data["group"].to_numpy()


def test_aux_k_doubles_for_pairs():
    y = np.array([1, 0, 0, 0, 1, 1, 0, 0])
    g = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    assert aux_k(y, g) == pytest.approx(3 / 8 * 2)


def test_permutation_importance_ranks_signal_first():
    _, X, y, g = _synthetic()
    imp = permutation_importance_cv(X, y, g, make_factory("logistic", PARAMS), (0, 1), 3, 2, 1)
    assert imp["variable"].iloc[0] == "a" and imp["top_rate"].iloc[0] == 1.0


def test_pick_top_pair_skips_near_identical_partner():
    imp = pd.DataFrame({"variable": ["x", "x_twin", "z"]})
    near = pd.DataFrame({"var_a": ["x"], "var_b": ["x_twin"], "r": [0.99]})
    assert pick_top_pair(imp, near) == ("x", "z")


def test_fn_types_one_per_defect():
    rows = []
    for pid, (m1, m2) in enumerate([(0.005, 0.5), (0.02, 0.5), (0.5, 0.01), (0.5, 0.5)]):
        rows += [{"model": "m1", "product_id": pid, "label": 1, "group": pid, "rank_frac_median": m1},
                 {"model": "m2", "product_id": pid, "label": 1, "group": pid, "rank_frac_median": m2}]
    out = fn_types(pd.DataFrame(rows), "m1", ["m1", "m2"], 0.014, 0.028)
    assert out["fn_type"].tolist() == list(FN_TYPES)


def test_fp_analysis_separates_structural():
    data = pd.DataFrame({"label": [1, 0, 0, 0, 0, 0, 0, 0], "group": [0, 0, 1, 1, 2, 2, 3, 3], "v": np.arange(8.0)})
    rank = np.array([0.0, 0.0, 0.05, 0.5, 0.5, 0.5, 0.5, 0.5])           # 혼합 쌍(0) 함께 표시 + 깨끗한 양품 1개
    summary, table = fp_analysis(data, rank, 0.10, ["v"])
    assert (summary["caught"], summary["structural_fp"], summary["condition_fp"]) == (1, 1, 1)
    assert table["condition_fp"].sum() == 1


def test_fn_concentration_counts_and_interaction_sums():
    data, X, y, _ = _synthetic()
    rank = np.where(data["a"] > 2.0, 0.0, 0.9)
    conc = fn_concentration(data, rank, 0.014, ["a"])
    assert conc["defects"].sum() == y.sum() and (conc["caught"] + conc["missed"]).sum() == y.sum()
    assert interaction_2x2(data, "a", "b")["rows"].sum() == len(data)


def test_domain_crosscheck_verdicts():
    data, *_ = _synthetic()
    vd = pd.DataFrame({"variable": ["a", "b"], "expected_relation": ["양방향: 테스트", "약함: 테스트"]})
    out = domain_crosscheck(data, vd, ["a", "b"]).set_index("variable")["verdict"]
    assert out["a"] == "일치" and out["b"].startswith("새 발견")


def test_pdp_2d_grid():
    _, X, y, _ = _synthetic()
    pdp = pdp_2d(make_factory("logistic", PARAMS), X, y, 0, "a", "b", 4)
    assert len(pdp) == 16 and pdp["mean_output"].between(0, 1).all()
    assert pdp.groupby("a")["mean_output"].mean().is_monotonic_increasing   # 신호 변수 a가 클수록 위험

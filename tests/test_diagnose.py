"""Diagnose 검증: §4 그룹 구조, 상수 변수, 불균형, 반올림 민감도, 인접도·불량 위치·분포 경고, 변수 사전."""
import json

import numpy as np
import pandas as pd
import pytest

import config
from src.framework.adapter import ProductTable, load_injection
from src.framework.diagnose import (
    check_variable_dictionary,
    constant_features,
    defect_concentration,
    diagnose_report,
    distribution_shift,
    group_structure,
    imbalance,
    near_identical_pairs,
    outlier_counts,
    rounding_sensitivity,
    twin_adjacency,
)

CFG = json.loads((config.CONFIGS / "injection.json").read_text(encoding="utf-8"))
EXACT_GROUPS = {"CN7": 606, "RG3": 591}


@pytest.fixture(scope="module")
def tables():
    return {
        (line, split): load_injection(pd.read_csv(config.PROJECT_ROOT / CFG["data"][split][line]), CFG["rename"], line, split)
        for line in EXACT_GROUPS for split in ("train", "test")
    }


@pytest.mark.parametrize("line, expected", [
    ("CN7", {"n_groups": 606, "size_1": 1, "size_3plus": 0, "pairs_0_0": 591, "pairs_0_1": 11, "pairs_1_1": 3, "defect_groups": 14}),
    ("RG3", {"n_groups": 591, "size_1": 0, "size_3plus": 0, "pairs_0_0": 566, "pairs_0_1": 25, "pairs_1_1": 0, "defect_groups": 25}),
])
def test_group_structure_matches_reference(tables, line, expected):
    s = group_structure(tables[(line, "train")])
    assert {k: s[k] for k in expected} == expected


@pytest.mark.parametrize("line", ["CN7", "RG3"])
def test_clamp_open_position_constant_in_labeled(tables, line):
    assert "Clamp_Open_Position" in constant_features(tables[(line, "train")])


def test_imbalance_cn7(tables):
    s = imbalance(tables[("CN7", "train")])
    assert (s["n_defects"], s["n_defect_groups"], s["n_missing"]) == (17, 14, 0)
    assert s["defect_rate"] == pytest.approx(17 / 1211)


@pytest.mark.parametrize("line", ["CN7", "RG3"])
def test_rounding_10_decimals_equals_exact(tables, line):
    r = rounding_sensitivity(tables[(line, "train")], CFG["diagnose"]["rounding_decimals"]).set_index("decimals")["n_groups"]
    assert r["exact"] == EXACT_GROUPS[line]
    assert r["10"] == EXACT_GROUPS[line]


def test_cn7_defects_concentrated_in_ids_47_to_120(tables):
    d = defect_concentration(tables[("CN7", "train")])
    assert (d["n_defects"], d["first_id"], d["last_id"]) == (17, 47, 120)


def test_variable_dictionary_covers_all_features(tables):
    vd = pd.read_csv(config.PROJECT_ROOT / CFG["variable_dictionary"])
    check_variable_dictionary(vd, tables[("CN7", "train")].feature_cols)


def test_variable_dictionary_rejects_broken(tables):
    vd = pd.read_csv(config.PROJECT_ROOT / CFG["variable_dictionary"])
    feats = tables[("CN7", "train")].feature_cols
    with pytest.raises(ValueError, match="누락 변수"):
        check_variable_dictionary(vd.iloc[1:], feats)
    with pytest.raises(ValueError, match="확신도"):
        check_variable_dictionary(vd.assign(confidence="높음"), feats)


def test_diagnose_report_keys(tables):
    summary, out = diagnose_report(tables[("CN7", "train")], tables[("CN7", "test")], CFG["diagnose"])
    assert summary["f1_ceiling"] == pytest.approx(0.756, abs=5e-4)
    assert set(out) == {"shift", "rounding", "outliers", "pairs"}
    assert "Clamp_Open_Position:train_constant" in summary["distribution_warning"]


# ── 합성 데이터 ───────────────────────────

def _synthetic() -> ProductTable:
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    data = pd.DataFrame({
        "product_id": range(200),
        "a": a,
        "b": 2 * a + rng.normal(scale=1e-3, size=200),
        "c": rng.normal(size=200),
        "k": 1.0,
        "label": 0,
    })
    data.loc[5, "c"] = 50.0
    data.loc[5, "label"] = 1
    return ProductTable(line="SYN", split="train", data=data, feature_cols=["a", "b", "c", "k"])


def test_near_identical_pairs_synthetic():
    pairs = near_identical_pairs(_synthetic(), 0.95)
    assert pairs[["var_a", "var_b"]].to_numpy().tolist() == [["a", "b"]]


def test_outlier_counts_synthetic():
    out = outlier_counts(_synthetic(), 3.0).set_index("variable")
    assert "k" not in out.index                                  # 상수 변수는 제외
    assert out.loc["c", "n_outliers"] >= 1
    assert out.loc["c", "outlier_defects"] == 1


def test_twin_adjacency_synthetic():
    def table(groups):
        data = pd.DataFrame({"product_id": range(len(groups)), "group": groups})
        return ProductTable(line="SYN", split="test", data=data, feature_cols=[])
    assert twin_adjacency(table([0, 0, 1, 1, 2, 2]), 0.9) == {"n_pairs": 3, "adjacent_share": 1.0, "row_order_trusted": True}
    assert twin_adjacency(table([0, 1, 0, 1, 2, 2]), 0.9)["adjacent_share"] == pytest.approx(1 / 3)


def test_distribution_shift_reasons():
    tr = pd.DataFrame({"const": 0.0, "mean": np.r_[np.zeros(50), np.ones(50)], "spread": np.r_[np.zeros(50), np.ones(50)],
                       "same": np.r_[np.zeros(50), np.ones(50)]})
    te = pd.DataFrame({"const": np.arange(100.0), "mean": tr["mean"] + 5, "spread": tr["spread"] * 4 - 1.5, "same": tr["same"]})
    mk = lambda d, s: ProductTable(line="SYN", split=s, data=d, feature_cols=list(tr.columns))
    out = distribution_shift(mk(tr, "train"), mk(te, "test"), 1.0, (0.5, 2.0)).set_index("variable")["warning"]
    assert out.to_dict() == {"const": "train_constant", "mean": "mean_shift", "spread": "spread_shift", "same": ""}

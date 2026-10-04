"""Adapter 단계 검증: 작업지시서 §4 기준 수치와 입력 검증 경로."""
import json

import pandas as pd
import pytest

import config
from src.framework.adapter import GROUP_COL, ID_COL, LABEL_COL, load_injection

CFG = json.loads((config.CONFIGS / "injection.json").read_text(encoding="utf-8"))
RAW = {v: k for k, v in CFG["rename"].items()}
N_FEATURES = 24  # D-006

EXPECTED_TRAIN = {
    "CN7": {"rows": 1211, "defects": 17, "groups": 606, "defect_groups": 14,
            "comp": {"1": 1, "0+0": 591, "0+1": 11, "1+1": 3, "3+": 0}},
    "RG3": {"rows": 1182, "defects": 25, "groups": 591, "defect_groups": 25,
            "comp": {"1": 0, "0+0": 566, "0+1": 25, "1+1": 0, "3+": 0}},
}
EXPECTED_TEST_ROWS = {"CN7": 35239, "RG3": 35941}


def _raw(split: str, line: str) -> pd.DataFrame:
    return pd.read_csv(config.PROJECT_ROOT / CFG["data"][split][line])


@pytest.fixture(scope="module")
def train_tables():
    return {line: load_injection(_raw("train", line), CFG["rename"], line, "train") for line in EXPECTED_TRAIN}


@pytest.mark.parametrize("line", list(EXPECTED_TRAIN))
def test_train_counts(train_tables, line):
    t, exp = train_tables[line], EXPECTED_TRAIN[line]
    assert len(t.data) == exp["rows"]
    assert int(t.data[LABEL_COL].sum()) == exp["defects"]
    assert len(t.feature_cols) == N_FEATURES
    assert t.data[GROUP_COL].nunique() == exp["groups"]


@pytest.mark.parametrize("line", list(EXPECTED_TRAIN))
def test_train_group_composition(train_tables, line):
    g = train_tables[line].data.groupby(GROUP_COL)[LABEL_COL].agg(["size", "sum"])
    pair = g["size"] == 2
    comp = {
        "1": int((g["size"] == 1).sum()),
        "0+0": int((pair & (g["sum"] == 0)).sum()),
        "0+1": int((pair & (g["sum"] == 1)).sum()),
        "1+1": int((pair & (g["sum"] == 2)).sum()),
        "3+": int((g["size"] >= 3).sum()),
    }
    assert comp == EXPECTED_TRAIN[line]["comp"]
    assert int((g["sum"] > 0).sum()) == EXPECTED_TRAIN[line]["defect_groups"]


def test_cn7_defect_id_range(train_tables):
    d = train_tables["CN7"].data
    ids = d.loc[d[LABEL_COL] == 1, ID_COL]
    assert (int(ids.min()), int(ids.max())) == (47, 120)


@pytest.mark.parametrize("line", list(EXPECTED_TEST_ROWS))
def test_test_table_grouped_by_same_rule(line):
    t = load_injection(_raw("test", line), CFG["rename"], line, "test")
    assert len(t.data) == EXPECTED_TEST_ROWS[line]
    assert not t.has_label
    assert len(t.feature_cols) == N_FEATURES
    # 같은 그룹 안은 공정변수 완전일치, 그룹 수 = 서로 다른 공정변수 조합 수
    assert (t.data.groupby(GROUP_COL)[t.feature_cols].nunique() == 1).all().all()
    assert t.data[GROUP_COL].nunique() == len(t.data[t.feature_cols].drop_duplicates())


def test_groups_deterministic(train_tables):
    again = load_injection(_raw("train", "CN7"), CFG["rename"], "CN7", "train")
    assert again.data[GROUP_COL].equals(train_tables["CN7"].data[GROUP_COL])


def _broken(case: str) -> pd.DataFrame:
    df = _raw("train", "CN7").head(6).copy()
    feat = next(c for c in df.columns if c not in CFG["rename"])
    if case == "dup_id":
        df[RAW[ID_COL]] = 0
    elif case == "non_binary":
        df[RAW[LABEL_COL]] = [0, 1, 2, 0, 0, 0]
    elif case == "non_numeric":
        df[feat] = "x"
    elif case == "missing":
        df.loc[df.index[0], feat] = None
    return df


@pytest.mark.parametrize("case, message", [
    ("dup_id", "중복 ID"),
    ("non_binary", "이진"),
    ("non_numeric", "비수치"),
    ("missing", "결측"),
])
def test_validation_stops_with_reason(case, message):
    with pytest.raises(ValueError, match=message):
        load_injection(_broken(case), CFG["rename"], "CN7", "train")


def test_test_split_rejects_label():
    with pytest.raises(ValueError, match="라벨 열이 있음"):
        load_injection(_raw("train", "CN7").head(6), CFG["rename"], "CN7", "test")

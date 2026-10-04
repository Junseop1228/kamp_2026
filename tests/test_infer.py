"""Infer 검증: 행 수, 명목·실제 표시 수, 순위·동점, 메타 기록, 제품군 짝 검사."""
import json

import numpy as np
import pandas as pd
import pytest

import config
from src.framework.adapter import GROUP_COL, ID_COL, load_injection
from src.framework.infer import FLAG_COL, PROBA_COL, RANK_COL, infer
from src.models.baselines import RandomScore

CFG = json.loads((config.CONFIGS / "injection.json").read_text(encoding="utf-8"))
EXPECTED = {"CN7": {"n_test": 35239, "nominal": 495}, "RG3": {"n_test": 35941, "nominal": 760}}


@pytest.fixture(scope="module")
def tables():
    return {
        (line, split): load_injection(pd.read_csv(config.PROJECT_ROOT / CFG["data"][split][line]), CFG["rename"], line, split)
        for line in EXPECTED for split in ("train", "test")
    }


@pytest.fixture(scope="module")
def results(tables):
    return {
        line: infer(tables[(line, "train")], tables[(line, "test")], RandomScore, config.SEEDS[0], CFG["limitations"])
        for line in EXPECTED
    }


@pytest.mark.parametrize("line", list(EXPECTED))
def test_rows_and_flag_counts(results, line):
    pred, meta = results[line]
    assert len(pred) == meta["n_test"] == EXPECTED[line]["n_test"]
    assert meta["n_flag_nominal"] == EXPECTED[line]["nominal"]
    assert int(pred[FLAG_COL].sum()) == meta["n_flag_actual"] >= meta["n_flag_nominal"]
    assert pred[ID_COL].is_unique and pred[GROUP_COL].notna().all()


@pytest.mark.parametrize("line", list(EXPECTED))
def test_rank_and_flag_consistent(results, line):
    pred, _ = results[line]
    assert pred.loc[pred[PROBA_COL].idxmax(), RANK_COL] == 1
    flagged = pred[pred[FLAG_COL] == 1]
    assert flagged[PROBA_COL].min() > pred.loc[pred[FLAG_COL] == 0, PROBA_COL].max()


class _TwoLevel:
    """테스트 전용 모델: 짝수 행 0.9, 홀수 행 0.1로 대량 동점을 만든다."""

    name = "two_level"

    def __init__(self, random_state: int) -> None:
        pass

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.where(np.asarray(X.index) % 2 == 0, 0.9, 0.1)


def test_ties_share_rank_and_are_all_flagged(tables):
    pred, meta = infer(tables[("CN7", "train")], tables[("CN7", "test")], _TwoLevel, 0)
    high = pred[PROBA_COL] == 0.9
    assert (pred.loc[high, RANK_COL] == 1).all()
    assert (pred.loc[~high, RANK_COL] == int(high.sum()) + 1).all()
    assert meta["n_flag_actual"] == int(high.sum()) > meta["n_flag_nominal"]


def test_meta_records(results):
    _, meta = results["CN7"]
    assert meta["k"] == pytest.approx(17 / 1211)
    assert (meta["n_train"], meta["n_train_defects"]) == (1211, 17)
    assert (meta["model"], meta["seed"]) == ("random_score", config.SEEDS[0])
    assert any("표준화" in s for s in meta["limitations"])
    assert meta["alarm"] is None and meta["distribution_warning"] is None


def test_line_mismatch_rejected(tables):
    with pytest.raises(ValueError, match="제품군 불일치"):
        infer(tables[("CN7", "train")], tables[("RG3", "test")], RandomScore, 0)

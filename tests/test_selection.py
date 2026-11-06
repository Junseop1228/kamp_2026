"""P5 검증: 선택 규칙, 하한 비교, 누수 대조 분할, 블록 분할, 교차검증 예측 요약, 보정 요약, 판정 확신도."""
import json

import numpy as np
import pandas as pd
import pytest

import config
from src.framework.adapter import GROUP_COL, LABEL_COL, load_injection
from src.framework.evaluate import block_split, calibration_summary, oof_summary, run_cv, select_model
from src.framework.infer import flag_rate_over_seeds
from src.models import make_factory

CFG = json.loads((config.CONFIGS / "injection.json").read_text(encoding="utf-8"))
ORDER = ["simple", "middle", "complex"]


def _scores(f1_by_model: dict) -> pd.DataFrame:
    return pd.DataFrame([{"model": m, "seed": s, "f1": v} for m, vals in f1_by_model.items() for s, v in enumerate(vals)])


def test_select_prefers_simplest_without_difference():
    scores = _scores({
        "simple": [0.40, 0.42, 0.38, 0.41], "middle": [0.30, 0.31, 0.29, 0.30],
        "complex": [0.43, 0.41, 0.40, 0.42], "floor": [0.0, 0.05, 0.0, 0.0],
    })
    chosen, table = select_model(scores, ORDER, "floor", 1)
    assert chosen == "simple"                                   # 최고는 complex지만 차이 없음 → 단순한 쪽
    assert table.set_index("model").loc["middle", "no_difference"] == False  # noqa: E712
    assert table["beats_floor"].all()


def test_select_keeps_best_when_clearly_better():
    scores = _scores({
        "simple": [0.10, 0.12, 0.11, 0.10], "middle": [0.30, 0.31, 0.29, 0.30],
        "complex": [0.31, 0.30, 0.30, 0.31], "floor": [0.0, 0.0, 0.0, 0.0],
    })
    chosen, _ = select_model(scores, ORDER, "floor", 1)
    assert chosen == "middle"


def test_beats_floor_false_when_indistinguishable():
    scores = _scores({m: [0.0, 0.05, 0.0, 0.02] for m in ORDER} | {"floor": [0.04, 0.0, 0.05, 0.0]})
    _, table = select_model(scores, ORDER, "floor", 1)
    assert not table["beats_floor"].any()


@pytest.fixture(scope="module")
def cn7():
    t = load_injection(pd.read_csv(config.PROJECT_ROOT / CFG["data"]["train"]["CN7"]), CFG["rename"], "CN7", "train")
    return t, t.data[t.feature_cols], t.data[LABEL_COL].to_numpy(), t.data[GROUP_COL].to_numpy()


def test_ignore_groups_splits_twins_across_folds(cn7):
    _, X, y, g = cn7
    recs = run_cv(X, y, g, make_factory("random_score", {}), (0,), 5, ignore_groups=True)
    fold_of = pd.Series(np.concatenate([[r["fold"]] * len(r["idx"]) for r in recs]), index=np.concatenate([r["idx"] for r in recs]))
    assert (fold_of.groupby(g[fold_of.index]).nunique() > 1).any()   # 짝이 다른 fold로 갈라진다 = 누수 경로


def test_block_split_cn7_event_is_not_learnable_across_blocks(cn7):
    _, X, y, g = cn7
    blocks = block_split(X, y, g, make_factory("logistic", CFG["models"]["params"]), 0, 5)
    assert blocks["eval_defects"].sum() == 17
    first = blocks.iloc[0]
    assert first["eval_defects"] == 17 and first["status"].startswith("학습 불가")   # 불량이 전부 첫 블록에 있다


def test_oof_summary_and_calibration(cn7):
    _, X, y, g = cn7
    recs = run_cv(X, y, g, make_factory("logistic", CFG["models"]["params"]), (0, 1), 5)
    oof = oof_summary(recs, len(y))
    assert len(oof) == len(y) and oof.notna().all().all()
    assert oof["flag_rate"].between(0, 1).all() and oof["rank_frac_median"].between(0, 1).all()
    cal = calibration_summary(oof["proba_mean"].to_numpy(), y)
    assert cal["defect_rate"] == pytest.approx(17 / 1211)


def test_flag_rate_over_seeds(cn7):
    t, *_ = cn7
    rate = flag_rate_over_seeds(t, t, make_factory("logistic", CFG["models"]["params"]), (0, 1, 2))
    assert set(np.unique(rate)) <= {0.0, 1.0}                       # 결정적 모델은 0 또는 1
    rnd = flag_rate_over_seeds(t, t, make_factory("random_score", {}), (0, 1, 2))
    assert ((rnd > 0) & (rnd < 1)).any()

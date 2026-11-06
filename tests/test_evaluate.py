"""Evaluate 검증: 판정 규칙(half-up, 최소 1, 경계 동점)과 CV 러너(그룹 보존, 재현, 학습 불가, 상·하한)."""
import json

import numpy as np
import pandas as pd
import pytest

import config
from src.framework.adapter import GROUP_COL, LABEL_COL, load_injection
from src.framework.evaluate import (
    dup_ceiling_f1,
    flag_top,
    n_flag,
    random_floor_f1,
    run_cv,
    score_repeats,
)

CFG = json.loads((config.CONFIGS / "injection.json").read_text(encoding="utf-8"))


# ── 판정 규칙 ─────────────────────────────

@pytest.mark.parametrize("k, n, expected", [
    (0.25, 10, 3),          # 2.5 → 3. 파이썬 round(2.5)는 2
    (0.625, 4, 3),          # 2.5 → 3
    (0.2, 10, 2),           # 정수 그대로
    (0.01, 10, 1),          # 0.1 → 0 → 최소 1
    (17 / 1211, 242, 3),    # CN7 불량률, 배치 242: 3.40 → 3
    (25 / 1182, 236, 5),    # RG3 불량률, 배치 236: 4.99 → 5
])
def test_n_flag_half_up_min_one(k, n, expected):
    assert n_flag(k, n) == expected


def test_n_flag_differs_from_builtin_round():
    assert round(2.5) == 2
    assert n_flag(0.25, 10) == 3


def test_flag_top_distinct_scores():
    flags = flag_top(np.array([0.1, 0.9, 0.5, 0.3]), 0.5)
    assert flags.dtype == bool
    assert flags.tolist() == [False, True, True, False]


def test_flag_top_includes_boundary_ties():
    scores = np.array([0.9, 0.8, 0.8, 0.1])
    flags = flag_top(scores, 0.5)
    assert n_flag(0.5, len(scores)) == 2     # 명목
    assert int(flags.sum()) == 3             # 실제: 경계 동점 포함
    assert flags.tolist() == [True, True, True, False]


def test_flag_top_all_tied_flags_all():
    assert flag_top(np.full(5, 0.3), 0.2).all()


def test_flag_top_minimum_one():
    flags = flag_top(np.array([0.1, 0.2, 0.3]), 0.01)
    assert flags.tolist() == [False, False, True]


# ── CV 러너 ───────────────────────────────

class _SeededNoise:
    """테스트 전용 모델: 입력과 무관한 seed 고정 난수 점수."""

    name = "seeded_noise"

    def __init__(self, random_state: int) -> None:
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.random.default_rng(self.random_state).random(len(X))


@pytest.fixture(scope="module")
def train_xyg():
    out = {}
    for line, rel in CFG["data"]["train"].items():
        t = load_injection(pd.read_csv(config.PROJECT_ROOT / rel), CFG["rename"], line, "train")
        out[line] = (t.data[t.feature_cols], t.data[LABEL_COL].to_numpy(), t.data[GROUP_COL].to_numpy())
    return out


def test_config_declares_implemented_rules():
    assert CFG["cv"] == {"max_folds": 5}
    assert CFG["threshold"] == {
        "main_k": "defect_rate", "aux_k": "auto_group_size",
        "tie_rule": "include_all", "rounding": "half_up", "min_flag": 1,
    }


@pytest.mark.parametrize("line, expected", [("CN7", 0.756), ("RG3", 0.667)])
def test_dup_ceiling_matches_reference(train_xyg, line, expected):
    _, y, g = train_xyg[line]
    assert dup_ceiling_f1(y, g) == pytest.approx(expected, abs=5e-4)


def test_random_floor_is_defect_rate(train_xyg):
    _, y, _ = train_xyg["CN7"]
    assert random_floor_f1(y) == pytest.approx(17 / 1211)


@pytest.mark.parametrize("line", ["CN7", "RG3"])
def test_cv_preserves_groups_and_covers_rows(train_xyg, line):
    X, y, g = train_xyg[line]
    records = run_cv(X, y, g, _SeededNoise, config.SEEDS, CFG["cv"]["max_folds"])
    for seed in config.SEEDS:
        recs = [r for r in records if r["seed"] == seed]
        assert len(recs) == 5
        idx = np.concatenate([r["idx"] for r in recs])
        assert sorted(idx.tolist()) == list(range(len(y)))           # 전 행이 평가 fold에 정확히 1번
        fold_of = pd.Series(np.concatenate([[r["fold"]] * len(r["idx"]) for r in recs]), index=idx)
        assert (fold_of.groupby(g[idx]).nunique() == 1).all()        # fold를 넘는 그룹 0개
        assert all(r["n_flag_actual"] >= r["n_flag_nominal"] for r in recs)


def test_cv_deterministic(train_xyg):
    X, y, g = train_xyg["CN7"]
    a = run_cv(X, y, g, _SeededNoise, (0, 1), CFG["cv"]["max_folds"])
    b = run_cv(X, y, g, _SeededNoise, (0, 1), CFG["cv"]["max_folds"])
    assert all(np.array_equal(ra["idx"], rb["idx"]) and np.array_equal(ra["flag"], rb["flag"]) for ra, rb in zip(a, b))


def test_cv_stops_when_not_learnable():
    X = pd.DataFrame({"v": np.arange(10.0)})
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    g = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8])                     # 불량 그룹 1개
    with pytest.raises(RuntimeError, match="학습 불가"):
        run_cv(X, y, g, _SeededNoise, (0,), 5)


def test_score_repeats_one_row_per_seed(train_xyg):
    X, y, g = train_xyg["RG3"]
    scores = score_repeats(run_cv(X, y, g, _SeededNoise, config.SEEDS, CFG["cv"]["max_folds"]), y)
    assert scores["seed"].tolist() == list(config.SEEDS)
    assert scores[["f1", "pr_auc"]].apply(lambda s: s.between(0, 1)).all().all()

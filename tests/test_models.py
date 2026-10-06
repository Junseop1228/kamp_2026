"""모델 계약 검증(핸드오프 §6, KI-004): 등록부 = 설정, 출력 모양·범위·결측, seed 재현, 배치 구성 무관, 사전확률 보정."""
import json

import numpy as np
import pandas as pd
import pytest

import config
from src.models import REGISTRY, make_factory
from src.models.baselines import RandomScore
from src.models.candidates import prior_correct

CFG = json.loads((config.CONFIGS / "injection.json").read_text(encoding="utf-8"))
PARAMS = CFG["models"]["params"]
NAMES = sorted(REGISTRY)

_rng = np.random.default_rng(0)
X = pd.DataFrame({"a": _rng.normal(size=60), "b": _rng.normal(size=60), "k": 0.0})
Y = np.r_[np.ones(6, dtype=int), np.zeros(54, dtype=int)]
X.loc[:5, ["a", "b"]] += 2.0                                    # 불량 6행을 바깥으로


def test_registry_matches_config():
    declared = {CFG["models"]["floor"], *CFG["models"]["complexity_order"]}
    assert declared == set(REGISTRY)
    assert set(PARAMS) == set(REGISTRY)
    assert CFG["models"]["class_weight"] == "balanced" and CFG["models"]["probability"] == "prior_correction"


@pytest.mark.parametrize("name", NAMES)
def test_contract_output(name):
    p = make_factory(name, PARAMS)(random_state=0).fit(X, Y).predict_proba(X)
    assert isinstance(p, np.ndarray) and p.shape == (len(X),)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize("name", NAMES)
def test_contract_same_seed_same_output(name):
    make = make_factory(name, PARAMS)
    a = make(random_state=3).fit(X, Y).predict_proba(X)
    b = make(random_state=3).fit(X, Y).predict_proba(X)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", NAMES)
def test_contract_independent_of_batch_composition(name):
    model = make_factory(name, PARAMS)(random_state=0).fit(X, Y)
    full = model.predict_proba(X)
    assert np.allclose(model.predict_proba(X.iloc[10:20]), full[10:20], rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", ["deviation_score", "guidebook_dae"])
def test_train_constant_column_is_ignored(name):
    model = make_factory(name, PARAMS)(random_state=0).fit(X, Y)
    moved = X.assign(k=np.linspace(-50, 50, len(X)))           # train 상수 변수가 test에서 크게 변해도
    assert np.allclose(model.predict_proba(moved), model.predict_proba(X))


def test_random_score_seed_changes_output():
    assert not np.array_equal(RandomScore(random_state=0).predict_proba(X), RandomScore(random_state=1).predict_proba(X))


def test_prior_correct_formula_and_order():
    r = 17 / 1194
    assert prior_correct(np.array([0.5]), r)[0] == pytest.approx(r / (1 + r))
    p = np.array([0.0, 0.2, 0.5, 0.9, 1.0])
    q = prior_correct(p, r)
    assert q[0] == 0.0 and q[-1] == 1.0 and np.all(np.diff(q) > 0)

"""모델 계약 검증(핸드오프 §6): 출력 모양·범위·결측, seed 재현. P5 모델은 MODELS에 추가한다."""
import numpy as np
import pandas as pd
import pytest

from src.models.baselines import RandomScore

MODELS = [RandomScore]
X = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0)[::-1]})
Y = np.r_[np.ones(5, dtype=int), np.zeros(45, dtype=int)]


@pytest.mark.parametrize("model_cls", MODELS)
def test_contract_output(model_cls):
    assert isinstance(model_cls.name, str)
    p = model_cls(random_state=0).fit(X, Y).predict_proba(X)
    assert isinstance(p, np.ndarray) and p.shape == (len(X),)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize("model_cls", MODELS)
def test_contract_same_seed_same_output(model_cls):
    a = model_cls(random_state=3).fit(X, Y).predict_proba(X)
    b = model_cls(random_state=3).fit(X, Y).predict_proba(X)
    assert np.array_equal(a, b)


def test_random_score_seed_changes_output():
    a = RandomScore(random_state=0).predict_proba(X)
    b = RandomScore(random_state=1).predict_proba(X)
    assert not np.array_equal(a, b)


def test_random_score_independent_of_batch_composition():
    model = RandomScore(random_state=0)
    full = model.predict_proba(X)
    assert np.array_equal(model.predict_proba(X.iloc[10:20]), full[10:20])
    assert not np.array_equal(model.predict_proba(X.iloc[10:20]), model.predict_proba(X.iloc[20:30]))

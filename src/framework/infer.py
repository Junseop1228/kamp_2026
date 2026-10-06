"""Infer 단계: 모델을 labeled 전체로 다시 학습해 test 전체를 한 배치로 판정한다. 이 단계에서 새로 정하는 규칙은 없다."""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from src.framework.adapter import GROUP_COL, ID_COL, LABEL_COL, ProductTable
from src.framework.evaluate import flag_top, n_flag

PROBA_COL = "proba"
RANK_COL = "rank"
FLAG_COL = "flag"
FLAG_RATE_COL = "flag_rate"


def _check_pair(train: ProductTable, test: ProductTable) -> None:
    if train.line != test.line:
        raise ValueError(f"제품군 불일치: train {train.line} / test {test.line}")
    if train.feature_cols != test.feature_cols:
        raise ValueError(f"[{test.line}] train과 test의 공정변수 열이 다르다")


def _fit_predict(train: ProductTable, test: ProductTable, make_model: Callable[..., Any], seed: int):
    y = train.data[LABEL_COL].to_numpy()
    model = make_model(random_state=seed).fit(train.data[train.feature_cols], y)
    proba = np.asarray(model.predict_proba(test.data[test.feature_cols]), dtype=float)
    if proba.shape != (len(test.data),):
        raise ValueError(f"[{test.line}] predict_proba 출력 모양 {proba.shape} ≠ ({len(test.data)},)")
    return model, proba, float(y.mean())


def infer(
    train: ProductTable,
    test: ProductTable,
    make_model: Callable[..., Any],
    seed: int,
    limitations: Sequence[str] = (),
) -> tuple[pd.DataFrame, dict]:
    """labeled 전체로 다시 학습하고 test 전체를 한 배치로 판정한다. k = labeled 전체 불량률, 판정 규칙은 CV와 같다."""
    _check_pair(train, test)
    model, proba, k = _fit_predict(train, test, make_model, seed)
    flags = flag_top(proba, k)
    y = train.data[LABEL_COL].to_numpy()
    pred = pd.DataFrame({
        ID_COL: test.data[ID_COL].to_numpy(),
        PROBA_COL: proba,
        RANK_COL: pd.Series(proba).rank(method="min", ascending=False).astype(int).to_numpy(),
        FLAG_COL: flags.astype(int),
        GROUP_COL: test.data[GROUP_COL].to_numpy(),
    })
    meta = {
        "line": test.line,
        "model": getattr(model, "name", type(model).__name__),
        "seed": seed,
        "k": k,
        "n_train": int(len(y)),
        "n_train_defects": int(y.sum()),
        "n_test": int(len(pred)),
        "n_flag_nominal": n_flag(k, len(pred)),
        "n_flag_actual": int(flags.sum()),
        "limitations": list(limitations),
        "alarm": None,                 # P7 Prioritize에서 채운다
        "distribution_warning": None,  # P4 Diagnose에서 채운다
    }
    return pred, meta


def flag_rate_over_seeds(train: ProductTable, test: ProductTable, make_model: Callable[..., Any], seeds: Sequence[int]) -> np.ndarray:
    """판정 확신도(D-009): seed마다 다시 학습해 표시한 비율. 결정적 모델이면 0 또는 1만 나온다."""
    _check_pair(train, test)
    flags = []
    for seed in seeds:
        _, proba, k = _fit_predict(train, test, make_model, seed)
        flags.append(flag_top(proba, k))
    return np.mean(flags, axis=0)

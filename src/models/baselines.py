"""프레임워크 내장 기준모델. 모델 계약(핸드오프 §6): __init__(random_state), fit(X, y), predict_proba(X).

점수는 제품 하나의 정보로만 정한다(KI-004). 같은 배치의 다른 제품이나 배치 안 위치를 쓰지 않는다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


def _good_nonconstant(X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, pd.Index]:
    """양품 행과, 양품 안에서 값이 둘 이상인 열. train 상수 변수는 test에서 변해도 점수를 지배하지 않게 뺀다."""
    good = X[np.asarray(y) == 0]
    cols = good.columns[good.nunique() > 1]
    return good, cols


class RandomScore:
    """무작위 점수 기준모델. 입력값을 보지 않으므로 무작위 하한 F1(≈ 불량률)을 실측으로 확인하는 데 쓴다."""

    name = "random_score"

    def __init__(self, random_state: int) -> None:
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RandomScore":
        """학습하지 않는다. 계약을 맞추려고 자기 자신을 돌려준다."""
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """제품(행 번호)마다 seed별로 고정된 0~1 난수. 배치 구성과 무관해 fold끼리 점수가 독립이다.

        배치 안 위치로 난수를 주면 같은 seed의 모든 fold가 같은 위치를 표시해, 행 순서에 몰린 불량과 상관이 생긴다.
        """
        pos = np.asarray(X.index, dtype=np.int64)
        return np.random.default_rng(self.random_state).random(int(pos.max()) + 1)[pos]


class DeviationScore:
    """양품 분포로부터의 거리. 양품의 평균·표준편차로 z를 만들고 RMS z를 s/(1+s)로 0~1에 올린다. 라벨은 양품 선별에만 쓴다."""

    name = "deviation_score"

    def __init__(self, random_state: int) -> None:
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "DeviationScore":
        """양품 행으로 변수별 평균·표준편차를 잡는다."""
        good, self.cols_ = _good_nonconstant(X, y)
        self.mu_ = good[self.cols_].mean()
        self.sd_ = good[self.cols_].std()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """위험 점수(확률 아님). 단조 변환이라 순위는 RMS z와 같다."""
        z = (X[self.cols_] - self.mu_) / self.sd_
        s = np.sqrt((z ** 2).mean(axis=1)).to_numpy()
        return s / (1.0 + s)


class GuidebookDAE:
    """가이드북 1부 잡음제거 오토인코더의 계약형 재구현(D-009). 양품만 학습하고 복원오차가 클수록 위험하다."""

    name = "guidebook_dae"

    def __init__(self, random_state: int, input_drop: float = 0.3, n_copies: int = 10, hidden=(15, 5, 15),
                 batch_size: int = 30, max_epochs: int = 30, validation_fraction: float = 0.2, patience: int = 5) -> None:
        self.random_state = random_state
        self.input_drop, self.n_copies, self.hidden = input_drop, n_copies, tuple(hidden)
        self.batch_size, self.max_epochs = batch_size, max_epochs
        self.validation_fraction, self.patience = validation_fraction, patience

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GuidebookDAE":
        """양품을 MinMax로 맞추고, 입력 일부를 가린(Dropout 대체) 복제본으로 원본을 복원하도록 학습한다."""
        good, self.cols_ = _good_nonconstant(X, y)
        self.scaler_ = MinMaxScaler().fit(good[self.cols_].to_numpy(float))
        clean = np.tile(self.scaler_.transform(good[self.cols_].to_numpy(float)), (self.n_copies, 1))
        keep = np.random.default_rng(self.random_state).random(clean.shape) >= self.input_drop
        noisy = np.where(keep, clean / (1.0 - self.input_drop), 0.0)
        self.net_ = MLPRegressor(
            hidden_layer_sizes=self.hidden, activation="relu", solver="adam", batch_size=self.batch_size,
            max_iter=self.max_epochs, early_stopping=True, validation_fraction=self.validation_fraction,
            n_iter_no_change=self.patience, random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.net_.fit(noisy, clean)
        self.ref_ = float(np.median(self._error(self.scaler_.transform(good[self.cols_].to_numpy(float)))))
        return self

    def _error(self, Z: np.ndarray) -> np.ndarray:
        return ((self.net_.predict(Z) - Z) ** 2).mean(axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """위험 점수(확률 아님): 복원오차 e를 e / (e + 학습 양품 오차 중앙값)으로 0~1에 올린다."""
        e = self._error(self.scaler_.transform(X[self.cols_].to_numpy(float)))
        return e / (e + self.ref_)

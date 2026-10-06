"""비교 후보 모델. 지도학습 모델은 불균형 가중치(balanced)로 학습하고, 출력은 사전확률 보정한 불량 확률이다(D-009)."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def prior_correct(p: np.ndarray, ratio: float) -> np.ndarray:
    """균형 가중치로 학습한 확률을 실제 클래스 비율로 되돌린다. ratio = 불량 수 / 양품 수. 단조 변환이라 순위는 그대로다."""
    p = np.asarray(p, dtype=float)
    return p * ratio / (p * ratio + (1.0 - p))


class _BalancedClassifier:
    """불균형 가중치 학습 + 사전확률 보정 공통부. 하위 클래스는 _make()만 정의한다."""

    name = ""

    def __init__(self, random_state: int, **params) -> None:
        self.random_state = random_state
        self.params = params

    def _make(self):
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_BalancedClassifier":
        """학습 불량 비율을 기억하고 가중치 학습한다."""
        y = np.asarray(y)
        self.ratio_ = float(y.sum() / (len(y) - y.sum()))
        self.est_ = self._make().fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """사전확률 보정한 불량 확률."""
        return prior_correct(self.est_.predict_proba(X)[:, 1], self.ratio_)


class LogisticModel(_BalancedClassifier):
    """표준화 + L2 로지스틱 회귀."""

    name = "logistic"

    def _make(self):
        return make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", random_state=self.random_state, **self.params)
        )


class RandomForestModel(_BalancedClassifier):
    """얕은 랜덤포레스트. 스레드 병렬은 확률 합산 순서가 바뀌어 재실행 결과가 비트 단위로 달라지므로 n_jobs=1(KI-005)."""

    name = "random_forest"

    def _make(self):
        return RandomForestClassifier(class_weight="balanced", n_jobs=1, random_state=self.random_state, **self.params)


class HistGBModel(_BalancedClassifier):
    """작은 잎의 히스토그램 그래디언트 부스팅. 조기종료 없음(설정값 고정)."""

    name = "hist_gb"

    def _make(self):
        return HistGradientBoostingClassifier(
            class_weight="balanced", early_stopping=False, random_state=self.random_state, **self.params
        )


class SoftVoteEnsemble:
    """구성 모델의 보정 확률을 평균한다. 배치 안 순위 평균은 배치 구성에 의존해 계약 위반이라 쓰지 않는다(D-009)."""

    name = "soft_vote_ensemble"

    def __init__(self, random_state: int, members: Sequence[Callable]) -> None:
        self.random_state = random_state
        self.members = list(members)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SoftVoteEnsemble":
        """구성 모델을 같은 seed로 각각 학습한다."""
        self.fitted_ = [make(random_state=self.random_state).fit(X, y) for make in self.members]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """구성 모델 불량 확률의 산술평균."""
        return np.mean([m.predict_proba(X) for m in self.fitted_], axis=0)

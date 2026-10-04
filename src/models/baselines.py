"""프레임워크 내장 기준모델. 모델 계약(핸드오프 §6): __init__(random_state), fit(X, y), predict_proba(X)."""
from __future__ import annotations

import numpy as np
import pandas as pd


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

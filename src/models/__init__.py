"""모델 슬롯: 이름 → 클래스 등록부와, 설정값(models.params)으로 make_model(random_state=...) 팩토리를 만드는 함수."""
from __future__ import annotations

from functools import partial
from typing import Callable

from src.models.baselines import DeviationScore, GuidebookDAE, RandomScore
from src.models.candidates import HistGBModel, LogisticModel, RandomForestModel, SoftVoteEnsemble

REGISTRY = {
    cls.name: cls
    for cls in (RandomScore, DeviationScore, LogisticModel, RandomForestModel, HistGBModel, GuidebookDAE, SoftVoteEnsemble)
}


def make_factory(name: str, params_by_model: dict) -> Callable:
    """설정의 모델별 파라미터로 팩토리를 만든다. 앙상블은 구성 모델 이름을 팩토리로 바꿔 넘긴다."""
    params = dict(params_by_model.get(name, {}))
    if name == SoftVoteEnsemble.name:
        params["members"] = [make_factory(m, params_by_model) for m in params["members"]]
    return partial(REGISTRY[name], **params)

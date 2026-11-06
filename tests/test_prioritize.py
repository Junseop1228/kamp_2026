"""Prioritize 검증(D-012): 검사량 표, 무작위 결합표, 배치 평균·알람."""
import numpy as np
import pandas as pd
import pytest

from src.framework.prioritize import alarm_scan, batch_means, budget_table, sampling_combo


def test_budget_table_counts_and_lift():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    rank = np.array([0.0, 0.5, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    t = budget_table(y, rank, [0.2], 0.1, 0.2).set_index("budget")
    assert (t.loc["20%", "inspected"], t.loc["20%", "caught"], t.loc["20%", "missed"]) == (2, 1, 1)
    assert t.loc["20%", "recall"] == 0.5 and t.loc["20%", "lift"] == pytest.approx(0.5 / 0.2)
    assert list(t.index) == ["20%", "1k", "aux_k"]


def test_sampling_combo_expected_catches():
    budget = pd.DataFrame([{"budget": "10%", "inspected": 100, "caught": 6, "missed": 4}])
    t = sampling_combo(budget, 1000, [0.0, 0.10]).set_index("random_rate")
    assert t.loc[0.0, "expected_caught"] == 6
    assert t.loc[0.10, "expected_caught"] == pytest.approx(6.4) and t.loc[0.10, "inspected"] == pytest.approx(190)


def test_batch_means_merges_small_tail():
    b = batch_means(np.ones(240), 100)
    assert b["rows"].tolist() == [100, 140]                     # 꼬리 40행(절반 미만)은 앞 배치에 붙인다
    assert batch_means(np.ones(260), 100)["rows"].tolist() == [100, 100, 60]


def test_alarm_scan_threshold():
    proba = np.r_[np.full(100, 0.01), np.full(100, 0.05)]
    out = alarm_scan(proba, 100, 0.02)
    assert out["alarm"].tolist() == [False, True]

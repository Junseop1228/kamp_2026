"""Report 단계: 결과 표(DataFrame)를 받아 보고서용 그림(matplotlib Figure)을 만든다. 파일 읽기·쓰기는 run.py가 한다."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "font.family": ["Malgun Gothic", "DejaVu Sans"],
    "axes.unicode_minus": False, "figure.dpi": 110, "savefig.dpi": 160, "axes.grid": True, "grid.alpha": 0.3,
})
FN_COLORS = {"잡힘": "#2e7d32", "경계": "#9ccc65", "모델 차이": "#ffb300", "신호 없음": "#9e9e9e"}


def fig_pair_share(groups: dict) -> plt.Figure:
    """제품군·분할별 동일입력 쌍에 속한 행 비율(1장: 생산단위)."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    labels, vals = zip(*[(f"{line} {split}", g[split]["rows_in_multi_share"]) for line, g in groups.items() for split in ("train", "test")])
    ax.bar(labels, vals, color=["#1565c0", "#90caf9"] * len(groups))
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("동일입력 쌍에 속한 행 비율")
    ax.set_title("labeled는 거의 전부 쌍(2캐비티), unlabeled는 약 1/3만 쌍")
    fig.tight_layout()
    return fig


def fig_defect_positions(y: np.ndarray, line: str) -> plt.Figure:
    """파일 순서상 불량 위치(1장: 시간 관계 대리 지표)."""
    fig, ax = plt.subplots(figsize=(7, 1.8))
    pos = np.flatnonzero(np.asarray(y) == 1)
    ax.vlines(pos, 0, 1, color="#c62828", lw=1)
    ax.set_xlim(0, len(y))
    ax.set_yticks([])
    ax.set_xlabel("labeled 파일 행 순서")
    ax.set_title(f"{line} 불량 {len(pos)}개의 파일상 위치")
    fig.tight_layout()
    return fig


def fig_model_f1(scores: pd.DataFrame, order: list, floor: str, ceiling: float, chosen: str, line: str) -> plt.Figure:
    """후보별 F1(5×10 반복의 평균 ± 표준편차)과 무작위 하한·중복 상한(2장)."""
    s = scores.groupby("model")["f1"].agg(["mean", "std"]).reindex(order)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = ["#1565c0" if m == chosen else "#90a4ae" for m in order]
    ax.bar(order, s["mean"], yerr=s["std"], color=colors, capsize=3)
    ax.axhline(scores.loc[scores["model"] == floor, "f1"].mean(), color="#6d4c41", ls="--", label="무작위 하한")
    ax.axhline(ceiling, color="#c62828", ls=":", label=f"중복 상한 {ceiling:.3f}")
    ax.set_ylabel("F1 (5-fold × 10회)")
    ax.set_title(f"{line} 후보 모델 비교 (파란색 = 선택)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_importance(imp: pd.DataFrame, line: str, top: int = 10) -> plt.Figure:
    """permutation 중요도 신뢰도(상위 5위에 든 비율)(3장)."""
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(d["variable"], d["top_rate"], color="#1565c0")
    ax.set_xlim(0, 1)
    ax.set_xlabel("50회 학습 중 상위 5위에 든 비율")
    ax.set_title(f"{line} 영향 변수(선택 모델, PR-AUC 하락 기준)")
    fig.tight_layout()
    return fig


def fig_pdp(pdp: pd.DataFrame, line: str) -> plt.Figure:
    """두 상위 변수의 2D 부분의존도(3장: 상호작용)."""
    va, vb = [c for c in pdp.columns if c != "mean_output"]
    grid = pdp.pivot_table(index=vb, columns=va, values="mean_output", aggfunc="mean").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap="Reds")
    ax.set_xticks(range(len(grid.columns)), [f"{v:.1f}" for v in grid.columns], rotation=45, fontsize=7)
    ax.set_yticks(range(len(grid.index)), [f"{v:.1f}" for v in grid.index], fontsize=7)
    ax.set_xlabel(va)
    ax.set_ylabel(vb)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="평균 불량 확률")
    ax.set_title(f"{line} 2D 부분의존도")
    fig.tight_layout()
    return fig


def fig_fn_types(counts: dict) -> plt.Figure:
    """제품군별 FN 유형 구성(3장: 잘 되는 조건과 실패하는 조건)."""
    fig, ax = plt.subplots(figsize=(6, 2.6))
    lines = list(counts)
    left = np.zeros(len(lines))
    for t, color in FN_COLORS.items():
        v = np.array([counts[line][t] for line in lines], dtype=float)
        ax.barh(lines, v, left=left, color=color, label=t)
        for i, (lv, vv) in enumerate(zip(left, v)):
            if vv:
                ax.text(lv + vv / 2, i, int(vv), ha="center", va="center", fontsize=8)
        left += v
    ax.set_xlabel("불량 수")
    ax.set_title("불량별 FN 유형(신호 없음 = 기계가 원리적으로 모르는 몫)")
    ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.65))
    fig.tight_layout()
    return fig


def fig_budget(budgets: dict) -> plt.Figure:
    """검사량별 Recall: 모델 우선순위 vs 무작위(4장)."""
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for line, b in budgets.items():
        d = b[b["budget"].str.endswith("%")]
        ax.plot(d["random_recall"], d["recall"], marker="o", label=f"{line} 모델 우선순위")
    ax.plot([0, 0.2], [0, 0.2], color="#6d4c41", ls="--", label="무작위 검사")
    ax.set_xlabel("검사 비율")
    ax.set_ylabel("Recall (잡은 불량 / 전체 불량)")
    ax.set_title("검사량 대비 불량 포착")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_alarm(batches: pd.DataFrame, threshold: float, line: str) -> plt.Figure:
    """labeled 파일 순서 배치의 평균 출력과 알람 기준(4장)."""
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(batches["batch"].astype(str), batches["mean_output"], color=np.where(batches["alarm"], "#c62828", "#90a4ae"))
    ax.axhline(threshold, color="#c62828", ls="--", label="알람 기준(교차검증 배치 평균의 최댓값)")
    ax.set_xlabel("labeled 배치(파일 순서)")
    ax.set_ylabel("배치 평균 불량 확률")
    ax.set_title(f"{line} 배치 알람 검증")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

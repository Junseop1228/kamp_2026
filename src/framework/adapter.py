"""Adapter 단계: 원천 테이블을 프레임워크 공통 형식(ProductTable)으로 바꾼다.

파일을 읽지 않는다. 읽기는 루트 run.py가 맡고, 여기서는 DataFrame만 받는다(AGENTS.md §6).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ID_COL = "product_id"
LABEL_COL = "label"
TIME_COL = "time"
GROUP_COL = "group"
RESERVED_COLS = (ID_COL, LABEL_COL, TIME_COL, GROUP_COL)
SPLITS = ("train", "test")


@dataclass
class ProductTable:
    """제품군 하나·분할 하나의 제품 단위 테이블. 행 1개 = 제품 1개."""

    line: str
    split: str
    data: pd.DataFrame
    feature_cols: list[str]

    @property
    def has_label(self) -> bool:
        """라벨 열 보유 여부. train만 True다."""
        return LABEL_COL in self.data.columns


def load_injection(df: pd.DataFrame, rename: dict[str, str], line: str, split: str) -> ProductTable:
    """사출성형 원천 DataFrame을 이름 변경 → 검증 → 그룹 부여 순으로 ProductTable로 만든다."""
    if split not in SPLITS:
        raise ValueError(f"split은 {SPLITS} 중 하나여야 한다: {split!r}")
    data = df.rename(columns=rename).copy()
    feature_cols = [c for c in data.columns if c not in RESERVED_COLS]
    table = ProductTable(line=line, split=split, data=data, feature_cols=feature_cols)
    validate_table(table)
    if GROUP_COL not in table.data.columns:
        table.data[GROUP_COL] = assign_groups(table.data, feature_cols)
    return table


def validate_table(table: ProductTable) -> None:
    """입력 검증. 문제를 전부 모아 사유와 함께 ValueError로 멈춘다."""
    data = table.data
    problems: list[str] = []

    if ID_COL not in data.columns:
        problems.append(f"ID 열({ID_COL}) 없음")
    elif data[ID_COL].duplicated().any():
        problems.append(f"중복 ID {int(data[ID_COL].duplicated().sum())}개")

    if table.split == "train":
        if not table.has_label:
            problems.append("train에 라벨 열 없음")
        else:
            values = set(data[LABEL_COL].dropna().unique().tolist())
            if not values <= {0, 1}:
                problems.append(f"라벨이 0/1 이진이 아님: {sorted(values, key=str)}")
    elif table.has_label:
        problems.append("test에 라벨 열이 있음")

    if not table.feature_cols:
        problems.append("공정변수 열 없음")
    non_numeric = [c for c in table.feature_cols if not pd.api.types.is_numeric_dtype(data[c])]
    if non_numeric:
        problems.append(f"비수치 열: {non_numeric}")

    checked = [c for c in (ID_COL, LABEL_COL, *table.feature_cols) if c in data.columns]
    n_missing = data[checked].isna().sum()
    n_missing = n_missing[n_missing > 0]
    if not n_missing.empty:
        problems.append(f"결측: {n_missing.to_dict()}")

    if problems:
        raise ValueError(f"[{table.line}/{table.split}] 입력 검증 실패: " + "; ".join(problems))


def assign_groups(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """공정변수 원값이 전부 같은 행끼리 같은 그룹 번호를 준다. 반올림 없음, 번호는 첫 등장 순서."""
    return df.groupby(feature_cols, sort=False).ngroup().rename(GROUP_COL)

"""실행 진입점: 제품군마다 Adapter → Diagnose → Evaluate → Infer를 1회 실행하고 results/에 저장한다.

실행: conda run -n kamp2026 python run.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config
from src.framework.adapter import GROUP_COL, LABEL_COL, load_injection
from src.framework.diagnose import check_variable_dictionary, diagnose_report
from src.framework.evaluate import run_cv, score_repeats
from src.framework.infer import infer
from src.models.baselines import RandomScore

CONFIG_FILE = config.CONFIGS / "injection.json"
MODEL = RandomScore  # 얇은 관통용. P5에서 선택 모델로 교체한다.


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False, lineterminator="\n")


def main() -> None:
    """설정 파일의 제품군을 하나씩 독립 실행한다(D-004)."""
    cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    variables = pd.read_csv(config.PROJECT_ROOT / cfg["variable_dictionary"])
    for line in cfg["data"]["train"]:
        train, test = (
            load_injection(pd.read_csv(config.PROJECT_ROOT / cfg["data"][split][line]), cfg["rename"], line, split)
            for split in ("train", "test")
        )
        check_variable_dictionary(variables, train.feature_cols)
        out = config.RESULTS / line.lower()
        out.mkdir(parents=True, exist_ok=True)

        diag, diag_tables = diagnose_report(train, test, cfg["diagnose"])
        _write_json(out / "diagnose_summary.json", diag)
        for name, df in diag_tables.items():
            _write_csv(out / f"diagnose_{name}.csv", df)

        X, y, g = train.data[train.feature_cols], train.data[LABEL_COL].to_numpy(), train.data[GROUP_COL].to_numpy()
        scores = score_repeats(run_cv(X, y, g, MODEL, config.SEEDS, cfg["cv"]["max_folds"]), y)
        scores.insert(0, "model", MODEL.name)
        _write_csv(out / "cv_scores.csv", scores)

        warnings = diag["distribution_warning"]
        limitations = cfg["limitations"] + (
            [f"train·test 분포 차이 경고 {len(warnings)}개 변수(diagnose_shift.csv). 예측 확률의 절대값은 해석하지 않는다."]
            if warnings else []
        )
        pred, meta = infer(train, test, MODEL, config.SEEDS[0], limitations)
        meta["distribution_warning"] = warnings
        _write_csv(out / "predictions.csv", pred)
        _write_json(out / "predictions_meta.json", meta)

        adj = diag["twin_adjacency"]["train"]
        f1 = scores["f1"]
        print(
            f"[{line}] 진단: 쌍 인접 {adj['adjacent_share']:.3f}(행 순서 신뢰 {adj['row_order_trusted']}), "
            f"분포 경고 {len(warnings)}개 | {MODEL.name} CV F1 {f1.mean():.4f} ± {f1.std(ddof=1):.4f} "
            f"(하한 {diag['f1_floor']:.4f}, 상한 {diag['f1_ceiling']:.4f}) | "
            f"test {meta['n_test']}행, 표시 {meta['n_flag_actual']}개 → {out.relative_to(config.PROJECT_ROOT).as_posix()}"
        )


if __name__ == "__main__":
    main()

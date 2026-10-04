"""실행 진입점: 제품군마다 Adapter → Evaluate → Infer를 1회 실행하고 results/에 저장한다.

실행: conda run -n kamp2026 python run.py
"""
from __future__ import annotations

import json

import pandas as pd

import config
from src.framework.adapter import GROUP_COL, LABEL_COL, load_injection
from src.framework.evaluate import dup_ceiling_f1, random_floor_f1, run_cv, score_repeats
from src.framework.infer import infer
from src.models.baselines import RandomScore

CONFIG_FILE = config.CONFIGS / "injection.json"
MODEL = RandomScore  # 얇은 관통용. P5에서 선택 모델로 교체한다.


def main() -> None:
    """설정 파일의 제품군을 하나씩 독립 실행한다(D-004)."""
    cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    for line in cfg["data"]["train"]:
        tables = {
            split: load_injection(pd.read_csv(config.PROJECT_ROOT / cfg["data"][split][line]), cfg["rename"], line, split)
            for split in ("train", "test")
        }
        train = tables["train"]
        X, y, g = train.data[train.feature_cols], train.data[LABEL_COL].to_numpy(), train.data[GROUP_COL].to_numpy()

        scores = score_repeats(run_cv(X, y, g, MODEL, config.SEEDS, cfg["cv"]["max_folds"]), y)
        scores.insert(0, "model", MODEL.name)
        pred, meta = infer(train, tables["test"], MODEL, config.SEEDS[0], cfg["limitations"])

        out = config.RESULTS / line.lower()
        out.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out / "cv_scores.csv", index=False, lineterminator="\n")
        pred.to_csv(out / "predictions.csv", index=False, lineterminator="\n")
        (out / "predictions_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )

        f1 = scores["f1"]
        print(
            f"[{line}] {MODEL.name} CV F1 {f1.mean():.4f} ± {f1.std(ddof=1):.4f} "
            f"(하한 {random_floor_f1(y):.4f}, 상한 {dup_ceiling_f1(y, g):.4f}) | "
            f"test {meta['n_test']}행, 표시 {meta['n_flag_actual']}개 → {out.relative_to(config.PROJECT_ROOT).as_posix()}"
        )


if __name__ == "__main__":
    main()

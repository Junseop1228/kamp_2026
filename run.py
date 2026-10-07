"""실행 진입점: 제품군마다 Adapter → Diagnose → Evaluate(후보 비교·선택) → Analyze → Infer를 1회 실행하고 results/에 저장한다.

실행: conda run -n kamp2026 python run.py   (CPU 기준 약 15분, 대부분 guidebook_dae 학습)
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.framework.adapter import GROUP_COL, ID_COL, LABEL_COL, load_injection
from src.framework.analyze import (
    FN_TYPES, aux_k, domain_crosscheck, fn_concentration, fn_types, fp_analysis, interaction_2x2, pdp_2d,
    permutation_importance_cv, pick_top_pair,
)
from src.framework.diagnose import check_variable_dictionary, diagnose_report
from src.framework.evaluate import (
    block_split, calibration_summary, oof_summary, run_cv, score_repeats, select_model,
)
from src.framework.infer import FLAG_RATE_COL, flag_rate_over_seeds, infer
from src.framework.prioritize import alarm_scan, budget_table, sampling_combo
from src.framework import report
from src.models import REGISTRY, make_factory
from src.models.candidates import SoftVoteEnsemble, _BalancedClassifier

CONFIG_FILE = config.CONFIGS / "injection.json"


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False, lineterminator="\n")


def _is_probability(name: str) -> bool:
    return issubclass(REGISTRY[name], (_BalancedClassifier, SoftVoteEnsemble))


def _xyg(train):
    return train.data[train.feature_cols], train.data[LABEL_COL].to_numpy(), train.data[GROUP_COL].to_numpy()


def evaluate_line(train, cfg: dict, diag: dict, out: Path):
    """후보 전부를 5×10 교차검증 → 사전 등록 선택 규칙 → 선택 모델의 보고 전용 분석."""
    X, y, g = _xyg(train)
    params, order, floor = cfg["models"]["params"], cfg["models"]["complexity_order"], cfg["models"]["floor"]
    max_folds, ro = cfg["cv"]["max_folds"], cfg["report_only"]

    scores, oof = [], []
    for name in [floor, *order]:
        rec = run_cv(X, y, g, make_factory(name, params), config.SEEDS, max_folds)
        scores.append(score_repeats(rec, y).assign(model=name))
        oof.append(oof_summary(rec, len(y)).assign(model=name, product_id=train.data[ID_COL].to_numpy(), label=y, group=g))
        print(f"  [{train.line}] {name} CV F1 {scores[-1]['f1'].mean():.4f}", flush=True)
    scores, oof = pd.concat(scores, ignore_index=True), pd.concat(oof, ignore_index=True)
    chosen, table = select_model(scores, order, floor, cfg["selection"]["std_ddof"])
    _write_csv(out / "cv_scores.csv", scores[["model", "seed", "f1", "pr_auc", "n_flag_nominal", "n_flag_actual"]])
    _write_csv(out / "selection.csv", table)
    _write_csv(out / "oof_by_model.csv", oof[["model", "product_id", "label", "group", "proba_mean", "flag_rate", "rank_frac_median"]])

    make = make_factory(chosen, params)
    grouped = scores[scores["model"] == chosen].set_index("seed")
    leak = score_repeats(run_cv(X, y, g, make, config.SEEDS, max_folds, ignore_groups=True), y).set_index("seed")
    _write_csv(out / "leakage_contrast.csv", pd.DataFrame({
        "seed": grouped.index, "f1_grouped": grouped["f1"].to_numpy(), "f1_ignoring_groups": leak["f1"].to_numpy(),
        "pr_auc_grouped": grouped["pr_auc"].to_numpy(), "pr_auc_ignoring_groups": leak["pr_auc"].to_numpy(),
    }))
    sens = score_repeats(run_cv(X.drop(columns=ro["sensitivity_drop"]), y, g, make, config.SEEDS, max_folds), y)
    _write_csv(out / "sensitivity_20.csv", pd.DataFrame({
        "seed": grouped.index, "f1_24": grouped["f1"].to_numpy(), "f1_20": sens["f1"].to_numpy(),
    }))
    blocks = None
    if diag["twin_adjacency"]["train"]["row_order_trusted"]:
        blocks = block_split(X, y, g, make, config.SEEDS[0], ro["block_split"]["n_blocks"])
        _write_csv(out / "block_split.csv", blocks)

    chosen_oof = oof[oof["model"] == chosen].reset_index(drop=True)
    _write_csv(out / "oof_predictions.csv", chosen_oof[["product_id", "label", "group", "proba_mean", "flag_rate", "rank_frac_median"]])
    row = table.set_index("model").loc[chosen]
    summary = {
        "selected_model": chosen,
        "output_type": "probability(prior_correction)" if _is_probability(chosen) else "risk_score",
        "f1_mean": float(row["f1_mean"]), "f1_std": float(row["f1_std"]),
        "beats_floor": bool(row["beats_floor"]),
        "any_candidate_beats_floor": bool(table["beats_floor"].any()),
        "f1_ignoring_groups_mean": float(leak["f1"].mean()),
        "f1_20_inputs_mean": float(sens["f1"].mean()),
        "block_split": None if blocks is None else blocks.to_dict("records"),
        "calibration": calibration_summary(chosen_oof["proba_mean"].to_numpy(), y) if _is_probability(chosen) else None,
    }
    _write_json(out / "model_summary.json", summary)
    return chosen, summary, scores, oof


def analyze_line(train, cfg: dict, chosen: str, scores: pd.DataFrame, oof: pd.DataFrame, near_pairs: pd.DataFrame,
                 variables: pd.DataFrame, interpretable: bool, out: Path) -> dict:
    """선택 모델의 영향요인·상호작용·FN 유형과 집중 조건·FP 조건·도메인 대조·2D 부분의존도(D-011)."""
    X, y, g = _xyg(train)
    an, params = cfg["analyze"], cfg["models"]["params"]
    make = make_factory(chosen, params)
    imp = permutation_importance_cv(X, y, g, make, config.SEEDS, cfg["cv"]["max_folds"], an["perm_repeats"], an["top_n"])
    _write_csv(out / "importance.csv", imp)
    var_a, var_b = pick_top_pair(imp, near_pairs)
    top = imp["variable"].head(an["top_n"]).tolist()
    k, aux = float(y.mean()), aux_k(y, g)
    rank = oof.loc[oof["model"] == chosen, "rank_frac_median"].to_numpy()          # train 행 순서와 같다

    fn = fn_types(oof, chosen, cfg["models"]["complexity_order"], k, aux)
    fp_summary, fp_table = fp_analysis(train.data, rank, an["fp_budget"], [var_a, var_b])
    _write_csv(out / "interaction_2x2.csv", interaction_2x2(train.data, var_a, var_b))
    _write_csv(out / "fn_types.csv", fn)
    _write_csv(out / "fn_concentration.csv", fn_concentration(train.data, rank, k, [var_a, var_b]))
    _write_csv(out / "fp_conditions.csv", fp_table)
    _write_csv(out / "domain_crosscheck.csv", domain_crosscheck(train.data, variables, top))
    _write_csv(out / "pdp_2d.csv", pdp_2d(make, X, y, config.SEEDS[0], var_a, var_b, an["pdp_grid"]))

    leak_rows = []
    for name in an["leakage_explore_models"]:
        ig = score_repeats(run_cv(X, y, g, make_factory(name, params), config.SEEDS, cfg["cv"]["max_folds"], ignore_groups=True), y)
        leak_rows.append({"model": name, "f1_grouped_mean": float(scores.loc[scores["model"] == name, "f1"].mean()),
                          "f1_ignoring_groups_mean": float(ig["f1"].mean()), "note": "사후 탐색(D-011)"})
    _write_csv(out / "leakage_explore.csv", pd.DataFrame(leak_rows))

    summary = {
        "interpretable": interpretable, "top_variables": top, "pair": [var_a, var_b], "k": k, "aux_k": aux,
        "fn_type_counts": fn["fn_type"].value_counts().reindex(list(FN_TYPES), fill_value=0).astype(int).to_dict(),
        "fp_at_budget": fp_summary,
    }
    _write_json(out / "analysis_summary.json", summary)
    return summary


def prioritize_line(train, cfg: dict, chosen: str, oof: pd.DataFrame, test_proba: np.ndarray, out: Path) -> dict:
    """검사량 표, 무작위 추가 검사 결합표, 배치 알람(D-012). 비용 가정 없이 개수만 낸다."""
    X, y, g = _xyg(train)
    pr = cfg["prioritize"]
    chosen_oof = oof[oof["model"] == chosen]
    budget = budget_table(y, chosen_oof["rank_frac_median"].to_numpy(), pr["budgets"], float(y.mean()), aux_k(y, g))
    _write_csv(out / "budget_table.csv", budget)
    _write_csv(out / "sampling_combo.csv", sampling_combo(budget, len(y), pr["sampling_rates"]))
    rec = run_cv(X, y, g, make_factory(chosen, cfg["models"]["params"]), config.SEEDS, cfg["cv"]["max_folds"])
    reference = [float(np.mean(r["proba"])) for r in rec]
    threshold, batch_size = max(reference), int(np.median([len(r["idx"]) for r in rec]))
    labeled = alarm_scan(chosen_oof["proba_mean"].to_numpy(), batch_size, threshold)
    tested = alarm_scan(test_proba, batch_size, threshold)
    _write_csv(out / "alarm_labeled.csv", labeled)
    _write_csv(out / "alarm_test.csv", tested)
    summary = {
        "threshold": threshold, "reference_batches": len(reference), "batch_size": batch_size,
        "labeled_alarm_batches": labeled.loc[labeled["alarm"], "batch"].astype(int).tolist(),
        "test_batches": int(len(tested)), "test_alarms": int(tested["alarm"].sum()),
    }
    _write_json(out / "prioritize_summary.json", summary)
    return summary


def make_report(cfg: dict) -> None:
    """results/의 표를 읽어 보고서용 그림을 results/figures/에 만든다(PNG 메타데이터의 버전 문자열은 지워 재현성 유지)."""
    fig_dir = config.RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    lines = list(cfg["data"]["train"])
    res = {line: config.RESULTS / line.lower() for line in lines}
    load = lambda line, name: json.loads((res[line] / name).read_text(encoding="utf-8"))
    figs = {"fig1_pair_share": report.fig_pair_share({line: load(line, "diagnose_summary.json")["groups"] for line in lines}),
            "fig3_fn_types": report.fig_fn_types({line: load(line, "analysis_summary.json")["fn_type_counts"] for line in lines}),
            "fig4_budget": report.fig_budget({line: pd.read_csv(res[line] / "budget_table.csv") for line in lines})}
    for line in lines:
        oof = pd.read_csv(res[line] / "oof_predictions.csv")
        chosen = load(line, "model_summary.json")["selected_model"]
        tag = line.lower()
        figs[f"fig1_defect_positions_{tag}"] = report.fig_defect_positions(oof["label"].to_numpy(), line)
        figs[f"fig2_model_f1_{tag}"] = report.fig_model_f1(pd.read_csv(res[line] / "cv_scores.csv"), cfg["models"]["complexity_order"],
                                                         cfg["models"]["floor"], load(line, "diagnose_summary.json")["f1_ceiling"], chosen, line)
        figs[f"fig3_importance_{tag}"] = report.fig_importance(pd.read_csv(res[line] / "importance.csv"), line)
        figs[f"fig3_pdp_{tag}"] = report.fig_pdp(pd.read_csv(res[line] / "pdp_2d.csv"), line)
        figs[f"fig4_alarm_{tag}"] = report.fig_alarm(pd.read_csv(res[line] / "alarm_labeled.csv"), load(line, "prioritize_summary.json")["threshold"], line)
    for name, fig in figs.items():
        fig.savefig(fig_dir / f"{name}.png", metadata={"Software": None})
        report.plt.close(fig)


def write_run_record() -> None:
    """실행 기록: 환경, seed, 시각, 커밋 해시. 시각이 들어가므로 재현성 비교에서는 이 파일만 뺀다."""
    import matplotlib, numpy, sklearn  # noqa: E401
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=config.PROJECT_ROOT).stdout.strip() or None
    except OSError:
        commit = None
    _write_json(config.RESULTS / "run_record.json", {
        "command": "python run.py", "finished_at": datetime.now().isoformat(timespec="seconds"), "git_commit": commit,
        "python": sys.version.split()[0], "platform": platform.platform(), "seeds": list(config.SEEDS),
        "packages": {"numpy": numpy.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
    })


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

        chosen, summary, scores, oof = evaluate_line(train, cfg, diag, out)
        near = diag_tables["pairs"].query("split == 'train'")
        analysis = analyze_line(train, cfg, chosen, scores, oof, near, variables, summary["any_candidate_beats_floor"], out)

        warnings = diag["distribution_warning"]
        limitations = cfg["limitations"] + (
            [f"train·test 분포 차이 경고 {len(warnings)}개 변수(diagnose_shift.csv). 예측 확률의 절대값은 해석하지 않는다."]
            if warnings else []
        )
        if not summary["any_candidate_beats_floor"]:
            limitations.append("어떤 후보 모델도 무작위 기준모델과 차이가 없다(selection.csv의 beats_floor). 기계 신호만으로는 학습되지 않는 제품군이다.")
        make = make_factory(chosen, cfg["models"]["params"])
        pred, meta = infer(train, test, make, config.SEEDS[0], limitations)
        pred[FLAG_RATE_COL] = flag_rate_over_seeds(train, test, make, config.SEEDS)
        priority = prioritize_line(train, cfg, chosen, oof, pred["proba"].to_numpy(), out)
        _write_csv(out / "predictions.csv", pred)
        _write_json(out / "predictions_meta.json", meta)

        print(
            f"[{line}] 선택 {chosen} CV F1 {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f} "
            f"(무작위보다 나음 {summary['beats_floor']}) | 상위 변수 {analysis['top_variables'][:3]} | "
            f"FN 유형 {analysis['fn_type_counts']} | test {meta['n_test']}행, 표시 {meta['n_flag_actual']}개 "
            f"→ {out.relative_to(config.PROJECT_ROOT).as_posix()}",
            flush=True,
        )


    make_report(cfg)
    write_run_record()


if __name__ == "__main__":
    main()

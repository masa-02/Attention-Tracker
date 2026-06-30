import argparse
from pathlib import Path

import pandas as pd

import analyze_qwen_phase2 as base


EXPECTED_RUNS = 5
FAMILY_NAME = "Gemma"
RUN_PREFIX = "gemma-phase2"
EXCLUDED_NOTE = "Gemma 4 / Gemma 4 MoE 系は現在のmanifest対象外で、この分析対象成果物にも含まれていません。"


def validate_gemma_runs(runs: list[dict]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if len(runs) != EXPECTED_RUNS:
        errors.append(f"expected {EXPECTED_RUNS} Gemma Phase2 eval runs, found {len(runs)}")

    for run in runs:
        run_id = run["run_id"]
        prompts_len = len(run["prompts"])
        scores_len = len(run["scores"])
        spans_len = len(run["spans"])
        if prompts_len != base.EXPECTED_PROMPTS:
            errors.append(f"{run_id}: prompts.parquet has {prompts_len} rows")
        if scores_len != base.EXPECTED_PROMPTS:
            errors.append(f"{run_id}: attention_tracker_scores.parquet has {scores_len} rows")
        if spans_len != base.EXPECTED_TOKEN_SPANS:
            errors.append(f"{run_id}: token_spans.parquet has {spans_len} rows")
        if run["attention_keys"] != base.EXPECTED_ATTENTION_KEYS:
            errors.append(f"{run_id}: attention_summary keys differ: {sorted(run['attention_keys'])}")

        injection_non_null = int(run["scores"]["mean_injection_mass"].notna().sum())
        if injection_non_null:
            warnings.append(f"{run_id}: mean_injection_mass has {injection_non_null} values")
        else:
            warnings.append(f"{run_id}: mean_injection_mass is empty; injection span was not collected")

    return errors, warnings


def gemma_sort_key(model: str) -> tuple[int, float, str]:
    name = model.lower()
    generation = 2 if "gemma-2" in name else 3 if "gemma-3" in name else 9
    return (generation, base.derive_params_b(model), model)


def gemma_group(model: str) -> str:
    name = model.lower()
    if "gemma-2" in name:
        return "Gemma 2"
    if "gemma-3" in name:
        return "Gemma 3"
    return "Gemma"


def add_gemma_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "model" in df.columns:
        df["model_group"] = df["model"].map(gemma_group)
        df["params_b"] = df["model"].map(base.derive_params_b)
        df = df.sort_values(["model_group", "params_b", "model"])
    return df


def group_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("model_group", as_index=False)
        .agg(
            models=("model", "count"),
            mean_auc=("auc", "mean"),
            mean_auprc=("auprc", "mean"),
            mean_fpr=("fpr", "mean"),
            mean_fnr=("fnr", "mean"),
            mean_selected_fraction=("selected_fraction", "mean"),
            mean_selected_depth=("selected_depth_mean", "mean"),
        )
        .sort_values("model_group")
    )


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    head_summary: pd.DataFrame,
    hard_samples: pd.DataFrame,
    validation_warnings: list[str],
    plot_paths: list[Path],
) -> None:
    best = base.describe_best_models(metrics)
    groups = group_summary(metrics)
    lines: list[str] = []
    lines.append("# Gemma Phase2 Attention Tracker Analysis")
    lines.append("")
    lines.append("## 概要")
    lines.append("")
    lines.append(
        f"- 対象は `{RUN_PREFIX}-*` の {len(metrics)} モデルです。再推論は行わず、既存のPhase2成果物だけを読みました。"
    )
    lines.append("- Attention Tracker標準の `instruction_mass`, `data_mass`, `ratio`, `entropy`, `focus_score` を中心に分析しています。")
    lines.append("- `mean_injection_mass` は全runで欠損しており、今回の分析対象外です。")
    lines.append(f"- {EXCLUDED_NOTE}")
    lines.append("")
    lines.append("## モデル別性能")
    lines.append("")
    lines.append("| model | group | params_b | auc | auprc | fpr | fnr | selected_heads | selected_depth |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in metrics.iterrows():
        lines.append(
            "| {model} | {group} | {params} | {auc} | {auprc} | {fpr} | {fnr} | {heads} | {depth} |".format(
                model=base.model_display_name(row["model"]),
                group=row["model_group"],
                params=base.fmt_float(row["params_b"], 1),
                auc=base.fmt_float(row["auc"], 3),
                auprc=base.fmt_float(row["auprc"], 3),
                fpr=base.fmt_float(row["fpr"], 3),
                fnr=base.fmt_float(row["fnr"], 3),
                heads=int(row["selected_heads"]),
                depth=base.fmt_float(row["selected_depth_mean"], 3),
            )
        )
    lines.append("")
    lines.append("## 主要な観察")
    lines.append("")
    lines.append(
        "- AUC最高は `{}` で AUC `{}`、FPR `{}`、FNR `{}` です。".format(
            base.model_display_name(best["best_auc"]["model"]),
            base.fmt_float(best["best_auc"]["auc"]),
            base.fmt_float(best["best_auc"]["fpr"]),
            base.fmt_float(best["best_auc"]["fnr"]),
        )
    )
    lines.append(
        "- 固定thresholdでFPRが最も低いのは `{}` で FPR `{}` です。".format(
            base.model_display_name(best["lowest_fpr"]["model"]),
            base.fmt_float(best["lowest_fpr"]["fpr"]),
        )
    )
    lines.append(
        "- FPRが最も高いのは `{}` で FPR `{}` です。AUCが高くても固定thresholdの校正は別問題です。".format(
            base.model_display_name(best["highest_fpr"]["model"]),
            base.fmt_float(best["highest_fpr"]["fpr"]),
        )
    )
    lines.append(
        "- AUCが最も低いのは `{}` で AUC `{}` です。".format(
            base.model_display_name(best["lowest_auc"]["model"]),
            base.fmt_float(best["lowest_auc"]["auc"]),
        )
    )
    lines.append("")
    lines.append("## 世代比較")
    lines.append("")
    lines.append("| group | models | mean_auc | mean_fpr | mean_fnr | mean_selected_fraction | mean_selected_depth |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in groups.iterrows():
        lines.append(
            "| {group} | {models} | {auc} | {fpr} | {fnr} | {fraction} | {depth} |".format(
                group=row["model_group"],
                models=int(row["models"]),
                auc=base.fmt_float(row["mean_auc"]),
                fpr=base.fmt_float(row["mean_fpr"]),
                fnr=base.fmt_float(row["mean_fnr"]),
                fraction=base.fmt_float(row["mean_selected_fraction"]),
                depth=base.fmt_float(row["mean_selected_depth"]),
            )
        )
    lines.append("")
    lines.append("## Attention / Head傾向")
    lines.append("")
    for _, row in head_summary.iterrows():
        lines.append(
            "- `{}`: selected heads `{}` / `{}` ({})、平均深度 `{}`、early/mid/late = `{}` / `{}` / `{}`。".format(
                base.model_display_name(row["model"]),
                int(row["selected_heads"]),
                int(row["total_heads"]),
                base.fmt_float(row["selected_fraction"]),
                base.fmt_float(row["selected_depth_mean"]),
                base.fmt_float(row["early_fraction"]),
                base.fmt_float(row["mid_fraction"]),
                base.fmt_float(row["late_fraction"]),
            )
        )
    lines.append("")
    lines.append("## Hard Samples")
    lines.append("")
    lines.append("| sample_index | label | FP models | FN models | avg_risk | preview |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for _, row in hard_samples.head(12).iterrows():
        lines.append(
            "| {sample} | {label} | {fp} | {fn} | {risk} | {preview} |".format(
                sample=int(row["sample_index"]),
                label=int(row["label"]),
                fp=int(row["false_positive_models"]),
                fn=int(row["false_negative_models"]),
                risk=base.fmt_float(row["avg_risk"]),
                preview=str(row["text_preview"]),
            )
        )
    lines.append("")
    lines.append("## 生成物")
    lines.append("")
    for path in sorted(plot_paths):
        lines.append(f"- `{path.relative_to(output_dir)}`")
    lines.append("")
    lines.append("## 検証メモ")
    lines.append("")
    lines.append(f"- {EXPECTED_RUNS}モデルすべてを検出しました。")
    lines.append("- 各runで `prompts.parquet` 116行、`attention_tracker_scores.parquet` 116行、`token_spans.parquet` 464行を確認しました。")
    lines.append("- `attention_summary.safetensors` の必須キーを確認しました。")
    for warning in validation_warnings:
        lines.append(f"- warning: {warning}")
    lines.append("")

    (output_dir / "gemma_phase2_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Gemma Phase2 Attention Tracker artifacts.")
    parser.add_argument("--phase2-dir", default="outputs/phase2")
    parser.add_argument("--result-runs-dir", default="result/deepset/prompt-injections/runs")
    parser.add_argument("--output-dir", default="result/analysis/gemma_phase2")
    args = parser.parse_args()

    phase2_dir = Path(args.phase2_dir)
    result_runs_dir = Path(args.result_runs_dir)
    output_dir = Path(args.output_dir)
    base.ensure_dir(output_dir)

    eval_dirs = sorted(path for path in phase2_dir.glob(f"{RUN_PREFIX}-*-seed0-phase2") if path.is_dir())
    runs = [base.load_run(path, result_runs_dir) for path in eval_dirs]
    runs.sort(key=lambda run: gemma_sort_key(str(run["metadata"].iloc[0]["model"])))

    validation_errors, validation_warnings = validate_gemma_runs(runs)
    if validation_errors:
        for error in validation_errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    samples = pd.concat([run["sample_table"] for run in runs], ignore_index=True)
    samples = add_gemma_groups(samples)
    head_summary, depth_profile = base.summarize_heads(runs)
    head_summary = add_gemma_groups(head_summary)
    depth_profile = add_gemma_groups(depth_profile)
    model_metrics = base.summarize_models(runs, head_summary)
    model_metrics = add_gemma_groups(model_metrics)
    score_distribution = add_gemma_groups(base.summarize_scores(samples))
    attention_summary = add_gemma_groups(base.summarize_attention(samples))
    risk_corr = base.build_risk_correlation(samples)
    hard_samples = base.build_hard_samples(samples)

    base.write_tables(
        output_dir,
        model_metrics,
        score_distribution,
        attention_summary,
        head_summary,
        depth_profile,
        risk_corr,
        hard_samples,
    )
    plot_paths = base.write_plots(
        output_dir,
        runs,
        samples,
        model_metrics,
        attention_summary,
        head_summary,
        depth_profile,
        risk_corr,
    )
    write_report(output_dir, model_metrics, head_summary, hard_samples, validation_warnings, plot_paths)

    csv_names = [
        "model_metrics.csv",
        "score_distribution.csv",
        "attention_mass_summary.csv",
        "head_selection_summary.csv",
        "depth_profile.csv",
        "risk_correlation.csv",
        "hard_samples.csv",
    ]
    output_errors = base.validate_output_files(output_dir, csv_names, plot_paths)
    if output_errors:
        for error in output_errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print(f"Wrote Gemma Phase2 analysis to {output_dir}")
    print(f"Models: {len(model_metrics)}")
    print(f"CSV files: {len(csv_names)}")
    print(f"PNG files: {len(plot_paths)}")
    for warning in validation_warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()

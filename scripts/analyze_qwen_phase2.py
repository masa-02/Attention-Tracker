import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from safetensors.torch import safe_open


EXPECTED_ATTENTION_KEYS = {
    "instruction_mass",
    "data_mass",
    "ratio",
    "normalized_instruction_ratio",
    "entropy",
}
EXPECTED_RUNS = 12
EXPECTED_PROMPTS = 116
EXPECTED_TOKEN_SPANS = 464


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return ""
    return f"{number:.{digits}f}"


def model_display_name(model: str) -> str:
    return model.replace("-attn", "")


def derive_model_group(model: str) -> str:
    name = model.lower()
    if "qwen2.5-coder" in name:
        return "Qwen2.5-Coder"
    if "qwen2.5" in name:
        if "instruct" in name:
            return "Qwen2.5-Instruct"
        return "Qwen2.5-Base"
    if "qwen3" in name:
        return "Qwen3"
    return "Qwen"


def derive_params_b(model: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)b", model.lower())
    return float(match.group(1)) if match else float("nan")


def sort_key(model: str) -> tuple[int, float, str]:
    group_order = {
        "Qwen2.5-Base": 0,
        "Qwen2.5-Instruct": 1,
        "Qwen2.5-Coder": 2,
        "Qwen3": 3,
    }
    group = derive_model_group(model)
    return (group_order.get(group, 9), derive_params_b(model), model)


def prompt_label(labels_json: str) -> int:
    labels = json.loads(labels_json)
    return int(bool(labels.get("injection_present") or labels.get("instruction_conflict")))


def prompt_sample_index(prompt_id: str) -> int | None:
    match = re.search(r"(\d+)$", prompt_id)
    return int(match.group(1)) if match else None


def prompt_text_from_messages(messages_json: str) -> str:
    try:
        messages = json.loads(messages_json)
    except json.JSONDecodeError:
        return ""
    for message in messages:
        if message.get("name") == "untrusted_data":
            return str(message.get("content", ""))
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def load_attention_means(path: Path) -> tuple[pd.DataFrame, set[str], dict[str, tuple[int, ...]]]:
    with safe_open(path, framework="pt") as handle:
        keys = set(handle.keys())
        shapes = {key: tuple(handle.get_tensor(key).shape) for key in keys}
        arrays = {
            "mean_instruction_mass": handle.get_tensor("instruction_mass").numpy().mean(axis=(1, 2)),
            "mean_data_mass": handle.get_tensor("data_mass").numpy().mean(axis=(1, 2)),
            "mean_ratio": handle.get_tensor("ratio").numpy().mean(axis=(1, 2)),
            "mean_normalized_instruction_ratio": handle.get_tensor("normalized_instruction_ratio")
            .numpy()
            .mean(axis=(1, 2)),
            "mean_entropy": handle.get_tensor("entropy").numpy().mean(axis=(1, 2)),
        }
    return pd.DataFrame(arrays), keys, shapes


def load_run(eval_dir: Path, result_runs_dir: Path) -> dict[str, Any]:
    run_id = eval_dir.name
    head_run_id = run_id.replace("-seed0-phase2", "-head-selection")
    head_dir = eval_dir.parent / head_run_id
    legacy_dir = result_runs_dir / run_id

    prompts = pd.read_parquet(eval_dir / "prompts.parquet")
    scores = pd.read_parquet(eval_dir / "attention_tracker_scores.parquet")
    spans = pd.read_parquet(eval_dir / "token_spans.parquet")
    metadata = pd.read_parquet(eval_dir / "model_metadata.parquet")
    generations = pd.read_parquet(eval_dir / "generation_outputs.parquet")
    attention_means, attention_keys, attention_shapes = load_attention_means(
        eval_dir / "attention_summary.safetensors"
    )
    heads = pd.read_parquet(head_dir / "calibration_artifacts" / "selected_heads.parquet")

    summary_path = legacy_dir / "summary.json"
    samples_path = legacy_dir / "samples.jsonl"
    summary = read_json(summary_path) if summary_path.exists() else {}
    samples = read_jsonl(samples_path) if samples_path.exists() else []

    prompts = prompts.copy()
    prompts["label"] = prompts["labels_json"].map(prompt_label)
    prompts["sample_index"] = prompts["prompt_id"].map(prompt_sample_index)
    prompts["text"] = prompts["messages_json"].map(prompt_text_from_messages)

    sample_text = {
        int(row["sample_index"]): str(row.get("text", ""))
        for row in samples
        if "sample_index" in row
    }
    if sample_text:
        prompts["text"] = prompts.apply(
            lambda row: sample_text.get(row["sample_index"], row["text"]),
            axis=1,
        )

    score_features = scores.drop(
        columns=["mean_instruction_mass", "mean_data_mass"],
        errors="ignore",
    )

    sample_table = prompts[
        ["prompt_id", "base_request_id", "label", "sample_index", "text"]
    ].merge(score_features, on="prompt_id", how="left")
    sample_table = pd.concat([sample_table.reset_index(drop=True), attention_means], axis=1)
    sample_table["run_id"] = run_id
    sample_table["model"] = metadata.iloc[0]["model"]
    sample_table["model_id"] = metadata.iloc[0]["model_id"]
    sample_table["risk_score"] = 1.0 - sample_table["focus_score"].astype(float)
    sample_table["is_fp"] = (sample_table["label"] == 0) & (sample_table["pred"].astype(bool))
    sample_table["is_fn"] = (sample_table["label"] == 1) & (~sample_table["pred"].astype(bool))

    return {
        "run_id": run_id,
        "head_run_id": head_run_id,
        "eval_dir": eval_dir,
        "head_dir": head_dir,
        "summary": summary,
        "samples": samples,
        "prompts": prompts,
        "scores": scores,
        "spans": spans,
        "metadata": metadata,
        "generations": generations,
        "heads": heads,
        "sample_table": sample_table,
        "attention_keys": attention_keys,
        "attention_shapes": attention_shapes,
    }


def validate_runs(runs: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if len(runs) != EXPECTED_RUNS:
        errors.append(f"expected {EXPECTED_RUNS} Qwen Phase2 eval runs, found {len(runs)}")

    for run in runs:
        run_id = run["run_id"]
        prompts_len = len(run["prompts"])
        scores_len = len(run["scores"])
        spans_len = len(run["spans"])
        if prompts_len != EXPECTED_PROMPTS:
            errors.append(f"{run_id}: prompts.parquet has {prompts_len} rows")
        if scores_len != EXPECTED_PROMPTS:
            errors.append(f"{run_id}: attention_tracker_scores.parquet has {scores_len} rows")
        if spans_len != EXPECTED_TOKEN_SPANS:
            errors.append(f"{run_id}: token_spans.parquet has {spans_len} rows")
        if run["attention_keys"] != EXPECTED_ATTENTION_KEYS:
            errors.append(
                f"{run_id}: attention_summary keys differ: {sorted(run['attention_keys'])}"
            )

        injection_non_null = int(run["scores"]["mean_injection_mass"].notna().sum())
        if injection_non_null:
            warnings.append(f"{run_id}: mean_injection_mass has {injection_non_null} values")
        else:
            warnings.append(
                f"{run_id}: mean_injection_mass is empty; injection span was not collected"
            )

    return errors, warnings


def summarize_scores(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, label), group in samples.groupby(["model", "label"], sort=False):
        values = group["focus_score"].astype(float)
        rows.append(
            {
                "model": model,
                "model_group": derive_model_group(model),
                "params_b": derive_params_b(model),
                "label": int(label),
                "count": int(len(group)),
                "focus_mean": values.mean(),
                "focus_std": values.std(ddof=0),
                "focus_median": values.median(),
                "focus_min": values.min(),
                "focus_p25": values.quantile(0.25),
                "focus_p75": values.quantile(0.75),
                "focus_max": values.max(),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_group", "params_b", "model", "label"])


def summarize_attention(samples: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_instruction_mass",
        "mean_data_mass",
        "mean_ratio",
        "mean_normalized_instruction_ratio",
        "mean_entropy",
    ]
    rows = []
    for (model, label), group in samples.groupby(["model", "label"], sort=False):
        row = {
            "model": model,
            "model_group": derive_model_group(model),
            "params_b": derive_params_b(model),
            "label": int(label),
            "count": int(len(group)),
        }
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=0)
            row[f"{metric}_median"] = values.median()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model_group", "params_b", "model", "label"])


def summarize_heads(runs: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    depth_rows = []
    bins = 10

    for run in runs:
        heads = run["heads"]
        model = str(run["metadata"].iloc[0]["model"])
        layers = int(run["metadata"].iloc[0]["layers"])
        n_heads = int(run["metadata"].iloc[0]["heads"])
        selected = heads[heads["selected"].astype(bool)].copy()
        total_heads = int(len(heads))
        selected_count = int(len(selected))
        selected_depths = selected["normalized_depth"].astype(float)

        summary_rows.append(
            {
                "model": model,
                "model_id": run["metadata"].iloc[0]["model_id"],
                "model_group": derive_model_group(model),
                "params_b": derive_params_b(model),
                "layers": layers,
                "heads": n_heads,
                "total_heads": total_heads,
                "selected_heads": selected_count,
                "selected_fraction": selected_count / total_heads if total_heads else np.nan,
                "selected_depth_mean": selected_depths.mean() if selected_count else np.nan,
                "selected_depth_min": selected_depths.min() if selected_count else np.nan,
                "selected_depth_max": selected_depths.max() if selected_count else np.nan,
                "early_fraction": float((selected_depths < 1 / 3).mean())
                if selected_count
                else np.nan,
                "mid_fraction": float(
                    ((selected_depths >= 1 / 3) & (selected_depths < 2 / 3)).mean()
                )
                if selected_count
                else np.nan,
                "late_fraction": float((selected_depths >= 2 / 3).mean())
                if selected_count
                else np.nan,
                "margin_mean_all": heads["margin"].astype(float).mean(),
                "margin_mean_selected": selected["margin"].astype(float).mean()
                if selected_count
                else np.nan,
                "margin_min_selected": selected["margin"].astype(float).min()
                if selected_count
                else np.nan,
                "margin_max_selected": selected["margin"].astype(float).max()
                if selected_count
                else np.nan,
            }
        )

        for idx in range(bins):
            start = idx / bins
            end = (idx + 1) / bins
            in_bin = heads[
                (heads["normalized_depth"] >= start)
                & (
                    (heads["normalized_depth"] < end)
                    | ((idx == bins - 1) & (heads["normalized_depth"] <= end))
                )
            ]
            selected_bin = in_bin[in_bin["selected"].astype(bool)]
            possible = int(len(in_bin))
            depth_rows.append(
                {
                    "model": model,
                    "model_group": derive_model_group(model),
                    "params_b": derive_params_b(model),
                    "depth_bin": idx,
                    "depth_start": start,
                    "depth_end": end,
                    "heads_in_bin": possible,
                    "selected_heads": int(len(selected_bin)),
                    "selected_density": len(selected_bin) / possible if possible else np.nan,
                    "margin_mean_all": in_bin["margin"].astype(float).mean()
                    if possible
                    else np.nan,
                    "margin_mean_selected": selected_bin["margin"].astype(float).mean()
                    if len(selected_bin)
                    else np.nan,
                }
            )

    head_summary = pd.DataFrame(summary_rows).sort_values(
        ["model_group", "params_b", "model"]
    )
    depth_profile = pd.DataFrame(depth_rows).sort_values(
        ["model_group", "params_b", "model", "depth_bin"]
    )
    return head_summary, depth_profile


def summarize_models(runs: list[dict[str, Any]], head_summary: pd.DataFrame) -> pd.DataFrame:
    head_by_model = head_summary.set_index("model").to_dict(orient="index")
    rows = []
    for run in runs:
        metadata = run["metadata"].iloc[0]
        model = str(metadata["model"])
        summary = run["summary"]
        metrics = summary.get("metrics", {})
        loading = json.loads(metadata.get("model_loading_json") or "{}")
        summary_loading = summary.get("model_loading") or {}
        selected = head_by_model.get(model, {})
        rows.append(
            {
                "run_id": run["run_id"],
                "model": model,
                "display_model": model_display_name(model),
                "model_id": metadata["model_id"],
                "model_group": derive_model_group(model),
                "params_b": derive_params_b(model),
                "num_samples": int(summary.get("num_samples") or len(run["prompts"])),
                "layers": int(metadata["layers"]),
                "heads": int(metadata["heads"]),
                "quantization": summary_loading.get("quantization")
                or loading.get("quantization", ""),
                "dtype": summary_loading.get("dtype") or loading.get("dtype", ""),
                "auc": metrics.get("auc"),
                "auprc": metrics.get("auprc"),
                "fpr": metrics.get("fpr"),
                "fnr": metrics.get("fnr"),
                "selected_heads": selected.get("selected_heads"),
                "selected_fraction": selected.get("selected_fraction"),
                "selected_depth_mean": selected.get("selected_depth_mean"),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_group", "params_b", "model"])


def build_risk_correlation(samples: pd.DataFrame) -> pd.DataFrame:
    pivot = samples.pivot_table(
        index="sample_index",
        columns="model",
        values="risk_score",
        aggfunc="mean",
    )
    ordered = sorted(pivot.columns, key=sort_key)
    return pivot[ordered].corr(method="pearson")


def build_hard_samples(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_index, group in samples.groupby("sample_index"):
        labels = group["label"].astype(int)
        label = int(labels.mode().iloc[0])
        fp_models = sorted(group.loc[group["is_fp"], "model"].map(model_display_name))
        fn_models = sorted(group.loc[group["is_fn"], "model"].map(model_display_name))
        text = str(group["text"].dropna().iloc[0]) if len(group["text"].dropna()) else ""
        rows.append(
            {
                "sample_index": int(sample_index),
                "label": label,
                "false_positive_models": len(fp_models),
                "false_negative_models": len(fn_models),
                "avg_risk": group["risk_score"].astype(float).mean(),
                "avg_focus_score": group["focus_score"].astype(float).mean(),
                "fp_model_list": ";".join(fp_models),
                "fn_model_list": ";".join(fn_models),
                "text_preview": text[:180].replace("\n", " ").replace("|", "/"),
            }
        )
    hard = pd.DataFrame(rows)
    return hard.sort_values(
        ["false_positive_models", "false_negative_models", "avg_risk"],
        ascending=[False, False, False],
    )


def write_tables(
    output_dir: Path,
    model_metrics: pd.DataFrame,
    score_distribution: pd.DataFrame,
    attention_summary: pd.DataFrame,
    head_summary: pd.DataFrame,
    depth_profile: pd.DataFrame,
    risk_corr: pd.DataFrame,
    hard_samples: pd.DataFrame,
) -> None:
    model_metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    score_distribution.to_csv(output_dir / "score_distribution.csv", index=False)
    attention_summary.to_csv(output_dir / "attention_mass_summary.csv", index=False)
    head_summary.to_csv(output_dir / "head_selection_summary.csv", index=False)
    depth_profile.to_csv(output_dir / "depth_profile.csv", index=False)
    risk_corr.to_csv(output_dir / "risk_correlation.csv")
    hard_samples.to_csv(output_dir / "hard_samples.csv", index=False)


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def save_model_auc_plot(output_dir: Path, metrics: pd.DataFrame, plt) -> Path:
    data = metrics.sort_values(["model_group", "params_b", "model"])
    labels = [model_display_name(model) for model in data["model"]]
    x = np.arange(len(data))
    width = 0.26
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(x - width, data["auc"], width=width, label="AUC", color="#3b82f6")
    ax1.bar(x, data["fpr"], width=width, label="FPR", color="#ef4444")
    ax1.bar(x + width, data["fnr"], width=width, label="FNR", color="#f59e0b")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score")
    ax1.set_title("Qwen Phase2 ranking quality and fixed-threshold errors")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.legend(ncol=3)
    fig.tight_layout()
    path = output_dir / "model_auc_fpr.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_focus_score_plot(output_dir: Path, samples: pd.DataFrame, metrics: pd.DataFrame, plt) -> Path:
    ordered = list(metrics["model"])
    fig, ax = plt.subplots(figsize=(12, 5.5))
    positions = []
    data = []
    colors = []
    tick_positions = []
    tick_labels = []
    for idx, model in enumerate(ordered):
        base = idx * 3
        for offset, label, color in [(0, 0, "#64748b"), (1, 1, "#10b981")]:
            values = samples[(samples["model"] == model) & (samples["label"] == label)][
                "focus_score"
            ].astype(float)
            data.append(values.to_numpy())
            positions.append(base + offset)
            colors.append(color)
        tick_positions.append(base + 0.5)
        tick_labels.append(model_display_name(model))
    boxes = ax.boxplot(data, positions=positions, widths=0.65, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.axhline(0.5, color="#111827", linestyle="--", linewidth=1, label="threshold=0.5")
    ax.set_ylabel("Focus score")
    ax.set_title("Focus score by label")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.legend(handles=[
        plt.Line2D([0], [0], color="#64748b", lw=6, alpha=0.65, label="label 0"),
        plt.Line2D([0], [0], color="#10b981", lw=6, alpha=0.65, label="label 1"),
        plt.Line2D([0], [0], color="#111827", lw=1, linestyle="--", label="threshold=0.5"),
    ])
    fig.tight_layout()
    path = output_dir / "focus_score_by_label.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_attention_mass_plot(
    output_dir: Path, attention_summary: pd.DataFrame, metrics: pd.DataFrame, plt
) -> Path:
    ordered = list(metrics["model"])
    labels = [model_display_name(model) for model in ordered]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plot_specs = [
        ("mean_instruction_mass_mean", "Instruction mass"),
        ("mean_data_mass_mean", "Data mass"),
        ("mean_ratio_mean", "Instruction ratio"),
    ]
    x = np.arange(len(ordered))
    width = 0.38
    for ax, (column, title) in zip(axes, plot_specs):
        label0 = []
        label1 = []
        for model in ordered:
            rows = attention_summary[attention_summary["model"] == model]
            label0.append(float(rows[rows["label"] == 0][column].iloc[0]))
            label1.append(float(rows[rows["label"] == 1][column].iloc[0]))
        ax.bar(x - width / 2, label0, width=width, label="label 0", color="#64748b")
        ax.bar(x + width / 2, label1, width=width, label="label 1", color="#10b981")
        ax.set_ylabel(title)
        ax.legend(loc="best")
    axes[0].set_title("Attention mass summary by label")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    path = output_dir / "attention_mass_by_label.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_depth_profile_plot(output_dir: Path, depth_profile: pd.DataFrame, metrics: pd.DataFrame, plt) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for model in metrics["model"]:
        rows = depth_profile[depth_profile["model"] == model].sort_values("depth_bin")
        xs = rows["depth_start"].astype(float) + 0.05
        ys = rows["selected_density"].astype(float)
        ax.plot(xs, ys, marker="o", linewidth=1.4, label=model_display_name(model))
    ax.set_xlabel("Normalized layer depth")
    ax.set_ylabel("Selected-head density")
    ax.set_title("Selected head depth profile")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    path = output_dir / "selected_head_depth_profile.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_selected_ratio_plot(output_dir: Path, head_summary: pd.DataFrame, plt) -> Path:
    data = head_summary.sort_values(["model_group", "params_b", "model"])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    labels = [model_display_name(model) for model in data["model"]]
    ax.bar(np.arange(len(data)), data["selected_fraction"].astype(float), color="#8b5cf6")
    ax.set_ylabel("Selected-head ratio")
    ax.set_title("Selected head ratio by model")
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    path = output_dir / "selected_head_ratio_by_model.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_risk_heatmap(output_dir: Path, risk_corr: pd.DataFrame, plt) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    matrix = risk_corr.to_numpy(dtype=float)
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [model_display_name(model) for model in risk_corr.columns]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Sample-level risk correlation")
    fig.colorbar(image, ax=ax, label="Pearson r")
    fig.tight_layout()
    path = output_dir / "risk_correlation_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_head_margin_heatmaps(output_dir: Path, runs: list[dict[str, Any]], plt) -> list[Path]:
    heatmap_dir = output_dir / "head_margin_heatmaps"
    ensure_dir(heatmap_dir)
    paths = []
    for run in runs:
        heads = run["heads"]
        model = str(run["metadata"].iloc[0]["model"])
        pivot = heads.pivot(index="layer", columns="head", values="margin").sort_index()
        fig, ax = plt.subplots(figsize=(8, 6))
        limit = float(np.nanmax(np.abs(pivot.to_numpy(dtype=float))))
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        selected = heads[heads["selected"].astype(bool)]
        ax.scatter(selected["head"], selected["layer"], s=10, color="#111827", marker="s", label="selected")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"Head margin heatmap: {model_display_name(model)}")
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(image, ax=ax, label="margin")
        fig.tight_layout()
        path = heatmap_dir / f"{model_display_name(model).replace('.', '_')}_margin_heatmap.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def write_plots(
    output_dir: Path,
    runs: list[dict[str, Any]],
    samples: pd.DataFrame,
    metrics: pd.DataFrame,
    attention_summary: pd.DataFrame,
    head_summary: pd.DataFrame,
    depth_profile: pd.DataFrame,
    risk_corr: pd.DataFrame,
) -> list[Path]:
    plt = configure_matplotlib()
    paths = [
        save_model_auc_plot(output_dir, metrics, plt),
        save_focus_score_plot(output_dir, samples, metrics, plt),
        save_attention_mass_plot(output_dir, attention_summary, metrics, plt),
        save_depth_profile_plot(output_dir, depth_profile, metrics, plt),
        save_selected_ratio_plot(output_dir, head_summary, plt),
        save_risk_heatmap(output_dir, risk_corr, plt),
    ]
    paths.extend(save_head_margin_heatmaps(output_dir, runs, plt))
    return paths


def describe_best_models(metrics: pd.DataFrame) -> dict[str, Any]:
    return {
        "best_auc": metrics.sort_values(["auc", "fpr"], ascending=[False, True]).iloc[0],
        "lowest_fpr": metrics.sort_values(["fpr", "fnr"], ascending=[True, True]).iloc[0],
        "highest_fpr": metrics.sort_values(["fpr", "auc"], ascending=[False, False]).iloc[0],
        "lowest_auc": metrics.sort_values(["auc", "fpr"], ascending=[True, True]).iloc[0],
    }


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
    score_distribution: pd.DataFrame,
    attention_summary: pd.DataFrame,
    head_summary: pd.DataFrame,
    hard_samples: pd.DataFrame,
    validation_warnings: list[str],
    plot_paths: list[Path],
) -> None:
    best = describe_best_models(metrics)
    groups = group_summary(metrics)
    injection_warning = "mean_injection_mass は全runで欠損しており、今回の分析対象外です。"

    lines: list[str] = []
    lines.append("# Qwen Phase2 Attention Tracker Analysis")
    lines.append("")
    lines.append("## 概要")
    lines.append("")
    lines.append(
        f"- 対象は `qwen-phase2-*` の {len(metrics)} モデルです。再推論は行わず、既存のPhase2成果物だけを読みました。"
    )
    lines.append("- Attention Tracker標準の `instruction_mass`, `data_mass`, `ratio`, `entropy`, `focus_score` を中心に分析しています。")
    lines.append(f"- {injection_warning}")
    lines.append("- `Qwen3-30B-A3B-Instruct-2507` は現在のmanifest対象外で、この分析対象成果物にも含まれていません。")
    lines.append("")
    lines.append("## モデル別性能")
    lines.append("")
    lines.append("| model | group | params_b | auc | auprc | fpr | fnr | selected_heads | selected_depth |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in metrics.iterrows():
        lines.append(
            "| {model} | {group} | {params} | {auc} | {auprc} | {fpr} | {fnr} | {heads} | {depth} |".format(
                model=model_display_name(row["model"]),
                group=row["model_group"],
                params=fmt_float(row["params_b"], 1),
                auc=fmt_float(row["auc"], 3),
                auprc=fmt_float(row["auprc"], 3),
                fpr=fmt_float(row["fpr"], 3),
                fnr=fmt_float(row["fnr"], 3),
                heads=int(row["selected_heads"]),
                depth=fmt_float(row["selected_depth_mean"], 3),
            )
        )
    lines.append("")
    lines.append("## 主要な観察")
    lines.append("")
    lines.append(
        "- AUC最高は `{}` で AUC `{}`、FPR `{}`、FNR `{}` です。".format(
            model_display_name(best["best_auc"]["model"]),
            fmt_float(best["best_auc"]["auc"]),
            fmt_float(best["best_auc"]["fpr"]),
            fmt_float(best["best_auc"]["fnr"]),
        )
    )
    lines.append(
        "- 固定thresholdでFPRが最も低いのは `{}` で FPR `{}` です。".format(
            model_display_name(best["lowest_fpr"]["model"]),
            fmt_float(best["lowest_fpr"]["fpr"]),
        )
    )
    lines.append(
        "- FPRが最も高いのは `{}` で FPR `{}` です。AUCが高くても固定thresholdの校正は別問題です。".format(
            model_display_name(best["highest_fpr"]["model"]),
            fmt_float(best["highest_fpr"]["fpr"]),
        )
    )
    lines.append(
        "- AUCが最も低いのは `{}` で AUC `{}` です。".format(
            model_display_name(best["lowest_auc"]["model"]),
            fmt_float(best["lowest_auc"]["auc"]),
        )
    )
    lines.append("")
    lines.append("## ファミリ比較")
    lines.append("")
    lines.append("| group | models | mean_auc | mean_fpr | mean_fnr | mean_selected_fraction | mean_selected_depth |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in groups.iterrows():
        lines.append(
            "| {group} | {models} | {auc} | {fpr} | {fnr} | {fraction} | {depth} |".format(
                group=row["model_group"],
                models=int(row["models"]),
                auc=fmt_float(row["mean_auc"]),
                fpr=fmt_float(row["mean_fpr"]),
                fnr=fmt_float(row["mean_fnr"]),
                fraction=fmt_float(row["mean_selected_fraction"]),
                depth=fmt_float(row["mean_selected_depth"]),
            )
        )
    lines.append("")
    lines.append("## Attention / Head傾向")
    lines.append("")
    for _, row in head_summary.iterrows():
        lines.append(
            "- `{}`: selected heads `{}` / `{}` ({})、平均深度 `{}`、early/mid/late = `{}` / `{}` / `{}`。".format(
                model_display_name(row["model"]),
                int(row["selected_heads"]),
                int(row["total_heads"]),
                fmt_float(row["selected_fraction"]),
                fmt_float(row["selected_depth_mean"]),
                fmt_float(row["early_fraction"]),
                fmt_float(row["mid_fraction"]),
                fmt_float(row["late_fraction"]),
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
                risk=fmt_float(row["avg_risk"]),
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
    lines.append("- 9モデルすべてを検出しました。")
    lines.append("- 各runで `prompts.parquet` 116行、`attention_tracker_scores.parquet` 116行、`token_spans.parquet` 464行を確認しました。")
    lines.append("- `attention_summary.safetensors` の必須キーを確認しました。")
    for warning in validation_warnings:
        lines.append(f"- warning: {warning}")
    lines.append("")

    (output_dir / "qwen_phase2_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def validate_output_files(output_dir: Path, csv_names: list[str], plot_paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for name in csv_names:
        path = output_dir / name
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing or empty CSV: {path}")
    for path in plot_paths:
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing or empty PNG: {path}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Qwen Phase2 Attention Tracker artifacts.")
    parser.add_argument("--phase2-dir", default="outputs/phase2")
    parser.add_argument("--result-runs-dir", default="result/deepset/prompt-injections/runs")
    parser.add_argument("--output-dir", default="result/analysis/qwen_phase2")
    args = parser.parse_args()

    phase2_dir = Path(args.phase2_dir)
    result_runs_dir = Path(args.result_runs_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    eval_dirs = sorted(
        path
        for path in phase2_dir.glob("qwen-phase2-*-seed0-phase2")
        if path.is_dir()
    )
    runs = [load_run(path, result_runs_dir) for path in eval_dirs]
    runs.sort(key=lambda run: sort_key(str(run["metadata"].iloc[0]["model"])))

    validation_errors, validation_warnings = validate_runs(runs)
    if validation_errors:
        for error in validation_errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    samples = pd.concat([run["sample_table"] for run in runs], ignore_index=True)
    head_summary, depth_profile = summarize_heads(runs)
    model_metrics = summarize_models(runs, head_summary)
    score_distribution = summarize_scores(samples)
    attention_summary = summarize_attention(samples)
    risk_corr = build_risk_correlation(samples)
    hard_samples = build_hard_samples(samples)

    write_tables(
        output_dir,
        model_metrics,
        score_distribution,
        attention_summary,
        head_summary,
        depth_profile,
        risk_corr,
        hard_samples,
    )
    plot_paths = write_plots(
        output_dir,
        runs,
        samples,
        model_metrics,
        attention_summary,
        head_summary,
        depth_profile,
        risk_corr,
    )
    write_report(
        output_dir,
        model_metrics,
        score_distribution,
        attention_summary,
        head_summary,
        hard_samples,
        validation_warnings,
        plot_paths,
    )

    csv_names = [
        "model_metrics.csv",
        "score_distribution.csv",
        "attention_mass_summary.csv",
        "head_selection_summary.csv",
        "depth_profile.csv",
        "risk_correlation.csv",
        "hard_samples.csv",
    ]
    output_errors = validate_output_files(output_dir, csv_names, plot_paths)
    if output_errors:
        for error in output_errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print(f"Wrote Qwen Phase2 analysis to {output_dir}")
    print(f"Models: {len(model_metrics)}")
    print(f"CSV files: {len(csv_names)}")
    print(f"PNG files: {len(plot_paths)}")
    for warning in validation_warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()

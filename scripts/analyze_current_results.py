import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


SAMPLE_RE = re.compile(
    r'"sample_index"\s*:\s*(?P<sample_index>\d+).*?'
    r'"label"\s*:\s*(?P<label>\d+).*?'
    r'"prediction"\s*:\s*(?P<prediction>true|false).*?'
    r'"focus_score"\s*:\s*(?P<focus_score>[-+0-9.eE]+).*?'
    r'"threshold"\s*:\s*(?P<threshold>[-+0-9.eE]+)'
)


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def fmt(value, digits=3):
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return round(float(value), digits)


def extract_json_string(line, field):
    marker = f'"{field}"'
    start = line.find(marker)
    if start < 0:
        return ""
    colon = line.find(":", start + len(marker))
    if colon < 0:
        return ""
    quote = line.find('"', colon + 1)
    if quote < 0:
        return ""
    out = []
    escaped = False
    for char in line[quote + 1 :]:
        if escaped:
            out.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            break
        out.append(char)
    return "".join(out)


def parse_samples(path):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = SAMPLE_RE.search(line)
            if not match:
                continue
            rows.append(
                {
                    "sample_index": int(match.group("sample_index")),
                    "label": int(match.group("label")),
                    "prediction": match.group("prediction") == "true",
                    "focus_score": float(match.group("focus_score")),
                    "threshold": float(match.group("threshold")),
                    "text": extract_json_string(line, "text"),
                }
            )
    return rows


def mean(values):
    values = list(values)
    if not values:
        return None
    return float(np.mean(values))


def median(values):
    values = list(values)
    if not values:
        return None
    return float(np.median(values))


def best_threshold(samples):
    pos = [row["focus_score"] for row in samples if row["label"] == 1]
    neg = [row["focus_score"] for row in samples if row["label"] == 0]
    if not pos or not neg:
        return {"threshold": None, "fnr": None, "fpr": None, "tpr": None}

    best = None
    for threshold in sorted({row["focus_score"] for row in samples}):
        tp = sum(score <= threshold for score in pos)
        fp = sum(score <= threshold for score in neg)
        tpr = tp / len(pos)
        fpr = fp / len(neg)
        fnr = 1.0 - tpr
        youden = tpr - fpr
        if best is None or youden > best["youden"]:
            best = {
                "threshold": threshold,
                "fnr": fnr,
                "fpr": fpr,
                "tpr": tpr,
                "youden": youden,
            }
    return best


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def corr_matrix(series_by_model, spearman=False):
    models = sorted(series_by_model)
    rows = []
    for left in models:
        row = {"model": left}
        left_map = series_by_model[left]
        for right in models:
            right_map = series_by_model[right]
            common = sorted(set(left_map) & set(right_map))
            if len(common) < 2:
                row[right] = ""
                continue
            x = np.asarray([left_map[idx] for idx in common], dtype=float)
            y = np.asarray([right_map[idx] for idx in common], dtype=float)
            if spearman:
                x = rankdata(x)
                y = rankdata(y)
            if np.std(x) == 0 or np.std(y) == 0:
                row[right] = ""
            else:
                row[right] = fmt(float(np.corrcoef(x, y)[0, 1]), 3)
        rows.append(row)
    return rows, ["model"] + models


def load_eval_runs(result_dir):
    eval_root = result_dir / "deepset" / "prompt-injections" / "runs"
    runs = []
    for summary_path in sorted(eval_root.glob("*/summary.json")):
        summary = read_json(summary_path)
        if not summary.get("run_id", "").startswith("corllm-"):
            continue
        samples_path = summary_path.parent / "samples.jsonl"
        samples = parse_samples(samples_path) if samples_path.exists() else []
        runs.append({"summary_path": summary_path, "summary": summary, "samples": samples})
    return runs


def load_head_runs(result_dir):
    head_root = result_dir / "head-selection" / "runs"
    by_model = defaultdict(list)
    for path in sorted(head_root.glob("*/head_selection.json")):
        data = read_json(path)
        by_model[data["model"]].append({"path": path, "data": data})

    preferred = {}
    for model, items in by_model.items():
        items.sort(
            key=lambda item: (
                0 if item["path"].parent.name.startswith("corllm-") else 1,
                -item["path"].stat().st_mtime,
            )
        )
        preferred[model] = items[0]
    return preferred


def summarize_head_map(head_data):
    normal = np.asarray(head_data["normal_mean"], dtype=float)
    attack = np.asarray(head_data["attack_mean"], dtype=float)
    diff = np.asarray(head_data["diff_mean"], dtype=float)
    selected = [tuple(pair) for pair in head_data.get("selected_heads") or []]
    layers, heads = diff.shape
    selected_values = np.asarray([diff[layer, head] for layer, head in selected], dtype=float)
    normal_selected = np.asarray([normal[layer, head] for layer, head in selected], dtype=float)
    attack_selected = np.asarray([attack[layer, head] for layer, head in selected], dtype=float)
    depths = np.asarray([layer / (layers - 1) if layers > 1 else 0.0 for layer, _ in selected])

    layer_counts = defaultdict(int)
    for layer, _ in selected:
        layer_counts[layer] += 1
    peak_layer = max(layer_counts, key=layer_counts.get) if layer_counts else None

    return {
        "layers": layers,
        "heads": heads,
        "total_heads": layers * heads,
        "selected_heads": len(selected),
        "selected_fraction": len(selected) / (layers * heads) if layers and heads else 0.0,
        "global_diff_mean": float(np.mean(diff)),
        "global_diff_pos_fraction": float(np.mean(diff > 0)),
        "selected_diff_mean": float(np.mean(selected_values)) if len(selected_values) else None,
        "selected_normal_mean": float(np.mean(normal_selected)) if len(normal_selected) else None,
        "selected_attack_mean": float(np.mean(attack_selected)) if len(attack_selected) else None,
        "selected_depth_mean": float(np.mean(depths)) if len(depths) else None,
        "selected_depth_min": float(np.min(depths)) if len(depths) else None,
        "selected_depth_max": float(np.max(depths)) if len(depths) else None,
        "early_fraction": float(np.mean(depths < 1 / 3)) if len(depths) else None,
        "mid_fraction": float(np.mean((depths >= 1 / 3) & (depths < 2 / 3))) if len(depths) else None,
        "late_fraction": float(np.mean(depths >= 2 / 3)) if len(depths) else None,
        "peak_layer": peak_layer,
        "peak_layer_depth": peak_layer / (layers - 1) if peak_layer is not None and layers > 1 else None,
        "peak_layer_selected_heads": layer_counts[peak_layer] if peak_layer is not None else 0,
    }


def depth_profile(model, head_data, bins=10):
    diff = np.asarray(head_data["diff_mean"], dtype=float)
    selected = [tuple(pair) for pair in head_data.get("selected_heads") or []]
    layers, heads = diff.shape
    selected_by_bin = defaultdict(list)
    for layer, head in selected:
        depth = layer / (layers - 1) if layers > 1 else 0.0
        idx = min(int(depth * bins), bins - 1)
        selected_by_bin[idx].append(float(diff[layer, head]))

    rows = []
    for idx in range(bins):
        start = idx / bins
        end = (idx + 1) / bins
        layer_indices = [
            layer
            for layer in range(layers)
            if start <= (layer / (layers - 1) if layers > 1 else 0.0) < end
            or (idx == bins - 1 and layer == layers - 1)
        ]
        values = diff[layer_indices, :].reshape(-1) if layer_indices else np.asarray([])
        possible = len(layer_indices) * heads
        selected_values = selected_by_bin[idx]
        rows.append(
            {
                "model": model,
                "depth_bin": idx,
                "depth_start": start,
                "depth_end": end,
                "layers_in_bin": len(layer_indices),
                "mean_diff_all_heads": float(np.mean(values)) if len(values) else None,
                "positive_fraction_all_heads": float(np.mean(values > 0)) if len(values) else None,
                "selected_heads": len(selected_values),
                "selected_density": len(selected_values) / possible if possible else None,
                "mean_diff_selected_heads": float(np.mean(selected_values)) if selected_values else None,
            }
        )
    return rows


def build_report(output_dir, model_rows, head_rows, hard_rows):
    lines = []
    lines.append("# Current Result Analysis")
    lines.append("")
    lines.append("Scope: structured `corllm-*` Attention Tracker runs currently present under `result/`.")
    lines.append("")
    lines.append("Caveats:")
    lines.append("- No `attention_summary.npz` files were present, so test-set full attention maps are not available.")
    lines.append("- Attention-map comparison uses mean maps saved by `select_head.py` on the head-selection `llm` dataset.")
    lines.append("- Best-threshold metrics are diagnostics on the current test outputs, not reportable calibrated test metrics.")
    lines.append("")
    lines.append("## Model Metrics")
    lines.append("")
    lines.append("| model | auc | auprc | fixed_fnr | fixed_fpr | pos_mean_focus | neg_mean_focus | best_threshold | best_fnr | best_fpr |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in model_rows:
        lines.append(
            "| {model} | {auc} | {auprc} | {fixed_fnr} | {fixed_fpr} | {pos_mean_focus} | {neg_mean_focus} | {best_threshold} | {best_fnr} | {best_fpr} |".format(
                **{key: row.get(key, "") for key in row}
            )
        )
    lines.append("")
    lines.append("## Head Map Summary")
    lines.append("")
    lines.append("| model | layers | heads | selected_heads | selected_fraction | selected_depth_mean | early | mid | late | selected_diff_mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in head_rows:
        lines.append(
            "| {model} | {layers} | {heads} | {selected_heads} | {selected_fraction} | {selected_depth_mean} | {early_fraction} | {mid_fraction} | {late_fraction} | {selected_diff_mean} |".format(
                **{key: row.get(key, "") for key in row}
            )
        )
    lines.append("")
    lines.append("## Common Hard Samples")
    lines.append("")
    lines.append("| sample_index | label | false_positive_models | false_negative_models | avg_risk | text_preview |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for row in hard_rows[:12]:
        lines.append(
            "| {sample_index} | {label} | {false_positive_models} | {false_negative_models} | {avg_risk} | {text_preview} |".format(
                **{key: row.get(key, "") for key in row}
            )
        )
    lines.append("")
    report_path = output_dir / "current_result_analysis.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def maybe_plot(output_dir, model_rows, depth_rows, pearson_rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    paths = []

    models = [row["model"] for row in model_rows]
    auc = [float(row["auc"]) for row in model_rows]
    fpr = [float(row["fixed_fpr"]) for row in model_rows]
    plt.figure(figsize=(8, 4.5))
    plt.scatter(fpr, auc)
    for model, x, y in zip(models, fpr, auc):
        plt.annotate(model.replace("-attn", ""), (x, y), fontsize=7)
    plt.xlabel("Fixed-threshold FPR")
    plt.ylabel("AUC")
    plt.title("Ranking quality vs fixed-threshold false positives")
    plt.tight_layout()
    path = output_dir / "auc_vs_fixed_fpr.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    plt.figure(figsize=(9, 4.8))
    for model in models:
        rows = [row for row in depth_rows if row["model"] == model]
        xs = [float(row["depth_start"]) + 0.05 for row in rows]
        ys = [float(row["selected_density"]) for row in rows]
        plt.plot(xs, ys, marker="o", linewidth=1, label=model.replace("-attn", ""))
    plt.xlabel("Normalized layer depth")
    plt.ylabel("Selected-head density")
    plt.title("Important-head depth profile")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    path = output_dir / "selected_head_depth_profile.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    labels = [field for field in pearson_rows[0] if field != "model"] if pearson_rows else []
    if labels:
        matrix = np.asarray([[float(row[label]) for label in labels] for row in pearson_rows])
        plt.figure(figsize=(7.5, 6.5))
        plt.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        plt.colorbar(label="Pearson r")
        plt.xticks(range(len(labels)), [label.replace("-attn", "") for label in labels], rotation=60, ha="right", fontsize=7)
        plt.yticks(range(len(labels)), [label.replace("-attn", "") for label in labels], fontsize=7)
        plt.title("Per-sample risk score correlation")
        plt.tight_layout()
        path = output_dir / "risk_correlation_pearson.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)

    return paths


def main():
    parser = argparse.ArgumentParser(description="Analyze current Attention Tracker result artifacts.")
    parser.add_argument("--result-dir", default="result")
    parser.add_argument("--output-dir", default="result/analysis/current")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_runs = load_eval_runs(result_dir)
    head_runs = load_head_runs(result_dir)

    model_rows = []
    head_rows = []
    depth_rows = []
    risk_by_model = {}
    samples_by_index = defaultdict(lambda: {"labels": [], "texts": [], "predictions": [], "risks": []})

    for run in eval_runs:
        summary = run["summary"]
        model = summary["model"]
        samples = run["samples"]
        pos_scores = [row["focus_score"] for row in samples if row["label"] == 1]
        neg_scores = [row["focus_score"] for row in samples if row["label"] == 0]
        best = best_threshold(samples)
        metrics = summary["metrics"]

        model_rows.append(
            {
                "model": model,
                "run_id": summary["run_id"],
                "num_samples": summary["num_samples"],
                "auc": fmt(metrics["auc"]),
                "auprc": fmt(metrics["auprc"]),
                "fixed_fnr": fmt(metrics["fnr"]),
                "fixed_fpr": fmt(metrics["fpr"]),
                "pos_mean_focus": fmt(mean(pos_scores)),
                "neg_mean_focus": fmt(mean(neg_scores)),
                "pos_median_focus": fmt(median(pos_scores)),
                "neg_median_focus": fmt(median(neg_scores)),
                "best_threshold": fmt(best["threshold"]),
                "best_fnr": fmt(best["fnr"]),
                "best_fpr": fmt(best["fpr"]),
            }
        )

        risk_by_model[model] = {}
        for row in samples:
            risk = 1.0 - row["focus_score"]
            risk_by_model[model][row["sample_index"]] = risk
            sample_entry = samples_by_index[row["sample_index"]]
            sample_entry["labels"].append(row["label"])
            sample_entry["texts"].append(row["text"])
            sample_entry["predictions"].append((model, row["prediction"]))
            sample_entry["risks"].append(risk)

        head_item = head_runs.get(model)
        if head_item:
            head_summary = summarize_head_map(head_item["data"])
            head_rows.append(
                {
                    "model": model,
                    "head_run": head_item["path"].parent.name,
                    **{key: fmt(value) if isinstance(value, float) or value is None else value for key, value in head_summary.items()},
                }
            )
            depth_rows.extend(
                {
                    key: fmt(value) if isinstance(value, float) or value is None else value
                    for key, value in row.items()
                }
                for row in depth_profile(model, head_item["data"])
            )

    model_rows.sort(key=lambda row: row["model"])
    head_rows.sort(key=lambda row: row["model"])
    depth_rows.sort(key=lambda row: (row["model"], row["depth_bin"]))

    pearson_rows, corr_fields = corr_matrix(risk_by_model, spearman=False)
    spearman_rows, _ = corr_matrix(risk_by_model, spearman=True)

    hard_rows = []
    for sample_index, entry in samples_by_index.items():
        if not entry["labels"]:
            continue
        label = max(set(entry["labels"]), key=entry["labels"].count)
        false_positive_models = [
            model for model, pred in entry["predictions"] if label == 0 and pred
        ]
        false_negative_models = [
            model for model, pred in entry["predictions"] if label == 1 and not pred
        ]
        text = next((text for text in entry["texts"] if text), "")
        hard_rows.append(
            {
                "sample_index": sample_index,
                "label": label,
                "false_positive_models": len(false_positive_models),
                "false_negative_models": len(false_negative_models),
                "avg_risk": fmt(mean(entry["risks"])),
                "models": ";".join(false_positive_models or false_negative_models),
                "text_preview": text[:140].replace("|", "/").replace("\n", " "),
            }
        )
    hard_rows.sort(
        key=lambda row: (
            -row["false_positive_models"],
            -row["false_negative_models"],
            -float(row["avg_risk"] or 0),
        )
    )

    write_csv(
        output_dir / "model_metrics.csv",
        model_rows,
        [
            "model",
            "run_id",
            "num_samples",
            "auc",
            "auprc",
            "fixed_fnr",
            "fixed_fpr",
            "pos_mean_focus",
            "neg_mean_focus",
            "pos_median_focus",
            "neg_median_focus",
            "best_threshold",
            "best_fnr",
            "best_fpr",
        ],
    )
    write_csv(
        output_dir / "head_map_summary.csv",
        head_rows,
        [
            "model",
            "head_run",
            "layers",
            "heads",
            "total_heads",
            "selected_heads",
            "selected_fraction",
            "global_diff_mean",
            "global_diff_pos_fraction",
            "selected_diff_mean",
            "selected_normal_mean",
            "selected_attack_mean",
            "selected_depth_mean",
            "selected_depth_min",
            "selected_depth_max",
            "early_fraction",
            "mid_fraction",
            "late_fraction",
            "peak_layer",
            "peak_layer_depth",
            "peak_layer_selected_heads",
        ],
    )
    write_csv(
        output_dir / "depth_profile.csv",
        depth_rows,
        [
            "model",
            "depth_bin",
            "depth_start",
            "depth_end",
            "layers_in_bin",
            "mean_diff_all_heads",
            "positive_fraction_all_heads",
            "selected_heads",
            "selected_density",
            "mean_diff_selected_heads",
        ],
    )
    write_csv(output_dir / "risk_correlation_pearson.csv", pearson_rows, corr_fields)
    write_csv(output_dir / "risk_correlation_spearman.csv", spearman_rows, corr_fields)
    write_csv(
        output_dir / "hard_samples_fixed_threshold.csv",
        hard_rows,
        [
            "sample_index",
            "label",
            "false_positive_models",
            "false_negative_models",
            "avg_risk",
            "models",
            "text_preview",
        ],
    )
    plot_paths = maybe_plot(output_dir, model_rows, depth_rows, pearson_rows)
    build_report(output_dir, model_rows, head_rows, hard_rows)

    print(f"Wrote analysis to {output_dir}")
    print(f"Models: {len(model_rows)}")
    print(f"Head maps: {len(head_rows)}")
    if plot_paths:
        print("Plots:")
        for path in plot_paths:
            print(f"- {path}")


if __name__ == "__main__":
    main()

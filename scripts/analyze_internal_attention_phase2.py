from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from safetensors.torch import safe_open
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


ATTENTION_KEYS = {
    "instruction_mass",
    "data_mass",
    "normalized_instruction_ratio",
    "ratio",
    "entropy",
}
DEPTH_METRICS = (
    "instruction_mass",
    "data_mass",
    "normalized_instruction_ratio",
    "entropy",
)


@dataclass
class ModelMeta:
    model: str
    display_model: str
    model_id: str
    provider: str
    family: str
    generation: str
    variant: str
    params_b: float
    layers: int
    heads: int
    quantization: str
    dtype: str


@dataclass
class RunData:
    run_id: str
    eval_dir: Path
    head_dir: Path
    meta: ModelMeta
    prompts: pd.DataFrame
    scores: pd.DataFrame
    spans: pd.DataFrame
    heads: pd.DataFrame
    sample_table: pd.DataFrame
    attention_arrays: dict[str, np.ndarray]
    attention_keys: set[str]
    summary: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_json(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def fmt_float(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number) or math.isinf(number):
        return ""
    return f"{number:.{digits}f}"


def display_model_name(model: str) -> str:
    return str(model).removesuffix("-attn")


def derive_params_b(model: str, model_id: str = "") -> float:
    text = f"{model} {model_id}".lower()
    match = re.search(r"(\d+(?:\.\d+)?)b", text)
    return float(match.group(1)) if match else float("nan")


def normalize_model_metadata(row: pd.Series, summary: dict[str, Any]) -> ModelMeta:
    model = str(row["model"])
    display = display_model_name(model)
    model_id = str(row["model_id"])
    key = f"{model} {model_id}".lower()

    if "qwen" in key:
        family = "Qwen"
        if "qwen2.5" in key or "qwen2_5" in key:
            generation = "Qwen2.5"
        elif "qwen3" in key:
            generation = "Qwen3"
        else:
            generation = "Qwen"

        if "coder" in key and "instruct" in key:
            variant = "coder-instruct"
        elif "instruct" in key:
            variant = "instruct"
        else:
            variant = "base"
    elif "gemma" in key:
        family = "Gemma"
        if "gemma-2" in key:
            generation = "Gemma2"
        elif "gemma-3" in key:
            generation = "Gemma3"
        else:
            generation = "Gemma"
        variant = "it" if "-it" in key or "it-" in key else "base"
    else:
        family = "Other"
        generation = "Other"
        variant = "unknown"

    loading = parse_json(row.get("model_loading_json"), {})
    summary_loading = summary.get("model_loading") or {}
    quantization = summary_loading.get("quantization") or loading.get("quantization", "")
    dtype = summary_loading.get("dtype") or loading.get("dtype", "")

    return ModelMeta(
        model=model,
        display_model=display,
        model_id=model_id,
        provider=str(row.get("provider", "")),
        family=family,
        generation=generation,
        variant=variant,
        params_b=derive_params_b(model, model_id),
        layers=int(row["layers"]),
        heads=int(row["heads"]),
        quantization=str(quantization),
        dtype=str(dtype),
    )


def model_sort_key(meta: ModelMeta) -> tuple[int, int, int, float, str]:
    family_order = {"Qwen": 0, "Gemma": 1, "Other": 9}
    generation_order = {
        "Qwen2.5": 0,
        "Qwen3": 1,
        "Gemma2": 0,
        "Gemma3": 1,
        "Gemma": 2,
    }
    variant_order = {
        "base": 0,
        "instruct": 1,
        "coder-instruct": 2,
        "it": 1,
        "unknown": 9,
    }
    return (
        family_order.get(meta.family, 9),
        generation_order.get(meta.generation, 9),
        variant_order.get(meta.variant, 9),
        meta.params_b,
        meta.model,
    )


def prompt_label(labels_json: str) -> int:
    labels = parse_json(labels_json, {})
    return int(bool(labels.get("injection_present") or labels.get("instruction_conflict")))


def prompt_sample_index(prompt_id: str) -> int | None:
    match = re.search(r"(\d+)$", str(prompt_id))
    return int(match.group(1)) if match else None


def prompt_text_from_messages(messages_json: str) -> str:
    messages = parse_json(messages_json, [])
    for message in messages:
        if message.get("name") == "untrusted_data":
            return str(message.get("content", ""))
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def safe_auc(labels: pd.Series, scores: pd.Series) -> float:
    if len(set(labels.astype(int))) < 2:
        return float("nan")
    return float(roc_auc_score(labels.astype(int), scores.astype(float)))


def safe_auprc(labels: pd.Series, scores: pd.Series) -> float:
    if len(set(labels.astype(int))) < 2:
        return float("nan")
    return float(average_precision_score(labels.astype(int), scores.astype(float)))


def load_attention_arrays(path: Path) -> tuple[dict[str, np.ndarray], set[str]]:
    with safe_open(path, framework="pt") as handle:
        keys = set(handle.keys())
        arrays = {key: handle.get_tensor(key).numpy() for key in keys}

    if "normalized_instruction_ratio" not in arrays and "ratio" in arrays:
        arrays["normalized_instruction_ratio"] = arrays["ratio"]
        keys.add("normalized_instruction_ratio")
    if "ratio" not in arrays and "normalized_instruction_ratio" in arrays:
        arrays["ratio"] = arrays["normalized_instruction_ratio"]
        keys.add("ratio")

    missing = ATTENTION_KEYS - set(arrays)
    if missing:
        raise ValueError(f"{path} is missing attention keys: {sorted(missing)}")

    shape = arrays["ratio"].shape
    for key in ATTENTION_KEYS:
        if arrays[key].shape != shape:
            raise ValueError(f"{path}: {key} shape {arrays[key].shape} differs from ratio shape {shape}")

    return arrays, keys


def required_files(eval_dir: Path, head_dir: Path) -> dict[str, Path]:
    return {
        "prompts": eval_dir / "prompts.parquet",
        "scores": eval_dir / "attention_tracker_scores.parquet",
        "spans": eval_dir / "token_spans.parquet",
        "metadata": eval_dir / "model_metadata.parquet",
        "generations": eval_dir / "generation_outputs.parquet",
        "attention": eval_dir / "attention_summary.safetensors",
        "heads": head_dir / "calibration_artifacts" / "selected_heads.parquet",
    }


def load_run(eval_dir: Path, result_runs_dir: Path) -> RunData:
    run_id = eval_dir.name
    head_dir = eval_dir.parent / run_id.replace("-seed0-phase2", "-head-selection")
    files = required_files(eval_dir, head_dir)
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{run_id}: missing artifact files: {', '.join(missing)}")

    legacy_dir = result_runs_dir / run_id
    summary = read_json(legacy_dir / "summary.json")
    samples_jsonl = read_jsonl(legacy_dir / "samples.jsonl")

    prompts = pd.read_parquet(files["prompts"])
    scores = pd.read_parquet(files["scores"])
    spans = pd.read_parquet(files["spans"])
    metadata = pd.read_parquet(files["metadata"])
    heads = pd.read_parquet(files["heads"])
    attention_arrays, attention_keys = load_attention_arrays(files["attention"])

    if len(metadata) != 1:
        raise ValueError(f"{run_id}: model_metadata.parquet must contain exactly one row.")
    if len(prompts) != len(scores):
        raise ValueError(f"{run_id}: prompts rows {len(prompts)} != score rows {len(scores)}")
    if attention_arrays["ratio"].shape[0] != len(prompts):
        raise ValueError(
            f"{run_id}: attention rows {attention_arrays['ratio'].shape[0]} != prompts rows {len(prompts)}"
        )

    meta = normalize_model_metadata(metadata.iloc[0], summary)

    prompts = prompts.copy()
    prompts["label"] = prompts["labels_json"].map(prompt_label)
    prompts["sample_index"] = prompts["prompt_id"].map(prompt_sample_index)
    prompts["text"] = prompts["messages_json"].map(prompt_text_from_messages)

    sample_text = {
        str(row["prompt_id"]): str(row.get("text", ""))
        for row in samples_jsonl
        if "prompt_id" in row
    }
    if sample_text:
        prompts["text"] = prompts.apply(
            lambda row: sample_text.get(str(row["prompt_id"]), row["text"]),
            axis=1,
        )

    means = {
        "mean_instruction_mass": np.nanmean(attention_arrays["instruction_mass"], axis=(1, 2)),
        "mean_data_mass": np.nanmean(attention_arrays["data_mass"], axis=(1, 2)),
        "mean_ratio": np.nanmean(attention_arrays["ratio"], axis=(1, 2)),
        "mean_normalized_instruction_ratio": np.nanmean(
            attention_arrays["normalized_instruction_ratio"], axis=(1, 2)
        ),
        "mean_entropy": np.nanmean(attention_arrays["entropy"], axis=(1, 2)),
    }
    mean_df = pd.DataFrame(means)

    score_features = scores.drop(
        columns=[
            "mean_instruction_mass",
            "mean_data_mass",
            "mean_injection_mass",
        ],
        errors="ignore",
    )
    sample_table = prompts[
        ["prompt_id", "base_request_id", "label", "sample_index", "text"]
    ].merge(score_features, on="prompt_id", how="left")
    sample_table = pd.concat([sample_table.reset_index(drop=True), mean_df], axis=1)
    sample_table["run_id"] = run_id
    sample_table["model"] = meta.model
    sample_table["display_model"] = meta.display_model
    sample_table["family"] = meta.family
    sample_table["generation"] = meta.generation
    sample_table["variant"] = meta.variant
    sample_table["params_b"] = meta.params_b
    sample_table["risk_score"] = 1.0 - sample_table["focus_score"].astype(float)
    sample_table["pred"] = sample_table["pred"].astype(bool)
    sample_table["is_fp"] = (sample_table["label"].astype(int) == 0) & sample_table["pred"]
    sample_table["is_fn"] = (sample_table["label"].astype(int) == 1) & (~sample_table["pred"])

    return RunData(
        run_id=run_id,
        eval_dir=eval_dir,
        head_dir=head_dir,
        meta=meta,
        prompts=prompts,
        scores=scores,
        spans=spans,
        heads=heads,
        sample_table=sample_table,
        attention_arrays=attention_arrays,
        attention_keys=attention_keys,
        summary=summary,
    )


def discover_runs(
    phase2_dir: Path,
    result_runs_dir: Path,
    include_run_id: list[str] | None,
) -> tuple[list[RunData], list[str]]:
    if include_run_id:
        candidates = [phase2_dir / run_id for run_id in include_run_id]
    else:
        candidates = sorted(path for path in phase2_dir.glob("*-seed0-phase2") if path.is_dir())

    runs: list[RunData] = []
    warnings: list[str] = []
    for candidate in candidates:
        try:
            runs.append(load_run(candidate, result_runs_dir))
        except Exception as exc:
            warnings.append(f"Skip {candidate.name}: {exc}")

    runs.sort(key=lambda run: model_sort_key(run.meta))
    return runs, warnings


def shared_prompt_ids(runs: list[RunData]) -> list[str]:
    if not runs:
        return []
    common = set(runs[0].sample_table["prompt_id"].astype(str))
    for run in runs[1:]:
        common &= set(run.sample_table["prompt_id"].astype(str))
    return sorted(common)


def filtered_samples(run: RunData, prompt_ids: set[str]) -> pd.DataFrame:
    return run.sample_table[run.sample_table["prompt_id"].astype(str).isin(prompt_ids)].copy()


def filtered_attention_indexes(run: RunData, prompt_ids: set[str]) -> np.ndarray:
    mask = run.sample_table["prompt_id"].astype(str).isin(prompt_ids).to_numpy()
    return np.where(mask)[0]


def model_base_row(run: RunData) -> dict[str, Any]:
    meta = run.meta
    return {
        "run_id": run.run_id,
        "model": meta.model,
        "display_model": meta.display_model,
        "model_id": meta.model_id,
        "provider": meta.provider,
        "family": meta.family,
        "generation": meta.generation,
        "variant": meta.variant,
        "params_b": meta.params_b,
        "layers": meta.layers,
        "heads": meta.heads,
        "quantization": meta.quantization,
        "dtype": meta.dtype,
    }


def build_model_inventory(runs: list[RunData], prompt_ids: set[str]) -> pd.DataFrame:
    rows = []
    for run in runs:
        row = model_base_row(run)
        row.update(
            {
                "num_prompts": int(len(run.prompts)),
                "shared_prompts": int(filtered_samples(run, prompt_ids).shape[0]),
                "num_spans": int(len(run.spans)),
                "attention_keys": ",".join(sorted(run.attention_keys)),
                "attention_shape": "x".join(str(dim) for dim in run.attention_arrays["ratio"].shape),
                "head_selection_run_id": run.head_dir.name,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_heads(runs: list[RunData], depth_bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    for run in runs:
        heads = run.heads.copy()
        heads["selected"] = heads["selected"].astype(bool)
        selected = heads[heads["selected"]]
        selected_depths = selected["normalized_depth"].astype(float)
        total_heads = int(len(heads))
        selected_count = int(len(selected))

        row = model_base_row(run)
        row.update(
            {
                "total_heads": total_heads,
                "selected_heads": selected_count,
                "selected_fraction": selected_count / total_heads if total_heads else np.nan,
                "selected_depth_mean": selected_depths.mean() if selected_count else np.nan,
                "selected_depth_std": selected_depths.std(ddof=0) if selected_count else np.nan,
                "selected_depth_min": selected_depths.min() if selected_count else np.nan,
                "selected_depth_max": selected_depths.max() if selected_count else np.nan,
                "early_fraction": float((selected_depths < 1 / 3).mean()) if selected_count else np.nan,
                "mid_fraction": float(
                    ((selected_depths >= 1 / 3) & (selected_depths < 2 / 3)).mean()
                ) if selected_count else np.nan,
                "late_fraction": float((selected_depths >= 2 / 3).mean()) if selected_count else np.nan,
                "margin_mean_all": heads["margin"].astype(float).mean(),
                "margin_mean_selected": selected["margin"].astype(float).mean() if selected_count else np.nan,
                "margin_max_selected": selected["margin"].astype(float).max() if selected_count else np.nan,
            }
        )
        summary_rows.append(row)

        for depth_bin in range(depth_bins):
            start = depth_bin / depth_bins
            end = (depth_bin + 1) / depth_bins
            in_bin = heads[
                (heads["normalized_depth"].astype(float) >= start)
                & (
                    (heads["normalized_depth"].astype(float) < end)
                    | ((depth_bin == depth_bins - 1) & (heads["normalized_depth"].astype(float) <= end))
                )
            ]
            selected_bin = in_bin[in_bin["selected"]]
            depth_row = model_base_row(run)
            depth_row.update(
                {
                    "depth_bin": depth_bin,
                    "depth_start": start,
                    "depth_end": end,
                    "heads_in_bin": int(len(in_bin)),
                    "selected_heads": int(len(selected_bin)),
                    "selected_density": len(selected_bin) / len(in_bin) if len(in_bin) else np.nan,
                    "margin_mean_all": in_bin["margin"].astype(float).mean() if len(in_bin) else np.nan,
                    "margin_mean_selected": selected_bin["margin"].astype(float).mean()
                    if len(selected_bin)
                    else np.nan,
                }
            )
            depth_rows.append(depth_row)

    return pd.DataFrame(summary_rows), pd.DataFrame(depth_rows)


def build_model_metrics(runs: list[RunData], head_summary: pd.DataFrame, prompt_ids: set[str]) -> pd.DataFrame:
    head_by_model = head_summary.set_index("model").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for run in runs:
        samples = filtered_samples(run, prompt_ids)
        labels = samples["label"].astype(int)
        risks = samples["risk_score"].astype(float)
        preds = samples["pred"].astype(bool)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) else np.nan
        fnr = fn / (fn + tp) if (fn + tp) else np.nan
        benign = samples[labels == 0]
        attack = samples[labels == 1]
        risk_delta = attack["risk_score"].astype(float).mean() - benign["risk_score"].astype(float).mean()
        pooled_std = samples["risk_score"].astype(float).std(ddof=0)

        row = model_base_row(run)
        row.update(
            {
                "num_samples": int(len(samples)),
                "auc": safe_auc(labels, risks),
                "auprc": safe_auprc(labels, risks),
                "fpr": float(fpr),
                "fnr": float(fnr),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "mean_focus_benign": benign["focus_score"].astype(float).mean(),
                "mean_focus_attack": attack["focus_score"].astype(float).mean(),
                "mean_risk_benign": benign["risk_score"].astype(float).mean(),
                "mean_risk_attack": attack["risk_score"].astype(float).mean(),
                "risk_delta_attack_minus_benign": risk_delta,
                "risk_cohens_d": risk_delta / pooled_std if pooled_std and not math.isnan(pooled_std) else np.nan,
            }
        )
        row.update(
            {
                key: head_by_model.get(run.meta.model, {}).get(key)
                for key in (
                    "selected_heads",
                    "selected_fraction",
                    "selected_depth_mean",
                    "early_fraction",
                    "mid_fraction",
                    "late_fraction",
                    "margin_mean_all",
                    "margin_mean_selected",
                )
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def attention_mass_by_label(runs: list[RunData], prompt_ids: set[str]) -> pd.DataFrame:
    metrics = (
        "focus_score",
        "risk_score",
        "mean_instruction_mass",
        "mean_data_mass",
        "mean_ratio",
        "mean_normalized_instruction_ratio",
        "mean_entropy",
    )
    rows: list[dict[str, Any]] = []
    for run in runs:
        samples = filtered_samples(run, prompt_ids)
        for label, group in samples.groupby("label", sort=True):
            row = model_base_row(run)
            row.update({"label": int(label), "count": int(len(group))})
            for metric in metrics:
                values = group[metric].astype(float)
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=0)
                row[f"{metric}_median"] = values.median()
            rows.append(row)
    return pd.DataFrame(rows)


def attention_delta_by_depth(runs: list[RunData], prompt_ids: set[str], depth_bins: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        sample_indexes = filtered_attention_indexes(run, prompt_ids)
        samples = run.sample_table.iloc[sample_indexes].reset_index(drop=True)
        labels = samples["label"].astype(int).to_numpy()
        benign_indexes = np.where(labels == 0)[0]
        attack_indexes = np.where(labels == 1)[0]
        if not len(benign_indexes) or not len(attack_indexes):
            continue

        layer_bins = np.minimum(
            (np.arange(run.meta.layers) / run.meta.layers * depth_bins).astype(int),
            depth_bins - 1,
        )
        for metric in DEPTH_METRICS:
            array = run.attention_arrays[metric][sample_indexes]
            for depth_bin in range(depth_bins):
                layer_mask = layer_bins == depth_bin
                benign_mean = float(np.nanmean(array[benign_indexes][:, layer_mask, :]))
                attack_mean = float(np.nanmean(array[attack_indexes][:, layer_mask, :]))
                row = model_base_row(run)
                row.update(
                    {
                        "metric": metric,
                        "depth_bin": depth_bin,
                        "depth_start": depth_bin / depth_bins,
                        "depth_end": (depth_bin + 1) / depth_bins,
                        "benign_mean": benign_mean,
                        "attack_mean": attack_mean,
                        "delta_attack_minus_benign": attack_mean - benign_mean,
                        "abs_delta": abs(attack_mean - benign_mean),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def group_summary(model_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        model_metrics.groupby(["family", "generation", "variant"], as_index=False)
        .agg(
            models=("model", "count"),
            mean_params_b=("params_b", "mean"),
            mean_auc=("auc", "mean"),
            mean_auprc=("auprc", "mean"),
            mean_fpr=("fpr", "mean"),
            mean_fnr=("fnr", "mean"),
            mean_risk_delta=("risk_delta_attack_minus_benign", "mean"),
            mean_selected_fraction=("selected_fraction", "mean"),
            mean_selected_depth=("selected_depth_mean", "mean"),
        )
        .sort_values(["family", "generation", "variant"])
    )


def build_sample_risk_correlation(runs: list[RunData], prompt_ids: set[str]) -> pd.DataFrame:
    rows = []
    for run in runs:
        samples = filtered_samples(run, prompt_ids)
        rows.append(samples[["prompt_id", "model", "risk_score"]])
    all_samples = pd.concat(rows, ignore_index=True)
    pivot = all_samples.pivot_table(index="prompt_id", columns="model", values="risk_score", aggfunc="mean")
    ordered = [run.meta.model for run in runs if run.meta.model in pivot.columns]
    return pivot[ordered].corr(method="pearson")


def selected_head_set(run: RunData) -> set[tuple[int, int]]:
    heads = run.heads[run.heads["selected"].astype(bool)]
    return {
        (int(row["layer"]), int(row["head"]))
        for _, row in heads.iterrows()
    }


def density_vector(depth_profile: pd.DataFrame, model: str) -> np.ndarray:
    rows = depth_profile[depth_profile["model"] == model].sort_values("depth_bin")
    return rows["selected_density"].fillna(0.0).astype(float).to_numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def build_signature_vectors(
    runs: list[RunData],
    attention_delta: pd.DataFrame,
    depth_profile: pd.DataFrame,
    depth_bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        features: dict[str, float] = {}
        for metric in DEPTH_METRICS:
            metric_rows = attention_delta[
                (attention_delta["model"] == run.meta.model)
                & (attention_delta["metric"] == metric)
            ].sort_values("depth_bin")
            values = metric_rows["delta_attack_minus_benign"].astype(float).to_list()
            values = (values + [0.0] * depth_bins)[:depth_bins]
            for idx, value in enumerate(values):
                features[f"{metric}_delta_bin_{idx}"] = float(value)

        head_rows = depth_profile[depth_profile["model"] == run.meta.model].sort_values("depth_bin")
        for idx in range(depth_bins):
            if idx < len(head_rows):
                row = head_rows.iloc[idx]
                features[f"selected_density_bin_{idx}"] = float(row["selected_density"])
                features[f"margin_mean_all_bin_{idx}"] = float(row["margin_mean_all"])
            else:
                features[f"selected_density_bin_{idx}"] = 0.0
                features[f"margin_mean_all_bin_{idx}"] = 0.0

        base = model_base_row(run)
        base.update(features)
        rows.append(base)

    df = pd.DataFrame(rows).fillna(0.0)
    feature_cols = [col for col in df.columns if col.endswith(tuple(str(i) for i in range(depth_bins)))]
    if feature_cols:
        matrix = df[feature_cols].astype(float)
        std = matrix.std(axis=0, ddof=0).replace(0, 1.0)
        df[feature_cols] = (matrix - matrix.mean(axis=0)) / std
    return df


def similarity_matrix_from_vectors(vectors: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = {
        "run_id",
        "model",
        "display_model",
        "model_id",
        "provider",
        "family",
        "generation",
        "variant",
        "params_b",
        "layers",
        "heads",
        "quantization",
        "dtype",
    }
    feature_cols = [col for col in vectors.columns if col not in metadata_cols]
    matrix = vectors[feature_cols].astype(float).to_numpy()
    models = list(vectors["model"])
    sim = np.eye(len(models), dtype=float)
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            value = cosine(matrix[i], matrix[j])
            sim[i, j] = value
            sim[j, i] = value
    return pd.DataFrame(sim, index=models, columns=models)


def build_head_overlap(
    runs: list[RunData],
    depth_profile: pd.DataFrame,
    signature_similarity: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_sets = {run.meta.model: selected_head_set(run) for run in runs}
    density_vectors = {run.meta.model: density_vector(depth_profile, run.meta.model) for run in runs}

    for left, right in combinations(runs, 2):
        left_set = selected_sets[left.meta.model]
        right_set = selected_sets[right.meta.model]
        exact_jaccard = np.nan
        exact_overlap = np.nan
        if left.meta.layers == right.meta.layers and left.meta.heads == right.meta.heads:
            union = left_set | right_set
            exact_overlap = len(left_set & right_set)
            exact_jaccard = exact_overlap / len(union) if union else np.nan

        left_density = density_vectors[left.meta.model]
        right_density = density_vectors[right.meta.model]
        left_bins = set(np.where(left_density > 0)[0])
        right_bins = set(np.where(right_density > 0)[0])
        bin_union = left_bins | right_bins
        row = {
            "model_a": left.meta.model,
            "model_b": right.meta.model,
            "display_model_a": left.meta.display_model,
            "display_model_b": right.meta.display_model,
            "family_a": left.meta.family,
            "family_b": right.meta.family,
            "generation_a": left.meta.generation,
            "generation_b": right.meta.generation,
            "variant_a": left.meta.variant,
            "variant_b": right.meta.variant,
            "params_b_a": left.meta.params_b,
            "params_b_b": right.meta.params_b,
            "same_shape": bool(left.meta.layers == right.meta.layers and left.meta.heads == right.meta.heads),
            "same_qwen25_size": bool(
                left.meta.family == right.meta.family == "Qwen"
                and left.meta.generation == right.meta.generation == "Qwen2.5"
                and left.meta.params_b == right.meta.params_b
            ),
            "selected_a": len(left_set),
            "selected_b": len(right_set),
            "exact_overlap": exact_overlap,
            "exact_jaccard": exact_jaccard,
            "normalized_depth_density_cosine": cosine(left_density, right_density),
            "normalized_depth_bin_jaccard": len(left_bins & right_bins) / len(bin_union)
            if bin_union
            else np.nan,
            "signature_cosine": float(signature_similarity.loc[left.meta.model, right.meta.model]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def paired_effects(
    model_metrics: pd.DataFrame,
    attention_mass: pd.DataFrame,
    head_overlap: pd.DataFrame,
    variant_a: str,
    variant_b: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    qwen25 = model_metrics[
        (model_metrics["family"] == "Qwen")
        & (model_metrics["generation"] == "Qwen2.5")
        & (model_metrics["variant"].isin([variant_a, variant_b]))
    ]
    for params_b, group in qwen25.groupby("params_b"):
        if set(group["variant"]) != {variant_a, variant_b}:
            continue
        left = group[group["variant"] == variant_a].iloc[0]
        right = group[group["variant"] == variant_b].iloc[0]
        left_attention = attention_mass[
            (attention_mass["model"] == left["model"])
            & (attention_mass["label"] == 1)
        ].iloc[0]
        right_attention = attention_mass[
            (attention_mass["model"] == right["model"])
            & (attention_mass["label"] == 1)
        ].iloc[0]
        overlap = head_overlap[
            (
                (head_overlap["model_a"] == left["model"]) & (head_overlap["model_b"] == right["model"])
            )
            | (
                (head_overlap["model_a"] == right["model"]) & (head_overlap["model_b"] == left["model"])
            )
        ]
        overlap_row = overlap.iloc[0] if len(overlap) else {}
        rows.append(
            {
                "params_b": float(params_b),
                "variant_a": variant_a,
                "variant_b": variant_b,
                "model_a": left["model"],
                "model_b": right["model"],
                "auc_delta_b_minus_a": float(right["auc"] - left["auc"]),
                "fpr_delta_b_minus_a": float(right["fpr"] - left["fpr"]),
                "fnr_delta_b_minus_a": float(right["fnr"] - left["fnr"]),
                "risk_delta_b_minus_a": float(
                    right["risk_delta_attack_minus_benign"]
                    - left["risk_delta_attack_minus_benign"]
                ),
                "attack_instruction_ratio_delta_b_minus_a": float(
                    right_attention["mean_normalized_instruction_ratio_mean"]
                    - left_attention["mean_normalized_instruction_ratio_mean"]
                ),
                "selected_fraction_delta_b_minus_a": float(
                    right["selected_fraction"] - left["selected_fraction"]
                ),
                "selected_depth_delta_b_minus_a": float(
                    right["selected_depth_mean"] - left["selected_depth_mean"]
                ),
                "exact_head_jaccard": float(overlap_row.get("exact_jaccard", np.nan)),
                "normalized_depth_cosine": float(
                    overlap_row.get("normalized_depth_density_cosine", np.nan)
                ),
                "signature_cosine": float(overlap_row.get("signature_cosine", np.nan)),
            }
        )
    return pd.DataFrame(rows).sort_values("params_b") if rows else pd.DataFrame()


def classify_text_factor(text: str) -> str:
    lowered = text.lower()
    factors: list[str] = []
    german_markers = (" ich ", " was ", " wie ", " mein ", " meine ", "nicht", "über", "ﾃ", " mﾃ")
    if any(marker in f" {lowered} " for marker in german_markers):
        factors.append("german_or_mojibake")
    if len(text) >= 180:
        factors.append("long_text")
    if "?" in text:
        factors.append("question")
    if any(word in lowered for word in ("ignore", "previous instruction", "output", "say ")):
        factors.append("instruction_like_word")
    if any(word in lowered for word in ("website", "marketing", "customer", "social media", "kunden")):
        factors.append("web_or_marketing")
    return ";".join(factors) if factors else "none"


def build_hard_samples(runs: list[RunData], prompt_ids: set[str]) -> pd.DataFrame:
    samples = pd.concat([filtered_samples(run, prompt_ids) for run in runs], ignore_index=True)
    rows: list[dict[str, Any]] = []
    for prompt_id, group in samples.groupby("prompt_id", sort=False):
        label = int(group["label"].mode().iloc[0])
        fps = group[group["is_fp"]]
        fns = group[group["is_fn"]]
        text = str(group["text"].dropna().iloc[0]) if len(group["text"].dropna()) else ""
        fp_families = sorted(set(fps["family"]))
        fn_families = sorted(set(fns["family"]))
        rows.append(
            {
                "prompt_id": prompt_id,
                "sample_index": prompt_sample_index(prompt_id),
                "label": label,
                "false_positive_models": int(len(fps)),
                "false_negative_models": int(len(fns)),
                "false_positive_families": int(len(fp_families)),
                "false_negative_families": int(len(fn_families)),
                "fp_family_list": ";".join(fp_families),
                "fn_family_list": ";".join(fn_families),
                "fp_model_list": ";".join(fps["display_model"].astype(str).sort_values()),
                "fn_model_list": ";".join(fns["display_model"].astype(str).sort_values()),
                "avg_risk": float(group["risk_score"].astype(float).mean()),
                "risk_std": float(group["risk_score"].astype(float).std(ddof=0)),
                "text_len": len(text),
                "text_factor": classify_text_factor(text),
                "text_preview": text[:220].replace("\n", " ").replace("|", "/"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["false_positive_families", "false_positive_models", "false_negative_models", "avg_risk"],
        ascending=[False, False, False, False],
    )


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
            "font.size": 9,
        }
    )
    return plt


def save_heatmap(path: Path, matrix: pd.DataFrame, title: str, colorbar_label: str, plt) -> Path:
    labels = [display_model_name(model) for model in matrix.columns]
    fig_width = max(7.0, len(labels) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.82))
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_model_metrics(output_dir: Path, metrics: pd.DataFrame, plt) -> Path:
    data = metrics.sort_values(["family", "generation", "variant", "params_b", "model"])
    x = np.arange(len(data))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.bar(x - width, data["auc"], width=width, label="AUROC", color="#2563eb")
    ax.bar(x, data["fpr"], width=width, label="FPR@0.5", color="#dc2626")
    ax.bar(x + width, data["fnr"], width=width, label="FNR@0.5", color="#f59e0b")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Attention Tracker metrics by model")
    ax.set_xticks(x)
    ax.set_xticklabels(data["display_model"], rotation=45, ha="right")
    ax.legend(ncol=3)
    fig.tight_layout()
    path = output_dir / "model_metrics_auc_fpr_fnr.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_focus_by_label(output_dir: Path, samples: pd.DataFrame, metrics: pd.DataFrame, plt) -> Path:
    ordered = list(metrics["model"])
    positions: list[float] = []
    data: list[np.ndarray] = []
    colors: list[str] = []
    ticks: list[float] = []
    labels: list[str] = []
    for index, model in enumerate(ordered):
        base = index * 3
        for offset, label, color in [(0, 0, "#64748b"), (1, 1, "#10b981")]:
            values = samples[
                (samples["model"] == model) & (samples["label"].astype(int) == label)
            ]["focus_score"].astype(float)
            data.append(values.to_numpy())
            positions.append(base + offset)
            colors.append(color)
        ticks.append(base + 0.5)
        labels.append(display_model_name(model))
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    boxes = ax.boxplot(data, positions=positions, widths=0.65, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.axhline(0.5, color="#111827", linestyle="--", linewidth=1)
    ax.set_ylabel("Focus score")
    ax.set_title("Focus score by label")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#64748b", lw=6, alpha=0.65, label="benign"),
            plt.Line2D([0], [0], color="#10b981", lw=6, alpha=0.65, label="attack"),
            plt.Line2D([0], [0], color="#111827", lw=1, linestyle="--", label="threshold=0.5"),
        ],
        ncol=3,
    )
    fig.tight_layout()
    path = output_dir / "focus_score_by_label.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_attention_delta(output_dir: Path, attention_delta: pd.DataFrame, metrics: pd.DataFrame, plt) -> Path:
    metric = "normalized_instruction_ratio"
    data = (
        attention_delta[attention_delta["metric"] == metric]
        .groupby("model", as_index=False)
        .agg(delta=("delta_attack_minus_benign", "mean"))
    )
    data = metrics[["model", "display_model", "family", "generation", "variant", "params_b"]].merge(
        data, on="model", how="left"
    )
    fig, ax = plt.subplots(figsize=(13, 4.8))
    colors = np.where(data["delta"].astype(float) >= 0, "#10b981", "#dc2626")
    ax.bar(np.arange(len(data)), data["delta"].astype(float), color=colors)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_ylabel("Attack - benign")
    ax.set_title("Normalized instruction ratio delta by model")
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data["display_model"], rotation=45, ha="right")
    fig.tight_layout()
    path = output_dir / "attention_mass_delta_by_group.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_depth_profile(output_dir: Path, attention_delta: pd.DataFrame, metrics: pd.DataFrame, plt) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for _, row in metrics.iterrows():
        model = row["model"]
        data = attention_delta[
            (attention_delta["model"] == model)
            & (attention_delta["metric"] == "normalized_instruction_ratio")
        ].sort_values("depth_bin")
        if data.empty:
            continue
        xs = (data["depth_start"].astype(float) + data["depth_end"].astype(float)) / 2
        ys = data["delta_attack_minus_benign"].astype(float)
        ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=3, label=row["display_model"])
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xlabel("Normalized layer depth")
    ax.set_ylabel("Attack - benign ratio")
    ax.set_title("Attention routing depth profile")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    path = output_dir / "normalized_depth_profile.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_selected_density(output_dir: Path, depth_profile: pd.DataFrame, metrics: pd.DataFrame, plt) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for _, row in metrics.iterrows():
        model = row["model"]
        data = depth_profile[depth_profile["model"] == model].sort_values("depth_bin")
        xs = (data["depth_start"].astype(float) + data["depth_end"].astype(float)) / 2
        ys = data["selected_density"].astype(float)
        ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=3, label=row["display_model"])
    ax.set_xlabel("Normalized layer depth")
    ax.set_ylabel("Selected-head density")
    ax.set_title("Selected head density by normalized depth")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    path = output_dir / "selected_head_density_by_depth.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def matrix_from_pair_rows(rows: pd.DataFrame, models: list[str], value_column: str) -> pd.DataFrame:
    matrix = pd.DataFrame(np.eye(len(models)), index=models, columns=models, dtype=float)
    for _, row in rows.iterrows():
        if row["model_a"] in matrix.index and row["model_b"] in matrix.index:
            matrix.loc[row["model_a"], row["model_b"]] = float(row[value_column])
            matrix.loc[row["model_b"], row["model_a"]] = float(row[value_column])
    return matrix


def write_plots(
    output_dir: Path,
    runs: list[RunData],
    samples: pd.DataFrame,
    metrics: pd.DataFrame,
    attention_delta: pd.DataFrame,
    depth_profile: pd.DataFrame,
    head_overlap: pd.DataFrame,
    risk_corr: pd.DataFrame,
    signature_similarity: pd.DataFrame,
) -> list[Path]:
    plt = configure_matplotlib()
    plot_paths = [
        plot_model_metrics(output_dir, metrics, plt),
        plot_focus_by_label(output_dir, samples, metrics, plt),
        plot_attention_delta(output_dir, attention_delta, metrics, plt),
        plot_depth_profile(output_dir, attention_delta, metrics, plt),
        plot_selected_density(output_dir, depth_profile, metrics, plt),
        save_heatmap(
            output_dir / "sample_risk_correlation_heatmap.png",
            risk_corr,
            "Sample-level risk correlation",
            "Pearson r",
            plt,
        ),
        save_heatmap(
            output_dir / "representation_signature_similarity_heatmap.png",
            signature_similarity,
            "Representation signature cosine similarity",
            "cosine",
            plt,
        ),
    ]

    qwen25_models = [
        run.meta.model
        for run in runs
        if run.meta.family == "Qwen" and run.meta.generation == "Qwen2.5"
    ]
    exact_rows = head_overlap[
        head_overlap["model_a"].isin(qwen25_models)
        & head_overlap["model_b"].isin(qwen25_models)
        & head_overlap["exact_jaccard"].notna()
    ]
    if len(exact_rows):
        exact_matrix = matrix_from_pair_rows(exact_rows, qwen25_models, "exact_jaccard")
        plot_paths.append(
            save_heatmap(
                output_dir / "exact_head_overlap_qwen25.png",
                exact_matrix,
                "Qwen2.5 exact selected-head overlap",
                "Jaccard",
                plt,
            )
        )

    depth_matrix = matrix_from_pair_rows(
        head_overlap, [run.meta.model for run in runs], "normalized_depth_density_cosine"
    )
    plot_paths.append(
        save_heatmap(
            output_dir / "normalized_depth_overlap.png",
            depth_matrix,
            "Selected-head normalized-depth overlap",
            "cosine",
            plt,
        )
    )
    return plot_paths


def write_tables(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> list[str]:
    use_df = df[columns].head(max_rows) if max_rows else df[columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in use_df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(fmt_float(value, 3))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    output_dir: Path,
    runs: list[RunData],
    shared_count: int,
    model_inventory: pd.DataFrame,
    metrics: pd.DataFrame,
    groups: pd.DataFrame,
    alignment_effects: pd.DataFrame,
    coder_effects: pd.DataFrame,
    head_summary: pd.DataFrame,
    hard_samples: pd.DataFrame,
    warnings: list[str],
    plot_paths: list[Path],
) -> None:
    best_auc = metrics.sort_values(["auc", "fpr"], ascending=[False, True]).iloc[0]
    lowest_fpr = metrics.sort_values(["fpr", "fnr"], ascending=[True, True]).iloc[0]
    highest_fpr = metrics.sort_values(["fpr", "auc"], ascending=[False, False]).iloc[0]

    lines: list[str] = [
        "# Attention-Tracker Phase2 内部表現比較",
        "",
        "## 概要",
        "",
        f"- 完全な Phase2 run を {len(runs)} 件読み込みました。",
        f"- 共通 prompt intersection は {shared_count} 件です。",
        "- 対象は attention routing 表現です。hidden state / MLP / MoE router は今回の成果物に含まれないため対象外です。",
        "- AUROC/AUPRC は表現分離、FPR/FNR は fixed threshold 0.5 の運用挙動として分けて解釈します。",
        "",
        "## 主な観察",
        "",
        f"- AUROC 最大は `{best_auc['display_model']}` の `{fmt_float(best_auc['auc'])}` です。",
        f"- fixed threshold 0.5 で FPR 最小は `{lowest_fpr['display_model']}` の `{fmt_float(lowest_fpr['fpr'])}` です。",
        f"- FPR 最大は `{highest_fpr['display_model']}` の `{fmt_float(highest_fpr['fpr'])}` です。AUROC が高くても threshold 校正がずれる場合があります。",
        "- Qwen2.5 は同サイズの Base/Instruct/Coder で exact layer/head overlap を比較できます。",
        "- Gemma は世代・サイズ差を normalized depth profile と hard false positive で比較します。",
        "",
        "## モデル一覧",
        "",
    ]
    lines.extend(
        markdown_table(
            model_inventory,
            ["display_model", "family", "generation", "variant", "params_b", "layers", "heads", "quantization"],
        )
    )
    lines.extend(["", "## モデル別指標", ""])
    lines.extend(
        markdown_table(
            metrics,
            [
                "display_model",
                "family",
                "generation",
                "variant",
                "auc",
                "auprc",
                "fpr",
                "fnr",
                "risk_delta_attack_minus_benign",
                "selected_fraction",
                "selected_depth_mean",
            ],
        )
    )
    lines.extend(["", "## グループ比較", ""])
    lines.extend(
        markdown_table(
            groups,
            [
                "family",
                "generation",
                "variant",
                "models",
                "mean_auc",
                "mean_fpr",
                "mean_fnr",
                "mean_risk_delta",
                "mean_selected_depth",
            ],
        )
    )
    if len(alignment_effects):
        lines.extend(["", "## Qwen2.5 Base → Instruct 差分", ""])
        lines.extend(
            markdown_table(
                alignment_effects,
                [
                    "params_b",
                    "auc_delta_b_minus_a",
                    "fpr_delta_b_minus_a",
                    "risk_delta_b_minus_a",
                    "attack_instruction_ratio_delta_b_minus_a",
                    "exact_head_jaccard",
                    "signature_cosine",
                ],
            )
        )
    if len(coder_effects):
        lines.extend(["", "## Qwen2.5 Instruct → Coder-Instruct 差分", ""])
        lines.extend(
            markdown_table(
                coder_effects,
                [
                    "params_b",
                    "auc_delta_b_minus_a",
                    "fpr_delta_b_minus_a",
                    "risk_delta_b_minus_a",
                    "attack_instruction_ratio_delta_b_minus_a",
                    "exact_head_jaccard",
                    "signature_cosine",
                ],
            )
        )
    lines.extend(["", "## Selected Head 概要", ""])
    lines.extend(
        markdown_table(
            head_summary,
            [
                "display_model",
                "selected_heads",
                "selected_fraction",
                "selected_depth_mean",
                "early_fraction",
                "mid_fraction",
                "late_fraction",
                "margin_mean_selected",
            ],
        )
    )
    lines.extend(["", "## Hard false positive / false negative samples", ""])
    lines.extend(
        markdown_table(
            hard_samples,
            [
                "sample_index",
                "label",
                "false_positive_families",
                "false_positive_models",
                "false_negative_models",
                "avg_risk",
                "text_factor",
                "text_preview",
            ],
            max_rows=15,
        )
    )
    lines.extend(["", "## 生成物", ""])
    for path in sorted(plot_paths):
        lines.append(f"- `{path.relative_to(output_dir)}`")
    lines.extend(["", "## 検証メモ", ""])
    for warning in warnings:
        lines.append(f"- warning: {warning}")
    if not warnings:
        lines.append("- 欠損 artifact による除外はありません。")
    lines.append("")

    (output_dir / "internal_attention_phase2_analysis.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def validate_outputs(output_dir: Path, csv_names: list[str], plot_paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for name in csv_names:
        path = output_dir / name
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing or empty CSV: {path}")
    for path in plot_paths:
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing or empty PNG: {path}")
    report = output_dir / "internal_attention_phase2_analysis.md"
    if not report.exists() or report.stat().st_size == 0:
        errors.append(f"missing or empty report: {report}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze cross-model Phase2 Attention Tracker artifacts."
    )
    parser.add_argument("--phase2-dir", default="outputs/phase2")
    parser.add_argument("--result-runs-dir", default="result/deepset/prompt-injections/runs")
    parser.add_argument("--output-dir", default="result/analysis/internal_attention_phase2")
    parser.add_argument("--depth-bins", type=int, default=10)
    parser.add_argument(
        "--include-run-id",
        action="append",
        default=None,
        help="Restrict analysis to an exact Phase2 eval run id. May be repeated.",
    )
    args = parser.parse_args()

    phase2_dir = Path(args.phase2_dir)
    result_runs_dir = Path(args.result_runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs, warnings = discover_runs(phase2_dir, result_runs_dir, args.include_run_id)
    if len(runs) < 2:
        raise SystemExit("At least two complete Phase2 eval runs are required.")

    shared_ids = set(shared_prompt_ids(runs))
    if not shared_ids:
        raise SystemExit("No shared prompt ids across selected runs.")

    model_inventory = build_model_inventory(runs, shared_ids)
    head_summary, depth_profile = summarize_heads(runs, args.depth_bins)
    model_metrics = build_model_metrics(runs, head_summary, shared_ids)
    model_metrics = model_metrics.sort_values(["family", "generation", "variant", "params_b", "model"])
    sample_rows = [filtered_samples(run, shared_ids) for run in runs]
    all_samples = pd.concat(sample_rows, ignore_index=True)
    attention_mass = attention_mass_by_label(runs, shared_ids)
    attention_delta = attention_delta_by_depth(runs, shared_ids, args.depth_bins)
    groups = group_summary(model_metrics)
    risk_corr = build_sample_risk_correlation(runs, shared_ids)
    signature_vectors = build_signature_vectors(runs, attention_delta, depth_profile, args.depth_bins)
    signature_similarity = similarity_matrix_from_vectors(signature_vectors)
    head_overlap = build_head_overlap(runs, depth_profile, signature_similarity)
    alignment_effects = paired_effects(
        model_metrics, attention_mass, head_overlap, "base", "instruct"
    )
    coder_effects = paired_effects(
        model_metrics, attention_mass, head_overlap, "instruct", "coder-instruct"
    )
    hard_samples = build_hard_samples(runs, shared_ids)

    tables = {
        "model_inventory.csv": model_inventory,
        "model_metrics.csv": model_metrics,
        "group_summary.csv": groups,
        "paired_alignment_effects.csv": alignment_effects,
        "paired_coder_effects.csv": coder_effects,
        "attention_mass_by_label.csv": attention_mass,
        "attention_delta_by_depth.csv": attention_delta,
        "head_selection_summary.csv": head_summary,
        "head_overlap.csv": head_overlap,
        "sample_risk_correlation.csv": risk_corr.reset_index().rename(columns={"index": "model"}),
        "hard_samples_cross_family.csv": hard_samples,
        "representation_signature_vectors.csv": signature_vectors,
        "representation_signature_similarity.csv": signature_similarity.reset_index().rename(columns={"index": "model"}),
    }
    write_tables(output_dir, tables)
    plot_paths = write_plots(
        output_dir,
        runs,
        all_samples,
        model_metrics,
        attention_delta,
        depth_profile,
        head_overlap,
        risk_corr,
        signature_similarity,
    )
    write_report(
        output_dir,
        runs,
        len(shared_ids),
        model_inventory,
        model_metrics,
        groups,
        alignment_effects,
        coder_effects,
        head_summary,
        hard_samples,
        warnings,
        plot_paths,
    )

    required_csv = [
        "model_inventory.csv",
        "model_metrics.csv",
        "group_summary.csv",
        "paired_alignment_effects.csv",
        "paired_coder_effects.csv",
        "attention_mass_by_label.csv",
        "attention_delta_by_depth.csv",
        "head_selection_summary.csv",
        "head_overlap.csv",
        "sample_risk_correlation.csv",
        "hard_samples_cross_family.csv",
    ]
    output_errors = validate_outputs(output_dir, required_csv, plot_paths)
    if output_errors:
        for error in output_errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print(f"Wrote internal attention Phase2 analysis to {output_dir}")
    print(f"Runs: {len(runs)}")
    print(f"Shared prompts: {len(shared_ids)}")
    print(f"CSV files: {len(tables)}")
    print(f"PNG files: {len(plot_paths)}")
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()

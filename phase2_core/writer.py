import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class Phase2Writer:
    def __init__(self, output_dir: str | Path, run_id: str):
        self.run_dir = Path(output_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_rows: list[dict[str, Any]] = []
        self.model_rows: list[dict[str, Any]] = []
        self.span_rows: list[dict[str, Any]] = []
        self.generation_rows: list[dict[str, Any]] = []
        self.behavior_rows: list[dict[str, Any]] = []
        self.score_rows: list[dict[str, Any]] = []
        self.attention_arrays: dict[str, list[np.ndarray]] = {}
        self._saw_injection_summary = False

    @staticmethod
    def _require_pandas():
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pandas and pyarrow are required for Phase2 parquet outputs. "
                "Install dependencies with `uv sync`."
            ) from exc
        return pd

    @staticmethod
    def _mean(summary: dict[str, np.ndarray], key: str) -> float | None:
        value = summary.get(key)
        if value is None:
            return None
        return float(np.nanmean(value))

    def add_sample(
        self,
        example,
        spans,
        decoded_spans: dict[str, str],
        details: dict[str, Any],
        attention_summary: dict[str, np.ndarray],
        model_output: dict[str, Any],
        seed: int,
        params: dict[str, Any],
    ) -> None:
        self.prompt_rows.append(
            {
                "prompt_id": example.prompt_id,
                "base_request_id": example.base_request_id,
                "task_type": example.task_type,
                "split": example.split,
                "messages_json": _json([message.to_dict() for message in example.messages]),
                "labels_json": _json(example.labels),
                "metadata_json": _json(example.metadata),
                "span_candidates_json": _json(example.span_candidates),
            }
        )

        for name, span in spans.items():
            self.span_rows.append(
                {
                    "prompt_id": example.prompt_id,
                    "span_name": name,
                    "start": int(span.start),
                    "end": int(span.end),
                    "source": span.source,
                    "decoded_text": decoded_spans.get(name, ""),
                }
            )

        self.generation_rows.append(
            {
                "prompt_id": example.prompt_id,
                "text": model_output.get("text", ""),
                "tokens_json": _json(model_output.get("tokens", [])),
                "max_new_tokens": int(params.get("max_output_tokens", 1)),
                "temperature": params.get("temperature"),
                "top_p": params.get("top_p"),
                "seed": int(seed),
            }
        )

        self.behavior_rows.append(
            {
                "prompt_id": example.prompt_id,
                "refusal": example.labels.get("refusal"),
                "attack_success": example.labels.get("injection_success"),
                "compliance_score": example.labels.get("compliance_score"),
                "judge_model": example.labels.get("judge_model"),
                "judge_prompt_version": example.labels.get("judge_prompt_version"),
                "human_verified": example.labels.get("human_verified"),
                "labels_json": _json(example.labels),
            }
        )

        ratio = attention_summary["normalized_instruction_ratio"]
        selected_count = int(details["selected_head_summary"]["count"])
        self.score_rows.append(
            {
                "prompt_id": example.prompt_id,
                "focus_score": float(details["focus_score"]),
                "threshold": float(details["threshold"]),
                "pred": bool(details["prediction"]),
                "selected_head_count": selected_count,
                "selected_head_ratio": (
                    selected_count / float(ratio.shape[0] * ratio.shape[1])
                    if ratio.size
                    else 0.0
                ),
                "mean_instruction_mass": self._mean(attention_summary, "instruction_mass"),
                "mean_data_mass": self._mean(attention_summary, "data_mass"),
                "mean_injection_mass": self._mean(attention_summary, "injection_mass"),
            }
        )

        self._append_attention_arrays(attention_summary)

    def _append_attention_arrays(self, summary: dict[str, np.ndarray]) -> None:
        base_keys = (
            "instruction_mass",
            "data_mass",
            "normalized_instruction_ratio",
            "ratio",
            "entropy",
        )
        for key in base_keys:
            self.attention_arrays.setdefault(key, []).append(summary[key].astype(np.float32))

        optional_keys = ("injection_mass", "normalized_injection_ratio")
        has_injection = "injection_mass" in summary
        self._saw_injection_summary = self._saw_injection_summary or has_injection
        shape = summary["ratio"].shape
        missing = np.full(shape, np.nan, dtype=np.float32)
        for key in optional_keys:
            value = summary[key].astype(np.float32) if has_injection else missing
            self.attention_arrays.setdefault(key, []).append(value)

    def finalize(self, model_config: dict[str, Any], config_path: str | Path, run_id: str) -> dict[str, Any]:
        pd = self._require_pandas()

        if self.attention_arrays:
            ratio_shape = self.attention_arrays["ratio"][0].shape
            layers, heads = int(ratio_shape[0]), int(ratio_shape[1])
        else:
            layers, heads = None, None

        self.model_rows.append(
            {
                "run_id": run_id,
                "model": model_config["model_info"]["name"],
                "model_id": model_config["model_info"]["model_id"],
                "provider": model_config["model_info"]["provider"],
                "config_path": str(config_path),
                "layers": layers,
                "heads": heads,
                "params_json": _json(model_config.get("params", {})),
            }
        )

        tables = {
            "prompts.parquet": self.prompt_rows,
            "model_metadata.parquet": self.model_rows,
            "token_spans.parquet": self.span_rows,
            "generation_outputs.parquet": self.generation_rows,
            "behavior_labels.parquet": self.behavior_rows,
            "attention_tracker_scores.parquet": self.score_rows,
        }
        for filename, rows in tables.items():
            pd.DataFrame(rows).to_parquet(self.run_dir / filename, index=False)

        attention_path = None
        if self.attention_arrays:
            from safetensors.torch import save_file

            tensors = {}
            for key, arrays in self.attention_arrays.items():
                if key in {"injection_mass", "normalized_injection_ratio"} and not self._saw_injection_summary:
                    continue
                tensors[key] = torch.from_numpy(np.stack(arrays, axis=0)).to(torch.float32)
            attention_path = self.run_dir / "attention_summary.safetensors"
            save_file(tensors, attention_path)

        return {
            "run_dir": str(self.run_dir),
            "prompts": len(self.prompt_rows),
            "attention_summary_path": str(attention_path) if attention_path else None,
            "tables": sorted(tables.keys()),
        }

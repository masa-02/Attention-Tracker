from pathlib import Path
from typing import Any

import numpy as np


def write_selected_heads_parquet(
    output_dir: str | Path,
    run_id: str,
    model_config: dict[str, Any],
    config_path: str | Path,
    normal_mean: np.ndarray,
    normal_std: np.ndarray,
    attack_mean: np.ndarray,
    attack_std: np.ndarray,
    selected_heads: list[list[int]] | None,
) -> str:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas and pyarrow are required for Phase2 parquet outputs. "
            "Install dependencies with `uv sync`."
        ) from exc

    selected = {tuple(pair) for pair in (selected_heads or [])}
    layers, heads = normal_mean.shape
    rows = []
    for layer in range(layers):
        normalized_depth = layer / (layers - 1) if layers > 1 else 0.0
        for head in range(heads):
            margin = float(
                normal_mean[layer, head]
                - attack_mean[layer, head]
                - (normal_std[layer, head] + attack_std[layer, head])
            )
            rows.append(
                {
                    "model": model_config["model_info"]["name"],
                    "model_id": model_config["model_info"]["model_id"],
                    "config_path": str(config_path),
                    "layer": int(layer),
                    "head": int(head),
                    "normalized_depth": float(normalized_depth),
                    "normal_mean": float(normal_mean[layer, head]),
                    "normal_std": float(normal_std[layer, head]),
                    "attack_mean": float(attack_mean[layer, head]),
                    "attack_std": float(attack_std[layer, head]),
                    "margin": margin,
                    "selected": (layer, head) in selected,
                }
            )

    artifact_dir = Path(output_dir) / run_id / "calibration_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "selected_heads.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return str(path)

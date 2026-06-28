from __future__ import annotations

import numpy as np
import torch


def compute_attention_summary(attention_maps, rng, injection_range=None):
    """Aggregate per-[layer, head] attention statistics for Phase2 outputs.

    The canonical Phase2 metric name is ``normalized_instruction_ratio``.
    ``ratio`` is kept as a legacy alias because the original Attention Tracker
    detector and existing tests use that name.
    """
    epsilon = 1e-8
    if not attention_maps:
        return {}
    step = attention_maps[0]
    num_layers = len(step)
    num_heads = step[0].shape[1]

    instruction_mass = np.zeros((num_layers, num_heads))
    data_mass = np.zeros((num_layers, num_heads))
    normalized_instruction_ratio = np.zeros((num_layers, num_heads))
    entropy = np.zeros((num_layers, num_heads))
    injection_mass = (
        np.zeros((num_layers, num_heads)) if injection_range is not None else None
    )
    normalized_injection_ratio = (
        np.zeros((num_layers, num_heads)) if injection_range is not None else None
    )

    for i, attn_layer in enumerate(step):
        attn_layer = attn_layer.to(torch.float32).numpy()
        full = attn_layer[0, :, -1, :]  # [H, T_key]

        inst = np.sum(full[:, rng[0][0]:rng[0][1]], axis=1)
        data = np.sum(full[:, rng[1][0]:rng[1][1]], axis=1)
        instruction_mass[i, :] = inst
        data_mass[i, :] = data
        normalized_instruction_ratio[i, :] = inst / (inst + data + epsilon)

        total = np.sum(full, axis=1, keepdims=True)
        probs = full / (total + epsilon)
        layer_entropy = -np.sum(probs * np.log(probs + epsilon), axis=1)
        entropy[i, :] = layer_entropy / np.log(full.shape[1] + epsilon)

        if injection_mass is not None:
            inj = np.sum(full[:, injection_range[0]:injection_range[1]], axis=1)
            injection_mass[i, :] = inj
            normalized_injection_ratio[i, :] = inj / (inst + inj + epsilon)

    normalized_instruction_ratio = np.nan_to_num(
        normalized_instruction_ratio, nan=0.0
    )
    summary = {
        "instruction_mass": np.nan_to_num(instruction_mass, nan=0.0),
        "data_mass": np.nan_to_num(data_mass, nan=0.0),
        "normalized_instruction_ratio": normalized_instruction_ratio,
        "ratio": normalized_instruction_ratio,
        "entropy": np.nan_to_num(entropy, nan=0.0),
    }
    if injection_mass is not None:
        summary["injection_mass"] = np.nan_to_num(injection_mass, nan=0.0)
        summary["normalized_injection_ratio"] = np.nan_to_num(
            normalized_injection_ratio, nan=0.0
        )
    return summary


class AttentionSummaryExtractor:
    def extract(self, attention_maps, spans):
        instruction_span = spans["system_instruction"]
        data_span = spans["untrusted_data"]
        injection_span = spans.get("injection_instruction")

        summary = compute_attention_summary(
            attention_maps,
            ((instruction_span.start, instruction_span.end), (data_span.start, data_span.end)),
            injection_range=(
                (injection_span.start, injection_span.end)
                if injection_span is not None
                else None
            ),
        )
        return summary

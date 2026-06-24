"""Equivalence test: compute_attention_summary vs the legacy process_attn path.

GPU-free. Verifies that the full-[L,H] ratio array reproduces, for the important
heads, exactly what process_attn + calc_head_scores would score. Run with:

    uv run python -m pytest tests/test_attention_summary.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector.utils import process_attn, calc_head_scores, compute_attention_summary


def _make_step(num_layers, num_heads, num_keys, seed=0):
    """List (len L) of attention tensors shaped [1, H, 1, T] (post get_last_attn)."""
    torch.manual_seed(seed)
    step = []
    for _ in range(num_layers):
        attn = torch.rand(1, num_heads, 1, num_keys)
        step.append(attn)
    return step


def test_ratio_matches_process_attn_for_important_heads():
    num_layers, num_heads, num_keys = 6, 4, 12
    step = _make_step(num_layers, num_heads, num_keys)
    attention_maps = [step]
    rng = ((1, 4), (6, 10))  # (instruction span, data span)

    summary = compute_attention_summary(attention_maps, rng)
    heatmap = process_attn(step, rng, "normalize_sum")

    heads = [(0, 0), (2, 1), (5, 3), (3, 2)]
    head_scores = calc_head_scores(heatmap, heads)
    for item in head_scores:
        l, h = item["layer"], item["head"]
        assert np.isclose(summary["ratio"][l, h], item["score"], atol=1e-6), (
            f"ratio[{l},{h}]={summary['ratio'][l, h]} != process_attn score {item['score']}"
        )


def test_shapes_and_entropy_bounds():
    num_layers, num_heads, num_keys = 5, 3, 9
    summary = compute_attention_summary([_make_step(num_layers, num_heads, num_keys)], ((0, 2), (3, 6)))
    for key in ("instruction_mass", "data_mass", "ratio", "entropy"):
        assert summary[key].shape == (num_layers, num_heads)
    assert summary["ratio"].min() >= 0.0 and summary["ratio"].max() <= 1.0
    assert summary["entropy"].min() >= 0.0 and summary["entropy"].max() <= 1.0 + 1e-6


def test_injection_mass_optional():
    summary_no = compute_attention_summary([_make_step(3, 2, 8)], ((0, 2), (3, 6)))
    assert "injection_mass" not in summary_no
    summary_yes = compute_attention_summary([_make_step(3, 2, 8)], ((0, 2), (3, 6)), injection_range=(6, 8))
    assert summary_yes["injection_mass"].shape == (3, 2)


if __name__ == "__main__":
    test_ratio_matches_process_attn_for_important_heads()
    test_shapes_and_entropy_bounds()
    test_injection_mass_optional()
    print("All attention summary equivalence tests passed.")

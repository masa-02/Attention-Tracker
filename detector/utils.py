import torch 
import numpy as np
from phase2_core.attention import compute_attention_summary as _phase2_compute_attention_summary

def process_attn(attention, rng, attn_func):
    heatmap = np.zeros((len(attention), attention[0].shape[1]))
    for i, attn_layer in enumerate(attention):
        attn_layer = attn_layer.to(torch.float32).numpy()

        if "sum" in attn_func:
            last_token_attn_to_inst = np.sum(attn_layer[0, :, -1, rng[0][0]:rng[0][1]], axis=1)
            attn = last_token_attn_to_inst
        
        elif "max" in attn_func:
            last_token_attn_to_inst = np.max(attn_layer[0, :, -1, rng[0][0]:rng[0][1]], axis=1)
            attn = last_token_attn_to_inst

        else: raise NotImplementedError
            
        last_token_attn_to_inst_sum = np.sum(attn_layer[0, :, -1, rng[0][0]:rng[0][1]], axis=1)
        last_token_attn_to_data_sum = np.sum(attn_layer[0, :, -1, rng[1][0]:rng[1][1]], axis=1)

        if "normalize" in attn_func:
            epsilon = 1e-8
            heatmap[i, :] = attn / (last_token_attn_to_inst_sum + last_token_attn_to_data_sum + epsilon)
        else:
            heatmap[i, :] = attn

    heatmap = np.nan_to_num(heatmap, nan=0.0)

    return heatmap


def calc_attn_score(heatmap, heads):
    if not heads:
        raise ValueError("important_heads is empty. Run head selection or set params.important_heads before detection.")
    score = np.mean([heatmap[l, h] for l, h in heads], axis=0)
    return score


def calc_head_scores(heatmap, heads):
    if not heads:
        raise ValueError("important_heads is empty. Run head selection or set params.important_heads before detection.")
    return [
        {
            "layer": int(layer),
            "head": int(head),
            "score": float(heatmap[layer, head]),
        }
        for layer, head in heads
    ]


def compute_attention_summary(attention_maps, rng, injection_range=None):
    """Backward-compatible wrapper for the Phase2 attention summary extractor."""
    return _phase2_compute_attention_summary(attention_maps, rng, injection_range)


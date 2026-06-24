import torch 
import numpy as np

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
    """Aggregate per-[layer, head] attention statistics from the last query token.

    Unlike ``process_attn``/``calc_head_scores`` which only score important heads,
    this returns full ``[L, H]`` arrays so cross-model head analyses are possible
    (plan section 5.2 / 9.4). Computed from the first generation step only, matching
    the detector's ``use_token="first"`` behaviour.

    Args:
    - attention_maps: list over generation steps; each is a list over layers of
      tensors shaped ``[1, H, 1, T_key]`` (after ``get_last_attn``).
    - rng: ((inst_start, inst_end), (data_start, data_end)) span ranges, exactly as
      consumed by ``process_attn`` (negative offsets allowed).
    - injection_range: optional (start, end) for an injected-instruction span. When
      None (current datasets have no injection annotation) injection_mass is omitted.

    Returns a dict of np.ndarray with shape ``[L, H]``: instruction_mass, data_mass,
    ratio (= inst / (inst + data + eps), plan section 9.4), entropy (normalized to
    [0, 1]); injection_mass is included only when injection_range is given.
    """
    epsilon = 1e-8
    if not attention_maps:
        return {}
    step = attention_maps[0]
    num_layers = len(step)
    num_heads = step[0].shape[1]

    instruction_mass = np.zeros((num_layers, num_heads))
    data_mass = np.zeros((num_layers, num_heads))
    ratio = np.zeros((num_layers, num_heads))
    entropy = np.zeros((num_layers, num_heads))
    injection_mass = (
        np.zeros((num_layers, num_heads)) if injection_range is not None else None
    )

    for i, attn_layer in enumerate(step):
        attn_layer = attn_layer.to(torch.float32).numpy()
        full = attn_layer[0, :, -1, :]  # [H, T_key]

        inst = np.sum(full[:, rng[0][0]:rng[0][1]], axis=1)
        data = np.sum(full[:, rng[1][0]:rng[1][1]], axis=1)
        instruction_mass[i, :] = inst
        data_mass[i, :] = data
        ratio[i, :] = inst / (inst + data + epsilon)

        total = np.sum(full, axis=1, keepdims=True)
        probs = full / (total + epsilon)
        layer_entropy = -np.sum(probs * np.log(probs + epsilon), axis=1)
        entropy[i, :] = layer_entropy / np.log(full.shape[1] + epsilon)

        if injection_mass is not None:
            injection_mass[i, :] = np.sum(
                full[:, injection_range[0]:injection_range[1]], axis=1
            )

    summary = {
        "instruction_mass": np.nan_to_num(instruction_mass, nan=0.0),
        "data_mass": np.nan_to_num(data_mass, nan=0.0),
        "ratio": np.nan_to_num(ratio, nan=0.0),
        "entropy": np.nan_to_num(entropy, nan=0.0),
    }
    if injection_mass is not None:
        summary["injection_mass"] = np.nan_to_num(injection_mass, nan=0.0)
    return summary


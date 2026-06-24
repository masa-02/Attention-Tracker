import torch
import torch.nn.functional as F


def _model_key(model_name, model_id):
    return f"{model_name} {model_id}".lower()


def tokenizer_kwargs_for_model(model_name, model_id):
    model_key = _model_key(model_name, model_id)
    kwargs = {"trust_remote_code": True}

    if "mistral" in model_key:
        kwargs["fix_mistral_regex"] = True

    if "gemma-4" in model_key or "gemma4" in model_key:
        kwargs["extra_special_tokens"] = {"video_token": "<|video|>"}

    return kwargs


def model_kwargs_for_model(model_name, model_id, device):
    model_key = _model_key(model_name, model_id)
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device,
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }

    # DeepSeek-V2-Lite ships older remote modeling code that calls the removed
    # DynamicCache.get_usable_length API. Current Transformers has a native
    # DeepseekV2 implementation, so prefer it over trust_remote_code.
    if "deepseek-v2" in model_key or "deepseek_v2" in model_key:
        kwargs["trust_remote_code"] = False

    return kwargs


def causal_model_class_for_model(model_name, model_id):
    model_key = _model_key(model_name, model_id)

    if "mistral-small-3.2" in model_key or "mistral3" in model_key:
        from transformers.models.mistral3 import Mistral3ForConditionalGeneration

        return Mistral3ForConditionalGeneration

    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


def get_last_attn(attn_map):
    for i, layer in enumerate(attn_map):
        attn_map[i] = layer[:, :, -1, :].unsqueeze(2)

    return attn_map

def sample_token(logits, top_k=None, top_p=None, temperature=1.0):
    # Optionally apply temperature
    logits = logits / temperature

    # Apply top-k sampling
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))  # Ensure top_k <= vocab size
        values, indices = torch.topk(logits, top_k)
        probs = F.softmax(values, dim=-1)
        next_token_id = indices[torch.multinomial(probs, 1)]

        return next_token_id

    return logits.argmax(dim=-1).squeeze()

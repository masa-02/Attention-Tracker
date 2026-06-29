import torch
import torch.nn.functional as F
from importlib.metadata import PackageNotFoundError


def _model_key(model_name, model_id):
    return f"{model_name} {model_id}".lower()


def _hub_kwargs_from_loading_config(loading_config):
    loading_config = loading_config or {}
    kwargs = {}
    for key in ("cache_dir", "local_files_only", "revision", "token"):
        if key in loading_config and loading_config[key] is not None:
            kwargs[key] = loading_config[key]
    return kwargs


def tokenizer_kwargs_for_model(model_name, model_id, loading_config=None):
    model_key = _model_key(model_name, model_id)
    kwargs = {"trust_remote_code": True}
    kwargs.update(_hub_kwargs_from_loading_config(loading_config))

    if "mistral" in model_key:
        kwargs["fix_mistral_regex"] = True

    if "gemma-4" in model_key or "gemma4" in model_key:
        kwargs["extra_special_tokens"] = {"video_token": "<|video|>"}

    return kwargs


def _dtype_from_name(value, default=torch.bfloat16):
    if value is None:
        return default
    normalized = str(value).lower()
    if normalized in {"auto", "none"}:
        return "auto" if normalized == "auto" else None
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "16bit"}:
        return torch.float16
    if normalized in {"fp32", "float32", "32bit"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _bitsandbytes_config(loading_config):
    quantization = str(loading_config.get("quantization", "none")).lower()
    if quantization in {"none", "no", "false", "0", "16bit", "bf16", "fp16"}:
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "Transformers BitsAndBytesConfig is required for quantized model loading. "
            "Install dependencies with `uv sync`."
        ) from exc

    if quantization in {"8bit", "int8", "8"}:
        try:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=bool(
                    loading_config.get("llm_int8_enable_fp32_cpu_offload", False)
                ),
            )
        except PackageNotFoundError as exc:
            raise ModuleNotFoundError(
                "bitsandbytes is required for 8bit model loading. Run `uv sync` "
                "after updating pyproject.toml."
            ) from exc

    if quantization in {"4bit", "nf4", "int4", "4"}:
        try:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_dtype_from_name(
                    loading_config.get("compute_dtype", loading_config.get("dtype", "bfloat16"))
                ),
                bnb_4bit_quant_type=str(loading_config.get("quant_type", "nf4")),
                bnb_4bit_use_double_quant=bool(loading_config.get("double_quant", True)),
            )
        except PackageNotFoundError as exc:
            raise ModuleNotFoundError(
                "bitsandbytes is required for 4bit model loading. Run `uv sync` "
                "after updating pyproject.toml."
            ) from exc

    raise ValueError(f"Unsupported quantization mode: {loading_config.get('quantization')}")


def model_kwargs_for_model(model_name, model_id, device, loading_config=None):
    loading_config = loading_config or {}
    model_key = _model_key(model_name, model_id)
    quantization_config = _bitsandbytes_config(loading_config)
    dtype = _dtype_from_name(loading_config.get("dtype", "bfloat16"))
    device_map = loading_config.get("device_map")
    if device_map is None:
        device_map = "auto" if quantization_config is not None else device

    kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    kwargs.update(_hub_kwargs_from_loading_config(loading_config))
    if dtype is not None:
        kwargs["dtype"] = dtype
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config

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

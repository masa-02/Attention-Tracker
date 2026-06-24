import json
from pathlib import Path

from models.attn_model import AttentionModel
from models.attn_model_nsys import AttentionModelNoSys


PROJECT_ROOT = Path(__file__).resolve().parent


def _yaml_module():
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required for YAML configs. Install dependencies with `uv sync` "
            "or `pip install pyyaml`."
        ) from exc
    return yaml


def open_config(config_path):
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            yaml = _yaml_module()
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config extension: {config_path}")
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def write_config(config_path, config):
    config_path = Path(config_path)
    with config_path.open("w", encoding="utf-8") as f:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            yaml = _yaml_module()
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        elif config_path.suffix.lower() == ".json":
            json.dump(config, f, indent=4)
            f.write("\n")
        else:
            raise ValueError(f"Unsupported config extension: {config_path}")


def _candidate_paths(value):
    path = Path(value)
    roots = [Path.cwd(), PROJECT_ROOT]

    if path.is_absolute():
        base_paths = [path]
    else:
        base_paths = [root / path for root in roots]
        base_paths.extend([
            PROJECT_ROOT / "configs" / "runtime" / path,
            PROJECT_ROOT / "configs" / "model_configs" / path,
            PROJECT_ROOT / "configs" / "model_configs" / f"{path}_config",
        ])

    candidates = []
    for base in base_paths:
        if base.suffix:
            candidates.append(base)
        else:
            candidates.extend([
                base,
                base.with_suffix(".yml"),
                base.with_suffix(".yaml"),
                base.with_suffix(".json"),
            ])
    return candidates


def resolve_config_path(model_name=None, config=None):
    if config:
        candidates = _candidate_paths(config)
    elif model_name:
        candidates = _candidate_paths(f"configs/model_configs/{model_name}_config.json")
    else:
        raise ValueError("Either --config or --model_name must be specified.")

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    checked = "\n".join(str(path) for path in candidates[:12])
    raise FileNotFoundError(f"Config not found. Checked:\n{checked}")


def load_runtime_config(model_name=None, config=None):
    config_path = resolve_config_path(model_name=model_name, config=config)
    return open_config(config_path), config_path


def create_model(config):
    provider = config["model_info"]["provider"].lower()
    if provider == 'attn-hf':
        model = AttentionModel(config)
    elif provider == 'attn-hf-no-sys':
        model = AttentionModelNoSys(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")
    return model

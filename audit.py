import json
from pathlib import Path

import numpy as np
import yaml


def dataset_path(dataset_name):
    return Path(*str(dataset_name).split("/"))


def default_run_id(model_name, seed):
    return f"{model_name}-{seed}"


def resolve_audit_dir(output_dir, dataset_name, run_id, audit_dir=None):
    if audit_dir:
        return Path(audit_dir)
    return Path(output_dir) / dataset_path(dataset_name) / "runs" / run_id


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")


def write_yaml(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_jsonable(payload), f, sort_keys=False, allow_unicode=True)

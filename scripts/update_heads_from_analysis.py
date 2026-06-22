import argparse
import ast
import json
from pathlib import Path


def parse_heads_by_model(analysis_path, select_k):
    heads_by_model = {}
    current_model = None
    waiting_for_heads = False

    for raw_line in analysis_path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("===== ") and line.endswith(" ====="):
            current_model = line.strip("= ").strip()
            waiting_for_heads = False
            continue

        if line == f"======== index pos (n={select_k}) =========":
            waiting_for_heads = True
            continue

        if waiting_for_heads and (line.startswith("[[") or line == "[]"):
            heads_by_model[current_model] = ast.literal_eval(line)
            waiting_for_heads = False

    return heads_by_model


def update_config(config_path, heads):
    with config_path.open() as f:
        config = json.load(f)

    config["params"]["important_heads"] = heads

    with config_path.open("w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")


def main(args):
    project_root = Path(__file__).resolve().parents[1]
    analysis_path = Path(args.analysis_file)
    if not analysis_path.is_absolute():
        analysis_path = project_root / analysis_path

    heads_by_model = parse_heads_by_model(analysis_path, args.select_k)
    if not heads_by_model:
        raise ValueError(f"No heads found for n={args.select_k} in {analysis_path}")

    for model_name, heads in heads_by_model.items():
        config_path = project_root / "configs" / "model_configs" / f"{model_name}_config.json"
        if not config_path.exists():
            print(f"Skip {model_name}: config not found at {config_path}")
            continue
        update_config(config_path, heads)
        print(f"Updated {config_path}: {len(heads)} heads from n={args.select_k}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update important_heads from select_head.py analysis output.")
    parser.add_argument("analysis_file", nargs="?", default="analysis_llama_family.txt")
    parser.add_argument("--select_k", default=4, type=int)
    main(parser.parse_args())

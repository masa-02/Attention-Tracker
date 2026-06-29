#!/usr/bin/env python
import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_runtime_config


def _repo_cache_name(model_id: str) -> str | None:
    if not model_id or "/" not in model_id:
        return None
    return "models--" + model_id.replace("/", "--")


def _candidate_cache_roots(config: dict) -> list[Path]:
    loading = config.get("model_loading", {})
    roots: list[Path] = []

    configured_cache = loading.get("cache_dir")
    if configured_cache:
        root = Path(str(configured_cache)).expanduser()
        roots.append(root)
        if root.name != "hub":
            roots.append(root / "hub")

    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        roots.append(Path(hf_hub_cache).expanduser())

    transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
    if transformers_cache:
        roots.append(Path(transformers_cache).expanduser())

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")

    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    seen = set()
    unique_roots = []
    for root in roots:
        resolved = root.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_roots.append(resolved)
    return unique_roots


def _safe_child(root: Path, child: Path) -> bool:
    try:
        child.relative_to(root)
        return True
    except ValueError:
        return False


def _remove_path(path: Path, dry_run: bool) -> str:
    if not path.exists():
        return "missing"
    if dry_run:
        return "dry-run"
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)
    return "removed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove one model's Hugging Face cache directory.")
    parser.add_argument("--config", required=True, help="Runtime YAML/JSON config path or stem.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-locks", action="store_true", default=True)
    args = parser.parse_args()

    config, config_path = load_runtime_config(config=args.config)
    model_id = config["model_info"]["model_id"]
    repo_name = _repo_cache_name(model_id)
    if repo_name is None:
        print(f"HF cache cleanup skip: model_id is not a Hub repo id: {model_id}")
        return 0

    print(f"HF cache cleanup for {model_id} ({config_path})")
    removed_any = False
    for root in _candidate_cache_roots(config):
        targets = [root / repo_name]
        if args.include_locks:
            targets.append(root / ".locks" / repo_name)

        for target in targets:
            target = target.resolve(strict=False)
            if not _safe_child(root, target):
                print(f"  skip unsafe target outside cache root: {target}")
                continue
            status = _remove_path(target, args.dry_run)
            if status in {"removed", "dry-run"}:
                removed_any = True
            print(f"  {status}: {target}")

    if not removed_any:
        print("  no cache entries found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

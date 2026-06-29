# Attention-Tracker Scripts

Use `scripts/common/` for reusable runners and `scripts/groups/<group>/` for experiment entry points.

## Common Runners

- `common/select_heads.sh`: run `select_head.py` for every config in a manifest and update `important_heads`.
- `common/run_dataset.sh`: run `run_dataset.py` for every config in a manifest.
- `common/pipeline_config_list.sh`: run head selection, then dataset evaluation.
- `common/pipeline_single_config.sh`: run head selection, dataset evaluation, and attention-map rendering for one YAML config.
- `common/render_config_list.sh`: render attention maps for every config in a manifest.
- `common/smoke_query.sh`: run one query with `run.py`.

## Model Groups

Each group script pins one manifest under `configs/runtime/manifests/` and delegates to `scripts/common/`.

- `groups/qwen/`
- `groups/llama/`
- `groups/gemma/`
- `groups/mistral/`
- `groups/deepseek/`
- `groups/moonlight/`
- `groups/core/`
- `groups/domain/`
- `groups/pilot/`
- `groups/all/`

Examples:

```bash
./scripts/groups/qwen/phase1.sh qwen-phase1
./scripts/groups/qwen/phase2.sh qwen-phase2
./scripts/groups/llama/select_heads.sh llm 30 analysis_llama_phase1_heads.txt 4
./scripts/groups/llama/run_dataset.sh deepset/prompt-injections 0
./scripts/groups/gemma/render_attention_maps.sh deepset 100 render/outputs/gemma
```

Keep new experiment entry points under `groups/<group>/`. Put only reusable implementation in `common/`.

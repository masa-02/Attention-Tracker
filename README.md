# Attention Tracker: Detecting Prompt Injection Attacks in LLMs

Welcome to the official repository for **"Attention Tracker: Detecting Prompt Injection Attacks in LLMs"**. This repository provides scripts and tools to identify important attention heads and evaluate prompt injection attacks on large language models (LLMs).

Project page: https://huggingface.co/spaces/TrustSafeAI/Attention-Tracker 

Paper: https://arxiv.org/abs/2411.00348 

---

## Features
- **Identify Important Heads**: Determine which attention heads are critical for detecting prompt injection attacks.
- **Run Experiments**: Execute experiments on datasets to evaluate the model's effectiveness.
- **Test Queries**: Test individual queries against your chosen model.

---

## Getting Started

### Prerequisites
1. Ensure you have Python and `uv` installed.
2. Install the dependencies from `pyproject.toml`:

   ```bash
   uv sync
   ```

   Alternatively, install the legacy dependency list:

   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Find important heads for the default models:

   ```bash
   ./scripts/find_heads.sh
   ```

2. To manually specify important heads, edit `configs/model_configs/{model_name}_config.json` and update `["params"]["important_heads"]`.

3. Run experiments on the [DeepSet Prompt Injection Dataset](https://huggingface.co/datasets/deepset/prompt-injections?row=19):

   ```bash
   ./scripts/run_dataset.sh
   ```

4. Test an individual query:

   ```bash
   uv run python run.py --model_name {model} --test_query "{query you want to test}"
   ```

### Important Head Selection

`select_head.py` can now write the selected heads back to the model config automatically.

```bash
uv run python3 -u select_head.py \
  --model_name llama3_8b-attn \
  --num_data 30 \
  --dataset llm \
  --select_k 4 \
  --update_config
```

New and commonly used parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `--model_name` | `qwen2-attn` | Model config stem under `configs/model_configs/{model_name}_config.json`. |
| `--num_data` | `10` | Number of normal samples and attack samples used for averaging. For example, `30` means 30 normal and 30 attack samples. |
| `--dataset` | none | Selection dataset. Supported values are `llm` and `deepset`. |
| `--select_k` | `4` | Threshold level used when selecting important heads. The analysis output prints this as `n={select_k}`. Larger values are stricter and usually select fewer heads. |
| `--update_config` | off | Update `params.important_heads` in the target model config using the heads selected at `--select_k`. |

If head selection has already been run and only the config update is needed, parse the analysis log instead of rerunning model inference:

```bash
uv run python scripts/update_heads_from_analysis.py analysis_llama_family.txt --select_k 4
```

### Llama Family

The Llama family configs are:

| Config name | Hugging Face model id |
| --- | --- |
| `llama3_8b-attn` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `llama3_70b-attn` | `meta-llama/Meta-Llama-3-70B-Instruct` |
| `llama3_1_8b-attn` | `meta-llama/Llama-3.1-8B-Instruct` |
| `llama3_2_1b-attn` | `meta-llama/Llama-3.2-1B-Instruct` |
| `llama3_2_3b-attn` | `meta-llama/Llama-3.2-3B-Instruct` |

`llama3_70b-attn` is configured but commented out in the default family scripts because it can exceed available GPU memory. Uncomment it in the script only when the environment has enough memory and model access.

Find and write important heads:

```bash
./scripts/find_llama_family_heads.sh [dataset] [num_data] [output_file] [select_k]
```

Defaults:

| Argument | Default | Description |
| --- | --- | --- |
| `dataset` | `llm` | Dataset passed to `select_head.py --dataset`. |
| `num_data` | `30` | Number of normal and attack samples used by head selection. |
| `output_file` | `analysis_llama_family.txt` | Analysis log path. |
| `select_k` | `4` | Threshold level passed to `select_head.py --select_k`. |

Render attention map figures:

```bash
./scripts/plot_llama_family_attention_maps.sh [dataset] [num_data] [output_dir]
```

Defaults are `dataset=deepset`, `num_data=100`, and `output_dir=render/outputs/llama_family`.

Run dataset evaluation:

```bash
./scripts/run_llama_family_dataset.sh [dataset_name] [seed]
```

Defaults are `dataset_name=deepset/prompt-injections` and `seed=0`. Models whose `important_heads` are empty are skipped.

### Qwen Family

The Qwen family configs are:

| Config name | Hugging Face model id |
| --- | --- |
| `qwen2_5_1_5b-attn` | `Qwen/Qwen2.5-1.5B` |
| `qwen2_5_3b-attn` | `Qwen/Qwen2.5-3B` |
| `qwen2_5_7b-attn` | `Qwen/Qwen2.5-7B` |
| `qwen2_5_14b-attn` | `Qwen/Qwen2.5-14B` |
| `qwen2_5_32b-attn` | `Qwen/Qwen2.5-32B` |
| `qwen3_32b-attn` | `Qwen/Qwen3-32B` |
| `qwen3_14b-attn` | `Qwen/Qwen3-14B` |
| `qwen3_8b-attn` | `Qwen/Qwen3-8B` |
| `qwen3_4b-attn` | `Qwen/Qwen3-4B` |
<!-- Qwen3.5 configs are kept for later work, but they are commented out in the default Qwen family scripts.
| `qwen3_5_4b-attn` | `Qwen/Qwen3.5-4B` |
| `qwen3_5_9b-attn` | `Qwen/Qwen3.5-9B` |
| `qwen3_5_27b-attn` | `Qwen/Qwen3.5-27B` |
-->

Find and write important heads:

```bash
./scripts/find_qwen_family_heads.sh [dataset] [num_data] [output_file] [select_k]
```

Defaults:

| Argument | Default | Description |
| --- | --- | --- |
| `dataset` | `llm` | Dataset passed to `select_head.py --dataset`. |
| `num_data` | `30` | Number of normal and attack samples used by head selection. |
| `output_file` | `analysis_qwen_family.txt` | Analysis log path. |
| `select_k` | `4` | Threshold level passed to `select_head.py --select_k`. |

Render attention map figures:

```bash
./scripts/plot_qwen_family_attention_maps.sh [dataset] [num_data] [output_dir]
```

Defaults are `dataset=deepset`, `num_data=100`, and `output_dir=render/outputs/qwen_family`.

Run dataset evaluation:

```bash
./scripts/run_qwen_family_dataset.sh [dataset_name] [seed]
```

Defaults are `dataset_name=deepset/prompt-injections` and `seed=0`. Models whose `important_heads` are empty are skipped.

### Attention Map Rendering

Render paper-style attention maps for one or more models:

```bash
./scripts/plot_attention_maps.sh [model_name|all] [dataset] [num_data] [output_dir]
```

Defaults are `model_name=all`, `dataset=deepset`, `num_data=100`, and `output_dir=render/outputs`. Outputs are written under `output_dir/{model_name}` and include:

- `{model}_{dataset}_normal_heads.png`
- `{model}_{dataset}_attack_heads.png`
- `{model}_{dataset}_diff_heads.png`
- `{model}_{dataset}_normal_tokens_layers.png`
- `{model}_{dataset}_attack_tokens_layers.png`
- `{model}_{dataset}_attention_maps.npz`

`render/plot_attention_maps.py` also supports `--instruction`, whose default is `Say xxxxxx`.

### License
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en)

### Citation
```
@misc{hung2024attentiontrackerdetectingprompt,
      title={Attention Tracker: Detecting Prompt Injection Attacks in LLMs}, 
      author={Kuo-Han Hung and Ching-Yun Ko and Ambrish Rawat and I-Hsin Chung and Winston H. Hsu and Pin-Yu Chen},
      year={2024},
      eprint={2411.00348},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2411.00348}, 
}
```

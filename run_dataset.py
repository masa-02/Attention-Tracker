import argparse
import os
import json
import random
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from audit import append_jsonl, dataset_path, default_run_id, resolve_audit_dir, write_json, write_yaml
from utils import create_model, load_runtime_config
from detector.attn import AttentionDetector
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from phase2_core import (
    AttentionSummaryExtractor,
    Phase2Writer,
    TokenSpan,
    TokenSpanMapper,
    attention_fields_from_example,
    deepset_to_prompt_example,
    load_prompt_jsonl,
    prompt_label_to_int,
)

ATTENTION_SUMMARY_VERSION = "at-attn-summary-v1"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _safe_auc(labels, scores):
    if len(set(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)

def _safe_auprc(labels, scores):
    if len(set(labels)) < 2:
        return float("nan")
    return average_precision_score(labels, scores)

def _legacy_span(trace, name, key):
    start, end = trace[key]
    return TokenSpan(start=int(start), end=int(end), source="legacy_trace")


def _resolve_phase2_spans(span_mapper, trace, input_ids, sample_instruction, data_text, injection, strict_spans):
    if strict_spans:
        return span_mapper.resolve_attention_tracker_spans(
            input_ids=input_ids,
            instruction=sample_instruction,
            data=data_text,
            injection=injection,
        )

    return {
        "system_instruction": _legacy_span(trace, "system_instruction", "instruction_range_abs"),
        "untrusted_data": _legacy_span(trace, "untrusted_data", "data_range_abs"),
        "last_input_token": TokenSpan(
            start=max(len(input_ids) - 1, 0),
            end=len(input_ids),
            source="derived",
        ),
        "assistant_prefix": TokenSpan(
            start=max(len(input_ids) - 1, 0),
            end=len(input_ids),
            source="derived",
        ),
    }


def _process_phase2_sample(
    *,
    record,
    details,
    detector,
    attention_maps,
    span_mapper,
    phase2_extractor,
    phase2_writer,
    strict_spans,
    sample_instruction,
    data_text,
    seed,
    params,
):
    if attention_maps is None:
        raise RuntimeError("Phase2 extraction requires attention maps")
    trace = details["trace"]
    input_ids = trace.get("input_ids")
    if not input_ids:
        raise RuntimeError("Phase2 strict span mapping requires trace.input_ids")

    spans = _resolve_phase2_spans(
        span_mapper=span_mapper,
        trace=trace,
        input_ids=input_ids,
        sample_instruction=sample_instruction,
        data_text=data_text,
        injection=record["fields"].get("injection"),
        strict_spans=strict_spans,
    )
    strict_range = (
        (spans["system_instruction"].start, spans["system_instruction"].end),
        (spans["untrusted_data"].start, spans["untrusted_data"].end),
    )
    details.update(detector.attn2details(attention_maps, strict_range))
    details["threshold"] = float(detector.threshold)
    detect = bool(details["focus_score"] <= details["threshold"])
    details["attention_summary"] = phase2_extractor.extract(attention_maps, spans)
    details["trace"]["phase2_span_source"] = "strict_subsequence" if strict_spans else "legacy_trace"
    details["prediction"] = bool(detect)

    decoded_spans = {
        name: span_mapper.decode_span(input_ids, span)
        for name, span in spans.items()
    }
    phase2_writer.add_sample(
        example=record["example"],
        spans=spans,
        decoded_spans=decoded_spans,
        details=details,
        attention_summary=details["attention_summary"],
        model_output=details["model_output"],
        seed=seed,
        params=params,
    )
    return detect

def main(args):
    model_config, config_path = load_runtime_config(
        model_name=args.model_name if not args.config else None,
        config=args.config,
    )
    runtime_config = model_config.get("runtime", {})
    audit_config = model_config.get("audit", {})
    phase2_config = model_config.get("phase2", {})

    prompt_jsonl = args.prompt_jsonl or phase2_config.get("prompt_jsonl")
    dataset_name = args.dataset_name if args.dataset_name is not None else runtime_config.get("dataset_name", "deepset/prompt-injections")
    if prompt_jsonl and args.dataset_name is None:
        dataset_name = f"prompt-jsonl/{Path(prompt_jsonl).stem}"
    seed = args.seed if args.seed is not None else int(runtime_config.get("seed", 0))
    instruction = args.instruction if args.instruction is not None else runtime_config.get("instruction", "Say xxxxxx")
    output_dir = args.output_dir if args.output_dir is not None else runtime_config.get("output_dir", "result")
    audit_enabled = bool(args.audit_log or audit_config.get("enabled", False))
    attn_summary_enabled = bool(args.attn_summary or audit_config.get("attention_summary", False))
    phase2_enabled = bool(args.phase2 or phase2_config.get("enabled", False))
    phase2_output_dir = args.phase2_output_dir or phase2_config.get("output_dir", "outputs/phase2")
    strict_spans = args.strict_spans if args.strict_spans is not None else bool(phase2_config.get("strict_spans", True))
    model_name = model_config["model_info"]["name"] if args.config else args.model_name
    run_id = args.run_id or audit_config.get("run_id") or default_run_id(model_name, seed)
    audit_dir = args.audit_dir or audit_config.get("audit_dir")

    set_seed(seed)

    output_logs = Path(output_dir) / dataset_path(dataset_name) / f"{model_name}-{seed}.json"
    output_result = Path(output_dir) / dataset_path(dataset_name) / "result.jsonl"

    model = create_model(config=model_config)
    model.print_model_info()

    if prompt_jsonl:
        prompt_examples = load_prompt_jsonl(prompt_jsonl)
        test_records = [
            {
                "example": example,
                "fields": attention_fields_from_example(example, instruction),
                "label": prompt_label_to_int(example),
                "text": attention_fields_from_example(example, instruction)["data"],
            }
            for example in prompt_examples
        ]
    else:
        dataset = load_dataset(dataset_name)
        test_data = dataset['test']
        test_records = []
        for sample_idx, data in enumerate(test_data):
            example = deepset_to_prompt_example(data, sample_idx, instruction, split="test")
            fields = attention_fields_from_example(example, instruction)
            test_records.append({
                "example": example,
                "fields": fields,
                "label": int(data["label"]),
                "text": data["text"],
            })
    
    detector = AttentionDetector(model, instruction=instruction)
    print("===================")
    print(f"Using detector: {detector.name}")

    labels, predictions, scores = [], [], []
    logs = []
    audit_run_dir = None
    samples_path = None
    attn_summary_accum = {"instruction_mass": [], "data_mass": [], "ratio": [], "entropy": []}
    attn_summary_index = []
    attn_summary_label = []
    attn_summary_span_source = []
    phase2_writer = Phase2Writer(phase2_output_dir, run_id) if phase2_enabled else None
    phase2_extractor = AttentionSummaryExtractor() if phase2_enabled else None
    span_mapper = TokenSpanMapper(model.tokenizer) if phase2_enabled else None
    if audit_enabled or attn_summary_enabled or phase2_enabled:
        audit_run_dir = resolve_audit_dir(output_dir, dataset_name, run_id, audit_dir=audit_dir)
        audit_run_dir.mkdir(parents=True, exist_ok=True)
    if audit_enabled or attn_summary_enabled or phase2_enabled:
        if audit_enabled:
            samples_path = audit_run_dir / "samples.jsonl"
            samples_path.write_text("", encoding="utf-8")
        write_yaml(audit_run_dir / "config_snapshot.yaml", {
            "config_path": str(config_path),
            "config": model_config,
            "runtime": {
                "dataset_name": dataset_name,
                "seed": seed,
                "instruction": instruction,
                "output_dir": output_dir,
            },
            "audit": {
                "enabled": audit_enabled,
                "attention_summary": attn_summary_enabled,
                "run_id": run_id,
                "audit_dir": str(audit_run_dir),
            },
            "phase2": {
                "enabled": phase2_enabled,
                "output_dir": phase2_output_dir,
                "prompt_jsonl": prompt_jsonl,
                "strict_spans": strict_spans,
            },
        })

    for sample_idx, record in enumerate(tqdm(test_records)):
        fields = record["fields"]
        sample_instruction = fields["instruction"] or instruction
        data_text = fields["data"]
        detector.instruction = sample_instruction
        detect, details = detector.detect(
            data_text,
            return_trace=(audit_enabled or phase2_enabled),
            return_full=(attn_summary_enabled and not phase2_enabled),
            return_attention=phase2_enabled,
        )
        attention_maps = details.pop("_attention_maps", None)

        if phase2_enabled:
            detect = _process_phase2_sample(
                record=record,
                details=details,
                detector=detector,
                attention_maps=attention_maps,
                span_mapper=span_mapper,
                phase2_extractor=phase2_extractor,
                phase2_writer=phase2_writer,
                strict_spans=strict_spans,
                sample_instruction=sample_instruction,
                data_text=data_text,
                seed=seed,
                params=model_config.get("params", {}),
            )

        details["prediction"] = bool(detect)
        score = details['focus_score']

        labels.append(record['label'])
        predictions.append(detect)
        scores.append(1-score)

        if attn_summary_enabled and details.get("attention_summary"):
            summary = details["attention_summary"]
            for key in attn_summary_accum:
                attn_summary_accum[key].append(summary[key])
            attn_summary_index.append(sample_idx)
            attn_summary_label.append(record['label'])
            span_source = details.get("trace", {}).get("span_source") if audit_enabled else None
            if phase2_enabled:
                span_source = details.get("trace", {}).get("phase2_span_source", span_source)
            attn_summary_span_source.append(span_source if span_source is not None else "")

        legacy_result = [detect, {"focus_score": score}]
        result_data = {
            "text": record['text'],
            "label": record['label'],
            "result": legacy_result
        }

        logs.append(result_data)
        if audit_enabled:
            append_jsonl(samples_path, {
                "sample_index": sample_idx,
                "prompt_id": record["example"].prompt_id,
                "text": record["text"],
                "label": record["label"],
                "prediction": bool(detect),
                "focus_score": score,
                "threshold": details["threshold"],
                "selected_head_summary": details["selected_head_summary"],
                "selected_head_scores": details["selected_head_scores"],
                "trace": details["trace"],
            })

    auc_score = _safe_auc(labels, scores)
    auprc_score = _safe_auprc(labels, scores)

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    auc_score = round(auc_score, 3)
    auprc_score = round(auprc_score, 3)
    fnr = round(fnr, 3)
    fpr = round(fpr, 3)

    print(f"AUC Score: {auc_score}; AUPRC Score: {auprc_score}; FNR: {fnr}; FPR: {fpr}")
    
    os.makedirs(os.path.dirname(output_logs), exist_ok=True)
    with open(output_logs, "w", encoding="utf-8") as f_out:
        f_out.write(json.dumps({"result": logs}, indent=4))

    os.makedirs(os.path.dirname(output_result), exist_ok=True)
    with open(output_result, "a", encoding="utf-8") as f_out:
        f_out.write(json.dumps({
            "model": model_name,
            "seed": seed,
            "auc": auc_score,
            "auprc": auprc_score,
            "fnr": fnr,
            "fpr": fpr
        }) + "\n")

    attention_summary_info = None
    if attn_summary_enabled and attn_summary_index:
        attn_summary_path = audit_run_dir / "attention_summary.npz"
        stacked = {key: np.stack(vals, axis=0) for key, vals in attn_summary_accum.items()}
        np.savez_compressed(
            attn_summary_path,
            sample_index=np.array(attn_summary_index),
            label=np.array(attn_summary_label),
            span_source=np.array(attn_summary_span_source),
            **stacked,
        )
        attention_summary_info = {
            "path": str(attn_summary_path),
            "metrics": list(attn_summary_accum.keys()),
            "shape": list(stacked["ratio"].shape),
            "extraction_version": ATTENTION_SUMMARY_VERSION,
        }

    phase2_info = None
    if phase2_writer is not None:
        phase2_info = phase2_writer.finalize(
            model_config=model_config,
            config_path=config_path,
            run_id=run_id,
        )

    if audit_enabled or attn_summary_enabled or phase2_enabled:
        write_json(audit_run_dir / "summary.json", {
            "model": model_name,
            "model_id": model_config["model_info"]["model_id"],
            "model_loading": model_config.get("model_loading", {}),
            "config_path": str(config_path),
            "dataset_name": dataset_name,
            "seed": seed,
            "instruction": instruction,
            "run_id": run_id,
            "num_samples": len(logs),
            "metrics": {
                "auc": auc_score,
                "auprc": auprc_score,
                "fnr": fnr,
                "fpr": fpr,
            },
            "legacy_result_path": str(output_logs),
            "legacy_summary_path": str(output_result),
            "samples_path": str(samples_path) if samples_path else None,
            "attention_summary": attention_summary_info,
            "phase2": phase2_info,
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt Injection Detection Script")
    
    parser.add_argument("--config", type=str, default=None,
                        help="Path or stem for a JSON/YAML runtime config.")
    parser.add_argument("--model_name", type=str, default="qwen2-attn",
                        help="Path to the model configuration file.")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Path to the dataset.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--audit-log", action="store_true")
    parser.add_argument("--attn-summary", action="store_true",
                        help="Write full [L,H] attention summary arrays to attention_summary.npz")
    parser.add_argument("--audit-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--phase2", action="store_true",
                        help="Write Phase2 parquet/safetensors artifacts.")
    parser.add_argument("--phase2-output-dir", type=str, default=None)
    parser.add_argument("--prompt-jsonl", type=str, default=None,
                        help="Phase2 PromptExample JSONL. When omitted, DeepSet rows are converted.")
    parser.add_argument("--strict-spans", action="store_true", default=None,
                        help="Resolve token spans with strict subsequence matching for Phase2 outputs.")
    
    args = parser.parse_args()

    main(args)

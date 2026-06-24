import numpy as np
from tqdm import tqdm

from .utils import process_attn, calc_attn_score, calc_head_scores, compute_attention_summary


class AttentionDetector():
    def __init__(self, model, pos_examples=None, neg_examples=None, use_token="first", instruction="Say xxxxxx", threshold=0.5):
        self.name = "attention"
        self.attn_func = "normalize_sum"
        self.model = model
        self.important_heads = model.important_heads
        if not self.important_heads:
            raise ValueError("important_heads is empty. Run head selection or set params.important_heads before detection.")
        self.instruction = instruction
        self.use_token = use_token
        self.threshold = threshold
        if pos_examples and neg_examples:
            pos_scores, neg_scores = [], []
            for prompt in tqdm(pos_examples, desc="pos_examples"):
                _, _, attention_maps, _, input_range, generated_probs = self.model.inference(
                    self.instruction, prompt, max_output_tokens=1
                )
                pos_scores.append(self.attn2score(attention_maps, input_range))

            for prompt in tqdm(neg_examples, desc="neg_examples"):
                _, _, attention_maps, _, input_range, generated_probs = self.model.inference(
                    self.instruction, prompt, max_output_tokens=1
                )
                neg_scores.append(self.attn2score(attention_maps, input_range))

            self.threshold = (np.mean(pos_scores) + np.mean(neg_scores)) / 2

        if pos_examples and not neg_examples:
            pos_scores = []
            for prompt in tqdm(pos_examples, desc="pos_examples"):
                _, _, attention_maps, _, input_range, generated_probs = self.model.inference(
                    self.instruction, prompt, max_output_tokens=1
                )
                pos_scores.append(self.attn2score(attention_maps, input_range))

            self.threshold = np.mean(pos_scores) - 4 * np.std(pos_scores)

    def attn2score(self, attention_maps, input_range):
        details = self.attn2details(attention_maps, input_range)
        return details["focus_score"]

    def attn2details(self, attention_maps, input_range):
        if not attention_maps:
            return {
                "focus_score": 0.0,
                "selected_head_scores": [],
                "selected_head_summary": {
                    "count": 0,
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                },
            }

        if self.use_token == "first":
            attention_maps = [attention_maps[0]]

        scores = []
        selected_scores = []
        for attention_map in attention_maps:
            heatmap = process_attn(
                attention_map, input_range, self.attn_func)
            score = calc_attn_score(heatmap, self.important_heads)
            scores.append(score)
            if not selected_scores:
                selected_scores = calc_head_scores(heatmap, self.important_heads)

        focus_score = sum(scores) if len(scores) > 0 else 0
        if selected_scores:
            head_values = [item["score"] for item in selected_scores]
            selected_head_summary = {
                "count": len(selected_scores),
                "mean": float(np.mean(head_values)),
                "min": float(np.min(head_values)),
                "max": float(np.max(head_values)),
            }
        else:
            selected_head_summary = {
                "count": 0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        return {
            "focus_score": float(focus_score),
            "selected_head_scores": selected_scores,
            "selected_head_summary": selected_head_summary,
        }

    def detect(self, data_prompt, return_trace=False, return_full=False):
        output = self.model.inference(
            self.instruction, data_prompt, max_output_tokens=1, return_trace=return_trace)
        if return_trace:
            _, _, attention_maps, _, input_range, _, trace = output
        else:
            _, _, attention_maps, _, input_range, _ = output

        details = self.attn2details(attention_maps, input_range)
        details["threshold"] = float(self.threshold)
        if return_trace:
            details["trace"] = trace
        if return_full:
            details["attention_summary"] = compute_attention_summary(attention_maps, input_range)
        return bool(details["focus_score"] <= self.threshold), details

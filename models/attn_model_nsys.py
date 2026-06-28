import torch
from .model import Model
import torch.nn.functional as F
from transformers import AutoTokenizer
from .utils import (
    causal_model_class_for_model,
    model_kwargs_for_model,
    sample_token,
    tokenizer_kwargs_for_model,
)
from phase2_core.spans import iter_text_token_candidates, select_subsequence_match

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AttentionModelNoSys(Model):
    def __init__(self, config):
        super().__init__(config)
        self.name = config["model_info"]["name"]
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        model_id = config["model_info"]["model_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            **tokenizer_kwargs_for_model(self.name, model_id),
        )
        model_class = causal_model_class_for_model(self.name, model_id)
        self.model = model_class.from_pretrained(
            model_id,
            **model_kwargs_for_model(self.name, model_id, device),
        ).eval()
        if config["params"]["important_heads"] == "all":
            attn_size = self.get_map_dim()
            self.important_heads = [[i, j] for i in range(
                attn_size[0]) for j in range(attn_size[1])]
        else:
            self.important_heads = config["params"]["important_heads"]
        
        self.top_k = 50
        self.top_p = None

    def get_map_dim(self):
        _, _, attention_maps, _, _, _ = self.inference("print hi", "")
        attention_map = attention_maps[0]
        return len(attention_map), attention_map[0].shape[1]

    @staticmethod
    def _find_token_span(input_ids, target_ids, occurrence="first"):
        return select_subsequence_match(input_ids, target_ids, occurrence=occurrence)

    def _find_text_span(self, input_ids, text, occurrence="first"):
        for _, target_ids in iter_text_token_candidates(self.tokenizer, text):
            span = self._find_token_span(input_ids, target_ids, occurrence=occurrence)
            if span is not None:
                return span
        return None

    def _dynamic_data_range(self, input_ids, instruction, data):
        instruction_range = self._find_text_span(input_ids, instruction, occurrence="first")
        data_range = self._find_text_span(input_ids, data, occurrence="last")
        if instruction_range is not None and data_range is not None:
            return (instruction_range, data_range), "subsequence"

        prefixed_data_range = self._find_text_span(input_ids, "Data: " + data, occurrence="last")
        if instruction_range is not None and prefixed_data_range is not None:
            return (instruction_range, prefixed_data_range), "subsequence_prefixed_data"

        return None

    @staticmethod
    def _absolute_range(rng, length):
        start, end = rng
        if start < 0:
            start = length + start
        if end < 0:
            end = length + end
        return max(start, 0), min(end, length)

    def _build_trace(self, text, input_ids, input_tokens, data_range, span_source, generated_tokens,
                     generated_probs, attention_maps, nonfinite_attention_count):
        token_count = len(input_tokens)
        instruction_range = data_range[0]
        untrusted_data_range = data_range[1]
        instruction_abs = self._absolute_range(instruction_range, token_count)
        data_abs = self._absolute_range(untrusted_data_range, token_count)
        attention_shape = None
        if attention_maps and attention_maps[0]:
            attention_shape = list(attention_maps[0][0].shape)

        return {
            "rendered_prompt": text,
            "input_ids": [int(token_id) for token_id in input_ids],
            "input_tokens": list(input_tokens),
            "input_token_count": token_count,
            "instruction_range": list(instruction_range),
            "data_range": list(untrusted_data_range),
            "instruction_range_abs": list(instruction_abs),
            "data_range_abs": list(data_abs),
            "instruction_tokens": list(input_tokens[instruction_abs[0]:instruction_abs[1]]),
            "data_tokens": list(input_tokens[data_abs[0]:data_abs[1]]),
            "span_source": span_source,
            "generated_token_ids": [int(token) for token in generated_tokens],
            "generated_tokens": [
                self.tokenizer.decode(token, skip_special_tokens=True)
                for token in generated_tokens
            ],
            "generated_probs": [float(prob) for prob in generated_probs],
            "attention_shape": attention_shape,
            "attention_steps": len(attention_maps),
            "nonfinite_attention_count": int(nonfinite_attention_count),
        }

    def inference(self, instruction, data, max_output_tokens=None, return_trace=False):
        data = "Data: " + data
        messages = [
            {"role": "user", "content": instruction + "\n" + data},
        ]

        chat_template_kwargs = {}
        if "gemma-4" in self.name or "gemma4" in self.name:
            chat_template_kwargs["enable_thinking"] = False

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )

        instruction_len = len(self.tokenizer.encode(instruction))
        data_len = len(self.tokenizer.encode(data))

        model_inputs = self.tokenizer(
            [text], return_tensors="pt").to(self.model.device)
        input_tokens = self.tokenizer.convert_ids_to_tokens(
            model_inputs['input_ids'][0])

        dynamic_range = self._dynamic_data_range(model_inputs["input_ids"][0].tolist(), instruction, data)
        if dynamic_range is not None:
            data_range, span_source = dynamic_range
        elif "gemma" in self.name:
            data_range = ((5, 5+instruction_len), (-4-data_len, -5))
            span_source = "fixed_offset"
        else:
            raise ValueError(
                f"Could not resolve instruction/data spans for {self.name}. "
                "Use a model name with a known fallback or inspect the chat template."
            )

        generated_tokens = []
        generated_probs = []  # Store probabilities of generated tokens
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask

        # Ensure attention maps are stored minimally to save memory
        attention_maps = []
        nonfinite_attention_count = 0

        if max_output_tokens != None:
            n_tokens = max_output_tokens
        else:
            n_tokens = self.max_output_tokens

        with torch.no_grad():  # Use no_grad to reduce memory usage
            for i in range(n_tokens):
                # Forward pass, optionally in half precision (mixed precision)
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    use_cache=False,
                )

                # Extract logits and compute probabilities
                logits = output.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                # next_token_id = logits.argmax(dim=-1).squeeze()
                next_token_id = sample_token(
                    logits[0], top_k=self.top_k, top_p=None, temperature=1.0)[0]

                generated_probs.append(probs[0, next_token_id.item()].item())
                generated_tokens.append(next_token_id.item())

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

                # Update input_ids and attention_mask for the next iteration
                input_ids = torch.cat(
                    (input_ids, next_token_id.unsqueeze(0).unsqueeze(0)), dim=-1)
                attention_mask = torch.cat(
                    (attention_mask, torch.tensor([[1]], device=input_ids.device)), dim=-1)

                # Detach attention maps early to reduce memory
                nonfinite_attention_count += sum(
                    int((~torch.isfinite(attention)).sum().item())
                    for attention in output['attentions']
                )
                attention_map = [attention.detach().cpu().half()
                                 for attention in output['attentions']]
                attention_map = [torch.nan_to_num(
                    attention, nan=0.0) for attention in attention_map]
                attention_maps.append(attention_map)

        output_tokens = [self.tokenizer.decode(
            token, skip_special_tokens=True) for token in generated_tokens]
        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True)

        if return_trace:
            trace = self._build_trace(
                text,
                model_inputs["input_ids"][0].tolist(),
                input_tokens,
                data_range,
                span_source,
                generated_tokens,
                generated_probs,
                attention_maps,
                nonfinite_attention_count,
            )
            return generated_text, output_tokens, attention_maps, input_tokens, data_range, generated_probs, trace

        return generated_text, output_tokens, attention_maps, input_tokens, data_range, generated_probs

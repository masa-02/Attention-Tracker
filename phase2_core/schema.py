import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class PromptSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class PromptMessage:
    role: str
    name: str
    content: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptMessage":
        missing = [key for key in ("role", "name", "content") if key not in data]
        if missing:
            raise PromptSchemaError(f"message is missing required keys: {missing}")
        if not all(isinstance(data[key], str) for key in ("role", "name", "content")):
            raise PromptSchemaError("message role/name/content must be strings")
        return cls(role=data["role"], name=data["name"], content=data["content"])

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "name": self.name, "content": self.content}


@dataclass(frozen=True)
class PromptExample:
    prompt_id: str
    base_request_id: str
    task_type: str
    split: str
    messages: list[PromptMessage]
    labels: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    span_candidates: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], line_no: int | None = None) -> "PromptExample":
        context = f" on line {line_no}" if line_no is not None else ""
        required = ("prompt_id", "base_request_id", "task_type", "split", "messages", "labels")
        missing = [key for key in required if key not in data]
        if missing:
            raise PromptSchemaError(f"prompt example{context} is missing required keys: {missing}")

        for key in ("prompt_id", "base_request_id", "task_type", "split"):
            if not isinstance(data[key], str) or not data[key]:
                raise PromptSchemaError(f"{key}{context} must be a non-empty string")

        if not isinstance(data["messages"], list) or not data["messages"]:
            raise PromptSchemaError(f"messages{context} must be a non-empty list")
        messages = [PromptMessage.from_dict(message) for message in data["messages"]]

        labels = data["labels"]
        if not isinstance(labels, dict):
            raise PromptSchemaError(f"labels{context} must be an object")

        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise PromptSchemaError(f"metadata{context} must be an object")

        span_candidates = data.get("span_candidates") or {}
        if not isinstance(span_candidates, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in span_candidates.items()
        ):
            raise PromptSchemaError(f"span_candidates{context} must be an object of string values")

        return cls(
            prompt_id=data["prompt_id"],
            base_request_id=data["base_request_id"],
            task_type=data["task_type"],
            split=data["split"],
            messages=messages,
            labels=labels,
            metadata=metadata,
            span_candidates=span_candidates,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "base_request_id": self.base_request_id,
            "task_type": self.task_type,
            "split": self.split,
            "messages": [message.to_dict() for message in self.messages],
            "labels": self.labels,
            "metadata": self.metadata,
            "span_candidates": self.span_candidates,
        }

    def message_by_name(self, name: str) -> PromptMessage | None:
        for message in self.messages:
            if message.name == name:
                return message
        return None


def load_prompt_jsonl(path: str | Path) -> list[PromptExample]:
    path = Path(path)
    examples: list[PromptExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PromptSchemaError(f"invalid JSON on line {line_no}: {exc}") from exc
            examples.append(PromptExample.from_dict(raw, line_no=line_no))
    return examples


def deepset_to_prompt_example(sample: dict[str, Any], index: int, instruction: str, split: str) -> PromptExample:
    label = int(sample["label"])
    text = str(sample["text"])
    return PromptExample(
        prompt_id=f"deepset-{split}-{index:06d}",
        base_request_id=f"deepset-{split}-{index:06d}",
        task_type="prompt_injection",
        split=split,
        messages=[
            PromptMessage(role="system", name="system_instruction", content=instruction),
            PromptMessage(role="user", name="untrusted_data", content=text),
        ],
        labels={
            "injection_present": bool(label),
            "instruction_conflict": bool(label),
            "injection_success": None,
            "harmful_intent": False,
            "jailbreak_present": False,
            "jailbreak_success": None,
        },
        metadata={
            "source": "deepset/prompt-injections",
            "original_label": label,
        },
        span_candidates={},
    )


def prompt_label_to_int(example: PromptExample) -> int:
    labels = example.labels
    return int(bool(labels.get("injection_present") or labels.get("instruction_conflict")))


def attention_fields_from_example(example: PromptExample, default_instruction: str) -> dict[str, str | None]:
    instruction_message = (
        example.message_by_name("system_instruction")
        or example.message_by_name("trusted_user_instruction")
    )
    data_message = example.message_by_name("untrusted_data")

    if instruction_message is None:
        instruction = default_instruction
    else:
        instruction = instruction_message.content

    if data_message is None:
        user_messages = [message for message in example.messages if message.role == "user"]
        if not user_messages:
            raise PromptSchemaError(f"{example.prompt_id}: no untrusted_data or user message found")
        data = user_messages[-1].content
    else:
        data = data_message.content

    injection = example.span_candidates.get("injection_instruction")
    if injection is None:
        injection_message = example.message_by_name("injection_instruction")
        injection = injection_message.content if injection_message is not None else None

    return {"instruction": instruction, "data": data, "injection": injection}

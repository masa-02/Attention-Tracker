from dataclasses import dataclass
from typing import Iterable


class SpanMappingError(ValueError):
    pass


class AmbiguousSpanError(SpanMappingError):
    pass


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int
    source: str = "subsequence"

    def to_list(self) -> list[int]:
        return [int(self.start), int(self.end)]


def find_subsequence_matches(input_ids: list[int], target_ids: list[int]) -> list[tuple[int, int]]:
    if not target_ids:
        return []
    width = len(target_ids)
    return [
        (index, index + width)
        for index in range(len(input_ids) - width + 1)
        if input_ids[index:index + width] == target_ids
    ]


def select_subsequence_match(
    input_ids: list[int],
    target_ids: list[int],
    occurrence: str = "first",
) -> tuple[int, int] | None:
    matches = find_subsequence_matches(input_ids, target_ids)
    if not matches:
        return None
    if occurrence == "last":
        return matches[-1]
    return matches[0]


def iter_text_token_candidates(tokenizer, text: str) -> Iterable[tuple[str, list[int]]]:
    candidates = [text]
    if text and not text.startswith(" "):
        candidates.append(" " + text)
    seen: set[tuple[int, ...]] = set()
    for candidate in candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        yield candidate, ids


class TokenSpanMapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def _find_matches(input_ids: list[int], target_ids: list[int]) -> list[tuple[int, int]]:
        return find_subsequence_matches(input_ids, target_ids)

    def _candidate_token_ids(self, text: str) -> Iterable[tuple[str, list[int]]]:
        yield from iter_text_token_candidates(self.tokenizer, text)

    def find_unique_span(self, input_ids: list[int], text: str, field_name: str) -> TokenSpan:
        for _, target_ids in self._candidate_token_ids(text):
            matches = self._find_matches(input_ids, target_ids)
            if not matches:
                continue
            if len(matches) > 1:
                raise AmbiguousSpanError(
                    f"Ambiguous token span for {field_name!r}: {sorted(matches)}"
                )
            start, end = matches[0]
            return TokenSpan(start=start, end=end)

        raise SpanMappingError(f"Could not resolve token span for {field_name!r}")

    def resolve_attention_tracker_spans(
        self,
        input_ids: list[int],
        instruction: str,
        data: str,
        injection: str | None = None,
    ) -> dict[str, TokenSpan]:
        spans = {
            "system_instruction": self.find_unique_span(
                input_ids, instruction, "system_instruction"
            ),
            "untrusted_data": self.find_unique_span(input_ids, data, "untrusted_data"),
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
        if injection:
            spans["injection_instruction"] = self.find_unique_span(
                input_ids, injection, "injection_instruction"
            )
        return spans

    def decode_span(self, input_ids: list[int], span: TokenSpan) -> str:
        return self.tokenizer.decode(
            input_ids[span.start:span.end],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

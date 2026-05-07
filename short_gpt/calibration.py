from typing import List, Optional

from datasets import load_dataset


def load_calibration_sentences(
    dataset_name: str = "mteb/stsbenchmark-sts",
    split: str = "train",
    text_columns: Optional[List[str]] = None,
    max_sentences: Optional[int] = 10_000,
    trust_remote_code: bool = True,
) -> List[str]:
    """
    Load a flat list of sentences from a HuggingFace dataset for use as
    calibration data in eval_importance().

    Defaults to STS-B (sentence pairs covering diverse topics, small and fast to load).
    Other good choices:
      - "snli"                   text_columns=["premise", "hypothesis"]
      - "multi_nli"              text_columns=["premise", "hypothesis"]
      - "ms_marco", "v2.1"      text_columns=["query"]
      - "mteb/stsbenchmark-sts"  text_columns=["sentence1", "sentence2"]  (default)

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to load ("train", "validation", "test").
        text_columns: Column name(s) to extract sentences from.
                      If None, auto-detects from known datasets.
        max_sentences: Cap on the number of sentences returned. None = no cap.
        trust_remote_code: Passed to load_dataset().

    Returns:
        List of sentence strings.
    """
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=trust_remote_code)

    if text_columns is None:
        text_columns = _infer_text_columns(dataset.column_names)

    sentences: List[str] = []
    for col in text_columns:
        sentences.extend(dataset[col])

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in sentences:
        if isinstance(s, str) and s.strip() and s not in seen:
            seen.add(s)
            unique.append(s.strip())

    if max_sentences is not None:
        unique = unique[:max_sentences]

    return unique


def _infer_text_columns(column_names: List[str]) -> List[str]:
    """Heuristically pick text columns from known column name patterns."""
    priority = [
        # STS-style
        ["sentence1", "sentence2"],
        # NLI-style
        ["premise", "hypothesis"],
        # Retrieval-style
        ["query", "passage"],
        # Generic
        ["text"],
        ["sentence"],
    ]
    for candidate in priority:
        if all(c in column_names for c in candidate):
            return candidate

    # Fallback: return the first string-typed column found
    for col in column_names:
        return [col]

    raise ValueError(f"Cannot infer text columns from: {column_names}. Pass text_columns explicitly.")

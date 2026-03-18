import re
import string
import unicodedata


def clean_text(text: str) -> str:
    """Lowercase, strip HTML tags, remove extra whitespace."""
    if not isinstance(text, str):
        return ""
    # Normalise unicode (e.g. curly quotes → ASCII)
    text = unicodedata.normalize("NFKD", text)
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E]", " ", text)
    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Remove all punctuation from a string."""
    return text.translate(str.maketrans("", "", string.punctuation))


def count_sentences(text: str) -> int:
    """Rough sentence count using punctuation heuristics."""
    sentences = re.split(r"[.!?]+", text)
    return max(1, len([s for s in sentences if s.strip()]))


def count_words(text: str) -> int:
    """Word count after basic cleaning."""
    return len(clean_text(text).split())


def truncate(text: str, max_words: int = 500) -> str:
    """Truncate to max_words for faster NLP processing."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns default instead of ZeroDivisionError."""
    if denominator == 0:
        return default
    return numerator / denominator
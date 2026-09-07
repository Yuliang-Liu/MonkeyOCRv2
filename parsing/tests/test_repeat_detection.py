import random

import pytest

from core_runner import detect_repeat_token


def _reference_detect_repeat_token(
    predicted_tokens: str,
    base_max_repeats: int = 4,
    window_size: int = 500,
    cut_from_end: int = 0,
    scaling_factor: float = 3.0,
) -> bool:
    """Reference implementation of the original backward-scan algorithm."""
    if cut_from_end > 0:
        predicted_tokens = predicted_tokens[:-cut_from_end]
    for seq_len in range(1, window_size // 2 + 1):
        candidate_seq = predicted_tokens[-seq_len:]
        max_repeats = int(base_max_repeats * (1 + scaling_factor / seq_len))
        repeat_count = 0
        pos = len(predicted_tokens) - seq_len
        if pos < 0:
            continue
        while pos >= 0:
            if predicted_tokens[pos : pos + seq_len] == candidate_seq:
                repeat_count += 1
                pos -= seq_len
            else:
                break
        if repeat_count > max_repeats:
            return True
    return False


@pytest.mark.parametrize(
    "value",
    ["", "abc", "A" * 20, "AB" * 20, "prefix-" + "xyz" * 30, "A" * 19 + "B"],
)
def test_repeat_detector_matches_reference(value):
    assert detect_repeat_token(value) == _reference_detect_repeat_token(value)


def test_repeat_detector_matches_reference_for_random_inputs():
    rng = random.Random(0)
    alphabet = "abcXYZ012"
    for _ in range(200):
        value = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 200)))
        cut = rng.randrange(0, 60)
        window = rng.randrange(2, 80)
        assert detect_repeat_token(value, window_size=window, cut_from_end=cut) == (
            _reference_detect_repeat_token(value, window_size=window, cut_from_end=cut)
        )

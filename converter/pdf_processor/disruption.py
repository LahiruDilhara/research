import random
import string
from typing import Optional


class LayoutDisruptor:
    """
    Generates jumbled, scrambled, or expanded character text streams for layout disruption.
    When overlaid as an invisible text layer (render_mode=3), copying and pasting text from the PDF
    results in garbled, unreadable character sequences while maintaining 100% visual legibility on screen.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def disrupt_word(self, word_text: str, mode: str = "shuffle", length_multiplier: float = 1.5) -> str:
        """
        Creates a disrupted version of word_text by shuffling, reversing, or expanding letters.
        """
        if len(word_text) <= 2:
            base = word_text[::-1]
        elif mode == "reverse":
            base = word_text[::-1]
        else:
            letters = list(word_text)
            random.shuffle(letters)
            if "".join(letters) == word_text and len(word_text) > 2:
                letters[0], letters[-1] = letters[-1], letters[0]
            base = "".join(letters)

        target_len = max(len(word_text), int(len(word_text) * length_multiplier))
        if len(base) < target_len:
            extra_chars = [random.choice(string.ascii_letters) for _ in range(target_len - len(base))]
            base += "".join(extra_chars)

        return base

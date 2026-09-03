import random
from typing import List, Optional


class ZeroWidthInjector:
    """
    Injects invisible zero-width Unicode characters inside words to break tokenization
    and string matching without changing the visual appearance of the text.
    """

    # Pool of invisible zero-width Unicode characters
    DEFAULT_ZERO_WIDTH_CHARS = [
        "\u200B",  # ZERO WIDTH SPACE (ZWSP)
        "\u200C",  # ZERO WIDTH NON-JOINER (ZWNJ)
        "\u200D",  # ZERO WIDTH JOINER (ZWJ)
        "\u2060",  # WORD JOINER (WJ)
        "\uFEFF",  # ZERO WIDTH NO-BREAK SPACE (ZWNBSP)
    ]

    def __init__(self, char_pool: Optional[List[str]] = None, seed: Optional[int] = None):
        self.char_pool = char_pool if char_pool else self.DEFAULT_ZERO_WIDTH_CHARS
        if seed is not None:
            random.seed(seed)

    def inject_into_word(self, word_text: str, zw_count: int = 2) -> str:
        """
        Inserts specified number of random zero-width characters between letters of word_text.
        """
        if len(word_text) <= 1 or zw_count <= 0:
            return word_text

        letters = list(word_text)
        result = []
        
        for i, char in enumerate(letters[:-1]):
            result.append(char)
            # Insert specified count of invisible characters
            for _ in range(zw_count):
                invisible_char = random.choice(self.char_pool)
                result.append(invisible_char)

        result.append(letters[-1])
        return "".join(result)

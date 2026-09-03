import random
from typing import Optional


class LayoutDisruptor:
    """
    Generates jumbled, scrambled, or reversed character text streams for layout disruption.
    When overlaid as an invisible text layer (render_mode=3), copying and pasting text from the PDF
    results in garbled, unreadable character sequences while maintaining 100% visual legibility on screen.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def disrupt_word(self, word_text: str, mode: str = "shuffle") -> str:
        """
        Creates a disrupted version of word_text by shuffling, reversing, or jumbling letters.
        """
        if len(word_text) <= 2:
            return word_text[::-1]

        if mode == "reverse":
            return word_text[::-1]
        
        # Default shuffle mode: swap character positions
        letters = list(word_text)
        random.shuffle(letters)
        # Ensure shuffled result is not accidentally identical to original
        if "".join(letters) == word_text and len(word_text) > 2:
            letters[0], letters[-1] = letters[-1], letters[0]
            
        return "".join(letters)

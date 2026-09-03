import random
from typing import Dict, Optional


class HomoglyphSubstitutor:
    """
    Substitutes Latin characters with visually identical Basic Cyrillic characters.
    Uses strictly standard Basic Cyrillic characters (U+0400 to U+045F) present in 100% of standard desktop
    and PDF fonts to guarantee zero missing glyph boxes or rendering artifacts.
    """

    # Strictly safe Basic Cyrillic homoglyph mappings (present in 100% of standard fonts)
    SAFE_HOMOGLYPH_MAP: Dict[str, str] = {
        # Lowercase Latin
        'a': '\u0430',  # Cyrillic small letter a
        'c': '\u0441',  # Cyrillic small letter es
        'e': '\u0435',  # Cyrillic small letter ie
        'i': '\u0456',  # Cyrillic small letter Ukrainian i
        'j': '\u0455',  # Cyrillic small letter dze
        'o': '\u043E',  # Cyrillic small letter o
        'p': '\u0440',  # Cyrillic small letter er
        's': '\u0455',  # Cyrillic small letter dze
        'x': '\u0445',  # Cyrillic small letter ha
        'y': '\u0443',  # Cyrillic small letter u

        # Uppercase Latin
        'A': '\u0410',  # Cyrillic capital letter A
        'B': '\u0412',  # Cyrillic capital letter Ve
        'C': '\u0421',  # Cyrillic capital letter Es
        'E': '\u0415',  # Cyrillic capital letter Ie
        'H': '\u041D',  # Cyrillic capital letter En
        'I': '\u0406',  # Cyrillic capital letter Byelorussian-Ukrainian I
        'J': '\u0408',  # Cyrillic capital letter Je
        'K': '\u041A',  # Cyrillic capital letter Ka
        'M': '\u041C',  # Cyrillic capital letter Em
        'O': '\u041E',  # Cyrillic capital letter O
        'P': '\u0420',  # Cyrillic capital letter Er
        'S': '\u0405',  # Cyrillic capital letter Dze
        'T': '\u0422',  # Cyrillic capital letter Te
        'X': '\u0425',  # Cyrillic capital letter Ha
        'Y': '\u0423',  # Cyrillic capital letter U
    }

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def substitute_word(self, word_text: str, ratio: float = 0.8) -> str:
        """
        Replaces Latin characters in word_text with visually identical Basic Cyrillic characters.
        """
        result = []
        for char in word_text:
            if char in self.SAFE_HOMOGLYPH_MAP and random.random() <= ratio:
                result.append(self.SAFE_HOMOGLYPH_MAP[char])
            else:
                result.append(char)
        return "".join(result)

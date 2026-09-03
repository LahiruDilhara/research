import re


def is_math_or_formula_text_fixed(text: str) -> bool:
    """
    Detects standalone mathematical equation blocks, formula lines, or math symbol expressions.
    Does NOT flag normal English hyphenated words ('paper-based', 'real-time') or slashes ('and/or').
    """
    clean_text = text.strip()
    
    # Standalone equation number format e.g. (1.1), (3.4)
    if re.match(r'^\s*\(\s*\d+(?:\.\d+)*\s*\)\s*$', clean_text):
        return True

    # Standalone math equation/formula indicators
    math_symbols = r'[=×÷≠≤≥\^_\{\}\\\|α-ωΑ-Ω∫∑√∂∇∝∞≈±]'
    if re.search(math_symbols, clean_text):
        # Check if it's an actual equation line (e.g. contains = or math sum/int)
        if re.search(r'[=≠≤≥∫∑√∂∇∝≈]', clean_text) or re.search(r'\\[a-zA-Z]+', clean_text):
            return True

    return False


def test_fix():
    p1 = "Human-Computer Interaction (HCI) has developed rapidly over the past few decades. Researchers and developers continue to look for natural, simple, and low-cost ways for people to interact with computers."
    p2 = "To address these limitations, this research project presents a customizable, paper-based virtual keyboard system. The system works using a single ordinary monocular RGB webcam, AprilTag visual marker tracking, and a lightweight PyTorch deep learning model."
    p3 = "E = m c^2 + \\alpha \\sum_{i=1}^n x_i"
    
    print("p1 (HCI intro):", is_math_or_formula_text_fixed(p1))
    print("p2 (paper-based):", is_math_or_formula_text_fixed(p2))
    print("p3 (equation):", is_math_or_formula_text_fixed(p3))


if __name__ == "__main__":
    test_fix()

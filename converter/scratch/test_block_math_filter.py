import re

def is_standalone_equation_block(block_text: str) -> bool:
    clean_text = block_text.strip()
    
    # Standalone equation number format e.g. (1.1), (3.4)
    if re.match(r'^\s*\(\s*\d+(?:\.\d+)*\s*\)\s*$', clean_text):
        return True

    # High math symbol density in standalone block
    math_symbols = re.findall(r'[=×÷≠≤≥\^_\{\}\\|α-ωΑ-Ω∫∑√∂∇∝∞≈±]', clean_text)
    if len(math_symbols) > 0 and len(clean_text) > 0:
        ratio = len(math_symbols) / len(clean_text)
        if ratio > 0.20:
            return True

    return False


def test_text_samples():
    sample1 = """
    Fingertip pixel movement in webcam images depends on two main factors: the distance
Z between the hand and the camera, and the physical hand size of the user. Because of
perspective projection, a fingertip movement of 10 mm produces a pixel shift xpixel = f · X
,
Z
where f is the focal length. As a result, fixed pixel thresholds fail when a user sits closer
to or farther from the camera, or between users with different hand sizes.
This problem was solved by implementing unitless hand scale normalization in
normalize landmarks.py. Wrist Landmark 0 (P0 ) is set as the local coordinate ori-
gin, removing absolute screen position offsets. Hand length Lhand (t) = ∥P9 (t) − P0 (t)∥2
is calculated as the Euclidean distance between wrist Landmark 0 and middle MCP
Landmark 9. All 21 joint landmark coordinates are then divided by this hand length:
    """

    sample2 = "x_{pixel} = f \\cdot \\frac{X}{Z} \\quad (3.2)"
    sample3 = "(4.15)"

    print("Sample 1 (Narrative with inline math): Standalone Eq =", is_standalone_equation_block(sample1))
    print("Sample 2 (Display equation block):    Standalone Eq =", is_standalone_equation_block(sample2))
    print("Sample 3 (Equation number block):     Standalone Eq =", is_standalone_equation_block(sample3))

if __name__ == "__main__":
    test_text_samples()



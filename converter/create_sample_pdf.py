import pymupdf as fitz


def create_test_pdf(filename="sample_test.pdf"):
    doc = fitz.open()

    # Page 1: Title, Header, TOC, and Body Paragraph 1
    page1 = doc.new_page(width=595, height=842)  # A4 standard size
    
    # Running Header
    page1.insert_text((50, 40), "JOURNAL OF COMPUTER VISION RESEARCH - VOL 12", fontsize=8, color=(0.4, 0.4, 0.4))
    
    # Document Title
    page1.insert_text((50, 100), "Deep Learning Models for Optical Hand Tracking", fontsize=18, color=(0, 0, 0))
    page1.insert_text((50, 125), "Author: Lahiru Dilhara", fontsize=11, color=(0.2, 0.2, 0.2))

    # Section Heading
    page1.insert_text((50, 160), "1. Introduction", fontsize=14, color=(0, 0, 0))

    # Body Paragraph 1
    para1 = (
        "Virtual input interaction systems represent an active area of research in modern computer vision. "
        "By utilizing a single monocular camera, standard paper surfaces can be transformed into interactive human "
        "computer interface keyboards. The skeletal landmark tracking pipeline processes real time temporal hand motion "
        "to detect continuous surface contact events accurately [1]. Our approach achieves real time CPU execution "
        "without requiring specialized depth sensors or wearable hardware components (Smith et al., 2021)."
    )
    rect1 = fitz.Rect(50, 180, 545, 300)
    page1.insert_textbox(rect1, para1, fontsize=10, align=0)

    # Section Heading 2
    page1.insert_text((50, 320), "2. Methodology", fontsize=14, color=(0, 0, 0))

    # Body Paragraph 2
    para2 = (
        "The proposed pipeline evaluates hand coordinates across consecutive video frames to determine physical touch "
        "impact timing. Scale normalization ensures robust operation across varying camera distances. Each active finger "
        "tip landmark is tracked continuously while planar homography maps image pixel coordinates directly to digital layout "
        "key definitions [2]. Experimental evaluations demonstrate high classification accuracy across diverse lighting conditions."
    )
    rect2 = fitz.Rect(50, 340, 545, 460)
    page1.insert_textbox(rect2, para2, fontsize=10, align=0)

    # Table of Contents sample
    page1.insert_text((50, 490), "Table of Contents", fontsize=12, color=(0, 0, 0))
    page1.insert_text((50, 510), "1. Introduction .................................................... 1", fontsize=9, color=(0.3, 0.3, 0.3))
    page1.insert_text((50, 525), "2. Methodology .................................................... 1", fontsize=9, color=(0.3, 0.3, 0.3))
    page1.insert_text((50, 540), "3. References ...................................................... 2", fontsize=9, color=(0.3, 0.3, 0.3))

    # Running Footer
    page1.insert_text((270, 810), "Page 1 of 2", fontsize=9, color=(0.4, 0.4, 0.4))

    # Page 2: References section
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 40), "JOURNAL OF COMPUTER VISION RESEARCH - VOL 12", fontsize=8, color=(0.4, 0.4, 0.4))

    page2.insert_text((50, 80), "References", fontsize=14, color=(0, 0, 0))
    ref1 = "[1] J. Smith, A. Johnson, 'Monocular Vision Keyboards', IEEE Trans on Vision, 2021."
    ref2 = "[2] L. Dilhara, 'Homography Alignment for Paper Surfaces', Research Thesis, 2024."
    page2.insert_text((50, 105), ref1, fontsize=9, color=(0.1, 0.1, 0.1))
    page2.insert_text((50, 125), ref2, fontsize=9, color=(0.1, 0.1, 0.1))

    page2.insert_text((270, 810), "Page 2 of 2", fontsize=9, color=(0.4, 0.4, 0.4))

    doc.save(filename)
    doc.close()
    print(f"Created sample PDF: {filename}")


if __name__ == "__main__":
    create_test_pdf()

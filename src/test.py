import numpy as np
from prep import prep
from embedding import embed_watermark
from extraction import extr

def test_prep():
    # Test parameters
    host_image = np.random.randint(0, 256, (8, 8))  # 8x8 test image
    k3, k4 = 12345, 54321
    
    # Test watermark preparation
    watermark = prep(host_image, k3, k4)
    assert isinstance(watermark, str), "Watermark should be a binary string."

    assert len(watermark) == 12 * (host_image.size // (2 * 4)), "Incorrect watermark length."
    print("prep module passed.")

def test_embd():
    # Test parameters
    host_image = np.random.randint(0, 256, (8, 8))  # 8x8 test image
    k3, k4 = 12345, 54321
    watermark = prep(host_image, k3, k4)
    
    # Test watermark embedding
    watermarked_image = embed_watermark(host_image, watermark)
    assert watermarked_image.shape == host_image.shape, "Watermarked image should match host image shape."
    print("embd module passed.")

    print(host_image)
    print(watermarked_image)

def test_extr():
    # Test parameters
    host_image = np.random.randint(0, 256, (8, 8))  # 8x8 test image
    k3, k4 = 12345, 54321
    watermark = prep(host_image, k3, k4)
    watermarked_image = embed_watermark(host_image, watermark)
    
    # Simulate tampering by modifying one block
    watermarked_image[0:2, 0:4] += 10
    
    # Test watermark extraction and tamper detection
    tampered_blocks = extr(watermarked_image, k3, k4)
    assert isinstance(tampered_blocks, set), "Tampered blocks should be a set."
    assert len(tampered_blocks) > 0, "There should be at least one tampered block."
    assert 0 in tampered_blocks, "block 0 should be detected as tampered!"
    print(tampered_blocks)

    print("extr module passed.")

def run_tests():
    test_prep()
    test_embd()
    test_extr()
    print("All tests passed.")

# Running tests
run_tests()

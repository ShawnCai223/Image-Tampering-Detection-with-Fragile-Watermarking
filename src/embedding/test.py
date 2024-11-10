import unittest
import numpy as np
from embedding import *

class TestFragileWatermarkEmbedding(unittest.TestCase):
    
    def test_convert_to_base9_digits(self):
        # Test conversion of 12-bit integer to base-9 digits
        
        self.assertEqual(convert_to_base_digits(2456, 9, 4), [3, 3, 2, 8])  # 2456 in base-9 is 3328
        self.assertEqual(convert_to_base_digits(1234, 9, 4), [1, 6, 2, 1])  # 1234 in base-9 is 1621
        self.assertEqual(convert_to_base_digits(0, 9, 4), [0, 0, 0, 0]
)     # 0 in base-9 is 0000
        self.assertEqual(convert_to_base_digits(6560, 9, 4), [8, 8, 8, 8])  # 6560 is the largest 12-bit number that fits within 4 base-9 digits

        
    def test_embed_digit_in_unit(self):
        # Test embedding a digit in a 2-pixel unit
        unit = np.array([10, 20])
        original_unit = unit.copy()
        embed_digit_in_unit(unit, 5)
        
        # Since we can't predict exact pixel values without running full calculations,
        # we check if the values in the unit have been modified as expected
        self.assertNotEqual(unit[0], original_unit[0])
        self.assertNotEqual(unit[1], original_unit[1])
        
    def test_embed_watermark(self):
        # Create a sample image (8x8 pixels) and watermark data
        image = np.ones((8, 8), dtype=int) * 100  # Initialize a grayscale image with all pixels as 100
        watermark_bits = [2456, 1234, 3456, 5678, 423, 4321, 1111, 999]  # Sample watermark data
        
        # Apply watermark embedding
        watermarked_image = embed_watermark(image, watermark_bits.copy())
        
        # Verify dimensions are unchanged
        self.assertEqual(watermarked_image.shape, image.shape)
        
        # Check that blocks have been modified (not identical to original image)
        changes_made = np.any(watermarked_image != image)
        self.assertTrue(changes_made)
        
    def test_complete_embedding_process(self):
        # Test the entire embedding process with known inputs and check pixel modifications
        image = np.array([
            [100, 101, 102, 103, 104, 105, 106, 107],
            [110, 111, 112, 113, 114, 115, 116, 117],
            [120, 121, 122, 123, 124, 125, 126, 127],
            [130, 131, 132, 133, 134, 135, 136, 137],
            [140, 141, 142, 143, 144, 145, 146, 147],
            [150, 151, 152, 153, 154, 155, 156, 157],
            [160, 161, 162, 163, 164, 165, 166, 167],
            [170, 171, 172, 173, 174, 175, 176, 177],
        ], dtype=int)
        
        watermark_bits = [2456, 1234, 3456, 5678]  # Only enough for four 2x4 blocks
        watermarked_image = embed_watermark(image, watermark_bits.copy())
        
        # Ensure watermarked image pixels have changed
        num_changes = np.sum(watermarked_image != image)
        self.assertGreater(num_changes, 0)
        
        # Check that only certain pixels (those in the first four blocks) were modified
        unchanged_area = watermarked_image[4:, 4:]  # Pixels outside the embedded blocks
        original_unchanged_area = image[4:, 4:]
        np.testing.assert_array_equal(unchanged_area, original_unchanged_area)

if __name__ == '__main__':
    unittest.main()

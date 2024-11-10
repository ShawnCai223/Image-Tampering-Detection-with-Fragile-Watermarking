import numpy as np

def embed_watermark(image, watermark_bits):
    """
    Embed fragile watermark data into the given image.
    
    Parameters:
        image (np.ndarray): Grayscale image array.
        watermark_bits (list): List of 12-bit watermark data to embed in each block.
    
    Returns:
        np.ndarray: Watermarked image.
    """
    # Copy image to avoid modifying the original
    watermarked_image = image.copy()
    
    # Block size (2x4)
    block_height, block_width = 2, 4
    
    # Iterate over each block in the image
    for i in range(0, image.shape[0], block_height):
        for j in range(0, image.shape[1], block_width):
            # Extract a block (2x4)
            block = watermarked_image[i:i+block_height, j:j+block_width]
            
            # [TODO: padding or ignore?] Ensure the block is complete (for edge cases)
            if block.shape[0] < block_height or block.shape[1] < block_width:
                continue
            
            # [TODO: maintain unchanged or warning?] Get the next 12-bit watermark data and convert to base-9 digits
            if watermark_bits == []:
                print("No enough watermark bits!")
                break
            watermark_12bit = watermark_bits.pop(0)
            watermark_4_base9 = convert_to_base_digits(watermark_12bit, 9, 4)

            # Embed watermark in each unit (2 pixels) of the block
            units = [block[:, col] for col in range(block_width)]  # U1, U2, U3, U4
            for unit, digit in zip(units, watermark_4_base9):
                embed_digit_in_unit(unit, digit)
                
    return watermarked_image

def convert_to_base_digits(value, base, nfill):
    # range check
    if value < 0 or value >= base**nfill:
        raise Warning("Base conversion out of range!")
    return [int(digit) for digit in np.base_repr(value, base=base).zfill(nfill)]

def embed_digit_in_unit(unit, d):
    """
    Embed a base-9 digit into a 2-pixel unit according to the fragile embedding scheme.
    
    Parameters:
        unit (np.ndarray): Array containing 2 pixels of the unit.
        d (int): Base-9 digit to embed.
    """
    # Compute F_embed [TODO: check what the {} stands for?]
    F_embed = (3**0 * unit[0] + 3**1 * unit[1]) % 3**2
    
    # Calculate x
    n = 2
    x = (d - F_embed + ((3**n - 1) // 2)) % 3**n
    
    # Convert x to base-3
    x_base3 = convert_to_base_digits(x, 3, n)
    x_base3 = [xi - 1 for xi in x_base3]  # Get x'' by subtracting 1 from each digit
    
    # Update unit pixels with x'' digits
    unit[0] += x_base3[1]
    unit[1] += x_base3[0]

# Example usage
# Load a grayscale image as a 2D numpy array (e.g., with OpenCV or PIL)
# image = np.array(...)  

# List of 12-bit watermark data for each block, e.g., [2456, 1234, ...]
# watermark_bits = [...]

# watermarked_image = embed_watermark(image, watermark_bits)
# Note: Save or display watermarked_image as needed.

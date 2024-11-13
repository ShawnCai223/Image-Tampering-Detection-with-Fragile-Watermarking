from prep import generate_random_sequence

from embedding import get_F_embed

def extr(watermarked_image, k3, k4):
    blocks = [watermarked_image[i:i+2, j:j+4] for i in range(0, watermarked_image.shape[0], 2)
              for j in range(0, watermarked_image.shape[1], 4)]
    TB = len(blocks)
    
    # Generate W_ran_2 for comparison
    W_ran_2 = generate_random_sequence(TB, k3)
    
    tampered_blocks = []
    for i, block in enumerate(blocks):
        F_ext = [get_F_embed(u) for u in block.T]
        # [TODO: 9-base to 3bits(<= 8) ?]
        extracted_digits = "".join([f"{f:03b}" for f in F_ext])

        EW_ran_2 = extracted_digits[:6]
        EW_recov = extracted_digits[6:]
        
        # Compare to authenticate
        if EW_ran_2 != W_ran_2[i]:
            tampered_blocks.append(i)
    
    # Neighborhood smoothing
    smoothed_tampered = set()
    for idx in tampered_blocks:
        neighbors = [(idx - 1) % TB, (idx + 1) % TB, (idx - TB//2) % TB, (idx + TB//2) % TB]
        if sum(1 for n in neighbors if n in tampered_blocks) > len(neighbors) // 2:
            smoothed_tampered.add(idx)
    
    return smoothed_tampered

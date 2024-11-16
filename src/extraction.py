from preparation import generate_random_sequence

from embedding import get_F_embed

block_height, block_width = 2, 4

def extr(watermarked_image, k3, k4):
    # [TODO: padding]
    blocks = [watermarked_image[i:i+2, j:j+4] for i in range(0, watermarked_image.shape[0], 2)
              for j in range(0, watermarked_image.shape[1], 4)]
    TB = len(blocks)
    
    # Generate W_ran_2 for comparison
    W_ran_2 = generate_random_sequence(TB, k3)
    print("extr ran: ", W_ran_2)
    
    tampered_blocks = []
    for i, block in enumerate(blocks):
        F_ext = [get_F_embed(block[:, col]) for col in range(block_width)]
        extracted_digits = "".join([f"{f:03b}" for f in F_ext])

        EW_ran_2 = extracted_digits[:6]
        EW_recov = extracted_digits[6:]
        
        # Compare to authenticate
        print(EW_ran_2, W_ran_2[i])
        if EW_ran_2 != W_ran_2[i]:
            tampered_blocks.append(i)
    
    # [TODO] Neighborhood smoothing
    smoothed_tampered = tampered_blocks
    
    return smoothed_tampered

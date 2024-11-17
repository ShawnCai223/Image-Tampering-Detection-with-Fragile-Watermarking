import numpy as np
from random import Random

def generate_random_sequence(TB, seed):
    rng = Random(seed)
    # [TODO SFMT process]
    return [''.join([str(rng.randint(0, 1)) for _ in range(6)]) for _ in range(TB)]

def avg_block_intensity(block):
    return int(np.mean(block))

def prep(host_image, k3, k4):
    # Divide the image into non-overlapping 2x4 blocks
    block_height, block_width = 2, 4
    pad_height = (block_height - (host_image.shape[0] % block_height)) % block_height
    pad_width = (block_width - (host_image.shape[1] % block_width)) % block_width
    # Apply padding
    host_image = np.pad(
        host_image,
        ((0, pad_height), (0, pad_width)),
        mode="constant",
        constant_values=0,
    )

    blocks = [host_image[i:i+2, j:j+4] for i in range(0, host_image.shape[0], 2)
              for j in range(0, host_image.shape[1], 4)]
    TB = len(blocks)
    
    # [TODO]Generate a random binary sequence W_ran_2_block
    W_ran_2_block = generate_random_sequence(TB, k3)

    #print("prep ran: ", W_ran_2_block)
    
    # Calculate block intensities and convert to binary
    block_intensities = [avg_block_intensity(block) for block in blocks]
    bin_representations = [f"{intensity:08b}"[:6] for intensity in block_intensities]
    
    # [TODO]Concatenate in controlled randomized manner using k4
    rng = Random(k4)
    W_recov_block = rng.sample(bin_representations, len(bin_representations))
    
    # Cascade W_ran_2_block and W_recov_block to create W_fragile+recov
    W_fragile_recov = "".join([W_ran_2_block[i] + W_recov_block[i] for i in range(TB)])

    return W_fragile_recov

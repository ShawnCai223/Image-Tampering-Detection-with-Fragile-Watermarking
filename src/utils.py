import numpy as np


def padding(image, block_height, block_width):
    pad_height = (block_height - (image.shape[0] % block_height)) % block_height
    pad_width = (block_width - (image.shape[1] % block_width)) % block_width

    image = np.pad(
        image,
        ((0, pad_height), (0, pad_width)),
        mode="constant",
        constant_values=0,
    )

    return image

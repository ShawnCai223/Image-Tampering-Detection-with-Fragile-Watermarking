import numpy as np
def calculate_F_embed(U):
    return sum((3**(k) * U[k]) for k in range(len(U))) % (3**len(U))

def embed_watermark(block, digits):
    for u, d in zip(block, digits):
        F_embed = calculate_F_embed(u)
        n = 2
        x = (d - F_embed + (3**n - 1) // 2) % 3**n
        x_base3 = [(x // (3**i)) % 3 - 1 for i in range(n-1, -1, -1)]
        for i in range(n):
            u[i] += x_base3[i]
    return block

def embd(host_image, W_fragile_recov):
    blocks = [host_image[i:i+2, j:j+4] for i in range(0, host_image.shape[0], 2)
              for j in range(0, host_image.shape[1], 4)]
    for i, block in enumerate(blocks):
        wat_12bit = W_fragile_recov[i*12:(i+1)*12]
        wat_9 = int(wat_12bit, 2)
        digits = [(wat_9 // (9**j)) % 9 for j in range(3, -1, -1)]

        for u, d in zip(block.T, digits):
            block = embed_watermark(block, digits)

    embd_image = np.ndarray(host_image.shape)

    for i in range(embd_image.shape[0]):
        for j in range(embd_image.shape[1]):
            bi, bj = i // 2, j // 4
            bn = bi * (embd_image.shape[1] // 4) + bj

            xi, xj = i % 2, j % 4
            embd_image[i][j] = blocks[bn][xi][xj]
    return embd_image

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from skimage.transform import resize


def evaluate_performance(original_image, watermarked_image, tampered_image, recovered_image, watermark,
                         extracted_watermark, tampered_blocks, true_tampered_blocks):
    """
    Evaluate the performance of the watermarking and tampering detection system.
    """

    # Flatten extracted_watermark if it contains tuples
    if isinstance(extracted_watermark, (tuple, list)):
        extracted_watermark = ''.join(
            map(str, [item if isinstance(item, int) else item[0] for item in extracted_watermark]))

    # Convert watermark and extracted_watermark to bit arrays
    watermark_bits = np.array(list(map(int, watermark)))
    extracted_bits = np.array(list(map(int, extracted_watermark)))

    # Ensure both arrays have the same length
    min_length = min(len(watermark_bits), len(extracted_bits))
    watermark_bits = watermark_bits[:min_length]
    extracted_bits = extracted_bits[:min_length]

    #Align watermarked image dimensions with original image
    if original_image.shape != watermarked_image.shape:
        watermarked_image = resize(watermarked_image, original_image.shape, preserve_range=True).astype(
            original_image.dtype)

    # 1. PSNR and SSIM for Watermarked Image
    psnr_watermarked = psnr(original_image, watermarked_image, data_range=255)
    ssim_watermarked = ssim(original_image, watermarked_image, data_range=255, multichannel=True)

    # 2. Bit Error Rate (BER)
    ber = np.sum(watermark_bits == extracted_bits) / len(watermark_bits)

    # 3. Tamper Detection Accuracy (TDeff)
    actual_set = set(true_tampered_blocks)
    detected_set = set(tampered_blocks)

    # Correctly detected blocks
    correctly_detected = actual_set.intersection(detected_set)

    # Calculate TDeff
    if len(actual_set) == 0:
        tamper_detection_accuracy = 0.0
    else:
        tamper_detection_accuracy = (len(correctly_detected) / len(actual_set)) * 100


    # 4. Self-Recovery Quality
    if original_image.shape != recovered_image.shape:
        recovered_image = resize(recovered_image, original_image.shape, preserve_range=True).astype(
            original_image.dtype)
    psnr_recovery = psnr(original_image, recovered_image, data_range=255)
    ssim_recovery = ssim(original_image, recovered_image, data_range=255, multichannel=True)

    # 5. Result Compilation
    metrics = {
        "PSNR_Watermarked": psnr_watermarked,
        "SSIM_Watermarked": ssim_watermarked,
        "Bit_Error_Rate": ber,
        "Tamper_Detection_Accuracy": tamper_detection_accuracy,
        "PSNR_Recovery": psnr_recovery,
        "SSIM_Recovery": ssim_recovery,
    }

    return metrics

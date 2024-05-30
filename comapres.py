import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(image1, image2):
    return psnr(image1, image2)

def calculate_ssim(image1, image2):
    return ssim(image1, image2, data_range=image2.max() - image2.min())

#Load infrared, visible, and fused images
image_vis_path = 'C:/Users/soma/Desktop/image/image2.png'
fused_image_path = 'C:/Users/soma/Desktop/image/fused_image.jpg'  # Update this path accordingly

image_vis = cv2.imread(image_vis_path, cv2.IMREAD_GRAYSCALE)
fused_image = cv2.imread(fused_image_path, cv2.IMREAD_GRAYSCALE)

#Check if the images are loaded correctly
if image_vis is None:
    print(f"Error loading visible image from {image_vis_path}")
if fused_image is None:
    print(f"Error loading fused image from {fused_image_path}")

#Calculate PSNR and SSIM if all images are loaded successfully
if image_vis is not None and fused_image is not None:
    psnr_vis_fused = calculate_psnr(image_vis, fused_image)
    ssim_vis_fused = calculate_ssim(image_vis, fused_image)

    print(f"PSNR between Visible and Fused Image: {psnr_vis_fused:.4f}")
    print(f"SSIM between Visible and Fused Image: {ssim_vis_fused:.4f}")

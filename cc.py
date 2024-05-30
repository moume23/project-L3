import cv2
import numpy as np

def calculate_correlation_coefficient(image1, image2):
    # Ensure the images have the same size and type
    if image1.shape != image2.shape or image1.dtype != image2.dtype:
        raise ValueError("Input images must have the same dimensions and type.")
    
    # Flatten the images to 1D arrays
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    
    # Calculate the mean of the flattened images
    mean1 = np.mean(image1_flat)
    mean2 = np.mean(image2_flat)
    
    # Calculate the correlation coefficient
    numerator = np.sum((image1_flat - mean1) * (image2_flat - mean2))
    denominator = np.sqrt(np.sum((image1_flat - mean1) ** 2) * np.sum((image2_flat - mean2) ** 2))
    
    if denominator == 0:
        raise ValueError("Denominator in correlation coefficient calculation is zero.")
    
    correlation_coefficient = numerator / denominator
    return correlation_coefficient

# Load the images (ensure they are grayscale for simplicity)
fused_image = cv2.imread('C:/Users/soma/Desktop/image/fused image IHS.jpg', cv2.IMREAD_GRAYSCALE)
visible_image = cv2.imread('C:/Users/soma/Desktop/image/image2.png', cv2.IMREAD_GRAYSCALE)

# Check if images are loaded properly
if fused_image is None or visible_image is None:
    raise FileNotFoundError("One or both images could not be loaded. Check the file paths.")

# Calculate the Correlation Coefficient
cc = calculate_correlation_coefficient(fused_image, visible_image)
print(f"Correlation Coefficient (CC): {cc}")

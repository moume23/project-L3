import cv2
import numpy as np

def bilateral_filter_decomposition(image, d, sigmaColor, sigmaSpace):
    base_layer = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
    detail_layer = cv2.subtract(image, base_layer)
    return base_layer, detail_layer

def fuse_images(visible_image, infrared_image, d=9, sigmaColor=75, sigmaSpace=75):
    # Decompose visible image
    B_vis, D_vis = bilateral_filter_decomposition(visible_image, d, sigmaColor, sigmaSpace)
    
    # Decompose infrared image
    B_ir, D_ir = bilateral_filter_decomposition(infrared_image, d, sigmaColor, sigmaSpace)
    
    # Fuse base layers
    B_fused = np.maximum(B_vis, B_ir)
    
    # Fuse detail layers
    D_fused = np.maximum(D_vis, D_ir)
    
    # Reconstruct fused image
    fused_image = B_fused + D_fused
    
    return fused_image

# Paths to your images
visible_image_path = 'C:/Users/lmoun/Desktop/image2.png'
infrared_image_path = 'C:/Users/lmoun/Desktop/image1.png'

# Read images
visible_image = cv2.imread(visible_image_path, cv2.IMREAD_GRAYSCALE)
infrared_image = cv2.imread(infrared_image_path, cv2.IMREAD_GRAYSCALE)

# Fuse images
fused_image = fuse_images(visible_image, infrared_image)

# Convert the fused grayscale image to a colored image using the visible image color information
visible_color = cv2.imread(visible_image_path)
visible_color_lab = cv2.cvtColor(visible_color, cv2.COLOR_BGR2LAB)
fused_image_lab = cv2.cvtColor(fused_image, cv2.COLOR_GRAY2BGR)
fused_image_lab = cv2.cvtColor(fused_image_lab, cv2.COLOR_BGR2LAB)

# Replace the L channel in LAB color space with the fused image
fused_image_lab[:, :, 0] = fused_image

# Convert back to BGR color space
colored_fused_image = cv2.cvtColor(fused_image_lab, cv2.COLOR_LAB2BGR)

# Save the result
output_path = 'colored_fused_image.jpg'
cv2.imwrite(output_path, colored_fused_image)

cv2.imshow('Fused Image Min', colored_fused_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

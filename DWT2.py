import pywt
import numpy as np
import cv2

def fuse_coeff(coeff1, coeff2, method='weighted_average', weight=0.5):
    if method == 'min':
        return np.minimum(coeff1, coeff2)
    elif method == 'max':
        return np.maximum(coeff1, coeff2)
    elif method == 'mean':
        return (coeff1 + coeff2) / 2
    elif method == 'weighted_average':
        return weight * coeff1 + (1 - weight) * coeff2
    elif method == 'energy':
        energy1 = np.abs(coeff1)
        energy2 = np.abs(coeff2)
        return np.where(energy1 > energy2, coeff1, coeff2)
    else:
        raise ValueError('Fusion method not recognized.')

def dwt2_fusion(channel1, channel2, method='weighted_average', weight=0.5):
    coeffs1 = pywt.wavedec2(channel1, 'db1', level=2)
    coeffs2 = pywt.wavedec2(channel2, 'db1', level=2)
    
    fused_coeffs = []
    for c1, c2 in zip(coeffs1, coeffs2):
        if isinstance(c1, tuple) and isinstance(c2, tuple):
            fused_subbands = tuple(fuse_coeff(sc1, sc2, method, weight) for sc1, sc2 in zip(c1, c2))
            fused_coeffs.append(fused_subbands)
        else:
            fused_coeffs.append(fuse_coeff(c1, c2, method, weight))
    
    return pywt.waverec2(fused_coeffs, 'db1')

def fuse_images(visible_img, infrared_img, fusion_rule='weighted_average', weight=0.5):
    # Convert infrared image to grayscale if it is not already
    if len(infrared_img.shape) == 3:
        infrared_img = cv2.cvtColor(infrared_img, cv2.COLOR_BGR2GRAY)
    
    # Split the visible image into RGB channels
    vis_r, vis_g, vis_b = cv2.split(visible_img)
    
    # Fuse each visible channel with the infrared image
    fused_r = dwt2_fusion(vis_r, infrared_img, fusion_rule, weight)
    fused_g = dwt2_fusion(vis_g, infrared_img, fusion_rule, weight)
    fused_b = dwt2_fusion(vis_b, infrared_img, fusion_rule, weight)
    
    # Merge the fused channels back into a color image
    fused_image = cv2.merge((fused_r, fused_g, fused_b)).astype(np.uint8)
    return fused_image

# Read images
visible_img_path = 'C:/Users/lmoun/Desktop/image2.png'
infrared_img_path = 'C:/Users/lmoun/Desktop/image1.png'

visible_img = cv2.imread(visible_img_path)
infrared_img = cv2.imread(infrared_img_path)

# Check if images were loaded successfully
if visible_img is None:
    raise FileNotFoundError(f"Visible image not found at {visible_img_path}")
if infrared_img is None:
    raise FileNotFoundError(f"Infrared image not found at {infrared_img_path}")

# Ensure the images are the same size
if visible_img.shape[:2] != infrared_img.shape[:2]:
    raise ValueError('Images must be of the same size.')

# Fuse the images
fused_image = fuse_images(visible_img, infrared_img, fusion_rule='weighted_average', weight=0.6)

# Save or display the fused image
cv2.imwrite('fused_image.jpg', fused_image)
cv2.imshow('Fused Image', fused_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


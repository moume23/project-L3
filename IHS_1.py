import cv2
import numpy as np

def rgb_to_ihs(image):
    """
    Convert an RGB image to IHS.
    """
    # Convert the image to float32 for more accurate calculations
    image = image.astype(np.float32) / 255.0

    # Separate the channels
    r, g, b = cv2.split(image)

    # Calculate the intensity
    i = (r + g + b) / 3.0

    # Calculate the hue and saturation
    min_val = np.minimum(np.minimum(r, g), b)
    s = 1 - 3 * min_val / (r + g + b + 1e-10)
    h = np.arccos((0.5 * (r - g + r - b)) / (np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-10))
    h[b > g] = 2 * np.pi - h[b > g]
    h = h / (2 * np.pi)

    return i, h, s

def ihs_to_rgb(i, h, s):
    """
    Convert an IHS image back to RGB.
    """
    h = h * 2 * np.pi

    r = np.zeros_like(i)
    g = np.zeros_like(i)
    b = np.zeros_like(i)

    idx = (h < 2 * np.pi / 3)
    b[idx] = i[idx] * (1 - s[idx])
    r[idx] = i[idx] * (1 + s[idx] * np.cos(h[idx]) / np.cos(np.pi / 3 - h[idx]))
    g[idx] = 3 * i[idx] - (r[idx] + b[idx])

    idx = (2 * np.pi / 3 <= h) & (h < 4 * np.pi / 3)
    h[idx] = h[idx] - 2 * np.pi / 3
    r[idx] = i[idx] * (1 - s[idx])
    g[idx] = i[idx] * (1 + s[idx] * np.cos(h[idx]) / np.cos(np.pi / 3 - h[idx]))
    b[idx] = 3 * i[idx] - (r[idx] + g[idx])

    idx = (4 * np.pi / 3 <= h) & (h < 2 * np.pi)
    h[idx] = h[idx] - 4 * np.pi / 3
    g[idx] = i[idx] * (1 - s[idx])
    b[idx] = i[idx] * (1 + s[idx] * np.cos(h[idx]) / np.cos(np.pi / 3 - h[idx]))
    r[idx] = 3 * i[idx] - (g[idx] + b[idx])

    # Clip to valid range
    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)

    # Convert to 8-bit image
    rgb_image = np.dstack((r, g, b)) * 255.0
    return rgb_image.astype(np.uint8)

# Load the visible and infrared images
visible_image_path = 'C:/Users/lmoun/Desktop/image2.png'
infrared_image_path = 'C:/Users/lmoun/Desktop/image1.png'

visible_image = cv2.imread(visible_image_path)
infrared_image = cv2.imread(infrared_image_path, cv2.IMREAD_GRAYSCALE)

# Check if the images are loaded successfully
if visible_image is None:
    raise FileNotFoundError(f"Could not load visible image from path: {visible_image_path}")
if infrared_image is None:
    raise FileNotFoundError(f"Could not load infrared image from path: {infrared_image_path}")

# Ensure the infrared image has the same size as the visible image
infrared_image = cv2.resize(infrared_image, (visible_image.shape[1], visible_image.shape[0]))

# Convert the visible image from RGB to IHS
i_visible, h, s = rgb_to_ihs(visible_image)

# Normalize the infrared image to match the intensity range of the visible image
infrared_image = cv2.normalize(infrared_image.astype(np.float32), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# Blend the infrared intensity with the visible intensity
# Give more weight to the visible intensity to retain more of the visible image's color information
alpha = 0.7
fused_i = alpha * i_visible + (1 - alpha) * infrared_image

# Convert the modified IHS image back to RGB
fused_image = ihs_to_rgb(fused_i, h, s)

# Save the fused image
cv2.imwrite('fused_image.jpg', fused_image)

# Display the images
cv2.imshow('Visible Image', visible_image)
cv2.imshow('Infrared Image', infrared_image)
cv2.imshow('Fused Image', fused_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

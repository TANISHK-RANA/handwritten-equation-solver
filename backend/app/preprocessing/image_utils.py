"""
Image preprocessing utilities for handwritten equation recognition.

Handles image loading, normalization, and basic preprocessing operations.
"""

import numpy as np
import cv2
from PIL import Image
import base64
import io
from typing import Tuple, Optional


def load_image_from_base64(base64_string: str) -> np.ndarray:
    """
    Load an image from a base64 encoded string.
    
    Args:
        base64_string: Base64 encoded image string (may include data URI prefix)
        
    Returns:
        Image as numpy array (BGR format)
    """
    # Remove data URI prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    image_data = base64.b64decode(base64_string)
    
    # Convert to PIL Image then to numpy array
    pil_image = Image.open(io.BytesIO(image_data))
    
    # Convert RGBA to RGB if needed
    if pil_image.mode == 'RGBA':
        # Create BLACK background (for canvas with white strokes)
        background = Image.new('RGB', pil_image.size, (0, 0, 0))
        background.paste(pil_image, mask=pil_image.split()[3])
        pil_image = background
    elif pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy array (RGB)
    image = np.array(pil_image)
    
    # Convert RGB to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image


def load_image_from_file(file_path: str) -> np.ndarray:
    """
    Load an image from a file path.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Image as numpy array (BGR format)
    """
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not load image from: {file_path}")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def apply_threshold(image: np.ndarray, 
                   threshold_type: str = 'otsu',
                   invert: bool = True) -> np.ndarray:
    """
    Apply thresholding to binarize the image.
    
    Args:
        image: Grayscale input image
        threshold_type: Type of thresholding ('otsu', 'adaptive', or 'binary')
        invert: If True, white characters on black background (MNIST format)
        
    Returns:
        Binary image
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        image = to_grayscale(image)
    
    if threshold_type == 'otsu':
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_type == 'adaptive':
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    else:  # binary
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Invert if needed (white on black for MNIST compatibility)
    if invert:
        binary = cv2.bitwise_not(binary)
    
    return binary


def denoise_image(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply denoising to the image.
    
    Args:
        image: Input image
        kernel_size: Size of the morphological kernel
        
    Returns:
        Denoised image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply opening (erosion followed by dilation) to remove small noise
    denoised = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    return denoised


def resize_with_padding(image: np.ndarray, 
                       target_size: Tuple[int, int] = (28, 28),
                       padding: int = 4) -> np.ndarray:
    """
    Resize image to target size while maintaining aspect ratio and adding padding.
    
    Args:
        image: Input image
        target_size: Target (height, width)
        padding: Padding to add around the character
        
    Returns:
        Resized and padded image
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate size with padding
    inner_h = target_h - 2 * padding
    inner_w = target_w - 2 * padding
    
    # Calculate scale to fit within inner area
    scale = min(inner_w / w, inner_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    if new_w > 0 and new_h > 0:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = image
    
    # Create output image (black background)
    output = np.zeros((target_h, target_w), dtype=np.uint8)
    
    # Calculate position to center the character
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    # Place resized image in center
    output[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return output


def center_character(image: np.ndarray) -> np.ndarray:
    """
    Center the character within the image based on its center of mass.
    
    Args:
        image: Binary image with white character on black background
        
    Returns:
        Centered image
    """
    # Find center of mass
    moments = cv2.moments(image)
    if moments['m00'] == 0:
        return image
    
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    
    h, w = image.shape[:2]
    target_cx, target_cy = w // 2, h // 2
    
    # Calculate shift needed
    shift_x = target_cx - cx
    shift_y = target_cy - cy
    
    # Create translation matrix
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    
    # Apply translation
    centered = cv2.warpAffine(image, M, (w, h), borderValue=0)
    
    return centered


def preprocess_image(image: np.ndarray, 
                    target_size: Tuple[int, int] = (28, 28)) -> np.ndarray:
    """
    Full preprocessing pipeline for a character image.
    
    Args:
        image: Input character image (already segmented, white on black)
        target_size: Target output size
        
    Returns:
        Preprocessed image ready for model input (white on black, 28x28)
    """
    # Convert to grayscale
    gray = to_grayscale(image)
    
    # The segmented image should already be binary (white on black)
    # Just ensure it's properly thresholded
    mean_val = np.mean(gray)
    
    if mean_val < 128:
        # Already white on black (canvas format) - just threshold to clean up
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    else:
        # Black on white - threshold and invert
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_not(binary)
    
    # Resize with padding
    resized = resize_with_padding(binary, target_size)
    
    # Center character
    centered = center_character(resized)
    
    return centered


def normalize_for_model(image: np.ndarray) -> np.ndarray:
    """
    Normalize image values for model input.
    
    Args:
        image: Input image (0-255 range)
        
    Returns:
        Normalized image (0-1 range, shape suitable for model)
    """
    # Normalize to [0, 1]
    normalized = image.astype(np.float32) / 255.0
    
    # Add channel dimension if needed
    if len(normalized.shape) == 2:
        normalized = normalized.reshape(*normalized.shape, 1)
    
    return normalized


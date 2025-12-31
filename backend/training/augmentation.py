"""
Data augmentation utilities for training.

Provides various transformations to increase dataset diversity and improve
model generalization.
"""

import numpy as np
import cv2
from typing import Tuple, Callable, List
import tensorflow as tf


class DataAugmentor:
    """
    Data augmentation class for handwritten character images.
    
    Supports:
    - Rotation
    - Scaling
    - Translation
    - Noise addition
    - Elastic distortion
    - Brightness/contrast adjustments
    """
    
    def __init__(self, 
                 rotation_range: float = 15.0,
                 scale_range: Tuple[float, float] = (0.85, 1.15),
                 translation_range: float = 3.0,
                 noise_std: float = 0.05,
                 elastic_alpha: float = 20.0,
                 elastic_sigma: float = 4.0):
        """
        Initialize augmentor with transformation parameters.
        
        Args:
            rotation_range: Max rotation in degrees (±)
            scale_range: Scale factor range (min, max)
            translation_range: Max translation in pixels (±)
            noise_std: Standard deviation of Gaussian noise
            elastic_alpha: Elastic distortion intensity
            elastic_sigma: Elastic distortion smoothness
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.noise_std = noise_std
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
    
    def rotate(self, image: np.ndarray) -> np.ndarray:
        """Apply random rotation."""
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=0)
        return rotated
    
    def scale(self, image: np.ndarray) -> np.ndarray:
        """Apply random scaling."""
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        scaled = cv2.resize(image, (new_w, new_h))
        
        # Pad or crop to original size
        if scale > 1:
            # Crop center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            scaled = scaled[start_h:start_h+h, start_w:start_w+w]
        else:
            # Pad to center
            result = np.zeros((h, w), dtype=image.dtype)
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            result[start_h:start_h+new_h, start_w:start_w+new_w] = scaled
            scaled = result
        
        return scaled
    
    def translate(self, image: np.ndarray) -> np.ndarray:
        """Apply random translation."""
        tx = np.random.uniform(-self.translation_range, self.translation_range)
        ty = np.random.uniform(-self.translation_range, self.translation_range)
        
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        h, w = image.shape[:2]
        translated = cv2.warpAffine(image, M, (w, h), borderValue=0)
        return translated
    
    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, self.noise_std, image.shape)
        noisy = image + noise
        return np.clip(noisy, 0, 1)
    
    def elastic_distortion(self, image: np.ndarray) -> np.ndarray:
        """Apply elastic distortion."""
        h, w = image.shape[:2]
        
        # Generate random displacement fields
        dx = np.random.uniform(-1, 1, (h, w))
        dy = np.random.uniform(-1, 1, (h, w))
        
        # Smooth with Gaussian filter
        dx = cv2.GaussianBlur(dx, (0, 0), self.elastic_sigma) * self.elastic_alpha
        dy = cv2.GaussianBlur(dy, (0, 0), self.elastic_sigma) * self.elastic_alpha
        
        # Create mesh grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Remap
        distorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
        
        return distorted
    
    def adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """Randomly adjust brightness."""
        factor = np.random.uniform(0.8, 1.2)
        adjusted = image * factor
        return np.clip(adjusted, 0, 1)
    
    def augment(self, image: np.ndarray, 
                apply_rotation: bool = True,
                apply_scale: bool = True,
                apply_translation: bool = True,
                apply_noise: bool = True,
                apply_elastic: bool = False,
                apply_brightness: bool = True) -> np.ndarray:
        """
        Apply a combination of augmentations.
        
        Args:
            image: Input image (28x28, normalized)
            apply_*: Flags to enable/disable specific augmentations
            
        Returns:
            Augmented image
        """
        result = image.copy()
        
        # Remove channel dimension if present
        if len(result.shape) == 3:
            result = result.squeeze()
        
        if apply_rotation and np.random.random() > 0.5:
            result = self.rotate(result)
        
        if apply_scale and np.random.random() > 0.5:
            result = self.scale(result)
        
        if apply_translation and np.random.random() > 0.5:
            result = self.translate(result)
        
        if apply_elastic and np.random.random() > 0.7:
            result = self.elastic_distortion(result)
        
        if apply_brightness and np.random.random() > 0.5:
            result = self.adjust_brightness(result)
        
        if apply_noise and np.random.random() > 0.5:
            result = self.add_noise(result)
        
        # Add channel dimension back
        result = result.reshape(28, 28, 1)
        
        return result


def create_augmentation_generator(x: np.ndarray, 
                                  y: np.ndarray, 
                                  batch_size: int = 32,
                                  augmentor: DataAugmentor = None):
    """
    Create a generator that yields augmented batches.
    
    Args:
        x: Input images
        y: Labels
        batch_size: Batch size
        augmentor: DataAugmentor instance
        
    Yields:
        Batches of (augmented_images, labels)
    """
    if augmentor is None:
        augmentor = DataAugmentor()
    
    num_samples = len(x)
    indices = np.arange(num_samples)
    
    while True:
        np.random.shuffle(indices)
        
        for start in range(0, num_samples, batch_size):
            batch_indices = indices[start:start + batch_size]
            
            batch_x = []
            batch_y = y[batch_indices]
            
            for idx in batch_indices:
                img = x[idx]
                # Apply augmentation with 50% probability
                if np.random.random() > 0.5:
                    img = augmentor.augment(img)
                batch_x.append(img)
            
            yield np.array(batch_x), batch_y


def get_keras_augmentation_layer():
    """
    Create a Keras augmentation layer for use in the model pipeline.
    
    Returns:
        Keras Sequential model with augmentation layers
    """
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ])
    
    return augmentation


if __name__ == "__main__":
    # Test augmentation
    print("Testing data augmentation...")
    
    # Create a sample image
    sample = np.zeros((28, 28, 1), dtype=np.float32)
    cv2.putText(
        sample[:, :, 0], "5", (5, 22), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1.0, 2
    )
    
    augmentor = DataAugmentor()
    
    # Generate augmented versions
    for i in range(5):
        augmented = augmentor.augment(sample)
        print(f"Augmented sample {i+1} shape: {augmented.shape}")


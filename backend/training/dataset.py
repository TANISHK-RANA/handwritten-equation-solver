"""
Dataset loading and preparation for CNN training.

Handles:
- MNIST dataset for digits (0-9)
- Custom/synthetic operator dataset (+, -, *, /, (, ), ^, √)
- Combining datasets into a unified training set
"""

import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


# Class labels - 18 classes total
# 0-9: digits, 10-13: basic operators, 14-17: advanced symbols
CLASS_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '(', ')', '^', '√']
NUM_CLASSES = len(CLASS_LABELS)

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST dataset for digits 0-9.
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    print(f"MNIST loaded: {len(x_train)} training, {len(x_test)} test samples")
    
    return x_train, y_train, x_test, y_test


def generate_synthetic_operators(num_samples_per_class: int = 6000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic operator symbols (+, -, *, /, (, ), ^, √).
    
    Creates images with various styles, sizes, and positions to simulate
    handwritten operators.
    
    Args:
        num_samples_per_class: Number of samples to generate per operator
        
    Returns:
        Tuple of (images, labels)
    """
    import cv2
    
    print(f"Generating synthetic operators ({num_samples_per_class} per class)...")
    
    # All operators including new ones
    operators = ['+', '-', '*', '/', '(', ')', '^', '√']
    operator_labels = [10, 11, 12, 13, 14, 15, 16, 17]  # Labels 10-17 for operators
    
    all_images = []
    all_labels = []
    
    for op, label in zip(operators, operator_labels):
        for i in range(num_samples_per_class):
            # Create blank image
            img = np.zeros((28, 28), dtype=np.uint8)
            
            # Random variations
            thickness = np.random.randint(1, 4)
            size = np.random.uniform(0.4, 0.9)
            offset_x = np.random.randint(-3, 4)
            offset_y = np.random.randint(-3, 4)
            
            center_x = 14 + offset_x
            center_y = 14 + offset_y
            
            if op == '+':
                # Horizontal line
                h_length = int(10 * size)
                cv2.line(img, (center_x - h_length, center_y), 
                        (center_x + h_length, center_y), 255, thickness)
                # Vertical line
                v_length = int(10 * size)
                cv2.line(img, (center_x, center_y - v_length), 
                        (center_x, center_y + v_length), 255, thickness)
                        
            elif op == '-':
                # Horizontal line only
                h_length = int(10 * size)
                cv2.line(img, (center_x - h_length, center_y), 
                        (center_x + h_length, center_y), 255, thickness)
                        
            elif op == '*':
                # Multiplication sign (X shape or asterisk)
                length = int(8 * size)
                # Diagonal lines
                cv2.line(img, (center_x - length, center_y - length), 
                        (center_x + length, center_y + length), 255, thickness)
                cv2.line(img, (center_x + length, center_y - length), 
                        (center_x - length, center_y + length), 255, thickness)
                        
            elif op == '/':
                # Forward slash
                length = int(10 * size)
                cv2.line(img, (center_x + length, center_y - length), 
                        (center_x - length, center_y + length), 255, thickness)
            
            elif op == '(':
                # Left parenthesis - curved arc
                height = int(20 * size)
                width = int(8 * size)
                # Draw arc for left parenthesis
                cv2.ellipse(img, (center_x + width//2, center_y), (width, height), 
                           0, 110, 250, 255, thickness)
            
            elif op == ')':
                # Right parenthesis - curved arc
                height = int(20 * size)
                width = int(8 * size)
                # Draw arc for right parenthesis
                cv2.ellipse(img, (center_x - width//2, center_y), (width, height), 
                           0, -70, 70, 255, thickness)
            
            elif op == '^':
                # Caret/exponent symbol - like an inverted V
                length = int(8 * size)
                # Left line going up
                cv2.line(img, (center_x - length, center_y + length//2), 
                        (center_x, center_y - length//2), 255, thickness)
                # Right line going down
                cv2.line(img, (center_x, center_y - length//2), 
                        (center_x + length, center_y + length//2), 255, thickness)
            
            elif op == '√':
                # Square root symbol
                height = int(16 * size)
                width = int(12 * size)
                # Draw the checkmark part (bottom left)
                cv2.line(img, (center_x - width, center_y), 
                        (center_x - width//3, center_y + height//3), 255, thickness)
                # Draw the main diagonal going up
                cv2.line(img, (center_x - width//3, center_y + height//3), 
                        (center_x, center_y - height//2), 255, thickness)
                # Draw the horizontal top line
                cv2.line(img, (center_x, center_y - height//2), 
                        (center_x + width, center_y - height//2), 255, thickness)
            
            # Add random noise
            noise = np.random.normal(0, 5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Random slight rotation
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
            img = cv2.warpAffine(img, M, (28, 28))
            
            all_images.append(img)
            all_labels.append(label)
    
    images = np.array(all_images, dtype='float32') / 255.0
    images = images.reshape(-1, 28, 28, 1)
    labels = np.array(all_labels)
    
    print(f"Generated {len(images)} operator samples (8 classes)")
    
    return images, labels


def load_custom_operator_dataset() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load custom operator dataset if available.
    
    Looks for saved operator images in the data directory.
    
    Returns:
        Tuple of (images, labels) or (None, None) if not available
    """
    operators_path = DATA_DIR / "operators"
    
    if not operators_path.exists():
        return None, None
    
    # This would load actual custom operator images
    # For now, return None to use synthetic data
    return None, None


def create_combined_dataset(
    synthetic_samples_per_operator: int = 6000,
    test_split: float = 0.15,
    validation_split: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a combined dataset with digits and operators.
    
    Args:
        synthetic_samples_per_operator: Number of synthetic samples per operator
        test_split: Fraction of data for testing
        validation_split: Fraction of training data for validation
        
    Returns:
        Tuple of (x_train, y_train, x_val, y_val, x_test, y_test)
    """
    # Load MNIST
    x_mnist_train, y_mnist_train, x_mnist_test, y_mnist_test = load_mnist()
    
    # Try to load custom operators, fallback to synthetic
    x_ops, y_ops = load_custom_operator_dataset()
    if x_ops is None:
        x_ops, y_ops = generate_synthetic_operators(synthetic_samples_per_operator)
    
    # Split operators into train/test
    num_ops = len(x_ops)
    num_ops_test = int(num_ops * test_split)
    
    # Shuffle operators
    indices = np.random.permutation(num_ops)
    x_ops = x_ops[indices]
    y_ops = y_ops[indices]
    
    x_ops_train = x_ops[num_ops_test:]
    y_ops_train = y_ops[num_ops_test:]
    x_ops_test = x_ops[:num_ops_test]
    y_ops_test = y_ops[:num_ops_test]
    
    # Combine datasets
    x_train = np.concatenate([x_mnist_train, x_ops_train])
    y_train = np.concatenate([y_mnist_train, y_ops_train])
    
    x_test = np.concatenate([x_mnist_test, x_ops_test])
    y_test = np.concatenate([y_mnist_test, y_ops_test])
    
    # Shuffle training data
    train_indices = np.random.permutation(len(x_train))
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    
    # Create validation split from training data
    num_val = int(len(x_train) * validation_split)
    x_val = x_train[:num_val]
    y_val = y_train[:num_val]
    x_train = x_train[num_val:]
    y_train = y_train[num_val:]
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_val = to_categorical(y_val, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)
    
    print(f"\nDataset prepared:")
    print(f"  Training: {len(x_train)} samples")
    print(f"  Validation: {len(x_val)} samples")
    print(f"  Test: {len(x_test)} samples")
    print(f"  Classes: {NUM_CLASSES}")
    
    return x_train, y_train, x_val, y_val, x_test, y_test


def save_dataset(x_train, y_train, x_val, y_val, x_test, y_test, save_dir: Optional[Path] = None):
    """
    Save the prepared dataset to disk.
    
    Args:
        All dataset arrays
        save_dir: Directory to save to (defaults to DATA_DIR)
    """
    if save_dir is None:
        save_dir = DATA_DIR
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        save_dir / "combined_dataset.npz",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test
    )
    
    print(f"Dataset saved to: {save_dir / 'combined_dataset.npz'}")


def load_saved_dataset(load_dir: Optional[Path] = None) -> Tuple:
    """
    Load a previously saved dataset.
    
    Args:
        load_dir: Directory to load from
        
    Returns:
        Tuple of (x_train, y_train, x_val, y_val, x_test, y_test)
    """
    if load_dir is None:
        load_dir = DATA_DIR
    
    data = np.load(load_dir / "combined_dataset.npz")
    
    return (
        data['x_train'],
        data['y_train'],
        data['x_val'],
        data['y_val'],
        data['x_test'],
        data['y_test']
    )


if __name__ == "__main__":
    # Test dataset creation
    print("Testing dataset creation...")
    x_train, y_train, x_val, y_val, x_test, y_test = create_combined_dataset(
        synthetic_samples_per_operator=1000  # Smaller for testing
    )
    
    print(f"\nShapes:")
    print(f"  x_train: {x_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  x_val: {x_val.shape}")
    print(f"  y_val: {y_val.shape}")
    print(f"  x_test: {x_test.shape}")
    print(f"  y_test: {y_test.shape}")


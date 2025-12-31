"""
Predictor module for CNN inference.

Handles loading the trained model and making predictions on preprocessed images.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Optional

from .cnn_model import CLASS_LABELS, INPUT_SHAPE


class EquationPredictor:
    """
    Predictor class for handwritten equation recognition.
    
    Loads the trained CNN model and provides methods for making predictions
    on individual characters or batches of characters.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model. If None, uses default path.
        """
        self.model = None
        self.model_path = model_path
        self.class_labels = CLASS_LABELS
        self.input_shape = INPUT_SHAPE
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if loading successful, False otherwise
        """
        try:
            path = Path(model_path)
            if not path.exists():
                print(f"Model not found at: {model_path}")
                return False
            
            self.model = tf.keras.models.load_model(str(path))
            self.model_path = model_path
            print(f"Model loaded successfully from: {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_for_prediction(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image for prediction.
        
        Args:
            image: Input image as numpy array (grayscale, any size)
            
        Returns:
            Preprocessed image ready for model input (1, 28, 28, 1)
        """
        import cv2
        
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to 28x28
        if image.shape != (28, 28):
            image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Reshape for model input (batch_size, height, width, channels)
        image = image.reshape(1, 28, 28, 1)
        
        return image
    
    def predict_single(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict a single character from an image.
        
        Args:
            image: Preprocessed image (28x28 grayscale)
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess
        processed = self.preprocess_for_prediction(image)
        
        # Predict
        predictions = self.model.predict(processed, verbose=0)
        
        # Get predicted class and confidence
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_class = self.class_labels[predicted_idx]
        
        return predicted_class, confidence
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Predict multiple characters from a batch of images.
        
        Args:
            images: List of images (each 28x28 grayscale)
            
        Returns:
            List of (predicted_class, confidence) tuples
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not images:
            return []
        
        # Preprocess all images
        processed_batch = np.zeros((len(images), 28, 28, 1), dtype=np.float32)
        for i, img in enumerate(images):
            processed = self.preprocess_for_prediction(img)
            processed_batch[i] = processed[0]
        
        # Batch prediction
        predictions = self.model.predict(processed_batch, verbose=0)
        
        # Extract results
        results = []
        for pred in predictions:
            predicted_idx = np.argmax(pred)
            confidence = float(pred[predicted_idx])
            predicted_class = self.class_labels[predicted_idx]
            results.append((predicted_class, confidence))
        
        return results
    
    def predict_equation(self, char_images: List[np.ndarray]) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Predict an entire equation from a list of character images.
        
        Args:
            char_images: List of segmented character images, sorted left-to-right
            
        Returns:
            Tuple of (equation_string, list of (char, confidence) pairs)
        """
        predictions = self.predict_batch(char_images)
        
        equation_string = ''.join([char for char, _ in predictions])
        
        return equation_string, predictions
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


# Global predictor instance for API use
_predictor_instance: Optional[EquationPredictor] = None


def get_predictor(model_path: Optional[str] = None) -> EquationPredictor:
    """
    Get or create the global predictor instance.
    
    Args:
        model_path: Path to model (only used for first initialization)
        
    Returns:
        EquationPredictor instance
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = EquationPredictor(model_path)
    elif model_path and not _predictor_instance.is_loaded():
        _predictor_instance.load_model(model_path)
    
    return _predictor_instance


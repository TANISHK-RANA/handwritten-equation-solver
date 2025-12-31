"""
CNN Model Architecture for Handwritten Character Recognition

This model recognizes 14 classes:
- Digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Operators: +, -, *, /
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# Class labels mapping
CLASS_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/']
NUM_CLASSES = len(CLASS_LABELS)
INPUT_SHAPE = (28, 28, 1)


def create_cnn_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Create a CNN model for handwritten character recognition.
    
    Architecture:
    - Input: 28x28x1 grayscale image
    - Conv2D(32) -> BatchNorm -> ReLU -> MaxPool
    - Conv2D(64) -> BatchNorm -> ReLU -> MaxPool
    - Conv2D(128) -> BatchNorm -> ReLU -> MaxPool
    - Flatten -> Dense(256) -> Dropout -> Dense(128) -> Dropout -> Dense(num_classes)
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer and loss function.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled model
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def load_trained_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model file (.h5 or SavedModel directory)
        
    Returns:
        Loaded Keras model
    """
    return tf.keras.models.load_model(model_path)


def get_model_summary(model):
    """Print model summary."""
    model.summary()


if __name__ == "__main__":
    # Test model creation
    model = create_cnn_model()
    model = compile_model(model)
    get_model_summary(model)
    print(f"\nModel created with {NUM_CLASSES} classes: {CLASS_LABELS}")


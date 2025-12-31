"""
Training script for the CNN model.

Trains the model on the combined dataset (MNIST + operators) and saves
the best performing model.
"""

import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.model.cnn_model import create_cnn_model, compile_model, CLASS_LABELS
from training.dataset import create_combined_dataset, save_dataset
from training.augmentation import DataAugmentor, create_augmentation_generator


# Paths
MODELS_DIR = Path(__file__).parent.parent / "models"
LOGS_DIR = Path(__file__).parent.parent / "logs"


def train_model(
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    use_augmentation: bool = True,
    synthetic_samples_per_operator: int = 6000,
    save_best_only: bool = True,
    early_stopping_patience: int = 5
):
    """
    Train the CNN model.
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        use_augmentation: Whether to use data augmentation
        synthetic_samples_per_operator: Number of synthetic operator samples
        save_best_only: Save only the best model
        early_stopping_patience: Epochs to wait before early stopping
        
    Returns:
        Trained model and training history
    """
    print("=" * 60)
    print("HANDWRITTEN EQUATION SOLVER - CNN TRAINING")
    print("=" * 60)
    
    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load/create dataset
    print("\n1. Preparing dataset...")
    x_train, y_train, x_val, y_val, x_test, y_test = create_combined_dataset(
        synthetic_samples_per_operator=synthetic_samples_per_operator
    )
    
    # Save dataset for future use
    save_dataset(x_train, y_train, x_val, y_val, x_test, y_test)
    
    # Create model
    print("\n2. Creating model...")
    model = create_cnn_model()
    model = compile_model(model, learning_rate=learning_rate)
    model.summary()
    
    # Callbacks
    print("\n3. Setting up callbacks...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / "equation_solver_model.h5"
    
    callbacks = [
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor='val_accuracy',
            save_best_only=save_best_only,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=str(LOGS_DIR / timestamp),
            histogram_freq=1
        )
    ]
    
    # Training
    print("\n4. Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Augmentation: {use_augmentation}")
    
    if use_augmentation:
        # Use data augmentation generator
        augmentor = DataAugmentor(
            rotation_range=15.0,
            scale_range=(0.85, 1.15),
            translation_range=3.0,
            noise_std=0.03
        )
        
        train_generator = create_augmentation_generator(
            x_train, y_train, batch_size, augmentor
        )
        
        steps_per_epoch = len(x_train) // batch_size
        
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    
    # Evaluation
    print("\n5. Evaluating model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Per-class accuracy
    print("\n6. Per-class accuracy:")
    predictions = model.predict(x_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    for i, label in enumerate(CLASS_LABELS):
        mask = true_classes == i
        if np.sum(mask) > 0:
            class_acc = np.mean(pred_classes[mask] == i)
            print(f"   {label}: {class_acc * 100:.2f}% ({np.sum(mask)} samples)")
    
    # Save final model
    final_model_path = MODELS_DIR / f"equation_solver_model_{timestamp}.h5"
    model.save(str(final_model_path))
    print(f"\n7. Model saved to: {final_model_path}")
    print(f"   Best model saved to: {model_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return model, history


def evaluate_model(model_path: str = None):
    """
    Evaluate a trained model on the test set.
    
    Args:
        model_path: Path to saved model
    """
    if model_path is None:
        model_path = MODELS_DIR / "equation_solver_model.h5"
    
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    
    # Load test data
    from training.dataset import load_saved_dataset
    try:
        _, _, _, _, x_test, y_test = load_saved_dataset()
    except:
        print("Saved dataset not found. Creating new dataset...")
        _, _, _, _, x_test, y_test = create_combined_dataset()
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    return test_accuracy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CNN for equation solving")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable augmentation")
    parser.add_argument("--samples-per-op", type=int, default=6000, 
                       help="Synthetic samples per operator")
    parser.add_argument("--evaluate", type=str, default=None,
                       help="Path to model to evaluate (skip training)")
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate_model(args.evaluate)
    else:
        train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            use_augmentation=not args.no_augmentation,
            synthetic_samples_per_operator=args.samples_per_op
        )


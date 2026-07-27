"""
Project: Fashion Image Classification using Neural Networks
==============================================================
This project builds a neural network model that classifies images
of clothing items into 10 categories using the Fashion MNIST dataset.

Algorithm used: Feedforward Neural Network (Dense layers)
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_and_explore_data():
    """Load the Fashion MNIST dataset and display an overview of it."""
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    print("=== Data Overview ===")
    print(f"Training samples: {train_images.shape[0]}")
    print(f"Testing samples: {test_images.shape[0]}")
    print(f"Image size: {train_images.shape[1]}x{train_images.shape[2]} pixels")
    print(f"Classes: {CLASS_NAMES}\n")

    return train_images, train_labels, test_images, test_labels


def preprocess_data(train_images, test_images):
    """Normalize pixel values from 0-255 to 0-1."""
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    print("=== Preprocessing ===")
    print("Data normalized successfully ✅\n")

    return train_images, test_images


def build_model():
    """Build a simple feedforward neural network."""
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("=== Model Architecture ===")
    model.summary()
    print()

    return model


def train_model(model, train_images, train_labels, epochs=5):
    """Train the neural network on the training data."""
    print("=== Training ===")
    model.fit(train_images, train_labels, epochs=epochs)
    print("\nModel trained successfully ✅\n")

    return model


def evaluate_model(model, test_images, test_labels):
    """Evaluate the model's performance on the test data."""
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)

    print("\n=== Model Evaluation ===")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}\n")

    return test_accuracy


def predict_sample(model, test_images, test_labels, index=0):
    """Predict the class of a single test image and compare to actual label."""
    predictions = model.predict(test_images)
    predicted_label = np.argmax(predictions[index])
    actual_label = test_labels[index]

    print("=== Sample Prediction ===")
    print(f"Predicted: {CLASS_NAMES[predicted_label]}")
    print(f"Actual:    {CLASS_NAMES[actual_label]}\n")

    return predicted_label


def main():
    # 1. Load and explore the data
    train_images, train_labels, test_images, test_labels = load_and_explore_data()

    # 2. Preprocess the data
    train_images, test_images = preprocess_data(train_images, test_images)

    # 3. Build the model
    model = build_model()

    # 4. Train the model
    model = train_model(model, train_images, train_labels, epochs=5)

    # 5. Evaluate the model
    evaluate_model(model, test_images, test_labels)

    # 6. Try a sample prediction
    predict_sample(model, test_images, test_labels, index=0)


if __name__ == "__main__":
    main()

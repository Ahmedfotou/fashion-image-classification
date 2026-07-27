# Fashion Image Classification

A machine learning project that classifies images of clothing items into 10 categories using a neural network, trained on the Fashion MNIST dataset.

## What This Project Does
The model takes a 28x28 grayscale image of a clothing item and predicts which of 10 categories it belongs to:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy

## Model Architecture
A feedforward neural network with:
- Flatten layer (converts 28x28 image into a 784-value array)
- Dense hidden layer (128 neurons, ReLU activation)
- Dense output layer (10 neurons, Softmax activation)

## Project Pipeline
1. Load the Fashion MNIST dataset (70,000 images total)
2. Normalize pixel values (0-255 → 0-1)
3. Build and compile the neural network
4. Train the model for 5 epochs
5. Evaluate accuracy on unseen test data
6. Predict the class of a sample image

## Results
- Training Accuracy: ~89%
- Test Accuracy: ~87%

## How to Run
```bash
pip install tensorflow numpy
python fashion_classifier.py
```

## Author
[Your name here]

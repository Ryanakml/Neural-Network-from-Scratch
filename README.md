# Neural Network from Scratch

## Overview

This project implements a neural network from scratch using Python without relying on high-level machine learning frameworks like TensorFlow or PyTorch. The goal is to understand the fundamental concepts of neural networks by building all components manually, including forward propagation, backpropagation, and gradient descent optimization.

## Features

- **Pure Python Implementation**: Built from the ground up using only NumPy for mathematical operations
- **Modular Architecture**: Clean, object-oriented design with separate classes for layers, activations, and loss functions
- **Multiple Activation Functions**: Support for ReLU, Sigmoid, Tanh, and Softmax
- **Various Loss Functions**: Implementation of Mean Squared Error, Cross-Entropy Loss
- **Gradient Descent Optimization**: Including variations like SGD, Adam, and RMSprop
- **Visualization Tools**: Training progress plots and decision boundary visualization
- **Educational Focus**: Well-commented code with step-by-step explanations

## Project Structure

```
Neural Network from Scratch/
├── src/
│   ├── neural_network.py      # Main neural network class
│   ├── layers.py              # Layer implementations
│   ├── activations.py         # Activation functions
│   ├── loss_functions.py      # Loss function implementations
│   ├── optimizers.py          # Optimization algorithms
│   └── utils.py               # Utility functions
├── examples/
│   ├── mnist_classification.py
│   ├── xor_problem.py
│   └── regression_example.py
├── tests/
│   └── test_neural_network.py
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.7 or higher
- Basic understanding of linear algebra and calculus
- Familiarity with machine learning concepts (optional but helpful)

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd "Neural Network from Scratch"
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   ```

3. **Activate Virtual Environment**
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Required Dependencies

The project uses minimal dependencies to maintain the "from scratch" philosophy:

```
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0  # Only for dataset loading and comparison
jupyter>=1.0.0        # For example notebooks
```

## Quick Start

### Basic Usage Example

```python
from src.neural_network import NeuralNetwork
from src.layers import DenseLayer
from src.activations import ReLU, Softmax
from src.loss_functions import CategoricalCrossentropy
import numpy as np

# Create a simple neural network
nn = NeuralNetwork()
nn.add(DenseLayer(784, 128))  # Input layer
nn.add(ReLU())
nn.add(DenseLayer(128, 64))   # Hidden layer
nn.add(ReLU())
nn.add(DenseLayer(64, 10))    # Output layer
nn.add(Softmax())

# Compile the model
nn.compile(
    loss=CategoricalCrossentropy(),
    learning_rate=0.001
)

# Train the model
nn.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions
predictions = nn.predict(X_test)
```

### Running Examples

1. **XOR Problem** (Classic non-linear problem):
   ```bash
   python examples/xor_problem.py
   ```

2. **MNIST Digit Classification**:
   ```bash
   python examples/mnist_classification.py
   ```

3. **Regression Example**:
   ```bash
   python examples/regression_example.py
   ```

## Key Concepts Implemented

### 1. Forward Propagation
- Matrix multiplication for linear transformations
- Activation function applications
- Layer-by-layer computation

### 2. Backpropagation
- Chain rule implementation
- Gradient computation for weights and biases
- Error propagation through layers

### 3. Optimization
- Gradient Descent
- Stochastic Gradient Descent (SGD)
- Adam Optimizer
- RMSprop

### 4. Regularization
- L1 and L2 regularization
- Dropout (planned feature)

## Learning Objectives

By studying this implementation, you will understand:

- How neural networks process information layer by layer
- The mathematics behind backpropagation
- How gradient descent optimizes network parameters
- The role of activation functions and loss functions
- Implementation details often abstracted by frameworks

## Testing

Run the test suite to verify implementations:

```bash
python -m pytest tests/ -v
```

## Contributing

Contributions are welcome! Areas for improvement:

- Additional activation functions
- More optimization algorithms
- Convolutional layers
- Recurrent layers
- Better visualization tools

## Educational Resources

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
- [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow
- [3Blue1Brown Neural Network Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by educational neural network implementations
- Built for learning and teaching purposes
- Thanks to the open-source community for mathematical insights


# Neural Network Regression (NumPy Only)

## Overview

This project demonstrates a simple neural network built from scratch using only NumPy, trained to fit a noisy cubic function.

## Architecture

- 1 Hidden Layer (10 neurons)
- Activation: ReLU
- Output Layer: Linear (for regression)
- Loss: Mean Squared Error (MSE)
- Optimizer: Manual SGD

## Files

- `model.py` — defines the neural network, activations, loss
- `train.ipynb` — training script with data generation and visualizations
- `README.md` — project documentation

## How to Run

1. Install requirements:
   ```
   pip install numpy matplotlib
   ```

2. Run `train.ipynb` to train the model and see plots.

## Output

- **Loss Curve**: MSE over epochs
- **Prediction Plot**: Network output vs. true function

## Notes

- Uses synthetic data: `y = x^3 + noise`
- Designed for educational/demo purposes

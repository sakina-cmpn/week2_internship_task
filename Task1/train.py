
# Train neural network on noisy cubic function y = x^3 + noise

import numpy as np
import matplotlib.pyplot as plt
from model import SimpleNeuralNetwork

# Generate synthetic data
np.random.seed(42)
X = np.linspace(-2, 2, 200).reshape(-1, 1)
y = X**3 + 0.3 * np.random.randn(*X.shape)

# Initialize model
model = SimpleNeuralNetwork(input_size=1, hidden_size=10, output_size=1, learning_rate=0.01)

# Training loop
epochs = 5000
losses = []

for epoch in range(epochs):
    loss = model.train_step(X, y)
    losses.append(loss)
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Plot loss curve
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Final predictions
y_pred = model.forward(X)

# Plot predictions vs true
plt.scatter(X, y, label="True Data", s=10)
plt.plot(X, y_pred, color='red', label="Model Prediction")
plt.title("Prediction vs True Function")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

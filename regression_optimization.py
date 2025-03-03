"""
This script performs a simple linear regression analysis using both brute-force search 
and gradient descent to estimate the best slope (m1) for a given dataset.
It compares both methods in terms of efficiency and visualizes results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(42)
sample_size = 100
x_vals = np.linspace(-10, 10, sample_size)
true_slope = 3.0  # Actual m1
intercept = 0.1  # Actual m2
noise = np.random.normal(0, 1, sample_size)  # Random noise

y_vals = true_slope * x_vals + intercept + noise  # Noisy linear data
dataset = pd.DataFrame({'x': x_vals, 'y': y_vals})

# Plot dataset
plt.scatter(dataset['x'], dataset['y'], label='Noisy Data', color='blue', alpha=0.5)
plt.plot(x_vals, true_slope * x_vals + intercept, label='True Line (No Noise)', color='red', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Generated Dataset')
plt.show()

# Brute-force approach to estimate best m1
m1_options = np.linspace(0, 5, 100)  # Range of m1 values
loss_results = []

for m in m1_options:
    predicted_y = m * dataset['x'] + intercept
    mse_loss = np.mean((dataset['y'] - predicted_y) ** 2)
    loss_results.append(mse_loss)

optimal_m1_brute = m1_options[np.argmin(loss_results)]  # Best m1 from brute-force search

# Plot loss curve for brute-force approach
plt.plot(m1_options, loss_results, label="Loss Curve")
plt.axvline(optimal_m1_brute, color='r', linestyle='dashed', label=f'Best m1: {optimal_m1_brute:.2f}')
plt.xlabel('m1 Values')
plt.ylabel('MSE Loss')
plt.title('Brute-force Search for m1')
plt.legend()
plt.show()

# Gradient Descent Implementation
estimated_m1 = np.random.uniform(0, 5)  # Initial guess for m1
learning_rate = 0.01
max_iters = 1000
tolerance = 1e-6  # Stopping condition
loss_history = []

for i in range(max_iters):
    predicted_y = estimated_m1 * dataset['x'] + intercept
    gradient = -2 * np.mean(dataset['x'] * (dataset['y'] - predicted_y))  # Compute gradient
    estimated_m1 -= learning_rate * gradient  # Update m1
    loss = np.mean((dataset['y'] - predicted_y) ** 2)  # Compute loss
    loss_history.append(loss)
    
    if len(loss_history) > 1 and abs(loss_history[-2] - loss_history[-1]) < tolerance:
        break  # Stop if loss improvement is minimal

# Plot loss curve for Gradient Descent
plt.plot(loss_history, label="Gradient Descent Loss")
plt.xlabel("Iterations")
plt.ylabel("MSE Loss")
plt.title("Loss Reduction Using Gradient Descent")
plt.legend()
plt.show()

# Efficiency comparison
brute_force_steps = len(m1_options)
gd_steps = len(loss_history)
efficiency_factor = brute_force_steps / gd_steps

# Display results
print(f"Best m1 (Brute-force): {optimal_m1_brute:.5f}")
print(f"Best m1 (Gradient Descent): {estimated_m1:.5f}")
print(f"Gradient Descent was {efficiency_factor:.2f} times faster than brute-force search.")

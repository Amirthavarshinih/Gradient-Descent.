import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegressionEstimator:
    """
    This class performs simple linear regression analysis using brute-force search 
    and gradient descent to estimate the best slope (m1) for a given dataset.
    """
    
    def __init__(self, x_values, y_values, intercept=0.1):
        """Initialize the dataset and model parameters."""
        self.x_values = x_values
        self.y_values = y_values
        self.intercept = intercept
    
    def generate_data(self, true_slope, noise_level=1.0):
        """Generate noisy linear data for regression analysis."""
        noise = np.random.normal(0, noise_level, len(self.x_values))
        self.y_values = true_slope * self.x_values + self.intercept + noise
        
    def plot_data(self, true_slope):
        """Visualize the generated dataset with noise and true line."""
        plt.scatter(self.x_values, self.y_values, label='Noisy Data', color='blue', alpha=0.5)
        plt.plot(self.x_values, true_slope * self.x_values + self.intercept, 
                 label='True Line (No Noise)', color='red', linestyle='dashed')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('Generated Dataset')
        plt.show()
    
    def brute_force_estimation(self, slope_range=(0, 5), steps=100):
        """Find the best slope (m1) using brute-force search."""
        m1_values = np.linspace(*slope_range, steps)
        losses = []
        
        for m in m1_values:
            predicted_y = m * self.x_values + self.intercept
            mse_loss = np.mean((self.y_values - predicted_y) ** 2)
            losses.append(mse_loss)
        
        optimal_m1 = m1_values[np.argmin(losses)]
        
        # Plot the loss curve
        plt.plot(m1_values, losses, label="Loss Curve")
        plt.axvline(optimal_m1, color='r', linestyle='dashed', label=f'Best m1: {optimal_m1:.2f}')
        plt.xlabel('Slope (m1)')
        plt.ylabel('MSE Loss')
        plt.title('Brute-force Search for Optimal m1')
        plt.legend()
        plt.show()
        
        return optimal_m1
    
    def gradient_descent(self, learning_rate=0.01, max_iters=1000, tolerance=1e-6):
        """Find the best slope (m1) using gradient descent."""
        estimated_m1 = np.random.uniform(0, 5)  # Random initial guess
        loss_history = []
        
        for i in range(max_iters):
            predicted_y = estimated_m1 * self.x_values + self.intercept
            gradient = -2 * np.mean(self.x_values * (self.y_values - predicted_y))
            estimated_m1 -= learning_rate * gradient
            loss = np.mean((self.y_values - predicted_y) ** 2)
            loss_history.append(loss)
            
            if len(loss_history) > 1 and abs(loss_history[-2] - loss_history[-1]) < tolerance:
                break  # Stop if loss improvement is minimal
        
        # Plot the loss reduction curve
        plt.plot(loss_history, label="Gradient Descent Loss")
        plt.xlabel("Iterations")
        plt.ylabel("MSE Loss")
        plt.title("Loss Reduction Using Gradient Descent")
        plt.legend()
        plt.show()
        
        return estimated_m1

# Main execution
if __name__ == "__main__":
    np.random.seed(42)
    sample_size = 100
    x_values = np.linspace(-10, 10, sample_size)
    true_slope = 3.0
    intercept = 0.1
    
    # Initialize estimator object
    regression_model = LinearRegressionEstimator(x_values, np.zeros(sample_size), intercept)
    regression_model.generate_data(true_slope)
    regression_model.plot_data(true_slope)
    
    # Estimate slope using brute-force search
    best_m1_brute = regression_model.brute_force_estimation()
    
    # Estimate slope using gradient descent
    best_m1_gd = regression_model.gradient_descent()
    
    # Compare efficiency
    print(f"Best m1 (Brute-force): {best_m1_brute:.5f}")
    print(f"Best m1 (Gradient Descent): {best_m1_gd:.5f}")

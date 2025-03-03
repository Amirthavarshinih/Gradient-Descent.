# Gradient-Descent.
Linear Regression Optimization: Brute-Force vs Gradient Descent
Overview
This project explores two approaches to optimizing the slope (m1) in a simple linear regression model:

Brute-force search – Iterates over a range of possible values and selects the best one based on Mean Squared Error (MSE).
Gradient Descent – Iteratively updates m1 using the gradient of the loss function to minimize MSE.
Key Features
Dataset Generation – Creates a synthetic dataset with noise.
Brute-force Optimization – Evaluates loss across a range of slope values.
Gradient Descent Implementation – Optimizes slope iteratively.
Comparison & Efficiency Analysis – Compares both methods in terms of accuracy and efficiency.
Visualizations – Plots data points, loss curves, and optimization results.
Implementation Details
Language: Python
Libraries Used: numpy, pandas, matplotlib
Programming Paradigm: Functional approach with minimal procedural programming.
Software Development Principles:
Modular, reusable, and well-commented code.
Follows best practices in software development.
How to Run
1. Clone this repository:
git clone https://github.com/Amirthavarshinih/Gradient-Descent.git
cd Gradient-Descent
2. Install dependencies:
pip install numpy pandas matplotlib
3. Run the script:

python regression_optimization.py
Expected Output
Scatter plot of the noisy dataset.
Loss curve for brute-force optimization.
Loss curve for gradient descent.
Efficiency comparison between both methods.
Best estimated slope values using both techniques.
Results & Insights
The brute-force method is simple but inefficient.
Gradient Descent converges faster and is computationally efficient.
Loss curve visualizations provide insights into the optimization process.
Contributing
Ensure code is well-documented.
Follow object-oriented or functional design principles.
Use GitHub for collaboration and version control.

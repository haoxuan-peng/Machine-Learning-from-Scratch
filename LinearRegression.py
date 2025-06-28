import comFunc
import numpy as np
import math, copy
import matplotlib.pyplot as plt
plt.style.use('E:/Code/Python/ML/.mplstyle')

# Computing the cost function for multiple features
def compute_cost_multi(x, y, w, b):
    """
    Computes the cost function for linear regression with multiple features.
    
    Args:
        x (ndarray (m, n)): Data, m examples, n features
        y (ndarray (m,)): target values
        w (ndarray (n,)): model parameters
        b (scalar): bias term
    Returns:
        cost (float): The cost of using w,b as the parameters for linear regression to fit the data points in x and y
    """
    m = len(y)
    
    predictions = comFunc.compute_model_output(x, w, b) # Compute predictions for all examples
    # Calculate the cost
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    
    return cost

# Computing gradients
def compute_gradients(x, y, w, b):
    """
    Computes the gradients for linear regression with multiple features.
    
    Args:
        x (ndarray (m, n)): Data, m examples, n features
        y (ndarray (m,)): target values
        w (ndarray (n,)): model parameters
        b (scalar): bias term
    
    Returns:
        dj_dw (ndarray (n,)): Gradient of the cost with respect to w
        dj_db (scalar): Gradient of the cost with respect to b
    """
    m = len(y)
    predictions = comFunc.compute_model_output(x, w, b)  # Compute predictions for all examples
    errors = predictions - y  # Calculate the errors
    
    dj_dw = (1 / m) * np.dot(x.T, errors)  # Gradient with respect to w
    dj_db = (1 / m) * np.sum(errors)  # Gradient with respect to b
    
    return dj_dw, dj_db

# Gradient descent function for multiple features
def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking num_iters gradient steps with learning rate alpha.
    
    Args:
        x (ndarray (m, n)): Data, m examples, n features
        y (ndarray (m,)): target values
        w_in (ndarray (n,)): initial values of model parameters
        b_in (scalar): initial value of bias term
        alpha (float): learning rate
        num_iters (int): number of iterations to run gradient descent
        
    Returns:
        w, b (ndarray (n,), scalar): updated model parameters after running gradient descent
        J_history (list): history of cost values
        p_history (list): history of parameters [w, b]        
    """
    w = copy.deepcopy(w_in)  # avoid modifying global w_in
    J_history = []
    p_history = []
    w = copy.deepcopy(w_in)  # Ensure w is a copy of w_in
    b = b_in
    
    for i in range(num_iters):
        # Calculate the gradient and update parameters using compute_gradients
        dj_dw, dj_db = compute_gradients(x, y, w, b)
        
        # Update Parameters
        w -= alpha * dj_dw
        b -= alpha * dj_db
        
        # Save cost J at each iteration
        if i < 100000:  # prevent excessive memory usage
            J_history.append(compute_cost_multi(x, y, w, b))
            p_history.append([w.copy(), b])
        
        if i % math.ceil(num_iters / 10) == 0 or i == num_iters - 1:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:.2f}, w: {w}, b: {b:.2f}")
        
    return w, b, J_history, p_history

# Z-score normalization for multiple features
def z_score_normalize_multi(x):
    """
    Computes x, zscore normalized by column for multiple features.
    
    Args:
        x (ndarray (m, n)): input data, m examples, n features
    
    Returns:
        x_norm (ndarray (m, n)): input normalized by column
        mu (ndarray (n,)): mean of each feature
        sigma (ndarray (n,)): standard deviation of each feature
    """
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    
    x_norm = (x - mu) / sigma # Numpy broadcasting for normalization
    
    return x_norm, mu, sigma

# Plot the feature versus target values
def plot_feature_vs_target(x, y, feature_names, target_name):
    """ Plots the feature versus target values. """
    plt.figure(figsize=(15, 4))
    
    for i in range(x.shape[1]):
        plt.subplot(1, x.shape[1], i+1)
        plt.scatter(x[:, i], y, marker='x', c='red', alpha=0.7)
        plt.xlabel(feature_names[i])
        plt.ylabel(target_name)
        plt.title(f'{feature_names[i]} vs {target_name}')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot cost versus iterations
def plot_cost_vs_iterations(J_history):
    """
    Plots the cost function versus iterations.
    
    Args:
        J_history (list): history of cost values
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(15, 4))
    
    # Check if J_history has enough iterations to split
    total_iters = len(J_history)
    if total_iters > 100:
        # Left figure: the first 100 iterations
        ax1.plot(J_history[:100])
        ax1.set_title('Cost vs. Iterations (Start)')
        
        # Right figure: the last 100 iterations
        mid_point = total_iters // 2
        ax2.plot(mid_point + np.arange(len(J_history[mid_point:])), J_history[mid_point:])
        ax2.set_title('Cost vs. Iterations (End)')
    else:
        # If J_history has less than 100 iterations, plot all in one figure
        ax1.plot(J_history)
        ax1.set_title('Cost vs. Iterations')
        ax2.set_visible(False)
    
    ax1.set_xlabel('Iterations steps')
    ax2.set_xlabel('Iterations steps')
    ax1.set_ylabel('Cost')
    ax2.set_ylabel('Cost')
    plt.show()

# Test
if __name__ == "__main__":
    
    # Training data
    x_train, y_train = comFunc.load_data('./LR_trainingData.txt')
    # Test data
    x_test, y_test = comFunc.load_data('./LR_testData.txt')
    print("Data loaded successfully.")
    
    # Feature names and target name
    feature_names = ['Population']
    target_name = 'Profit'
    
    # Plot the feature versus target values
    plot_feature_vs_target(x_train, y_train, feature_names, target_name)
    
    # Hyperparameters
    num_iters = 20000  # Number of iterations for gradient descent
    alpha = 0.001  # Learning rate for gradient descent
    # Parameters
    w_initial = np.zeros_like(x_train[0], dtype=float)  # Initialize weights to zeros
    b_initial = 0.  # Initial bias term
    
    # Train the model without normalization
    print("\nTraining without normalization...")
    w, b, J_history, p_history = gradient_descent(x_train, y_train, w_initial, b_initial, alpha, num_iters)
    print(f"Final parameters: w = {w}, b = {b:.2f}")
    # Plot cost vs iterations
    plot_cost_vs_iterations(J_history)
    # Plot cost history
    plt.figure(figsize=(8, 6))
    plt.plot(J_history)
    plt.title('Cost During Training without Normalization')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    train_predictions = comFunc.compute_model_output(x_train, w, b)
    test_predictions = comFunc.compute_model_output(x_test, w, b)
    plt.figure(figsize=(16, 6))
    # Plot the predicted values to see the linear fit
    plt.subplot(1, 2, 1)
    plt.plot(x_train, train_predictions, c = "b")
    plt.scatter(x_train, y_train, marker='x', c='r') 
    plt.title("Profits vs. Population per city (Without Normalization)")
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')
    plt.legend(['Predicted', 'Training Data'])
    plt.grid(True)
    # Plot the test predictions
    plt.subplot(1, 2, 2)
    plt.plot(x_test, test_predictions, c = "g")
    plt.scatter(x_test, y_test, marker='x', c='r')
    plt.title("Test Set: Profits vs. Population per city (Without Normalization)")
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')
    plt.legend(['Predicted', 'Test Data'])
    plt.grid(True)
    plt.show()
    
    # Train the model with normalization
    # Normalize the training data
    x_norm, mu, sigma = z_score_normalize_multi(x_train)
    x_test_norm = (x_test - mu) / sigma
    
    print("\nTraining with normalization...")
    w_norm, b_norm, J_history_norm, p_history_norm = gradient_descent(x_norm, y_train, w_initial, b_initial, alpha, num_iters)
    print(f"Final parameters: w = {w_norm}, b = {b_norm:.2f}")
    w_original = w_norm / sigma  # Denormalize weights
    b_original = b_norm - (w_norm * mu / sigma)
    
    # Plot cost vs iterations
    plot_cost_vs_iterations(J_history_norm)
    # Plot cost history
    plt.figure(figsize=(8, 6))
    plt.plot(J_history_norm)
    plt.title('Cost During Training with Normalization')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot the final model prediction vs actual values
    train_predictions_norm = comFunc.compute_model_output(x_train, w_original, b_original)
    test_predictions_norm = comFunc.compute_model_output(x_test, w_original, b_original)
    plt.figure(figsize=(16, 6))
    # Plot the predicted values to see the linear fit
    plt.subplot(1, 2, 1)
    plt.plot(x_train, train_predictions_norm, c = "b")
    plt.scatter(x_train, y_train, marker='x', c='r') 
    plt.title("Profits vs. Population per city (With Normalization)")
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')
    plt.legend(['Predicted', 'Training Data'])
    plt.grid(True)
    # Plot the test predictions
    plt.subplot(1, 2, 2)
    plt.plot(x_test, test_predictions_norm, c = "g")
    plt.scatter(x_test, y_test, marker='x', c='r')
    plt.title("Test Set: Profits vs. Population per city (With Normalization)")
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')
    plt.legend(['Predicted', 'Test Data'])
    plt.grid(True)
    plt.show()
    
    
    # Create a comparison plot
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    x_axis = np.arange(len(y_train))
    plt.plot(x_axis, train_predictions, 'bo-', label='Training Predictions')
    plt.plot(x_axis, y_train, 'rx-', label='Training Actual')
    plt.title('Training Set: Predicted vs Actual')
    plt.xlabel('Sample Index')
    plt.ylabel(target_name)
    plt.legend()
    plt.grid(True)
    
    # Test set predictions
    plt.subplot(1, 3, 2)
    x_axis_test = np.arange(len(y_test))
    plt.plot(x_axis_test, test_predictions, 'go-', label='Test Predictions')
    plt.plot(x_axis_test, y_test, 'rx-', label='Test Actual')
    plt.title('Test Set: Predicted vs Actual')
    plt.xlabel('Sample Index')
    plt.ylabel(target_name)
    plt.legend()
    plt.grid(True)
    
    # Predicted vs Actual values
    plt.subplot(1, 3, 3)
    plt.scatter(y_train, train_predictions, c='blue', alpha=0.7, label='Training')
    plt.scatter(y_test, test_predictions, c='green', alpha=0.7, label='Test')
    all_actual = np.concatenate([y_train, y_test])
    plt.plot([min(all_actual), max(all_actual)], [min(all_actual), max(all_actual)], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # # Print the comparison
    # print("\nTraining Set Comparison:")
    # for i in range(len(y_train)):
    #     print(f"Sample {i+1}: Actual = {y_train[i]}, Predicted = {train_predictions[i]:.2f}, Error = {abs(y_train[i] - train_predictions[i]):.2f}")
    
    # print("\nTest Set Comparison:")
    # for i in range(len(y_test)):
    #     print(f"Sample {i+1}: Actual = {y_test[i]}, Predicted = {test_predictions[i]:.2f}, Error = {abs(y_test[i] - test_predictions[i]):.2f}")
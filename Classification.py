import comFunc
import numpy as np
import math, copy
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
plt.style.use('E:/Code/Python/ML/.mplstyle')

dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#c00000', dlmagenta='#FF40FF', dlpurple='#7030A0')
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#c00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'
dlcolors =[dlblue, dlorange, dldarkred, dlmagenta, dlpurple]

# Feature mapping
def map_features(X1, X2):
    """
    Feature mapping function to polynomial features    
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)

# Plot data
def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    """
    Plots the data points X and y.
    
    Args:
        X (ndarray): Features of the dataset.
        y (ndarray): Target values of the dataset.
        pos_label (str): Label for positive examples.
        neg_label (str): Label for negative examples.
    """
    positive = y == 1
    negative = y == 0
    
    # Plot examples
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)

# Decision boundary
def plot_decision_boundary(w, b, X, y):
    # Credit to dibgerge on Github for this plotting code
     
    plot_data(X[:, 0:2], y)
    
    if X.shape[1] <= 2:
        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)
        
        plt.plot(plot_x, plot_y, c="b")
        
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        
        z = np.zeros((len(u), len(v)))

        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = sigmoid(np.dot(map_features(u[i], v[j]), w).item() + b)
        
        # important to transpose z before calling contour       
        z = z.T
        
        # Plot z = 0
        plt.contour(u, v, z, levels = [0.5], colors="g")
    
    plt.show()

# Draw a threshold at x=0.5
def draw_threshold(ax, x):
    """ Draws a threshold """
    
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.fill_between([xlim[0], x], [ylim[1], ylim[1]], alpha=0.2, color=dlblue)
    ax.fill_between([x, xlim[1]], [ylim[1], ylim[1]], alpha=0.2, color=dldarkred)
    ax.annotate("z >= 0", xy=[x, 0.5], xycoords='data', xytext=[30, 5], textcoords='offset points')
    
    d = FancyArrowPatch(
        posA=(x, 0.5), posB=(x+3, 0.5), color=dldarkred, arrowstyle='simple, head_width=5, head_length=10, tail_width=0.0'
    )
    ax.add_artist(d)
    ax.annotate("z < 0", xy=[x, 0.5], xycoords='data', xytext=[-50, 5], textcoords='offset points', ha='left')
    
    f = FancyArrowPatch(
        posA=(x, 0.5), posB=(x-3, 0.5), color=dlblue, arrowstyle='simple, head_width=5, head_length=10, tail_width=0.0'
    )
    
    ax.add_artist(f)

# Sigmoid function
def sigmoid(z):
    """
    Computes the sigmoid of z
    
    Args:
        z (ndarray): A scalar, numpy array of any size.
    
    Returns:
        g (ndarray): Sigmoid(z), with the same shape as z.
    """
    g = 1 / (1 + np.exp(-z))
    
    return g

# Computing the cost function
def compute_cost(X, y, w, b, lambda_=0):
    """
    Computes the cost over all examples
    
    Args:
        x: (ndarray Shape (m, n)) data, m examples by n features
        y: (array_like Shape (m,)) target value
        w: (array_like Shape (n,)) Values of parameters of the model
        b: scalar Values of bias parameter of the model
        lambda_: unused placeholder
        
    Returns:
        cost: (scalar) cost
    """
    m = len(y)
    cost = 0
    
    for i in range(m):
        z = comFunc.compute_model_output(X[i], w, b)
        f_wb = sigmoid(z)
        f_wb = np.clip(f_wb, 1e-15, 1 - 1e-15)
        cost += -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)
    
    total_cost = (1 / m) * (cost + 0.5 * lambda_ * np.sum(np.square(w)))
    
    return total_cost

# Computing gradients
def compute_gradients(X, y, w, b, lambda_=0):
    """
    Computes the gradient for logistic regression.
    
    Args:
        X (ndarray (m, n)): Data, m examples by n features
        y (ndarray (m, 1)): actual values
        w (ndarray (n, 1)): values of parameters of the model
        b (scalar): values of parameter of the model
        lambda_ (scalar): unused placeholder
    
    Returns:
        dj_dw (ndarray (n, 1)): The gradient of the cost w.r.t. the parameters w.
        dj_db (scalar): The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
      
    for i in range(m):
        f_wb_i = sigmoid(comFunc.compute_model_output(X[i], w, b))
        err_i = f_wb_i - y[i]
        
        dj_dw += err_i * X[i]
        dj_db += err_i
    
    dj_dw /= m
    dj_dw += (lambda_ / m) * w
    dj_db /= m
    
    return dj_dw, dj_db

# Gradient descent function
def gradient_descent(X, y, w_in, b_in, alpha, num_iters, lambda_=0):
    """
    Performs batch gradient descent to learn theta. Updates theta by takeing num_iters gradient steps with learning rate alpha
    
    Args:
        X (ndarray (m, n)): Data, m examples by n features
        y (ndarray (m, 1)): actual values
        w_in (ndarray (n, 1)): initial values of parameters of the model
        b_in (scalar): initial value of parameter of the model
        alpha (float): learning rate
        num_iters (int): number of iterations to run gradient descent
        lambda_ (scalar, float): regularization constant
    
    Returns:
        w (ndarray (n, 1)): Updated values of parameters of the model after running gradient descent
        b (scalar): Updated value of parameter of the model
        J_history (list): An array to store the cost J at each iteration
        w_history (list): An array to store the values of w at each iteration
    """
    # Number of training examples
    m = len(y)
    w = copy.deepcopy(w_in)  # Make a copy of w_in to avoid modifying the original
    b = b_in  # Make a copy of b_in to avoid modifying the original
    
    # An array to store the cost J and w's at each iteration
    J_history = []
    w_history = []
    
    for i in range(num_iters):
        # Calculate the gradients
        dj_dw, dj_db = compute_gradients(X, y, w, b, lambda_)
        
        # Update the parameters
        w -= alpha * dj_dw
        b -= alpha * dj_db
        
        # Save the cost J and w's at each iteration
        if i < 100000:  # Avoid overflow in J_history    
            cost = compute_cost(X, y, w, b, lambda_)
            J_history.append(cost)
        
        # Print cost every at intervals 10 times or as many iterations if less than 10
        if i % math.ceil(num_iters / 10) == 0 or i == num_iters - 1:
            w_history.append(w)
            print(f"Iteration {i:4}/{num_iters}: Cost {J_history[-1]:8.2f}")
    
    return w, b, J_history, w_history

# Prediction function
def predict(X, w, b):
    """
    Predicts the class label for each example in X using learned parameters w and b
    
    Args:
        X (ndarray (m, n)): Data, m examples by n features
        w (ndarray (n, 1)): learned parameters of the model
        b (scalar): learned parameter of the model
        
    Returns:
        p: (ndarray (m, 1)): The predictions for X using a threshold at 0.5
    """
    # Number of training examples
    m, n = X.shape
    p = np.zeros(m)
    
    for i in range(m):
        z_wb = comFunc.compute_model_output(X[i], w, b)
        f_wb = sigmoid(z_wb)
        if f_wb > 0.5:
            p[i] = 1
        else:
            p[i] = 0
    
    return p

# Z-score normalization function
def z_score_normalize(X):
    """
    Computes X, zscore normalized by column
    
    Args:
        X (ndarray (m, n)): input data, m examples, n features
    
    Returns:
        X_train1_norm (ndarray (m, n)): input normalized bt column
        mu1 (ndarray (n,)): mean of each feature
        sigma1 (ndarray (n,)): standard deviation of each feature
    """
    # Find the mean of each column/feature
    mu = np.mean(X, axis=0)
    # Find the standard deviation of each column/feature
    sigma = np.std(X, axis=0)
    X_train_norm = (X - mu) / sigma
    
    return X_train_norm, mu, sigma

# Denormalization function
def denormalize(X_norm, mu, sigma):
    """
    Denormalizes the normalized data X_norm using the mean mu and standard deviation sigma
    
    Args:
        X_norm (ndarray (m, n)): normalized data, m examples, n features
        mu (ndarray (n,)): mean of each feature
        sigma (ndarray (n,)): standard deviation of each feature
    
    Returns:
        X_denorm (ndarray (m, n)): denormalized data
    """
    X_denorm = X_norm * sigma + mu
    
    return X_denorm

if __name__ == "__main__":
    
    # Training1
    # Load training data
    X_train1, y_train1 = comFunc.load_data('./CL_training1.txt')
    print("Data1 loaded successfully.")
    print("Type of X_train1:", type(X_train1))
    print("Type of y_train1:", type(y_train1))
    print ('The shape of X_train1 is: ' + str(X_train1.shape))
    print ('The shape of y_train1 is: ' + str(y_train1.shape))
    print ('We have m = %d training examples\n' % (len(y_train1)))
    
    # Visualize the data
    plt.figure(figsize=(8, 6))
    plot_data(X_train1, y_train1, pos_label="Admitted", neg_label="Not admitted")
    plt.ylabel('Exam 2 score')
    plt.xlabel('Exam 1 score')
    plt.legend(loc='upper right')
    plt.title('Training Data1')
    plt.grid(True, alpha=0.3)
    plt.show()   
    
    # Initialization
    np.random.seed(1)
    initial_w = 0.01 * (np.random.rand(2) - 0.5)
    initial_b = -6.
    # Some hyperparameters
    iterations = 20000
    alpha = 0.0008
    # Train the model without normalization
    print("Training without normalization...")
    w1, b1, J_history1, _ = gradient_descent(X_train1 ,y_train1, initial_w, initial_b, alpha, iterations, 0)
    plot_decision_boundary(w1, b1, X_train1, y_train1)
    # Compute accuracy
    p1 = predict(X_train1, w1, b1)
    print('Train1 without normalization Accuracy: %f'%(np.mean(p1 == y_train1) * 100))
    
    # Plot cost history
    plt.figure(figsize=(8, 6))
    plt.plot(J_history1)
    plt.title('Cost During Training without Normalization - 1')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Train the model with normalization
    X_train1_norm, mu1, sigma1 = z_score_normalize(X_train1)
    initial_w = 0.01 * (np.random.rand(2) - 0.5)
    initial_b = 0.
    alpha = 0.005
    print(f"\nData1 Normalization parameters:")
    print(f"mu1: {mu1}")
    print(f"sigma1: {sigma1}")
    print("Training1 with normalization...")
    w1_norm, b1, J_history1_norm, _ = gradient_descent(X_train1_norm, y_train1, initial_w, initial_b, alpha, iterations, lambda_=0)
    # Denormalize the parameters for plotting
    plot_decision_boundary(w1_norm, b1, X_train1_norm, y_train1)
    # Compute accuracy
    p1_norm = predict(X_train1_norm, w1_norm, b1)
    print('Train1 with normalization Accuracy: %f'%(np.mean(p1_norm == y_train1) * 100))
    
    # Plot cost history
    plt.figure(figsize=(8, 6))
    plt.plot(J_history1_norm)
    plt.title('Cost During Training with Normalization - 1')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True, alpha=0.3)
    plt.show()


    # Training2
    # Load training data
    X_train2, y_train2 = comFunc.load_data('./CL_training2.txt')
    print("\n\nData2 loaded successfully.")
    print("Type of X_train2:", type(X_train2))
    print("Type of y_train2:", type(y_train2))
    print ('The shape of X_train2 is: ' + str(X_train2.shape))
    print ('The shape of y_train2 is: ' + str(y_train2.shape))
    print ('We have m = %d training examples' % (len(y_train2)))
    
    # Visualize the data
    plt.figure(figsize=(8, 6))
    plot_data(X_train2, y_train2, pos_label="Accepted", neg_label="Rejected")
    plt.ylabel('Microchip test 2')
    plt.xlabel('Microchip test 1')
    plt.legend(loc='upper right')
    plt.title('Training Data2')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Feature mapping
    X_train2_mapped = map_features(X_train2[:, 0], X_train2[:, 1])

    # Initialize fitting parameters
    np.random.seed(1)
    initial_w = np.random.rand(X_train2_mapped.shape[1]) - 0.5
    initial_b = 1.
    # Set regularization parameter lambda_ to 1 (you can try varying this)
    lambda_ = 0.01;                                          
    # Some gradient descent settings
    iterations = 20000
    alpha = 0.025

    print("Training2 without normalization...")
    w2, b2, J_history2, _ = gradient_descent(X_train2_mapped, y_train2, initial_w, initial_b, alpha, iterations, lambda_)
    plot_decision_boundary(w2, b2, X_train2_mapped, y_train2)
    # Compute accuracy
    p2 = predict(X_train2_mapped, w2, b2)
    print('Train2 without normalization Accuracy: %f'%(np.mean(p2 == y_train2) * 100))
    
    # Plot cost history
    plt.figure(figsize=(8, 6))
    plt.plot(J_history2)
    plt.title('Cost During Training without Normalization - 2')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Train the model with normalization
    X_train2_norm, mu2, sigma2 = z_score_normalize(X_train2)
    print(f"\nNormalization parameters:")
    print(f"mu2: {mu2}")
    print(f"sigma2: {sigma2}")
    
    # Feature mapping for normalized data
    X_train2_norm_mapped = map_features(X_train2_norm[:, 0], X_train2_norm[:, 1])
    print(X_train2_norm_mapped)
    
    w2_norm, b2_norm, J_history2_norm, _ = gradient_descent(X_train2_norm_mapped, y_train2, initial_w, initial_b, alpha, iterations, lambda_=0)
    # Compute accuracy
    p2_norm = predict(X_train2_norm_mapped, w2_norm, b2_norm)
    print('Train2 with normalization Accuracy: %f'%(np.mean(p2_norm == y_train2) * 100))
    
    # Plot cost history
    plt.figure(figsize=(8, 6))
    plt.plot(J_history2_norm)
    plt.title('Cost During Training with Normalization - 2')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True, alpha=0.3)
    plt.show()
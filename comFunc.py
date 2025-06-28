import numpy as np
import math, copy
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
plt.style.use('E:/Code/Python/ML/.mplstyle')

# Load data from a file
def load_data(file_path):
    """
    Loads data from a file and returns the features and target values.
    
    Args:
        file_path (str): Path to the data file.
        
    Returns:
        x (ndarray): Features of the dataset.
        y (ndarray): Target values of the dataset.
    """
    data = np.loadtxt(file_path, delimiter=',')
    x = data[:, :-1]  # All columns except the last one
    y = data[:, -1]   # Last column
    return x, y

# Compute model output
def compute_model_output(x, w, b):
    """ Computes the model output for given input x, weights w, and bias b. """
    
    return np.dot(x, w) + b
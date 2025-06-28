# Machine Learning Practice Repository

This repository contains my personal implementations of fundamental machine learning algorithms as part of my learning journey. The code demonstrates core concepts in supervised learning with practical examples.

## 📁 Repository Structure

```
ML/
├── LogisticRegression.py      # Logistic regression implementation
├── LinearRegression.py    # Linear regression implementation
├── comFunc.py            # Common utility functions
├── .mplstyle             # Plotting profile
├── requirements.txt      # Python dependencies
├── CL_training1.txt      # Classification dataset 1
├── CL_training2.txt      # Classification dataset 2
├── LR_trainingData.txt   # Linear regression training data
└── LR_testData.txt       # Linear regression test data
```

## 🚀 Features

### Linear Regression
- **Single and multiple feature support**
- **Gradient descent optimization**
- **Feature normalization (Z-score)**
- **Cost function visualization**
- **Training vs test performance comparison**
- **Prediction visualization**

### Logistic Regression (Classification)
- **Binary classification**
- **Sigmoid activation function**
- **Gradient descent optimization**
- **Feature mapping for polynomial features**
- **Decision boundary visualization**
- **Data normalization support**
- **Training accuracy evaluation**

### Common Utilities
- **Data loading from CSV files**
- **Model output computation**
- **Visualization tools**

## 📊 Datasets

- **Linear Regression**: Population vs Profit prediction
- **Classification Dataset 1**: Exam scores for university admission
- **Classification Dataset 2**: Microchip quality control (with polynomial features)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ML
```

2. Create and activate virtual environment:
```bash
python -m venv venv_ML
# On Windows
venv_ML\Scripts\activate
# On macOS/Linux
source venv_ML/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Linear Regression
```bash
python LinearRegression.py
```

### Logistic Regression
```bash
python LogisticRegression.py
```

Both scripts will:
- Load the respective datasets
- Train models with and without normalization
- Display cost function convergence
- Show prediction visualizations
- Output training accuracy (for classification)

## 📈 Key Algorithms Implemented

### Gradient Descent
- Batch gradient descent optimization
- Learning rate: configurable
- Iterations: 20,000 (default)
- Cost function monitoring

### Feature Normalization
- Z-score normalization for improved convergence
- Mean centering and standard deviation scaling
- Denormalization for final parameter interpretation

### Polynomial Feature Mapping
- Up to 6th degree polynomial features
- Used for complex decision boundaries
- Implemented in Classification dataset 2

## 📋 Dependencies

- `numpy >= 2.3.1` - Numerical computations
- `matplotlib >= 3.10.3` - Data visualization
- `contourpy >= 1.3.2` - Contour plotting support

## 🎯 Learning Objectives

This repository demonstrates understanding of:

- **Supervised learning fundamentals**
- **Cost function optimization**
- **Gradient descent algorithm**
- **Feature engineering and normalization**
- **Model evaluation and visualization**
- **Overfitting prevention techniques**

## 📝 Implementation Details

### Cost Functions
- **Linear Regression**: Mean Squared Error (MSE)
- **Logistic Regression**: Cross-entropy loss with regularization support

### Optimization
- **Algorithm**: Batch gradient descent
- **Learning rates**: Adaptive based on problem complexity
- **Convergence**: Cost function monitoring across iterations

### Visualization
- Training data scatter plots
- Cost function convergence curves
- Decision boundaries (classification)
- Prediction vs actual value comparisons

## 🔄 Future Improvements

- [ ] Add regularization (Ridge/Lasso) for linear regression
- [ ] Implement mini-batch gradient descent
- [ ] Add cross-validation
- [ ] Support for multi-class classification
- [ ] Feature selection algorithms
- [ ] Advanced optimization algorithms (Adam, RMSprop)

## 📚 Learning Resources

This implementation is based on fundamental machine learning concepts from:
- Andrew Ng's Machine Learning Course
- Pattern Recognition and Machine Learning (Bishop)
- Hands-On Machine Learning (Géron)

---

**Note**: This repo is for learning purposes and tracks my ML progress.
# Machine Learning
Subject taught by Eduardo Bezerra

##Supervised Learning

###Algorithms:
- Linear Regression
- Logistic Regression
obs : (Polynomial Features)
- KNN (K-NearestNeighbors)
- Decision Trees

###Cost Functions:
- Mean Squared Error (MSE)
- Binary? Cross Entropy (LogLoss)

###Model Evaluation
Classification:
- Confusion Matrix (TP, FP, TN, FN)
- Roc Curve (TPR, FPR)
- Thresholds
- Accuracy, Prediction Error
- Recall, Precision
- Precision-Recall Curve
- F1-score

Regression:
- MSE, RMSE, MAE
- Predicted x Actual
- Residual Plot
- R2-score

###Model Selection
- Two Way Holdout (train/test)
- Three Way Holdout (train/val/test)
- K-fold Cross Validation (HyperParams)
- Nested Cross Validation (Algorithm)
- Grid, Random, Bayes Search (HyperParams)

###Model Calibration
- Bins
- Mean Predict Value, Empirical Accuracy
- Calibration Curve
- Over x Under Confidence
- ECE Expected Calibration Error (Probabilistic MSE)
- Log Loss (Probabilistic CrossEntropy)
- Brier Score (Confidence x Reality)

Calibration Models:
- Platt Scaling (Sigmoid/LogisticRegression)
- Isotonic Regression (Monotonic/Non Parametric)
- Temperature Scaling

###Optimization
- Gradient Descent
- Sthocastic Gradient Descent
- Limited Memory BFGS

###Unbalanced Data (Classes)
- UnderSampling
- OverSampling (SMOTE)
- Threshold Adjusting

###Features
Encoding (Categorical)
- One Hot Encoding (Dummy)
- Frequency Encoding
- Ordinal Encoding
- Target Encoding

Scaling (Numerical)
- Min Max
- Standardization
- Robust Scaling
- Log Transform (Skewed)

###Errors
- Generalization Error (Theoretical)
- Empirical Error (Training)
- Validation Error (Testing)

Breakdown
- Bias and Variance
- Model Complexity
- Overfitting and Underfitting

###Regularization
- Lasso (L1) -> absolute
- Ridge (L2) -> squared
- Elastic Net (L1 & L2)

###Model Diagnostics
- Learning Curve (error metric x train data size)
- Loss Curve (loss x epochs)
- Validation Curve (val score x complexity)
- Feature Importance, Data Leakage

###Ensemble Models

## Deep Learning
- Tensors and Pytorch
- ANNs
- Perceptron
- MLP
- Backpropagation
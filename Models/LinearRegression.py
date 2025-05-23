'''
__________________________________________________________________________________________________________________________________________________________________________________
Understand Linear Regression Model:
Goal: Find linear relationship between input featues X and target ŷ using the following formula:

    ŷ = wX + b
    
    where:
        w = weights
        X = input features
        ŷ = target variable
        b = bias
__________________________________________________________________________________________________________________________________________________________________________________
Objective Function:
Use Mean Squared Error as the loss function to find w and b that minimizes the distance between predicted values and actual values
MSE = 1/n * np.sum((y - ŷ)^2)
__________________________________________________________________________________________________________________________________________________________________________________
Compute Gradients:
Use Gradient Descent to iteratively update w and b. Compute partial derivatives
dmse/dw = -2/n * np.sum((y - ŷ)x)
dmse/db = -2/n * np.sum(y - ŷ)
__________________________________________________________________________________________________________________________________________________________________________________
Implement Gradient Descent:
Use gradients to update parameters:

w := w - a * dmse/dw
b := b - a * dmse/db

where:
    a = learning rate
__________________________________________________________________________________________________________________________________________________________________________________
'''
import sys 
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from Utils.HelperFunctions import expand_polynomial_features


class LinearRegression:
    def __init__(self, learning_rate = 0.01, num_epochs = 100, regularization = None, lambda_ = 1.0, use_polynomial = False, degree = 1):
        self.lr = learning_rate
        self.epochs = num_epochs
        self.regularization = regularization
        self.lambda_ = lambda_
        self.use_poly = use_polynomial
        self.degree = degree
        self.b = 0
        self.loss_history = []
        
    def predict_raw(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.use_poly == True:
            X = expand_polynomial_features(X, self.degree)
        y_pred = np.dot(X, self.w) + self.b
        return y_pred
    
    def compute_loss(self, y_pred, y_actual):
        loss = 1/len(y_actual) * np.sum((y_actual - y_pred)**2)
        return loss
    
    def compute_gradients(self, X, y_actual, y_pred):
        dw = -2/len(y_actual) * (np.dot(np.transpose(X), (y_actual - y_pred)))
        db = -2/len(y_actual) * np.sum((y_actual - y_pred))
        if self.regularization == "L1":
            dw += self.lambda_ * np.sign(self.w)
        elif self.regularization == "L2":
            dw += self.lambda_ * self.w
        return dw, db
    
    def fit(self, X, y):
        if self.use_poly:
            X = expand_polynomial_features(X, self.degree)
        self.w = np.zeros(X.shape[1])
        for n in range(self.epochs):
            preds = np.dot(X, self.w) + self.b
            dw, db = self.compute_gradients(X, y, preds)
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
            loss = self.compute_loss(preds, y)
            self.loss_history.append(loss)
        return self.w, self.b

        
    def predict(self, X):
        preds = self.predict_raw(X)
        return preds
    
    def evaluate(self, y_pred, y):
        loss = self.compute_loss(y_pred, y)
        return loss
    
        
            
        
        
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

import numpy as np

class LinearRegression:
    def __init__(self, learning_rate = 0.01, num_epochs = 100):
        self.lr = learning_rate
        self.epochs = num_epochs
        self.b = 0
        
    def predict_raw(self, X):
        y_pred = (np.dot(self.w, X)) + self.b
        return y_pred
    
    def compute_loss(self, y_pred, y_actual):
        loss = 1/len(y_actual) * np.sum((y_actual - y_pred)**2)
        return loss
    
    def compute_gradients(self, X, y_actual, y_pred):
        dw = -2/len(y_actual) * (np.dot(np.transpose(X), (y_actual - y_pred)))
        db = -2/len(y_actual) * np.sum((y_actual - y_pred))
        return dw, db
    
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        for n in range(self.epochs):
            preds = self.predict_raw(X)
            dw,db =  self.compute_gradients(X, y, preds)
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
        return self.w, self.b
        
    def predict(self, X):
        preds = self.predict_raw(X)
        return preds
    
    def evaluate(self, y_pred, y):
        loss = self.compute_loss(y_pred, y)
        return loss
    
        
            
        
        
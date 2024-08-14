import pandas as pd
import numpy as np

class LinearRegression:
    """
    Implementation of a linear regression model using the least squares method and gradient descent algorithm.

    Provides methods for fitting the model to data, making predictions, calculating the cost, and feature scaling.
    """

    def __init__(self, learning_rate = 1e-3, iterations = 1000, fit_intercept = True, scale_features = True):
        """
        Initializes the linear regression model.

        parameters:
        learning_rate: float, optional
            The learning rate for the gradient descent algorithm; default is 0.001.
        iterations: int, optional
            The number of iterations the gradient descent algorithm runs for; default is 1000.
        fit_intercept: bool, optional
            Whether to include an intercept in the linear regression model's calculation; default is True.
        scale_features: bool, optional
            Whether to apply feature scaling; default is True
        
        attributes:
        learning_rate: float
            Stores the learning rate for the gradient descent algorithm.
        iterations: int
            Stores the number of iterations the gradient descent algorithm runs for.
        fit_intercept: bool
            Stores the decision whether to include an intercept in the linear regression model's calculation.
        scale_features: bool
            Stores the decision whether to apply feature scaling.
        weights: numpy.ndarray
            Stores coefficients for each independent variable and intercept term (if applicable) in the linear regression model; initialized to None.
        """

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.fit_intercept = fit_intercept
        self.scale_features = scale_features
        self.weights = None
    
    def fit(self, X, y):
        """
        Fits the linear regression model to the provided input features and target values.

        parameters:
        X: numpy.ndarray
            The input features with shape (m_examples, n_features).
        y: numpy.ndarray
            The target values with shape (m_examples,).

        returns:
        self: LinearRegression
            The instance of the linear regression model.
        """

        if self.scale_features:
            X = self.apply_feature_scaling(X)

        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0]),1), X), axis = 1)

        self.weights = np.zeros(X.shape[1],)

        for _ in range(self.iterations):
            self.gradient_descent(X, y)

        return self

    def predict(self, X):
        """
        Predicts target values for the provided input features.

        parameters:
        X: numpy.ndarray
            The input features with shape (m_examples, n_features).
        
        returns:
        numpy.ndarray
            The predicted values for the input features with shape (m_examples,)
        """

        return np.dot(self.weights, X)

    def calculate_cost(self, X, y):
        """
        Calculates the cost function for the linear regression model.

        parameters:
        X: numpy.ndarray
            The input features with shape (m_examples, n_features).
        y: numpy.ndarray
            The target values with shape (m_examples,).

        returns:
        float
            The value of the cost function
        """

        pass

    def gradient_descent(self, X, y):
        """
        Carries out the gradient descent algorithm to optimize the linear regression model's parameters.

        parameters:
        X: numpy.ndarray
            The input features with shape (m_examples, n_features).
        y: numpy.ndarray
            The target values with shape (m_examples,).
        """

        pass

    def apply_feature_scaling(self, X):
        """
        Applies feature scaling to the input features.

        parameters:
        X: numpy.ndarray
            The input features with shape (m_examples, n_features).
        
        returns:
        numpy.ndarray
            The scaled input features with shape (m_examples, n_features).
        """

        pass

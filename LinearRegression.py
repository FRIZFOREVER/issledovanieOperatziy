import pandas as pd
import numpy as np


class LinearRegression:
    def __init__(self, 
                 learning_rate=0.01, 
                 n_iterations=100,
                 loss=None,
                 grad=None,
                 regularisation=None,
                 lambda_=1):
        """
        Это конструктор объекта класса Линейной регрессии
        ---
        Параметры:
            learning_rate (float, optional): Скорость обучения, множитель градиента. 
            Стандатное значение - 0.01
        ---
            n_iterations (int, optional): Количество итераций градиентного спуска
            Стандартное значение - 100
        ---
            loss (_type_, optional): Конфигурабельная функция потерь.
            Функция, принимающая на вход два numpy array: 
            y_true - истинные значения
            y_pred - предсказания модели
            и возвращающая ошибку предсказания. 
        ---
            grad (_type_, optional): Конфигурабельная функция градиента.
            Функция, принимающая на вход:
            y_true - истинные значения
            y_pred - предсказания модели
            X - Значения на входе
            n_samples - Количество векторов в батче
            И возвращающая градиент весов и байеса в формате tuple: 
            (grad_weights, grad_bias)
        ---
            В случае, если в loss и grad были переданы функции - они будут использоваться вместо
            Стандартной. Иначе используется MSE и grad_MSE соответственно.
        ---
            regularisation (_type_, optional): Тип регуляризации, имеет два возможных значения
            'L1' - применяется Lasso регуляризация - штраф в виде линейной коомбинации весов модели
            'L2' - применяется Ridge регуляразация - штраф в виде линейной коомбинации квадратов весов модели
            Стандартное значение - None.
        ---
            lambda_ (int, optional): Параметр силы регуляризации.
            При невыбранном методе регуляризации данный параметр ничего не делает
            Стандартное значение - 1
        ---
        """
        self.__weights__ = None
        self.__bias__ = 0
        self.__learning_rate__ = learning_rate
        self.__n_iterations__ = n_iterations
        if (loss) and (grad):
            self.__loss__ = loss
            self.__grad__ = grad
        else:
            self.__loss__ = self.__mse__
            self.__grad__ = self.__mse_grad__
        self.__regularisation__ = regularisation
        self.__lambda__ = lambda_
        
    def fit(self, X: np.array, y:np.array):
        """
        Это метод, включающий в себя обучение модели на данных.
        ---
            Параметры:
            X - Матрица типа numpy array с размерностью Shape: (n_samples, n_features)
            y - Вектор true значений типа numpy array
"""
        try:
            n_samples, n_features = X.shape
        except Exception as e:
            print(f'{e}: Shape error')
            return e
        
        # Init model weights and bias 
        self.__bias__ = 0
        self.__weights__ = np.zeros((n_features, 1))
        
        for epoch in range(self.__n_iterations__):
            # Make prediction
            y_pred = self.predict(X)
            
            # Calculate gradient
            grad_weights, grad_bias = self.__grad__(y, y_pred, X, n_samples)
            
            # Apply regularisation
            if self.__regularisation__:
                grad_weights = self.__apply_regularisation__(grad_weights)
                
            # Update weights and bias
            self.__weights__ -= self.__learning_rate__ * grad_weights
            self.__bias__ -= self.__learning_rate__ * grad_bias

    def predict(self, X):
        """
        Метод, делающий предсказание модели
        ---
            Параметры:
            X - Матрица типа numpy array с размерностью Shape: (n_samples, n_features)
"""
        return X @ self.__weights__ + self.__bias__

    def __mse__(self, y_true: np.array, y_pred: np.array):
        return np.mean((y_pred - y_true) ** 2)

    def __mse_grad__(self, y_true, y_pred, X, n_samples):
        grad_weights = 2 / n_samples * X.T @ (y_pred - y_true)
        grad_bias = 2 / n_samples * np.sum(y_pred - y_true)
        return grad_weights, grad_bias

    def __apply_regularisation__(self, grad_weights):
            if self.__regularisation__ == 'L2':
                return grad_weights + np.sum(self.__weights__ ** 2)
            if self.__regularisation__ == 'L1':
                return grad_weights + (self.__lambda__ * np.sign(self.__weights__))

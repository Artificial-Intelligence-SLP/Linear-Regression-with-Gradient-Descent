import numpy as np
# Operations on numpy arrays are element-wise.
# Example:
# Let x = np.array([1, 5, 10])
# x*2 = [2, 10, 20]
# Let y = np.array([10, 10, 10])
# y - x = [9, 5, 0]
# We use numpy to handle numerical operations because it's crazy efficient
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def initialize_dataframe(source):
    df = None
    if source == 'insurance':
        df = pd.read_csv('insurance.csv')  # Reads CSV file into pandas dataframe
    elif source == 'salary':
        df = pd.read_csv('Salary_Data.csv')
    # Feel free to print the dataframe to see what it looks like.
    # print(df)
    return df


class LinearRegression:
    def __init__(self, dependent, independent, source='salary'):
        dataframe = initialize_dataframe(source)
        print('\nDataframe of {0} items and {1} columns initialized.'.format(len(dataframe), len(dataframe.columns)))
        # Initialize training/testing sets
        Y = np.array(dataframe[dependent]) / 1000  # Y has large values, risking overflow.
        X = np.array(dataframe[independent])  # We pass the column headers, and access columns as arrays with df['column_name']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, train_size=0.75, shuffle=True)
        # Model parameters
        self.b = [0, 1]
        print('Parameters randomized to {}.'.format(self.b))

    def model_loss(self, X, Y):  # X,Y, and predicted_Y are all numpy arrays
        predicted_Y = self.predict(X)
        error = Y - predicted_Y
        squared_error = error ** 2
        mean_squared_error = np.mean(squared_error)
        return mean_squared_error

    def predict(self, X):
        predicted_Y = self.b[0] + self.b[1] * X  # X is a numpy array, therefore Y is too
        return predicted_Y

    def update_parameters(self, learning_rate):
        predicted_Y = self.predict(self.X_train)
        gradient = [2 * np.mean(predicted_Y - self.Y_train),    # Be sure you are doing (y_hat - y)
                    2 * np.mean((predicted_Y - self.Y_train) * self.X_train)]
        self.b[0] = self.b[0] - (learning_rate * gradient[0])
        self.b[1] = self.b[1] - (learning_rate * gradient[1])
        return gradient  # This is just a list

    # Batch gradient descent includes ALL data to update parameters.
    # It provides the "smoothest" descent, but requires many computations to complete each step.
    def batch_gradient_descent(self):
        n_epoch = 0
        MAX_EPOCHS = 10000
        # Hyper-parameters
        LEARNING_RATE = 0.001
        CONVERGENCE_MEASURE = 0.005
        converged = False
        while not converged and n_epoch < MAX_EPOCHS:
            gradient = self.update_parameters(LEARNING_RATE)
            n_epoch += 1

            # The '%' or 'mod' operation returns the remainder from division.
            # The condition here returns True when n_epoch is a multiple of 1000 (1000, 2000, etc.)
            if n_epoch % 1000 == 0:
                print('Epoch Iter {} with gradient {}.'.format(n_epoch, gradient))

            gradient_norm = np.sqrt(gradient[0] ** 2 + gradient[1] ** 2)
            # Small only when both partial derivatives are near zero
            if gradient_norm < CONVERGENCE_MEASURE:
                converged = True
        # End while
        print('Converged:', converged)
        print('Number of Epochs:'.format(n_epoch))

    def display_plotted_data(self, x_label, y_label):
        # This is syntax from MatPlotLib library, plotting data is pretty straightforward :)
        plt.scatter(self.X_train, self.Y_train)
        plt.xlabel = x_label
        plt.ylabel = y_label
        plt.show()

    # Try these yourself! Not required. How would the above code change?
    def stochastic_gradient_descent(self):
        pass

    def mini_batch_gradient_descent(self):
        pass
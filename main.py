from linear_regression_learning import LinearRegression

if __name__ == '__main__':
    '''Conduct AI Learning (Linear Regression) on a salary data set'''
    # Train a linear regression model from network dataframe
    independent_var = 'YearsExperience'
    dependent_var = 'Salary'

    lr_model = LinearRegression(source='salary', dependent=dependent_var, independent=independent_var)
    lr_model.display_plotted_data(independent_var, dependent_var)
    lr_model.batch_gradient_descent()
    intercept, slope = lr_model.b[0], lr_model.b[1]
    loss = lr_model.model_loss(X=lr_model.X_test, Y=lr_model.Y_test)

    # Output learned parameters
    print("MODEL RESULTS",
          "\n\tIntercept: {0}\n\tSlope: {1}\n\tLoss: {2}".format(intercept, slope, loss))
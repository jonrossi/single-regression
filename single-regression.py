import numpy as np
import pandas as pd
from scipy import stats

# Define the Single-Variable Linear Regression model class.
class SLRegression(object):
    def __init__(self, learnrate = .01, tolerance = .000000001, max_iter = 10000):
        # learnrate (float): the learning rate for the regression.
        # tolerance (float): our threshold for defining "convergence."
        # max_iter (int): the maximum number of iterations we will allow.

        self.learnrate = learnrate
        self.tolerance = tolerance
        self.max_iter = max_iter

    def change_learnrate(self, new_learnrate):
        # new_learnrate (float): the new learning rate.
        self.learnrate = new_learnrate

    def change_tolerance(self, new_tolerance):
        # new_tolerance (float): the new tolerance.
        self.tolerance = new_tolerance

    def change_max_iter(self, new_max_iter):
        # new_max_iter (int): thenew maximum number of iterations.
        self.max_iter = new_max_iter

    # Define fit function.
    def fit(self, data):
        # data (array-like, shape = [m_observations, 2_columns]): the training data.

        converged = False
        m = data.shape[0]
            # converged (bool): whether algorithm has converged.
            # m (int): the number of samples.

        # Initialize other class variables.
        self.iter_ = 0
        self.theta0_ = 0
        self.theta1_ = 0

        # Compute the "cost" function J.
        J = (1.0/(2.0*m)) * sum([(self.theta0_ + self.theta1_*data[i][1] - data[i][0])**2 for i in range(m)])

        # Recursively update theta0 and theta1.
        while not converged:
            self.iter_ += 1

            # Calculate the partial derivatives of J with respect to theta0 and theta1.
            pdtheta0 = (1.0/m) * sum([(self.theta0_ + self.theta1_*data[i][1] - data[i][0]) for i in range(m)])
            pdtheta1 = (1.0/m) * sum([(self.theta0_ + self.theta1_*data[i][1] - data[i][0]) * data[i][1] for i in range(m)])

            # Subtract the learnrate * partial derivative from theta0 and theta1.
            temp0 = self.theta0_ - (self.learnrate * pdtheta0)
            temp1 = self.theta1_ - (self.learnrate * pdtheta1)

            # Update theta0 and theta1.
            self.theta0_ = temp0
            self.theta1_ = temp1

            # Compute the updated cost function, given new theta0 and theta1.
            new_J = (1.0/(2.0*m)) * sum([(self.theta0_ + self.theta1_*data[i][1] - data[i][0])**2 for i in range(m)])

            # Test for convergence.
            if abs(J - new_J) <= self.tolerance:
                converged = True
                print(('Model converged after %s iterations!') % (self.iter_))

            # Set old cost equal to new cost.
            J = new_J

            # Test whether we have hit max_iter.
            if self.iter_ == self.max_iter:
                converged = True
                print('Maximum iterations have been reached!')

        return self

    # Define a "point forecast" function.
    def point_forecast(self, x):
        return self.theta0_ + self.theta1_ * x

# Test the algorithm on a data set.
if __name__ == '__main__':
    # Load in the data.
    data = np.squeeze(np.array(pd.read_csv('sales_normalized.csv')))

    # Create a regression model with the default learning rate, tolerance, and maximum number of iterations.
    slregression = SLRegression()

    # Call the fit function and pass in our data.
    slregression.fit(data)

    # Print out the results.
    print(('After %s iterations, the model converged on Theta0 = %s and Theta1 = %s.') % (
    slregression.iter_, slregression.theta0_, slregression.theta1_))

    # Compare our model to scipy linregress model.
    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(data[:, 1], data[:, 0])
    print(('Scipy linear regression gives intercept: %s and slope = %s.') % (intercept, slope))

    # Test the model with a point forecast.
    print(('As an example, our algorithm gives y = %s, given x = .87.') % (
    slregression.point_forecast(.87)))  # Should be about .83.
    print('The true y-value for x = .87 is about .8368.')
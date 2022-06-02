# importing relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# data for training
x_train = [[6], [10], [2], [4], [6], [7], [0]]
y_train = [[82], [88], [56], [64], [77], [92], [23]]
# data for testing
x_test = [[1], [8], [5], [3]]
# data for % error
y_predict = [[41], [80], [59], [47]]

# calling object of LRM
model = linear_model.LinearRegression()
# data training
model.fit(x_train, y_train)

# testing the data
y_predicted = model.predict(x_test)

# printing relevant results
print("Mean squared value: ", mean_squared_error(y_predicted, y_predict))
print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

# plotting
plt.scatter(x_test, y_predict)
plt.plot(x_test, y_predicted)
plt.show()



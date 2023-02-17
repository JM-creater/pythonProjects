import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# load the data from csv
dataset = pd.read_csv("polydata_degree_2.csv")
# sort the data by area
dataset = dataset.sort_values(by=['area'])

# get input and output
x = dataset.iloc[:, 0:1].values  # starting point 0, and ends with 1
y = dataset.iloc[:, 1:2].values  # starting 1, and ends with 2. 2 is the output
print(x)
print(y)

# create the model
poly = PolynomialFeatures(degree=2)  # accuracy 28%
x_poly = poly.fit_transform(x)
pilreg = LinearRegression()
pilreg.fit(x_poly, y)
new_y = pilreg.predict(x_poly)
plt.scatter(x, y)
plt.plot(x, new_y, color = "blue")
plt.show()

# test the model
test_area = 600
test_price = pilreg.predict(poly.fit_transform([[test_area]]))
print("Php %.2f" % test_price)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('grade_salary.csv')

x = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3:4].values
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x_train)
poly.fit(x_train, y_train)

model = LinearRegression()
model.fit(x_poly, y_train)
#x_train = np.arange(0,len(x_train),1)

#plt.scatter(x_poly, y_train, color = 'red')
#plt.plot(x_train, model.predict(x_train), color = 'green')
#plt.show()

test_mod = [[2, 4.5, 4.8]]
model_salary = model.predict(poly.fit_transform(test_mod))
print("%.2f" % model_salary)
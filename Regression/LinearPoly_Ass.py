import math
import numpy as np
import matplotlib .pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

dataset = pd.read_csv("grade_salary.csv")
dataset.head()

#reg_model = linear_model.LinearRegression()

x = dataset.iloc[:, :4].values
y = dataset.iloc[:, 3:4].values
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
model = linear_model.LinearRegression()
model.fit(dataset[['gender', 'gwa_3rd_year_college', 'gwa_4th_year_college']], dataset.salary_after_5_years)
predictions = model.predict(x_test)
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, model.predict(x_train))
plt.show()

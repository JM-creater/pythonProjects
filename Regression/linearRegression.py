import math
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

dataset = pd.read_csv("yacht_prices.csv")
print(dataset)

# first get the media
median_cap = math.floor(dataset.capacity.median())
# put the median to the blank capacity
dataset.capacity = dataset.capacity.fillna(median_cap)


# build the linear model
reg_model = linear_model.LinearRegression()
reg_model.fit(dataset[['area', 'capacity', 'rooms']], dataset.price)

#dataset['price'] = reg_model.predict(dataset[['capacity']])
#plt.scatter(dataset.capacity, dataset.price)
#plt.show()
# test the model

test_area = 250
test_capacity = 10
test_rooms = 3
test_price = reg_model.predict([[test_area, test_capacity, test_rooms]])
print('$', test_price)
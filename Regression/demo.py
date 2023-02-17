#import the libraries
import math
import numpy as np
import matplotlib.pyplot as pls
from sklearn.linear_model import LinearRegression
import pandas as pd

# assigning the file to the object name dataset
dataset = pd.read_csv('grade_salary.csv')
print(dataset)

# to build the model
ml = LinearRegression()
ml.fit(dataset[['gender', 'gwa_3rd_year_college', 'gwa_4th_year_college']].values, dataset.salary_after_5_years)


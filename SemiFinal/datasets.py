# we will have the description of ML model
# using k-Nearest Neighbors algorithm

# objectives
# 1. Load data explore data
# 2. Visualize the dataset
# 3. Determine the number of neighbours
# 4. Predict the color of the fruit -apple, mandarin, orange and lemon

import numpy as np
# for data manipulation, analysis and reading our dataset
import pandas as pd
# for plotting and visualizing the data
import matplotlib.pyplot as plt

# import the data
fruit = pd.read_csv('fruit_data_with_colours.csv')

# 2nd step -> check the data using head command
print(fruit.head())
# 3rd step -> determine the number of piece of fruits(rows), attributes(columns) using shape command
print(fruit.shape)
# 4th step -> determine the fruits within the data using .unique() command
print(fruit['fruit_name'].unique())
# 5th step -> determine the count of fruits within the data using .value_counts() command
fruit['fruit_name'].value_counts()

# Sample 1: Based on matplotlib using simple bar graph
# Seaborn is a data visualization library in Python based on matplotlib
import seaborn as sns
# set the variable fruit as a independent variable
sns.countplot(x=fruit['fruit_name'], label="Count", palette="Set3")
plt.show()

# Sample 2: Using Bloxplot
# It will assess the distribution
fruit.drop('fruit_label', axis=1).plot(kind='box',
                                        subplots=True,
                                        layout=(2,2),
                                        sharex=False,
                                        sharey=False,
                                        figsize=(10,10),
                                        color ='c',
                                        patch_artist=True)
plt.suptitle("Box Plot for each input variable")
plt.savefig('fruits_boxplot')
plt.show()

# Sample 3: using Histogram
# PyLab is a module that belongs to the Python mathematics library Matplotlib.
# PyLab combines the numerical module numpy with the graphical plotting module pyplot
import pylab as pl
# To create a histogram, we will use pandas hist() method.
fruit.drop('fruit_label', axis=1).hist(bins=30,
                                        figsize=(10,10),
                                        color = "c",
                                        ec = "m",
                                        lw=0)
pl.suptitle("Histogram for each numeric input variable")
plt.savefig('fruits_histogram')
plt.show()

# Sample 4: Using Scatter matrix
from pandas.plotting import scatter_matrix
from matplotlib import cm
# The Gaussian distribution is also commonly called the "normal distribution" and is often described as a "bell-shaped curve".
# Colour_Score and Height seem to be closer to the Gaussian distribution
cmap = cm.get_cmap('gnuplot')
df = pd.DataFrame(np.random.randn(1000, 4), columns=['mass', 'width', 'height', 'color_score'])
scatter_matrix(df, alpha=0.2, cmap = cmap, figsize=(10,10), marker = '.', s=30, hist_kwds={'bins':10}, range_padding=0.05, color = 'm')
plt.suptitle('Scatter-matrix for each input variable')
plt.savefig('fruit_scatter_matrix')
plt.show()

# Step 7 -> Build a KNN classifier model to determine K
# import the KNeighborsClassifier module
from sklearn.neighbors import KNeighborsClassifier

# In order to understand the model performance, divide the dataset into a training set and a test set.
# The split is done by using the function
# train_test_split()
from sklearn.model_selection import train_test_split

# Split the dataset into two different datasets
# X -> for the independent features such as mass, width, height
# Y -> for the dependent feature ex: fruit name
X = fruit[['mass','width','height','color_score']]
Y = fruit['fruit_name']

# Next is to separate sets ex: X_train and X_test
# and so on
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Step 8 -> display the statistical summary

print(X_train.describe())
print(X_test.describe())

#Invoke the classifier and Training the model
# Now create a KNN classifier for making predictions
knn = KNeighborsClassifier()

# Train the model using the training sets
knn.fit(X_train, y_train)


KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                        metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                        weights='uniform')

# Evaluate the accuracy of the model for k=5
# Note the output above that by default the n_neighbors = 5
knn.score(X_test, y_test)
print("Accuracy for K=5 : ", knn.score(X_test, y_test))

# Evaluate the accuracy of the model for k=6
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
print("Accuracy for K=6 : ", knn.score(X_test, y_test))

# Evaluate the accuracy of the model for k=7
knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
print("Accuracy for K=7 : ", knn.score(X_test, y_test))

# Evaluate the accuracy of the model for k=8
knn = KNeighborsClassifier(n_neighbors = 8)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
print("Accuracy for K=8 : ", knn.score(X_test, y_test))

# Find the most appropriate K by plotting the accuracy for the various neighbours in a graph
neighbours = np.arange(1,10)
training_accuracy = np.empty(len(neighbours))
testing_accuracy = np.empty(len(neighbours))

for i in range(len(neighbours)):
    knn = KNeighborsClassifier(n_neighbors = i+1)
    knn.fit(X_train,y_train)
    training_accuracy[i] = knn.score(X_train,y_train)
    testing_accuracy[i] = knn.score(X_test,y_test)

plt.title('KNN - Accuracy for various neighbors')
plt.plot(neighbours, testing_accuracy, label = 'Testing Accuracy', color ='c')
plt.plot(neighbours, training_accuracy, label = 'Training accuracy', color ='m')
plt.legend()
plt.xlabel('No. of neighbours')
plt.ylabel('Accuracy')
plt.show()
plt.savefig('knn - accuracy vs no of neighbours')

# As a result, we can say that using KNN algorithm with K=7,
# we can estimate the "Colour" of a fruit from its "Mass", "Width", "Height","Color_Code" values with 66.67% accuracy


# pip install pandas
import pandas as pd
import numpy as np
# pip install matplotlib
import matplotlib.pyplot as plt
# pip install scikit-learn
from sklearn.linear_model import LogisticRegression

import pickle

# Load the CSV
dataset = pd.read_csv('dinoy_zombies.csv')
# print(dataset.head());
#
# Graph
# plt.scatter(dataset.has_wall, dataset.role)
# plt.show()

# Convert strings to numeric
dataset.sex = dataset.sex.replace(to_replace=['Male', 'Female'], value=[0, 1])
dataset.food = dataset.food.replace(to_replace=['No food', 'Food'], value=[0, 1])
dataset.rurality = dataset.rurality.replace(to_replace=['Rural', 'Suburban', 'Urban'], value=[0, 1, 2])
dataset.medication = dataset.medication.replace(to_replace=['No medication', 'Medication'], value=[0, 1])
dataset.sanitation = dataset.sanitation.replace(to_replace=['No sanitation', 'Sanitation' ], value=[0, 1])
dataset.zombie = dataset.zombie.replace(to_replace=['Human', 'Zombie'], value=[0, 1])


# Create the Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(dataset[['age', 'sex', 'rurality', 'food', 'medication', 'sanitation']], dataset.zombie)
# Save the model
with open('logistic.pk', 'wb') as f:
	pickle.dump(model, f)

# Test the model
test_age = 18
test_sex = 'Female'
test_rurality = 'Urban'
test_food = 'Food'
test_medication = 'Medication'
test_sanitation = 'Sanitation'

test_sex = 1 if test_sex == 'Female' else 0
test_food = 1 if test_food == 'Food' else 0
test_medication = 1 if test_medication == 'Medication' else 0
test_sanitation = 1 if test_sanitation == 'Sanitation' else 0

if(test_rurality == 'Suburban'):
    test_rurality = 1
elif(test_rurality == 'Urban'):
    test_rurality = 2
else:
    test_rurality = 0




output = model.predict_proba([[test_age, test_sex, test_rurality, test_food, test_medication, test_sanitation]])
print("HUMAN", "{:.4f}".format(output[0][0]))
print("ZOMBIE", "{:.4f}".format(output[0][1]))

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# Build a KNN classifier model to determine K
# import the KNeighborsClassifier module
# In order to understand the model performance, divide the dataset into a training set and a test set.
# The split is done by using the function
# train_test_split()
# Split the dataset into two different datasets
# X -> for the independent features such as mass, width, height
# Y -> for the dependent feature ex: fruit name

X = dataset[['age', 'sex', 'rurality', 'food', 'medication', 'sanitation']]
Y = dataset['zombie']

# Next is to separate sets ex: X_train and X_test
# and so on
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Step 8 -> display the statistical summary

print(X_train.describe())
print(X_test.describe())

# Invoke the classifier and Training the model
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

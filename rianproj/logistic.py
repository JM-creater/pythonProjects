# pip install pandas
import pandas as pd
import numpy as np
# pip install matplotlib
import matplotlib.pyplot as plt
# pip install scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import pickle

# Load the CSV
dataset = pd.read_csv('pubg_weapon.csv')
# print(dataset.head());
#
# Graph
# plt.scatter(dataset.has_wall, dataset.role)
# plt.show()

# Convert strings to numeric
dataset.bullet_type = dataset.bullet_type.replace(to_replace=[7.62, 5.56, 9, 0.45, 12, 0.3], value=[0, 1, 2, 3, 4, 5])
dataset.type = dataset.type.replace(to_replace=['Assault Rifle', 'Submachine Gun', 'Shotgun', 'Sniper Rifle', 'Pistol'], value=[0, 1, 2, 3, 4])

# Create the Logistic Regression Model
model = LogisticRegression(max_iter=100000)
model.fit(dataset[['bullet_type', 'damage', 'magazine', 'range', 'fire_rate', 'bullet_speed']].values, dataset.type)
# Save the model
with open('logistic.pk', 'wb') as f:
	pickle.dump(model, f)

# Test the model
test_bullet_type = 7.62
test_damage = 321
test_magazine = 32
test_range = 432
test_fire_rate = 432
test_bullet_speed = 432

if test_bullet_type == 5.56:
    test_bullet_type = 1
elif test_bullet_type == 9:
    test_bullet_type = 2
elif test_bullet_type == 0.45:
    test_bullet_type = 3
elif test_bullet_type == 12:
    test_bullet_type = 4
elif test_bullet_type == 0.3:
    test_bullet_type = 5
else:
    test_bullet_type = 0


output = model.predict_proba([[test_bullet_type, test_damage, test_magazine, test_range, test_fire_rate, test_bullet_speed]])
print("ASSAULT RIFLE", "{:.4f}".format(output[0][0]))
print("SUBMACHINE GUN", "{:.4f}".format(output[0][1]))
print("SHOTGUN", "{:.4f}".format(output[0][2]))
print("SNIPER RIFLE", "{:.4f}".format(output[0][3]))
print("PISTOL", "{:.4f}".format(output[0][4]))


# Build a KNN classifier model to determine K
# import the KNeighborsClassifier module
# In order to understand the model performance, divide the dataset into a training set and a test set.
# The split is done by using the function
# train_test_split()
# Split the dataset into two different datasets
# X -> for the independent features such as mass, width, height
# Y -> for the dependent feature ex: fruit name

X = dataset[['bullet_type', 'damage', 'magazine', 'range', 'fire_rate', 'bullet_speed']]
Y = dataset['type']

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

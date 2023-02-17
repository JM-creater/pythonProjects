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
dataset.wall_pen = dataset.wall_pen.replace(to_replace=['Low', 'Medium', 'High'], value=[0, 1, 2])
dataset.type = dataset.type.replace(to_replace=['Pistol', 'Shotgun', 'SMG', 'Assault Rifle', 'Sniper', 'Heavy'], value=[0, 1, 2, 3, 4, 5])

# Create the Logistic Regression Model
model = LogisticRegression(max_iter=50000000000000000000000000000000000000000)
model.fit(dataset[['wall_pen', 'HDMG', 'magazine', 'LDMG', 'price', 'BDMG']].values, dataset.type)
# Save the model
with open('logistic.pk', 'wb') as f:
	pickle.dump(model, f)

# Test the model
test_wall_pen = 'Low'
test_HDMG = 46
test_magazine = 25
test_LDMG = 14
test_price = 1000
test_BDMG= 26

if test_wall_pen == 'Medium':
    test_wall_pen = 1
elif test_wall_pen == 'High':
    test_wall_pen = 2
else:
    test_wall_pen = 0


output = model.predict_proba([[test_wall_pen, test_HDMG, test_magazine, test_LDMG, test_price, test_BDMG]])
print("PISTOL", "{:.4f}".format(output[0][0]))
print("SHOTGUN", "{:.4f}".format(output[0][1]))
print("SUBMACHINE GUN", "{:.4f}".format(output[0][2]))
print("ASSAULT RIFLE", "{:.4f}".format(output[0][3]))
print("SNIPER RIFLE", "{:.4f}".format(output[0][4]))
print("HEAVY", "{:.4f}".format(output[0][5]))

X = dataset[['wall_pen', 'HDMG', 'magazine', 'LDMG', 'price', 'BDMG']]
Y = dataset['type']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


print()
print(X_train.describe())
print()
print(X_test.describe())
print()

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

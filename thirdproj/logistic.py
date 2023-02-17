import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('b_depressed.csv')
# print(dataset.head())

# graph
# plt.scatter(dataset.employee_id, dataset.is_promoted)
# plt.show()

# Create the Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(dataset[['sex', 'abuse', 'neglect', 'lose_someone', 'traumatic_event', 'unstable']].values, dataset.depressed)
# Save the model
with open('logistic.pk', 'wb') as f:
    pickle.dump(model, f)

# Test the model
test_sex = 0
test_abuse = 0
test_neglect = 0
test_lose_someone = 0
test_traumatic_event = 0
test_unstable = 0

output = model.predict_proba([[test_sex, test_abuse, test_neglect, test_lose_someone, test_traumatic_event, test_unstable]])
print("Not Depressed ", "{:.4f}".format(output[0][0]))
print("Depressed ", "{:.4f}".format(output[0][1]))

X = dataset[['sex', 'abuse', 'neglect', 'lose_someone', 'traumatic_event', 'unstable']]
Y = dataset['depressed']

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
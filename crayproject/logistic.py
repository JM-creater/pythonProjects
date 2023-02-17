# pip install pandas
import pandas as pd
import numpy as np
# pip install matplotlib
import matplotlib.pyplot as plt
# pip install scikit-learn
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the CSV
dataset = pd.read_csv('CSA-Data.csv')
# print(dataset.head());

# Graph
# plt.scatter(dataset.temperature, dataset.prognosis)
# plt.show()

# Convert strings to numeric
dataset.safe = dataset.safe.replace(to_replace=['Disagree', 'Agree'], value=[0, 1])
dataset.abused = dataset.abused.replace(to_replace=['Disagree', 'Agree'], value=[0, 1])
dataset.sexual_abused = dataset.sexual_abused.replace(to_replace=['Disagree', 'Agree'], value=[0, 1])
dataset.teaching = dataset.teaching.replace(to_replace=['Disagree', 'Agree'], value=[0, 1])
dataset.grooming = dataset.grooming.replace(to_replace=['No', 'Yes'], value=[0, 1])
dataset.identify = dataset.identify.replace(to_replace=['No', 'Yes'], value=[0, 1])
dataset.recovering = dataset.recovering.replace(to_replace=['No', 'Yes'], value=[0, 1])
dataset.action = dataset.action.replace(to_replace=['No', 'Yes'], value=[0, 1])
dataset.Knowledge_Level = dataset.Knowledge_Level.replace(to_replace=['Beginner', 'Intermediate'], value=[0, 1])

# Create the Logistic Regression Model
model = LogisticRegression(max_iter=5000)
model.fit(dataset[['safe', 'abused', 'sexual_abused', 'teaching', 'grooming', 'identify', 'recovering', 'action']].values, dataset.Knowledge_Level)
# Save the model
with open('logistic.pk', 'wb') as f:
	pickle.dump(model, f)

# Test the model
test_safe = 'Disagree'
test_abused = 'Agree'
test_sexual_abused = 'Disagree'
test_teaching = 'Disagree'
test_grooming = 'Yes'
test_identify = 'No'
test_recovering = 'No'
test_action = 'Yes'

test_safe = 1 if test_safe == 'Agree' else 0
test_abused = 1 if test_abused == 'Agree' else 0
test_sexual_abused = 1 if test_sexual_abused == 'Agree' else 0
test_teaching = 1 if test_teaching == 'Agree' else 0
test_grooming = 1 if test_grooming == 'Yes' else 0
test_identify = 1 if test_identify == 'Yes' else 0
test_recovering = 1 if test_recovering == 'Yes' else 0
test_action = 1 if test_action== 'Yes' else 0

output = model.predict_proba([[test_safe, test_abused, test_sexual_abused, test_teaching, test_grooming, test_identify, test_recovering, test_action]])
print("Beginner ", "{:.4f}".format(output[0][0]))
print("Intermediate ", "{:.4f}".format(output[0][1]))


X = dataset[['safe', 'abused', 'sexual_abused', 'teaching', 'grooming', 'identify', 'recovering', 'action']]
Y = dataset['Knowledge_Level']

# Next is to separate sets ex: X_train and X_test
# and so on
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

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
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
print("Accuracy for K=6 : ", knn.score(X_test, y_test))

# Evaluate the accuracy of the model for k=7
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
print("Accuracy for K=7 : ", knn.score(X_test, y_test))

# Evaluate the accuracy of the model for k=8
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
print("Accuracy for K=8 : ", knn.score(X_test, y_test))

neighbours = np.arange(1, 10)
training_accuracy = np.empty(len(neighbours))
testing_accuracy = np.empty(len(neighbours))

for i in range(len(neighbours)):
    knn = KNeighborsClassifier(n_neighbors=i+1)
    knn.fit(X_train, y_train)
    training_accuracy[i] = knn.score(X_train, y_train)
    testing_accuracy[i] = knn.score(X_test, y_test)

plt.title('KNN - Accuracy for various neighbors')
plt.plot(neighbours, testing_accuracy, label='Testing Accuracy', color='c')
plt.plot(neighbours, training_accuracy, label='Training accuracy', color='m')
plt.legend()
plt.xlabel('No. of neighbours')
plt.ylabel('Accuracy')
plt.show()
plt.savefig('knn - accuracy vs no of neighbours')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the CSV
dataset = pd.read_csv('heart_failure_prediction.csv')
# print(dataset.head());

# Convert strings to numeric
dataset.sex = dataset.sex.replace(to_replace=['f', 'm'], value=[0, 1])

# Create the Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(dataset[['age', 'sex', 'chest_pain_type','resting_bp','chol_in_mg','fasting_blood_sugar','restecg','maxheart_rate_achieved']].values, dataset.Result)
# Save the model
with open('logistic.pk', 'wb') as f:
	pickle.dump(model, f)

# Test the model
test_age = 64
test_sex = f
test_chest_pain_type = 3
test_resting_bp = 145
test_chol_in_mg = 256
test_fasting_blood_sugar = 0
test_restecg = 1
test_maxheart_rate_achieved = 172

test_sex = 1 if test_sex == 'm' else 0

output = model.predict_proba([[test_age, test_sex, test_chest_pain_type, test_resting_bp, test_chol_in_mg, test_fasting_blood_sugar, test_restecg, test_maxheart_rate_achieved]])
print("Low Chance of Having Heart Failure: ", "{:.4f}".format(output[0][0]))
print("High Chance of Having Heart Failure: ", "{:.4f}".format(output[0][1]))

X = dataset[['age', 'sex', 'chest_pain_type', 'resting_bp', 'chol_in_mg', 'fasting_blood_sugar', 'restecg', 'maxheart_rate_achieved']]
Y = dataset['Result']

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
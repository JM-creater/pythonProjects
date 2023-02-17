import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

dataset = pd.read_csv('employee_promotion.csv')
# print(dataset.head())

# graph
# plt.scatter(dataset.employee_id, dataset.is_promoted)
# plt.show()

# Convert strings to numeric
dataset.education = dataset.education.replace(to_replace=['Master', 'Bachelor'], value=[0, 1])
dataset.gender = dataset.gender.replace(to_replace=['f', 'm'], value=[0, 1])
dataset.recruitment_channel = dataset.recruitment_channel.replace(to_replace=['sourcing', 'other'], value=[0, 1])
dataset.department = dataset.department.replace(to_replace=['Sales & Marketing', 'Operations', 'Technology', 'Analytics', 'R&D', 'Procurement', 'Finance', 'HR', 'Legal'], value=[0, 1, 2, 3, 4, 5, 6, 7, 8])


# Create the Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(dataset[['employee_id', 'department', 'region', 'education', 'gender', 'recruitment_channel', 'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'awards_won', 'avg_training_score']].values, dataset.is_promoted)
# Save the model
with open('logistic.pk', 'wb') as f:
    pickle.dump(model, f)

# Test the model
test_employee_id = 68082
test_department = 'Sales & Marketing'
test_region = 16
test_education = 'Bachelor'
test_gender = 'm'
test_recruitment_channel = 'sourcing'
test_no_of_trainings = 1
test_age = 32
test_previous_year_rating = 4
test_length_of_service = 6
test_awards_won = 0
test_avg_training_score = 44

if test_department == 'Operations':
    test_department = 1
elif test_department == 'Technology':
    test_department = 2
elif test_department == 'Analytics':
    test_department = 3
elif test_department == 'R&D':
    test_department = 4
elif test_department == 'Procurement':
    test_department = 5
elif test_department == 'Finance':
    test_department = 6
elif test_department == 'HR':
    test_department = 7
elif test_department == 'Legal':
    test_department = 8
else:
    test_department = 0

test_education = 1 if test_education == 'Master' else 0
test_gender = 1 if test_gender == 'f' else 0
test_recruitment_channel = 1 if test_recruitment_channel == 'other' else 0

output = model.predict_proba([[test_employee_id, test_department, test_region, test_education, test_gender, test_recruitment_channel, test_no_of_trainings, test_age, test_previous_year_rating, test_length_of_service, test_awards_won, test_avg_training_score]])
print("NO PROMOTION ", "{:.4f}".format(output[0][0]))
print("PROMOTION ", "{:.4f}".format(output[0][1]))

X = dataset[['employee_id', 'department', 'region', 'education', 'gender', 'recruitment_channel', 'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'awards_won', 'avg_training_score']]
Y = dataset['is_promoted']

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

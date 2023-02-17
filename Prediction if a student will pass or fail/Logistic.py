import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns

#Read the data
student = pd.read_csv('My_DataSet.csv')
# print (student.head())

#visualization
# sns.countplot(x=student['student_number'], label = 'Count')
# plt.show()

#Convert string to numeric
student.quizzes = student.quizzes.replace(to_replace=['no', 'yes'], value=[0, 1])
student.performance_task = student.performance_task.replace(to_replace=['no', 'yes'], value=[0,1])
student.activities = student.activities.replace(to_replace=['no', 'yes'], value=[0,1])
student.projects = student.projects.replace(to_replace=['no', 'yes'], value=[0,1])
student.submit_on_time = student.submit_on_time.replace(to_replace=['no', 'yes'], value=[0,1])
student.submit_late = student.submit_late.replace(to_replace=['no', 'yes'], value=[0,1])
student.paid = student.paid.replace(to_replace=['no', 'yes'], value=[0,1])
student.passed_or_failed = student.passed_or_failed.replace(to_replace=['failed', 'passed'], value=[0,1])

#Create the logistic model
model = LogisticRegression(max_iter=500)
model.fit(student[['student_number', 'quizzes', 'performance_task', 'activities', 'projects', 'submit_on_time', 'submit_late', 'paid']].values, student.passed_or_failed)

#save the model
with open('logistic.pk', 'wb') as f:
    pickle.dump(model, f)

#test the model
test_student_number = 2
test_quizzes = "yes"
test_performance_task = "yes"
test_activities = "no"
test_projects = "yes"
test_submit_on_time = "no"
test_submit_late = "yes"
test_paid = "yes"

#converting the yes or no to decimal (0 and 1)
test_quizzes = 1 if test_quizzes == "yes" else 0
test_performance_task = 1 if test_performance_task == "yes" else 0
test_activities = 1 if test_activities == "yes" else 0
test_projects = 1 if test_projects == "yes" else 0
test_submit_on_time = 1 if test_submit_on_time == "yes" else 0
test_submit_late = 1 if test_submit_late == "yes" else 0
test_paid = 1 if test_paid == "yes" else 0

#output
output = model.predict_proba([[test_student_number, test_quizzes, test_performance_task, test_activities, test_projects, test_submit_on_time, test_submit_late, test_paid]])
print("failed:", "{:.4f}".format(output[0][0]))
print("passed:", "{:.4f}".format(output[0][1]))

# #splitting the data
#
# X = student[['student_number', 'quizzes', 'performance_task', 'activities', 'projects', 'submit_on_time',
#           'submit_late', 'paid']]
# Y = student['passed_or_failed']
#
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state= 0)
#
# knn = KNeighborsClassifier()
# knn.fit(X_train,Y_train)
# KNeighborsClassifier(algorithm='auto', leaf_size= 30, metric = 'minkowski', metric_params=None,
#                            n_jobs=None, n_neighbors=5, weights='uniform')
# knn.score(X_test,Y_test)
# print("The accuracy of K=5 is",knn.score(X_test,Y_test))
#
# knn = KNeighborsClassifier(n_neighbors= 6)
# knn.fit(X_train,Y_train)
# knn.score(X_test,Y_test)
# print("The accuracy of K=6 is",knn.score(X_test,Y_test))
#
# knn = KNeighborsClassifier(n_neighbors= 7)
# knn.fit(X_train,Y_train)
# knn.score(X_test,Y_test)
# print("The accuracy of K=6 is",knn.score(X_test,Y_test))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle

# Load the CSV
dataset = pd.read_csv('breast-cancer.csv')

# Convert strings to numeric
dataset.diagnosis = dataset.diagnosis.replace(to_replace=['B', 'M'], value=[0, 1])

# graph
plt.scatter(dataset.id, dataset.output)
plt.show()

# Create the Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(dataset[['id', 'diagnosis', 'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean']], dataset.output)
# Save the model
with open('logistic.pk', 'wb') as f:
	pickle.dump(model, f)

# Test the model
test_id = 84300903
test_diagnosis = 'B'
test_radius_mean = 18.99
test_texture_mean = 15.90
test_perimeter_mean = 122.8
test_area_mean = 0.08474
test_smoothness_mean = 0.08474
test_compactness_mean = 0.1599
test_concavity_mean = 0.0869

test_diagnosis = 1 if test_diagnosis == 'M' else 0

output = model.predict_proba([[test_id, test_diagnosis, test_radius_mean, test_texture_mean,test_perimeter_mean,test_area_mean,test_smoothness_mean,test_compactness_mean,test_concavity_mean]])
print("Less Chance: ", "{:.4f}".format(output[0][0]))
print("More Chance: ", "{:.4f}".format(output[0][1]))
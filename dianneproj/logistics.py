# pip install pandas
import pandas as pd
import numpy as np
# pip install matplotlib
import matplotlib.pyplot as plt
# pip install scikit-learn
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the CSV
dataset = pd.read_csv('DogFM.csv')
#print(dataset.head());

# Graph
#plt.scatter(dataset.breed, dataset.findings)
#plt.show()

# Convert strings to numeric

dataset.breed = dataset.breed.replace(to_replace=['Akita', 'Anatolian Sheepdog', 'Bernese Mountain Dog', 'Bloodhound', 'Borzoi', 'Bullmastiff', 'Great Dane', 'Great Pyrenees', 'Great Swiss Mountain Dog',  'Irish Wolfhound', 'Kuvasz', 'Mastiff',  'Neopolitan Mastiff', 'Newfoundland', 'Otter Hound', 'Rottweiler', 'Saint Bernard', 'Afghan Hound', 'American Foxhound', 'Beauceron', 'Belgian Malinois', 'Belgian Sheepdog', 'Belgian Tervuren', 'Black And Tan Coonhound', 'Black Russian Terrier', 'Bouvier Des Flandres', 'Boxer', 'Briard', 'Chesapeake Bay Retriever', 'Clumber Spaniel', 'Collie (Rough) & (Smooth)', 'Curly Coated Retriever', 'Doberman Pinscher', 'English Foxhound', 'English Setter', 'German Shepherd Dog', 'German Shorthaired Pointer', 'German Wirehaired Pointer', 'Giant Schnauzer', 'Golden Retriever', 'Gordon Setter', 'Greyhound', 'Irish Setter', 'Komondor', 'Labrador Retriever', 'Old English Sheepdog (Bobtail)', 'Rhodesian Ridgeback', 'Scottish Deerhound', 'Spinone Italiano', 'Tibetan Mastiff', 'Poodle Standard','Weimaraner', 'Airdale Terrier', 'American Staffordshire Terrier', 'American Water Spaniel', 'Australian Cattle Dog', 'Australian Shepherd', 'Basset Hound', 'Bearded Collie', 'Border Collie', 'Brittany', 'Bull Dog', 'Bull Terrier', 'Canaan Dog', 'Chinese Shar Pei', 'Chow Chow', 'Cocker Spaniel-American', 'Cocker Spaniel-English', 'Dalmatian', 'English Springer Spaniel', 'Field Spaniel', 'Flat Coated Retriever', 'Finnish Spitz', 'Harrier', 'Ibizan Hound', 'Irish Terrier', 'Irish Water Spaniel', 'Keeshond', 'Kerry Blue Terrier', 'Norwegian Elkhound', 'Nova Scotia Duck Tolling Retriever', 'Petit Basset Griffon Vendeen', 'Pharaoh Hound', 'Plott Hound', 'Pointer', 'Polish Lowland Sheepdog', 'Portuguese Water Dog', 'Redbone Coonhound', 'Saluki', 'Samoyed', 'Siberian Husky', 'Soft-Coated Wheaten Terrier', 'Staffordshire Bull Terrier', 'Standard Schnauzer', 'Sussex Spaniel', 'Vizsla', 'Welsh Springer Spaniel', 'Wirehaired Pointing Griffon', 'American Eskimo', 'Australian Terrier', 'Basenji', 'Beagle', 'Bedlington Terrier', 'Bichon Frise', 'Border Terrier', 'Boston Terrier', 'Brussels Griffon', 'Cairn Terrier', 'Cardigan Welsh Corgi', 'Cavalier King Charles Spaniel', 'Dachshund', 'Dandie Dinmont Terrier', 'English Toy Spaniel', 'Fox Terrier ?? Smooth', 'Fox Terrier ?? Wirehair', 'French Bulldog', 'German Pinscher', 'Glen Imaal Terrier', 'Lakeland Terrier', 'Manchester Terrier (Standard)', 'Poodle Miniature', 'Pug', 'Puli', 'Schipperke', 'Scottish Terrier', 'Sealyham Terrier', 'Shetland Sheepdog (Sheltie)', 'Shiba Inu', 'Shih Tzu', 'Chihuahua', 'Maltese', 'Pomeranian', 'Yorkshire Terrier'], value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132])
dataset.findings = dataset.findings.replace(to_replace=['male', 'female'], value=[0, 1])
dataset.has_testicles = dataset.has_testicles.replace(to_replace=['no', 'yes'], value=[0, 1])
dataset.has_vagina = dataset.has_vagina.replace(to_replace=['no', 'yes'], value=[0, 1])
dataset.leg_raise = dataset.leg_raise.replace(to_replace=['no', 'yes'], value=[0, 1])
dataset.squatting_down = dataset.squatting_down.replace(to_replace=['no', 'yes'], value=[0, 1])

# Create the Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(dataset[['breed', 'height_inches', 'weight_lbs', 'has_testicles', 'has_vagina', 'leg_raise', 'squatting_down']].values, dataset.findings)
# Save the model
with open('logistic.pk', 'wb') as f:
    pickle.dump(model, f)

# Test the model
test_breed = 'Great Swiss Mountain Dog'
test_height_inches = 23
test_weight_lbs = 130
test_has_testicles = 'no'
test_has_vagina = 'yes'
test_leg_raise = 'no'
test_squatting_down = 'yes'

if test_breed == 'Anatolian Sheepdog':
    test_breed = 1
elif test_breed == 'Bernese Mountain Dog':
    test_breed = 2
elif test_breed == 'Bloodhound':
    test_breed = 3
elif test_breed == 'Borzoi':
    test_breed = 4
elif test_breed == 'Bullmastiff':
    test_breed = 5
elif test_breed == 'Great Dane':
    test_breed = 6
elif test_breed == 'Great Pyrenees':
    test_breed = 7
elif test_breed == 'Great Swiss Mountain Dog':
    test_breed = 8
elif test_breed == 'Irish Wolfhound':
    test_breed = 9
elif test_breed == 'Kuvasz':
    test_breed = 10
elif test_breed == 'Mastiff':
    test_breed = 11
elif test_breed == 'Neopolitan Mastiff':
    test_breed = 12
elif test_breed == 'Newfoundland':
    test_breed = 13
elif test_breed == 'Otter Hound':
    test_breed = 14
elif test_breed == 'Rottweiler':
    test_breed = 15
elif test_breed == 'Saint Bernard':
    test_breed = 16
elif test_breed == 'Afghan Hound':
    test_breed = 17
elif test_breed == 'American Foxhound':
    test_breed = 18
elif test_breed == 'Beauceron':
    test_breed = 19
elif test_breed == 'Belgian Malinois':
    test_breed = 20
elif test_breed == 'Belgian Sheepdog':
    test_breed = 21
elif test_breed == 'Belgian Tervuren':
    test_breed = 22
elif test_breed == 'Black And Tan Coonhound':
    test_breed = 23
elif test_breed == 'Black And Tan Coonhound':
    test_breed = 24
elif test_breed == 'Black Russian Terrier':
    test_breed = 25
elif test_breed == 'Bouvier Des Flandres':
    test_breed = 26
elif test_breed == 'Boxer':
    test_breed = 27
elif test_breed == 'Briard':
    test_breed = 28
elif test_breed == 'Chesapeake Bay Retriever':
    test_breed = 29
elif test_breed == 'Clumber Spaniel':
    test_breed = 30
elif test_breed == 'Collie (Rough) & (Smooth)':
    test_breed = 31
elif test_breed == 'Curly Coated Retriever':
    test_breed = 32
elif test_breed == 'Doberman Pinscher':
    test_breed = 33
elif test_breed == 'English Foxhound':
    test_breed = 34
elif test_breed == 'English Setter':
    test_breed = 35
elif test_breed == 'German Shepherd Dog':
    test_breed = 36
elif test_breed == 'German Shorthaired Pointer':
    test_breed = 37
elif test_breed == 'German Wirehaired Pointer':
    test_breed = 38
elif test_breed == 'Giant Schnauzer':
    test_breed = 39
elif test_breed == 'Golden Retriever':
    test_breed = 40
elif test_breed == 'Gordon Setter':
    test_breed = 41
elif test_breed == 'Greyhound':
    test_breed = 42
elif test_breed == 'Irish Setter':
    test_breed = 43
elif test_breed == 'Komondor':
    test_breed = 44
elif test_breed == 'Labrador Retriever':
    test_breed = 45
elif test_breed == 'Old English Sheepdog (Bobtail)':
    test_breed = 46
elif test_breed == 'Rhodesian Ridgeback':
    test_breed = 47
elif test_breed == 'Scottish Deerhound':
    test_breed = 48
elif test_breed == 'Spinone Italiano':
    test_breed = 49
elif test_breed == 'Tibetan Mastiff':
    test_breed = 50
elif test_breed == 'Poodle Standard':
    test_breed = 51
elif test_breed == 'Weimaraner':
    test_breed = 52
elif test_breed == 'Airdale Terrier':
    test_breed = 53
elif test_breed == 'American Staffordshire Terrier':
    test_breed = 54
elif test_breed == 'American Water Spaniel':
    test_breed = 55
elif test_breed == 'Australian Cattle Dog':
    test_breed = 56
elif test_breed == 'Australian Shepherd':
    test_breed = 57
elif test_breed == 'Basset Hound':
    test_breed = 58
elif test_breed == 'Bearded Collie':
    test_breed = 59
elif test_breed == 'Border Collie':
    test_breed = 60
elif test_breed == 'Brittany':
    test_breed = 61
elif test_breed == 'Bull Dog':
    test_breed = 62
elif test_breed == 'Bull Terrier':
    test_breed = 63
elif test_breed == 'Canaan Dog':
    test_breed = 64
elif test_breed == 'Chinese Shar Pei':
    test_breed = 65
elif test_breed == 'Chow Chow':
    test_breed = 66
elif test_breed == 'Cocker Spaniel-American':
    test_breed = 67
elif test_breed == 'Cocker Spaniel-English':
    test_breed = 68
elif test_breed == 'Dalmatian':
    test_breed = 69
elif test_breed == 'English Springer Spaniel':
    test_breed = 70
elif test_breed == 'Field Spaniel':
    test_breed = 71
elif test_breed == 'Flat Coated Retriever':
    test_breed = 72
elif test_breed == 'Finnish Spitz':
    test_breed = 73
elif test_breed == 'Harrier':
    test_breed = 74
elif test_breed == 'Ibizan Hound':
    test_breed = 75
elif test_breed == 'Irish Terrier':
    test_breed = 76
elif test_breed == 'Irish Water Spaniel':
    test_breed = 77
elif test_breed == 'Keeshond':
    test_breed = 78
elif test_breed == 'Kerry Blue Terrier':
    test_breed = 79
elif test_breed == 'Norwegian Elkhound':
    test_breed = 80
elif test_breed == 'Nova Scotia Duck Tolling Retriever':
    test_breed = 81
elif test_breed == 'Petit Basset Griffon Vendeen':
    test_breed = 82
elif test_breed == 'Pharaoh Hound':
    test_breed = 83
elif test_breed == 'Plott Hound':
    test_breed = 84
elif test_breed == 'Pointer':
    test_breed = 85
elif test_breed == 'Polish Lowland Sheepdog':
    test_breed = 86
elif test_breed == 'Portuguese Water Dog':
    test_breed = 87
elif test_breed == 'Redbone Coonhound':
    test_breed = 88
elif test_breed == 'Saluki':
    test_breed = 89
elif test_breed == 'Samoyed':
    test_breed = 90
elif test_breed == 'Siberian Husky':
    test_breed = 91
elif test_breed == 'Soft-Coated Wheaten Terrier':
    test_breed = 92
elif test_breed == 'Staffordshire Bull Terrier':
    test_breed = 93
elif test_breed == 'Standard Schnauzer':
    test_breed = 94
elif test_breed == 'Sussex Spaniel':
    test_breed = 95
elif test_breed == 'Vizsla':
    test_breed = 96
elif test_breed == 'Welsh Springer Spaniel':
    test_breed = 97
elif test_breed == 'Wirehaired Pointing Griffon':
    test_breed = 98
elif test_breed == 'American Eskimo':
    test_breed = 99
elif test_breed == 'Australian Terrier':
    test_breed = 100
elif test_breed == 'Basenji':
    test_breed = 101
elif test_breed == 'Beagle':
    test_breed = 102
elif test_breed == 'Bedlington Terrier':
    test_breed = 103
elif test_breed == 'Bichon Frise':
    test_breed = 104
elif test_breed == 'Border Terrier':
    test_breed = 105
elif test_breed == 'Boston Terrier':
    test_breed = 106
elif test_breed == 'Brussels Griffon':
    test_breed = 107
elif test_breed == 'Cairn Terrier':
    test_breed = 108
elif test_breed == 'Cardigan Welsh Corgi':
    test_breed = 109
elif test_breed == 'Cavalier King Charles Spaniel':
    test_breed = 110
elif test_breed == 'Dachshund':
    test_breed = 111
elif test_breed == 'Dandie Dinmont Terrier':
    test_breed = 112
elif test_breed == 'English Toy Spaniel':
    test_breed = 113
elif test_breed == 'Fox Terrier ?? Smooth':
    test_breed = 114
elif test_breed == 'Fox Terrier ?? Wirehair':
    test_breed = 115
elif test_breed == 'French Bulldog':
    test_breed = 116
elif test_breed == 'German Pinscher':
    test_breed = 117
elif test_breed == 'Glen Imaal Terrier':
    test_breed = 118
elif test_breed == 'Lakeland Terrier':
    test_breed = 119
elif test_breed == 'Manchester Terrier (Standard)':
    test_breed = 120
elif test_breed == 'Poodle Miniature':
    test_breed = 121
elif test_breed == 'Pug':
    test_breed = 122
elif test_breed == 'Puli':
    test_breed = 123
elif test_breed == 'Scottish Terrier':
    test_breed = 124
elif test_breed == 'Sealyham Terrier':
    test_breed = 125
elif test_breed == 'Shetland Sheepdog (Sheltie)':
    test_breed = 126
elif test_breed == 'Shiba Inu':
    test_breed = 127
elif test_breed == 'Shih Tzu':
    test_breed = 128
elif test_breed == 'Chihuahua':
    test_breed = 129
elif test_breed == 'Maltese':
    test_breed = 130
elif test_breed == 'Pomeranian':
    test_breed = 131
elif test_breed == 'Yorkshire Terrier':
    test_breed = 132
else:
    test_breed = 0

test_has_testicles = 1 if test_has_testicles == 'yes' else 0
test_has_vagina = 1 if test_has_vagina == 'yes' else 0
test_leg_raise = 1 if test_leg_raise == 'yes' else 0
test_squatting_down = 1 if test_squatting_down == 'yes' else 0

output = model.predict_proba([[test_breed, test_height_inches, test_weight_lbs, test_has_testicles, test_has_vagina, test_leg_raise, test_squatting_down]])
print("male", "{:.4f}".format(output[0][0]))
print("female", "{:.4f}".format(output[0][1]))

X = dataset[['breed', 'height_inches', 'weight_lbs', 'has_testicles', 'has_vagina', 'leg_raise', 'squatting_down']]
Y = dataset['findings']

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
knn = KNeighborsClassifier(n_neighbors =8)
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
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("../data/car.data")
print(data.head())
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
persons = le.fit_transform(list(data["persons"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

#x-list - features, y-list  -labels
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)
for i in range(1, 30):
    accM = 0
    for j in range(10):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

        #computationally intensive, have to find distance of the point to every point in training data
        #take top K of those points, thus useless to train beforehand and save model
        model = KNeighborsClassifier(n_neighbors=2*i+1)
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        accM += acc
    accM = accM/10
    print("accuracy with {} neighbours is {}".format(2*i+1, accM))

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_test, y_test)
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
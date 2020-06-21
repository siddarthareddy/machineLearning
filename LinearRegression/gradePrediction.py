import pandas as pd
import numpy as np
import  sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import matplotlib.style as style
import pickle

#use Pandas to read data from CSV files as datafrome to wrangle with data(data munging)
data = pd.read_csv("../data/student-mat.csv", sep=";")
# print(data.head)
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

#python has no inbuilt, fast arrays, they have lists
#numpy provides arrays which are nearly as fast as
#native array data structures in other languages
#integrates with pandas
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

#using matplotlib to visualize data
p = "failures"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()

#scikit learn provides repo of classical ML algorithms
#intergrates with numpy, matplotlib, scipy
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y, test_size=0.1)
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y, test_size=0.1)

    #select model
    linear = linear_model.LinearRegression()
    #run the model over data
    linear.fit(x_train, y_train)
    #find accuracy with test data
    acc = linear.score(x_test, y_test)
    print(acc)
    if acc > best:
        best = acc
        #pickle is used to serialize objects, in this case our model
        # save the model in linear object by serializing and dumping in a file
        with open("gradesModel.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("gradesModel.pickle", "rb")
#load the working model directly from the file
linear = pickle.load(pickle_in)
#find intercept and 5 coefficients of the linear model
print("Co: ", linear.coef_)
print("Intercept: ", linear.intercept_)
predictions = linear.predict(x_test)

#printed results
for x in range(len(predictions)):
    print(round(predictions[x]), x_test[x], y_test[x])
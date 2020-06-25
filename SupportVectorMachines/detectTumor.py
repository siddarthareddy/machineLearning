import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

#load the data set
cancer = datasets.load_breast_cancer()

# print(len(cancer.feature_names))
# print(cancer.target_names)

x = cancer.data
y = cancer.target

#split into test and train data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.2)

# print(x_train[0])

#SVM is used for classification, it works by creating hyperplane, which divides data
#kernel(a function of existing dimentions) is used to bring unclear data, a new dimension in which it can be classfied

#hard-margin can effect quality of classifier, overfitting the data, but poorly representing the model
#soft-margin to draw better hyperp lanes, thus better classifier, allowing few points to exist inside margins
#set the parameters for model
clf = svm.SVC(kernel="linear", C=2)
#train the model on the data
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)

clf = KNeighborsClassifier(n_neighbors=13)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
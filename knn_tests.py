import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import KNN

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    print(y_true)
    return accuracy


# print(X_train[21])
# print(y_train)

data= iris.data
names=iris.target_names

#k = 3
k = 5
classifier = KNN(k=k)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

# print(names)
# print("-------------------------------------------------------------")
# print(data)
# print("-------------------------------------------------------------")
# print(predictions)

print("The accuracy is ", accuracy(y_test, predictions))
print(predictions)

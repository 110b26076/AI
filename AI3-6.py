%matplotlib inline
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#讀取iris分類資料集
iris = datasets.load_iris()
X = iris.data
X = X[:, ::2]
Y = iris.target

np.random.seed(29)

#資料集中，2/3當作訓練集，1/3當作測試集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33)
knn_clf = KNeighborsClassifier(n_neighbors = 5)
knn_clf = knn_clf.fit(X_train, Y_train)
Y_predict  = knn_clf.predict(X_test)
score = accuracy_score(Y_test, Y_predict)
print("鳶尾花分類的預測準確率: ", score)

plt.figure(figsize = (6, 6))
colmap = np.array(["blue", "green", "red"])
plt.scatter(X_test[:, 0], X_test[:, 1], c = colmap[Y_test], s = 150, marker = "o", alpha = 0.5)
plt.scatter(X_test[:, 0], X_test[:, 1], c = colmap[Y_predict], s = 50, marker = "o", alpha = 0.5)
plt.xlabel("Sepal length", fontsize = 12)
plt.xlabel("Petal length", fontsize = 12)
plt.show()

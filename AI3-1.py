%matplotlib inline
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

np.random.seed(29)

#讀取Boston房價資料集
boston = datasets.load_boston()
print("資料集的特徵欄位名稱為: ", boston.feature_names)
X = boston.data
Y = boston.target

#資料集中，80%當作訓練集，20%當作測試集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
regression_clf = LinearRegression()
regression_clf.fit(X_train, Y_train)
Y_predict = regression_clf.predict(X_test)
score = r2_score(Y_test, Y_predict)
print("房價的預測準確率: ", score)

#劃出真實房價和預測房價的圖
plt.plot(Y_test, label = "Real Price")
plt.plot(Y_predict, label = "Predict Price")
plt.legend()

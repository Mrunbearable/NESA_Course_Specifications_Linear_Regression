import numpy as np
from sklearn.linear_model import LinearRegression


x = np.array([[2], [4], [6], [8], [10], [12], [14], [16]])
y = np.array([1, 3, 5, 7, 9, 11, 13, 15])
model = LinearRegression()
model.fit(x, y)
y_prediction = model.predict([[4]])
print(f"predicted target is: {y_prediction}")
y_prediction = model.predict([[5.5]])
print(f"predicted target is: {y_prediction}")
y_prediction = model.predict([[4.5]])
print(f"predicted target is: {y_prediction}")
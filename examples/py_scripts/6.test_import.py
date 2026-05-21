import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.linear_model import LinearRegression
import pickle
import os
if not os.path.exists("output"):
    os.makedirs("output")


training_data = pd.read_csv('examples/data/3.course_specifications_data.csv', delimiter=',')
x = np.array(training_data.iloc[:,1]).reshape(-1, 1)
y = np.array(training_data.iloc[:,0])

m = len(x)
print(f"Number of training examples is: {m}")
table = pd.DataFrame({
    training_data.columns[0]: x.flatten(),  # Flatten x for easy display
    training_data.columns[1]: y
})
print(table.head())

loaded_model = pickle.load(open('examples/output/my_saved_model.sav', 'rb'))
predict = np.array([4]).reshape(1, -1)
result = loaded_model.predict(predict)
print(result[0])
# 2.9999999999999982



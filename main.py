import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
#dataframe comes from Kaggle about honey production in the United States
df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

print(df.head())
#Get the mean of the total production of honey per year
prod_per_year = df.groupby('year').totalprod.mean().reset_index()
#create a scatterplot to see if there is a linear relationship between the year and total production
X = prod_per_year['year']
X = X.values.reshape(-1, 1)
print(X)

y = prod_per_year['totalprod']

plt.scatter(X, y)
plt.show()
#create a linear regression model using scikit-learn
regr = linear_model.LinearRegression()

regr.fit(X, y)

# print(regr.coef_, regr.intercept_)

y_predict = regr.predict(X)

plt.plot(X, y_predict)
plt.show()
#predict what honey production will look like in the year 2050 based on the dataset that stops at 2013
X_future = np.array(range(2013, 2050))

X_future = X_future.reshape(-1, 1)

print(X_future)

future_predict = regr.predict(X_future)

plt.plot(X_future, future_predict)
plt.show()

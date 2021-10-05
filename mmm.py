############################################################################################
#                                   Marketing Mix Modeling                                 #
#
# data contains - sales, price, adv1, adv2, adv3
############################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#load data
df = pd.read_csv("mmm_data.csv", thousands=r',')
pd.set_option('display.max_columns', None)
print(df.head())

#remove junk values from data
df = df.replace({'\t': '', '\\s':''}, regex=True)
#encode categorical features
df = pd.get_dummies(df)
print(df.head())

df.info()
print(df.describe())

#check for missing values
print(df.isnull().sum())
#impute missing values using univariate feature imputation
imp = SimpleImputer(missing_values=np.nan, strategy="median")
#impute missing values using multivariate feature imputation
#imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(df.iloc[:, 1:])
df = pd.concat([
	df.iloc[:, 0],
	pd.DataFrame(np.round(imp.transform(df.iloc[:, 1:])), columns=df.columns[1:])], axis=1)
print(df.head())

#Univariate analysis
sns.heatmap(df.corr(), annot= True)
sns.pairplot(df)
#plt.show()

#Adstock
def adstock(feature, r):
    adstock = np.zeros(len(feature))
    adstock[0] = feature[0]
    for i in range(1, len(feature)):
        adstock[i] = feature[i] + r*adstock[i-1]
    return adstock

adv1 = adstock(df.adv1, 0.50)
adv2 = adstock(df.adv2, 0.50)
adv3 = adstock(df.adv3, 0.50)
X = pd.concat([
	pd.DataFrame({"adv1": adv1, "adv2": adv2, "adv3": adv3}),
	df.iloc[:, np.r_[1]]], axis=1)
y = df.iloc[:, 0]

#split data into train & test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#standardization of features
scaler = StandardScaler().fit(X_train)
standardized_X_train = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)

#in case y is skewed then make y = log(y)
#y_train = np.log(y_train)

#linear regression model
lm = LinearRegression()
lm.fit(standardized_X_train, y_train)

#intercepts and coefficients
print(lm.intercept_)
coefficients = pd.DataFrame(np.transpose(lm.coef_), X.columns, columns=['Coefficient'])
print(coefficients)

#predicting sales on test set
y_pred = np.exp(lm.predict(standardized_X_test))
sns.scatterplot(y_test, y_pred)
#plt.show()

#evaluating the model
print('MAE: ', mean_absolute_error(y_test, y_pred))
print('MSE: ', mean_squared_error(y_test, y_pred))
print('RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R-Square: ', r2_score(y_test, y_pred))
print('Adjusted R-Square', 1-(1-r2_score(y_test, y_pred))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)))

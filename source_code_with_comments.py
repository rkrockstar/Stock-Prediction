
import math
import warnings
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import *
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore")

# This code block takes all the input values that re required

print("Input Start Date:")
starting_date = input()
start_date = datetime(year=int(starting_date[0:4]), month=int(starting_date[4:6]), day=int(starting_date[6:8]))  #The date string is converted into datetime format

print("Input number of days:")
number_of_days = int(input())
dti = pd.date_range(start_date, periods=number_of_days)  # This provides the indexes of the dates in the given range
print("Inout End Date:")
ending_date = input()
end_date = datetime(year=int(ending_date[0:4]), month=int(ending_date[4:6]), day=int(ending_date[6:8]))   #The date string is converted into datetime format
dti2 = pd.date_range(end_date, periods=1)  # This provides the indexe of the end date

# This code block takes the input file and loads into a dataframe

df = pd.read_csv("mycsv.csv", parse_dates={'dateTime': ['Date']}, index_col='dateTime')

# Here the data cleaning is done by sorting and removing null values

df.sort_index(axis=0, inplace=True)
df = df.fillna(0)  # placing 0 in place of null values

df2 = pd.DataFrame(df, index=dti2)  # Test value
df1 = pd.DataFrame(df, index=dti)   # Training dataset
df1 = df1.fillna(0)

X = df1[['Open', 'Adj Close']] # Considering 2 features
X1 = df2[['Open', 'Adj Close']]
y = df1['Close'] # Considering Closed price of the day for the given stock

print("Training Dataset for specific number of days from starting date:")
print("X dataset containing Open and Adj Close values:")
print(X)
print("Y dataset containing Close values:")
print(y)

start_test = start_date


X_train = X[X.index > start_test]
X_test = X[X.index >= start_test]
y_train = y[y.index > start_test]
y_test = y[y.index >= start_test]

# Converting to scalar values for training purposes

scalerX = StandardScaler().fit(X_train)
scalery = StandardScaler().fit(y_train)
X_train = scalerX.transform(X_train)
y_train = scalery.transform(y_train)
X_test = scalerX.transform(X_test)
y_test = scalery.transform(y_test)

# This method predicts the price of the given stock on specific date.
# This method also does K-fold cross validation, provides accuracy of the model and also provides training score using mean squared error.
# Graphs are plotted after the implementation of this model

def train_and_evaluate(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    # create a k-fold cross validation iterator of k=5 folds
    cv = KFold(X_train.shape[0], 5, shuffle=True, random_state=33)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    trainPredicted = clf.predict(X_train)
    predicted = clf.predict(X_test)
    predicted1 = clf.predict(X1)
    print(predicted1)
    accuracy = clf.score(X_test, y_test)  # Accuracy determines the most suitable model.
    print("Accuracy of the Model:")
    print(accuracy)
    trainScore = math.sqrt(
        mean_squared_error(scalery.inverse_transform(y_train), scalery.inverse_transform(trainPredicted)))
    testScore = math.sqrt(mean_squared_error(scalery.inverse_transform(y_test), scalery.inverse_transform(predicted)))
    print('Train Score: %.2f RMSE' % (trainScore)) #The RMSE (Root Mean Square Error) is the square root of the variance of the residuals.
    print('Test Score: %.2f RMSE' % (testScore))  #RMSE is an absolute measure of fit.
    plt.plot(scalery.inverse_transform(y_test))
    plt.plot(scalery.inverse_transform(predicted))
    plt.show()

# This method trains the model by fitting the training datasets to the model.
# This method also provides training and testing scores.
# Graphs are plotted after the implementation of this model

def train_linear(reg, X_train, y_train, X_test, y_test):
    reg.fit(X_train, y_train)
    trainPredicted = reg.predict(X_train)
    testPredicted = reg.predict(X_test)
    trainScore = math.sqrt(
        mean_squared_error(scalery.inverse_transform(y_train), scalery.inverse_transform(trainPredicted)))
    testScore = math.sqrt(
        mean_squared_error(scalery.inverse_transform(y_test), scalery.inverse_transform(testPredicted)))
    print('Train Score: %.2f RMSE' % (trainScore))
    print('Test Score: %.2f RMSE' % (testScore))
    plt.plot(scalery.inverse_transform(y_test))
    plt.plot(scalery.inverse_transform(testPredicted))
    plt.show()

# Implementation of different models is done here.
# I have implemented multiple models to check the most suitable/accurate model.
# KNN regressor gives the most accurate value with 2 neighbors. As dataset increases, please set the value appropriately.

print("Prediction for Linear Regression:")

reg = linear_model.LinearRegression()
train_linear(reg, X_train, y_train, X_test, y_test)

print("Prediction for Stochastic Gradient Descent Linear Model:")

clf_sgd = linear_model.SGDRegressor(loss='squared_loss', penalty='l2', random_state=42)
train_and_evaluate(clf_sgd, X_train, y_train, X_test, y_test)

print("Prediction for Epsilon-Support Vector Regression (Kernel type: Poly):")

clf_svr_poly = svm.SVR(kernel='poly')
train_and_evaluate(clf_svr_poly, X_train, y_train, X_test, y_test)

print("Prediction for Epsilon-Support Vector Regression (Kernel type: rbf):")

clf_svr_rbf = svm.SVR(kernel='rbf')
train_and_evaluate(clf_svr_rbf, X_train, y_train, X_test, y_test)

print("Prediction for K-Nearest Neighbor Regressor (With 2 neighbors):")

neigh = KNeighborsRegressor(n_neighbors=2)
train_and_evaluate(neigh, X_train, y_train, X_test, y_test)

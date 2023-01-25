# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:48:38 2022

@author: user
"""

import time

start_counter_ns = time.perf_counter_ns()

import pandas as pd
import numpy as np

df = pd.read_csv("kc_house_data.csv",parse_dates=['date'])

df['date'] = pd.to_numeric(pd.to_datetime(df['date']))

df.info()

def info():

    print("\n################### DATASET INFO ######################\n")

    df.info()  
    
    print("\n####################################################")
    print("\n####################################################")
    print("\n####################################################\n")

    i=0
    for column in df.columns:
        
        print(i,pd.api.types.infer_dtype(df[column]),
              "\t\tUniques:",df.iloc[:,i].nunique())
        
        i+=1
        
info()


X = df.loc[:,df.columns.difference(['price'],sort=False)].values

y = df.iloc[:,1:2].values


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

import sklearn.metrics as met

def accuracy(a,b,c):

    print(f"\n################# ACCURACY TEST NO.{c} ###################")

    print("\nMean absolute error:",round(met.mean_absolute_error(a, b),2))
    print("Mean absolute error%:",1-met.mean_absolute_error(a, b)/y_test.mean())

    print("\nVariance score:",round(met.explained_variance_score(a, b),2))

    print("\nMedian absolute error:",round(met.median_absolute_error(a, b),2))
    print("Median absolute error%:",1-met.median_absolute_error(a, b)/y_test.mean(),'\n')
    
print("############### ALL VARIABLES #################")    
accuracy(y_test,y_pred,1)


# Backwards Elimination:

import statsmodels.api as sm
    
list1=list(range(len(X[0])))

for i in range(22000):
        
    X_opt = np.array(X[:,list1],dtype=float)
        
    regressor_opt = sm.OLS(endog=y,exog=X_opt).fit()
        
    pvalues = list(regressor_opt.pvalues)
        
    q = max(pvalues)
        
    if q > 0.05:
            
        del list1[pvalues.index(q)]
            
    else:
            
        break
    
X_opt_train,X_opt_test,y_train,y_test = train_test_split(X_opt,y,test_size=0.25,random_state=0)

regressor.fit(X_opt_train,y_train)

y_opt_pred = regressor.predict(X_opt_test)

print("############### BACKWARDS ELIMINATION #################")
accuracy(y_test,y_opt_pred,2)

# Polynomial Regression:
    
from sklearn.preprocessing import PolynomialFeatures

regressor_poly = PolynomialFeatures(degree=3)

X_poly = regressor_poly.fit_transform(X)

X_pf_train,X_pf_test,y_train,y_test = train_test_split(X_poly,y,test_size=0.25,random_state=0)

regressor = LinearRegression()

regressor.fit(X_pf_train,y_train)

y_pf_pred = regressor.predict(X_pf_test)

print("############### POLY #################")
accuracy(y_test,y_pf_pred,3)


# SVR (Linear):
    
from sklearn.preprocessing import StandardScaler

standardscaler = StandardScaler()

X = standardscaler.fit_transform(X)

y = np.ravel(standardscaler.fit_transform(y.reshape(-1,1)))

from sklearn.svm import SVR

regressor=SVR(kernel='linear')

from sklearn.model_selection import train_test_split

X_linear_train,X_linear_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

regressor.fit(X_linear_train,y_train)

y_linear_pred=regressor.predict(X_linear_test)

print("############### SVR LINEAR #################")
accuracy(y_test,y_linear_pred,4)


# SVR (Poly):
    
print("############### SVR POLY #################")
regressor=SVR(kernel='poly',degree=3)

from sklearn.model_selection import train_test_split

X_poly_train,X_poly_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

regressor.fit(X_poly_train,y_train)

y_poly_pred=regressor.predict(X_poly_test)

accuracy(y_test,y_poly_pred,5)


# SVR (RBF):
    
regressor=SVR()

from sklearn.model_selection import train_test_split

X_rbf_train,X_rbf_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

regressor.fit(X_rbf_train,y_train)

y_rbf_pred=regressor.predict(X_rbf_test)

print("############### SVR RBF #################")
accuracy(y_test,y_rbf_pred,6)


# SVR (sigmoid):
    
regressor = SVR(kernel='sigmoid',coef0= 10)

from sklearn.model_selection import train_test_split

X_sig_train,X_sig_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

regressor.fit(X_sig_train,y_train)

y_sig_pred=regressor.predict(X_rbf_test)

print("############### SVR SIGMOID #################")
accuracy(y_test,y_sig_pred,7)


end_counter_ns = time.perf_counter_ns()

timer_ns = end_counter_ns - start_counter_ns
    
print(f"{timer_ns/1000000000} seconds")




"""
ٌRevise last 15 minutes
"""




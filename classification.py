# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 08:50:40 2022

@author: msipek3
"""

import csv
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import sklearn.linear_model as lm
import numpy as np
import pandas as pd
from sklearn import model_selection
#from toolbox_02450 import rlr_validate
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


#%%


attributeNames = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]

df = pd.read_csv(
        r"C:\Users\msipek3\Documents\dtu\Courses\Autumn 2022\02450 Intro to machine learning\Projects\library report2\Report 2\glass.data",
        names=attributeNames ,
        delimiter=",")



C = len(attributeNames)

X = df.to_numpy()

# Split dataset into features and target vector
type_column = attributeNames.index('Type')
y = X[:,type_column]

X_cols = np.arange(2,10)
X = X[:,X_cols]


N, M = X.shape


#%%
X = df.to_numpy()
type_column = attributeNames.index('Type')

X = X[0:163,:]

y = X[:,type_column]

X_cols = np.arange(2,10)
X = X[:,X_cols]


N, M = X.shape

#%%
# PREPROCESSING = TARGET COLUMN (NOT FLOAT = 0 / FLOAT PROCESSED = 1)

new_y = np.zeros(len(X))
ones = 0
zeros = 0

for i in range(len(y)):
    if(y[i] == 1 or y[i] == 3):
        new_y[i] = 1
        ones+=1
    else:
        new_y[i] = 0
        zeros+=1

print(new_y)
print(ones)
print(zeros)
print(zeros+ones)

y = new_y
#%%


K1 = 10 # for model selection
K2 = 10 # for optimal parameter selection

# K-fold crossvalidation
CV1 = model_selection.KFold(n_splits=K1, shuffle=True)



# Initialize variable
logistic_regession_test_error = np.zeros(K1)
#logreg_test_error_k1
knn_test_error = np.zeros(K1)
#
baseline_test_error = np.zeros(K1)

LogisticRegression(solver='liblinear')

optimal_lambdas_regression = []
optimal_nearest_neighbor = []

errors_baseline = []
#%%


k1=0
for par_index, test_index in CV1.split(X):
    print('*******************************************************')
    print('CV FOLD : ', k1+1)
    print()
    
    # extract training and test set for current CV fold
    X_training, y_training = X[par_index,:], y[par_index]
    X_test, y_test = X[test_index,:], y[test_index]
    
    CV2 = model_selection.KFold(n_splits=K2, shuffle=False)
    
    #--------------------------------Regularized - LogRegression ----------------------------------------#
    lambdas = np.power(10.,range(-5,5))
    logreg_gen_error_rate_s = np.zeros(len(lambdas))
    
    
    # INNER LOOP
    for s in range(0, len(lambdas)):
        k2 = 0
        logreg_val_error_rate = np.zeros(K2)
        #logreg_val_error_rate
        for train_index, val_index in CV2.split(X_training):

            # extract training and test set for current CV fold
            X_train, y_train = X_training[train_index,:], y_training[train_index]
            X_val, y_val = X_training[val_index,:], y_training[val_index]
        
            logreg_model = LogisticRegression(penalty='l2', C=1/lambdas[s], solver = 'liblinear')
            logreg_model = logreg_model.fit(X_train, y_train)

            logreg_y_val_estimated = logreg_model.predict(X_val).T
            logreg_val_error_rate[k2] = np.sum(logreg_y_val_estimated != y_val) / len(y_val)
            k2 = k2 + 1
            #print(k2)
        
        logreg_gen_error_rate_s[s] = np.sum(logreg_val_error_rate) / len(logreg_val_error_rate)
            
    logreg_min_error = np.min(logreg_gen_error_rate_s)
    opt_lambda_index = np.argmin(logreg_gen_error_rate_s)
    opt_lambda = lambdas[opt_lambda_index]
    
    optimal_lambdas_regression.append(opt_lambda)
    
    logreg_model = LogisticRegression(penalty='l2', C=lambdas[opt_lambda_index], solver = 'liblinear')
    logreg_model = logreg_model.fit(X_training, y_training)
    
    logreg_y_test_estimated = logreg_model.predict(X_test).T
    logistic_regession_test_error[k1] = np.sum(logreg_y_test_estimated != y_test) / len(y_test)
    
    

    #-------------------------------------------------------------- KNN ---------------------------------------------------------------#
    L=[1,2,3,4,5,6,7,8,9,10]
    knn_gen_error_rate_sum = np.zeros(len(L))
    
    
    for i in range(len(L)):  # testing for different number of N (neighbors)
        k3 = 0
        knn_val_error_rate = np.zeros(K2) ## error rate for each N
        
        for train_index, val_index in CV2.split(X_training): ## 
        
            X_train, y_train = X_training[train_index,:], y_training[train_index] 
            X_val, y_val = X_training[val_index,:], y_training[val_index]
            
            knclassifier = KNeighborsClassifier(n_neighbors=L[i])
            knclassifier = knclassifier.fit(X_train, y_train)
            
            knn_estimated = knclassifier.predict(X_val).T
            #print(len(y_val))
            #print(np.sum(knn_estimated != y_val) / len(y_val))
            knn_val_error_rate[k3] = np.sum(knn_estimated != y_val) / len(y_val)
            k3 = k3 + 1

        #print(knn_val_error_rate)
        knn_gen_error_rate_sum[i] = np.sum(knn_val_error_rate[i]) / len(knn_val_error_rate)
        
    knn_min_error = np.min(knn_gen_error_rate_sum)
    optimal_neighbors = np.argmin(knn_gen_error_rate_sum)
    optimal_n = L[optimal_neighbors]       
    
    optimal_nearest_neighbor.append(optimal_n)
    
    knclassifier_full = KNeighborsClassifier(n_neighbors=optimal_n)
    knclassifier_full = knclassifier_full.fit(X_training, y_training)
    
    knn_y_test_estimated = knclassifier_full.predict(X_test).T
    knn_test_error[k1] = np.sum(knn_y_test_estimated != y_test) / len(y_test)

    #----------------------------------------------------------------------------------------------------#
    
    #--------------------------------Baseline  ----------------------------------------------------------#
    class_1_count = y_training.sum() # class 1
    class_0_count = len(y_training) - y_training.sum() # class 0
    baseline_class = float(np.argmax([class_0_count, class_1_count]))

    baseline_test_error[k1] = np.sum(y_test != baseline_class) / len(y_test)
    
    #     print()
    #----------------------------------------------------------------------------------------------------#
    
    k1 = k1 + 1
    print()
    print()
    
    
    
    print('*******************************************************')

#%%

    #---------------------------STATISTICAL TESTS  ----------------------------------------------------------------#
gen_error_regularized_logreg = np.sum(logistic_regession_test_error) / len(logistic_regession_test_error)
gen_error_baseline = np.sum(baseline_test_error) /len(baseline_test_error)
gen_error_knn = np.sum(knn_test_error) /len(knn_test_error)

print('Generalized error Regularized logistic regression',np.round(100 * gen_error_regularized_logreg,decimals = 2), ' %' )
print('Generalized error baseline',np.round(100 * gen_error_baseline,decimals = 2), ' %'  )
print('Generalized error KNN',np.round(100 * gen_error_knn,decimals = 2), ' %'  )
    
z = (logistic_regession_test_error - baseline_test_error)
z_mean = z.mean()
#%%

r_knn_rlr = [y - x for y, x in zip(knn_test_error, logistic_regession_test_error)]
r_knn_baseline = [y - x for y, x in zip(knn_test_error, baseline_test_error)]
r_rlr_baseline = [y - x for y, x in zip(logistic_regession_test_error, baseline_test_error)] #  test statustic -> estimator of THETA



# distribution means
mean_r_knn_rlr = sum(r_knn_rlr)/len(r_knn_rlr)
mean_r_knn_baseline = sum(r_knn_baseline)/len(r_knn_baseline)
mean_r_rlr_baseline= sum(r_rlr_baseline)/len(r_rlr_baseline)

# distribution std
std_r_knn_rlr = 0
for i in range(0, len(r_knn_rlr)):
    std_r_knn_rlr += (r_knn_rlr[i]- mean_r_knn_rlr)**2/(len(r_knn_rlr)*(len(r_knn_rlr)-1))

std_r_knn_baseline = 0
for i in range(0, len(r_knn_baseline)):
    std_r_knn_baseline += (r_knn_baseline[i]- mean_r_knn_baseline)**2/(len(r_knn_baseline)*(len(r_knn_baseline)-1))
    
std_r_rlr_baseline = 0
for i in range(0, len(r_rlr_baseline)):
    std_r_rlr_baseline += (r_rlr_baseline[i]- mean_r_rlr_baseline)**2/(len(r_rlr_baseline)*(len(r_rlr_baseline)-1))

import numpy as np, scipy.stats as st

# compute confidence interval of model regression + ann vs regularized regression
alpha = 0.05
CIA_r_knn_rlr = st.t.interval(1-alpha, df=len(r_knn_rlr)-1, loc=mean_r_knn_rlr, scale=std_r_knn_rlr)  # Confidence interval
p_r_knn_rlr = 2*st.t.cdf( -np.abs( mean_r_knn_rlr )/st.sem(r_knn_rlr), df=len(r_knn_rlr)-1)  # p-value

CIA_r_knn_baseline = st.t.interval(1-alpha, df=len(r_knn_baseline)-1, loc=mean_r_knn_baseline, scale=std_r_knn_baseline)  # Confidence interval
p_r_knn_baseline = 2*st.t.cdf( -np.abs( mean_r_knn_baseline )/st.sem(r_knn_baseline), df=len(r_knn_baseline)-1)  # p-value

CIA_r_rlr_baseline = st.t.interval(1-alpha, df=len(r_rlr_baseline)-1, loc=mean_r_rlr_baseline, scale=std_r_rlr_baseline)  # Confidence interval
p_r_rlr_baseline = 2*st.t.cdf( -np.abs( mean_r_rlr_baseline )/st.sem(r_rlr_baseline), df=len(r_rlr_baseline)-1)  # p-value


#%%


# COMMON ITEMS
import numpy as np, scipy.stats as stats
plt.figure(figsize = [20, 5])
plt.subplots_adjust(wspace = 0.5)


### BASELINE VS REGULAR LOG REG

alpha = 0.05

r_rlr_baseline = [y - x for y, x in zip(logistic_regession_test_error, baseline_test_error)] #  test statustic -> estimator of THETA
mean_r_rlr_baseline= sum(r_rlr_baseline)/len(r_rlr_baseline)

sig =  (r_rlr_baseline - mean_r_rlr_baseline).std() / np.sqrt(K1)

zL = mean_r_rlr_baseline + sig * stats.t.ppf(alpha/2, K1);
zH = mean_r_rlr_baseline + sig * stats.t.ppf(1-alpha/2, K1);

fig = plt.figure()
plt.xticks(np.arange(len(knn_test_error)), np.arange(1, len(knn_test_error)+1))

fig.suptitle("Regularized logistic regression vs Non-regularized baseline logistic regression", fontsize=8)
plt.plot(np.concatenate((logistic_regession_test_error.reshape((len(logistic_regession_test_error), 1))*100, baseline_test_error.reshape((len(baseline_test_error), 1))*100), axis = 1))
plt.xlabel('Outer fold count')
plt.ylabel('Cross-validation error %')
txt="Regularized log regression vs baseline logistic regression"
#plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)



#%%

### KNN VS baseline
alpha = 0.05

sig =  (r_knn_baseline - mean_r_knn_baseline).std() / np.sqrt(K1)

zL = mean_r_knn_baseline + sig * stats.t.ppf(alpha/2, K1);
zH = mean_r_knn_baseline + sig * stats.t.ppf(1-alpha/2, K1);

fig = plt.figure()

fig.suptitle("K nearest neighboors vs Non-regularized baseline logistic regression", fontsize=8)
plt.xticks(np.arange(len(knn_test_error)), np.arange(1, len(knn_test_error)+1))

plt.plot(np.concatenate((knn_test_error.reshape((len(knn_test_error), 1))*100, baseline_test_error.reshape((len(baseline_test_error), 1))*100), axis = 1))
plt.xlabel('Outer fold count')
plt.ylabel('Cross-validation error %')
txt="K nearest neighboors vs baseline logistic regression"


###-------------------------------------------------------------------------------------------------------------------
#%%

### KNN VS REGULAR LOG REG
alpha = 0.05

sig =  (r_knn_rlr - mean_r_knn_rlr).std() / np.sqrt(K1)

zL = mean_r_knn_rlr + sig * stats.t.ppf(alpha/2, K1);
zH = mean_r_knn_rlr + sig * stats.t.ppf(1-alpha/2, K1);

fig = plt.figure()

fig.suptitle("K nearest neighboors vs  Regularized logistic regression", fontsize=8)
plt.xticks(np.arange(len(knn_test_error)), np.arange(1, len(knn_test_error)+1))

plt.plot(np.concatenate((knn_test_error.reshape((len(knn_test_error +1), 1))*100, logistic_regession_test_error.reshape((len(logistic_regession_test_error), 1))*100), axis = 1))
plt.xlabel('Outer fold count')
plt.ylabel('Cross-validation error %')
txt="K nearest neighboors vs baseline logistic regression"


#%%

# 5. Train a logistic regression model using a suitable value of λ (see previous exercise).
# Explain how the logistic regression model make a prediction. Are the
# same features deemed relevant as for the regression part of the report?


optimal_lambda = opt_lambda # from previous task

logistic_regression = LogisticRegression(penalty='l2', C=lambdas[opt_lambda_index], solver = 'liblinear')
logistic_regression = logreg_model.fit(X, y)

print('Weights for logistic regression  features deemed relevant:')

for m in range(M):
    print(attributeNames[m] + " " + str(np.round(logistic_regression.coef_[0][m], 5)))
    print()
    #print('{:>20} {:>20}'.format(attributeNames[m], ))
    
    #str(np.round(logistic_regression.coef_[0][m],3))

#%%

print(range(1,K1))
print(len(optimal_nearest_neighbor))
print(len(knn_gen_error_rate_sum))
print(len(optimal_lambdas_regression))
print(len(logreg_gen_error_rate_s))
print(len(baseline_test_error))\
    
#print(len(optimal_nearest_neighbor))
#%%

###-------------------------------------------------------------------------------------------------------------------
#         safe results in according format
###-------------------------------------------------------------------------------------------------------------------
 
d = {'outer fold' : range(1,K1+1), 'k-nn': optimal_nearest_neighbor, 'e_i1': knn_test_error, 'lambda_i': optimal_lambdas_regression, 'e_i2': logistic_regession_test_error, 'e_i3': baseline_test_error}
performance_table = pd.DataFrame(d)
# print and export to LaTeX
print(performance_table)
print(performance_table.iloc[:,2])
print(performance_table.iloc[:,4])
print(performance_table.iloc[:,5])
print(performance_table.to_latex())


#%%
import numpy as np
import matplotlib.pyplot as plt

plt.imshow(X)
plt.colorbar()
plt.show()
























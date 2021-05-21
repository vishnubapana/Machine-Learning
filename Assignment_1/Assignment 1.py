#!/usr/bin/env python
# coding: utf-8

# In[1]:


def f_true(x):
    y = 6.0 * (np.sin(x + 2) + np.sin(2*x + 4))
    return y


# In[2]:


import numpy as np
n = 750
X = np.random.uniform(-7.5, 7.5, n)
e = np.random.normal(0.0, 5.0, n)
y = f_true(X) + e


# In[3]:


import matplotlib.pyplot as plt
plt.figure()

plt.scatter(X, y, 12, marker = 'o')

x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker = 'None', color='r')


# In[4]:


from sklearn.model_selection import train_test_split
tst_frac = 0.3
val_frac = 0.1

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)
X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)

plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color = 'orange')
plt.scatter(X_val, y_val, 12, marker='o', color = 'green')
plt.scatter(X_tst, y_tst, 12, marker='o', color = 'blue')

# In[5]:


# X float(n, ): univariate data
# d int: degree of polynomial  
def polynomial_transform(X, d):
    van = np.vander(X, d, increasing = True)
    return van

# In[6]:


# Phi float(n, d): transformed data
# y   float(n,  ): labels

def train_model(Phi, y):
    return (np.linalg.inv(Phi.transpose()@Phi))@(Phi.transpose())@(y)


# In[7]:


# Phi float(n, d): transformed data
# y   float(n,  ): labels
# w   float(d,  ): linear regression model
def evaluate_model(Phi, y, w):
    n = len(y)
    summation = 0
    for i  in range (0,n):
        squared_diff = (y[i] - (w.transpose()@Phi[i]))**(2)
        summation = summation + squared_diff
    return (summation/n)

# In[8]:


w = {}               # Dictionary to store all the trained models
validationErr = {}   # Validation error of the models
testErr = {}         # Test error of all the models

for d in range(3, 25, 3):  # Iterate over polynomial degree
    Phi_trn = polynomial_transform(X_trn, d)                 # Transform training data into d dimensions
    w[d] = train_model(Phi_trn, y_trn)                       # Learn model on training data
    
    Phi_val = polynomial_transform(X_val, d)                 # Transform validation data into d dimensions
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d])  # Evaluate model on validation data

    Phi_tst = polynomial_transform(X_tst, d)           # Transform test data into d dimensions
    testErr[d] = evaluate_model(Phi_tst, y_tst, w[d])  # Evaluate model on test data

# Plot all the models
plt.figure()
plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([2, 25, 15, 60])

# In[]:
plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d in range(9, 25, 3):
    X_d = polynomial_transform(x_true, d)
    y_d = X_d @ w[d]
    plt.plot(x_true, y_d, marker='None', linewidth=2)
    
plt.legend(['true'] + list(range(9, 25, 3)))
plt.axis([-8, 8, -15, 15])
# In[9]:
# X float(n, ): univariate data
# B float(n, ): basis functions
# gamma float : standard deviation / scaling of radial basis kernel
import math
def radial_basis_transform(X, B, gamma=0.1):
    Phi = []
    for val in X:
        Z = []
        for mean in B:
            Z.append(math.exp((-gamma)*(np.power(val-mean,2))))
        Phi.append(Z)
    Phi = np.asarray(Phi)
    return Phi

# In[]:
# Phi float(n, d): transformed data
# y   float(n,  ): labels
# lam float      : regularization parameter
def train_ridge_model(Phi, y, lam):
    n = len(Phi)
    w = np.linalg.inv((Phi.transpose()@Phi)+((lam)*(np.eye(n))))@(Phi.transpose()@y)
    return w


# In[]:
w = {}               # Dictionary to store all the trained models
validationErr = {}   # Validation error of the models
testErr = {}         # Test error of all the models

ranges = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
for d in ranges:  # Iterate over polynomial degree
    Phi_trn = radial_basis_transform(X_trn, X_trn)                 # Transform training data into d dimensions
    w[d] = train_ridge_model(Phi_trn, y_trn, d)                       # Learn model on training data
    
    Phi_val = radial_basis_transform(X_val, X_trn)                 # Transform validation data into d dimensions
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d])  # Evaluate model on validation data
    
    Phi_tst = radial_basis_transform(X_tst, X_trn)           # Transform test data into d dimensions
    testErr[d] = evaluate_model(Phi_tst, y_tst, w[d])  # Evaluate model on test data

# Plot all the models
plt.figure()
plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)

# In[]:
plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d in ranges:
    X_d = radial_basis_transform(x_true, X_trn)
    y_d = X_d @ w[d]
    plt.plot(x_true, y_d, marker='None', linewidth=2)
    
plt.legend(['true'] + list(ranges))
plt.axis([-8, 8, -15, 15])

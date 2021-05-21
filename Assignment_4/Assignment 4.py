#!/usr/bin/env python
# coding: utf-8

# 
# 
# ---
# 
# <center><h1>Assignment 4</h1></center>
# 
# ---

# # 1. <font color='#556b2f'> **Support Vector Machines with Synthetic Data**</font>, 50 points. 

# For this problem, we will generate synthetic data for a nonlinear binary classification problem and partition it into training, validation and test sets. Our goal is to understand the behavior of SVMs with Radial-Basis Function (RBF) kernels with different values of $C$ and $\gamma$.

# In[1]:


# DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH DATA GENERATION, 
# MAKE A COPY OF THIS FUNCTION AND THEN EDIT
#
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def generate_data(n_samples, tst_frac=0.2, val_frac=0.2):
  # Generate a non-linear data set
  X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=42)
   
  # Take a small subset of the data and make it VERY noisy; that is, generate outliers
  m = 30
  np.random.seed(30)  # Deliberately use a different seed
  ind = np.random.permutation(n_samples)[:m]
  X[ind, :] += np.random.multivariate_normal([0, 0], np.eye(2), (m, ))
  y[ind] = 1 - y[ind]

  # Plot this data
  cmap = ListedColormap(['#b30065', '#178000'])  
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')       
  
  # First, we use train_test_split to partition (X, y) into training and test sets
  X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, 
                                                random_state=42)

  # Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
  X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, 
                                                random_state=42)
  
  return (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst)


# In[2]:


#
#  DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH VISUALIZATION, 
#  MAKE A COPY OF THIS FUNCTION AND THEN EDIT 
#

def visualize(models, param, X, y):
  # Initialize plotting
  if len(models) % 3 == 0:
    nrows = len(models) // 3
  else:
    nrows = len(models) // 3 + 1
    
  fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5.0 * nrows))
  cmap = ListedColormap(['#b30065', '#178000'])

  # Create a mesh
  xMin, xMax = X[:, 0].min() - 1, X[:, 0].max() + 1
  yMin, yMax = X[:, 1].min() - 1, X[:, 1].max() + 1
  xMesh, yMesh = np.meshgrid(np.arange(xMin, xMax, 0.01), 
                             np.arange(yMin, yMax, 0.01))

  for i, (p, clf) in enumerate(models.items()):
    # if i > 0:
    #   break
    r, c = np.divmod(i, 3)
    ax = axes[r, c]

    # Plot contours
    zMesh = clf.decision_function(np.c_[xMesh.ravel(), yMesh.ravel()])
    zMesh = zMesh.reshape(xMesh.shape)
    ax.contourf(xMesh, yMesh, zMesh, cmap=plt.cm.PiYG, alpha=0.6)

    if (param == 'C' and p > 0.0) or (param == 'gamma'):
      ax.contour(xMesh, yMesh, zMesh, colors='k', levels=[-1, 0, 1], 
                 alpha=0.5, linestyles=['--', '-', '--'])

    # Plot data
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')       
    ax.set_title('{0} = {1}'.format(param, p))


# In[3]:


# Generate the data
n_samples = 300    # Total size of data set 
(X_trn, y_trn), (X_val, y_val), (X_tst, y_tst) = generate_data(n_samples)


# ---
# ### **a**. (25 points)  The effect of the regularization parameter, $C$
# Complete the Python code snippet below that takes the generated synthetic 2-d data as input and learns non-linear SVMs. Use scikit-learn's [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) function to learn SVM models with **radial-basis kernels** for fixed $\gamma$ and various choices of $C \in \{10^{-3}, 10^{-2}\, \cdots, 1, \, \cdots\, 10^5\}$. The value of $\gamma$ is fixed to $\gamma = \frac{1}{d \cdot \sigma_X}$, where $d$ is the data dimension and $\sigma_X$ is the standard deviation of the data set $X$. SVC can automatically use these setting for $\gamma$ if you pass the argument gamma = 'scale' (see documentation for more details).
# 
# **Plot**: For each classifier, compute **both** the **training error** and the **validation error**. Plot them together, making sure to label the axes and each curve clearly.
# 
# **Discussion**: How do the training error and the validation error change with $C$? Based on the visualization of the models and their resulting classifiers, how does changing $C$ change the models? Explain in terms of minimizing the SVM's objective function $\frac{1}{2} \mathbf{w}^\prime \mathbf{w} \, + \, C \, \Sigma_{i=1}^n \, \ell(\mathbf{w} \mid \mathbf{x}_i, y_i)$, where $\ell$ is the hinge loss for each training example $(\mathbf{x}_i, y_i)$.
# 
# **Final Model Selection**: Use the validation set to select the best the classifier corresponding to the best value, $C_{best}$. Report the accuracy on the **test set** for this selected best SVM model. _Note: You should report a single number, your final test set accuracy on the model corresponding to $C_{best}$_.

# In[4]:


# Learn support vector classifiers with a radial-basis function kernel with 
# fixed gamma = 1 / (n_features * X.std()) and different values of C
C_range = np.arange(-3.0, 6.0, 1.0)
C_values = np.power(10.0, C_range)

models = dict()
trnErr = dict()
valErr = dict()

from sklearn import svm, metrics
from sklearn.metrics import mean_squared_error
i = 0
for C in C_values:
    clf = svm.SVC(C=C,kernel='rbf', gamma='scale')
    clf.fit(X_trn, y_trn)
    models[C]=clf
    y_pred_trn = clf.predict(X_trn)
    y_pred_val = clf.predict(X_val)
    trnErr[i] = mean_squared_error(y_trn,y_pred_trn)
    valErr[i] = mean_squared_error(y_val,y_pred_val)
    i+=1
    
plt.figure()
plt.plot(list(valErr.keys()), list(valErr.values()), marker='o', linewidth=3)
plt.plot(list(trnErr.keys()), list(trnErr.values()), marker='s', linewidth=3)
plt.xlabel('C value')
plt.ylabel('Error')
plt.xticks(list(trnErr.keys()), ('0.001', '0.01', '0.1', '1.0', '10.0','100.0','1000.0','10000.0','100000.0'))
plt.legend(['Validation Error', 'Train Error'])

visualize(models, 'C', X_trn, y_trn)

clf = svm.SVC(C=100,kernel='rbf', gamma='scale')
clf.fit(X_trn, y_trn)
y_pred_tst = clf.predict(X_tst)
tstErr = mean_squared_error(y_tst,y_pred_tst)
print("Test Accuracy: ", (1-tstErr)*100,"%")


# ## Discussion:
# Training and validation errors decrease as the value of C increases. However the validation error seems to increase after the C value passes 100. C works to decrease misclassification on the training set by creating harder margins around data. As C increases past 100 it seems to overfit causing the validaiton set to see increase in error.
# 
# ## Final Model Selection:
# C equals to 100 is the most optimal value of C that results in the lower train and validation error and the highest accuracy.

# ---
# ### **b**. (25 points)  The effect of the RBF kernel parameter, $\gamma$
# Complete the Python code snippet below that takes the generated synthetic 2-d data as input and learns various non-linear SVMs. Use scikit-learn's [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) function to learn SVM models with **radial-basis kernels** for fixed $C$ and various choices of $\gamma \in \{10^{-2}, 10^{-1}\, 1, 10, \, 10^2 \, 10^3\}$. The value of $C$ is fixed to $C = 10$.
# 
# **Plot**: For each classifier, compute **both** the **training error** and the **validation error**. Plot them together, making sure to label the axes and each curve clearly.
# 
# **Discussion**: How do the training error and the validation error change with $\gamma$? Based on the visualization of the models and their resulting classifiers, how does changing $\gamma$ change the models? Explain in terms of the functional form of the RBF kernel, $\kappa(\mathbf{x}, \,\mathbf{z}) \, = \, \exp(-\gamma \cdot \|\mathbf{x} - \mathbf{z} \|^2)$
# 
# **Final Model Selection**: Use the validation set to select the best the classifier corresponding to the best value, $\gamma_{best}$. Report the accuracy on the **test set** for this selected best SVM model. _Note: You should report a single number, your final test set accuracy on the model corresponding to $\gamma_{best}$_.

# In[5]:


# Learn support vector classifiers with a radial-basis function kernel with 
# fixed C = 10.0 and different values of gamma
gamma_range = np.arange(-2.0, 4.0, 1.0)
gamma_values = np.power(10.0, gamma_range)

models = dict()
trnErr = dict()
valErr = dict()
i = 0
for G in gamma_values:
    clf = svm.SVC(C=10,kernel='rbf', gamma=G)
    clf.fit(X_trn, y_trn)
    models[G]=clf
    y_pred_trn = clf.predict(X_trn)
    y_pred_val = clf.predict(X_val)
    trnErr[i] = mean_squared_error(y_trn,y_pred_trn)
    valErr[i] = mean_squared_error(y_val,y_pred_val)
    i+=1

plt.figure()
plt.plot(list(valErr.keys()), list(valErr.values()), marker='o', linewidth=3)
plt.plot(list(trnErr.keys()), list(trnErr.values()), marker='s', linewidth=3)
plt.xlabel('Gamma value')
plt.ylabel('Error')
plt.xticks(list(trnErr.keys()), ( '0.01', '0.1', '1.0', '10.0','100.0','1000.0'))
plt.legend(['Validation Error', 'Train Error'])

visualize(models, 'gamma', X_trn, y_trn)

clf = svm.SVC(C=10,kernel='rbf', gamma=1)
clf.fit(X_trn, y_trn)
y_pred_tst = clf.predict(X_tst)
tstErr = mean_squared_error(y_tst,y_pred_tst)
print("Test Error:", (1-tstErr)*100, "%")


# ## Discussion:
# The training and validation error decreases for smaller values of gamma, but after gamma passes 1 the validaiton error starts to increase. The gamma value seems to dictate the amount of influence each data point holds to its nearby datapoints, with smaller gamma values having the most influence and larger having the smallest influence.
# 
# ## Final Model Selection:
# The gamma value of 1 yields the most optimal function with the lowerst validation error.

# ---
# # 2. <font color='#556b2f'> **Breast Cancer Diagnosis with Support Vector Machines**</font>, 25 points. 

# For this problem, we will use the [Wisconsin Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) data set, which has already been pre-processed and partitioned into training, validation and test sets. Numpy's [loadtxt](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html) command can be used to load CSV files.

# In[6]:


train_dataset=np.loadtxt(open("wdbc_trn.csv", "rb"), delimiter=",")
X_trn=train_dataset[:,1:]
y_trn=train_dataset[:,0]
validation_dataset=np.loadtxt(open("wdbc_val.csv", "rb"), delimiter=",")
X_val=validation_dataset[:,1:]
y_val=validation_dataset[:,0]
test_dataset=np.loadtxt(open("wdbc_tst.csv", "rb"), delimiter=",")
X_tst=test_dataset[:,1:]
y_tst=test_dataset[:,0]


# Use scikit-learn's [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) function to learn SVM models with **radial-basis kernels** for **each combination** of $C \in \{10^{-2}, 10^{-1}, 1, 10^1, \, \cdots\, 10^4\}$ and $\gamma \in \{10^{-3}, 10^{-2}\, 10^{-1}, 1, \, 10, \, 10^2\}$. Print the tables corresponding to the training and validation errors.
# 
# **Final Model Selection**: Use the validation set to select the best the classifier corresponding to the best parameter values, $C_{best}$ and $\gamma_{best}$. Report the accuracy on the **test set** for this selected best SVM model. _Note: You should report a single number, your final test set accuracy on the model corresponding to $C_{best}$ and $\gamma_{best}$_.

# In[7]:


range_C = np.arange(-2.0, 5.0, 1.0)
values_C = np.power(10.0, range_C)
c_list=list(values_C)

range_g = np.arange(-3.0, 3.0, 1.0)
values_g = np.power(10.0, range_g)
g_list=list(values_g)

optimal_C=[]
optimal_G=[]
min_error=1
tst_error=1
for C,i in  enumerate(values_C):
    for G,j in enumerate(values_g):
        clf = svm.SVC(C = i,gamma=j,kernel='rbf')
        clf.fit(X_trn, y_trn)
        y_trnpred=clf.predict(X_trn)
        y_valpred=clf.predict(X_val)
        y_tstpred=clf.predict(X_tst)
        val_err= mean_squared_error(y_val,y_valpred)
        tst_error= mean_squared_error(y_tst,y_tstpred)
        if(val_err < min_error ):
            min_error=val_err
            optimal_C.clear()
            optimal_G.clear()
            optimal_C.append(c_list[C])
            optimal_G.append(g_list[G])
        elif (val_err == min_error ):
            optimal_C.append(c_list[C])
            optimal_G.append(g_list[G])
print('Optimal C values: ', optimal_C)                    
print('Optimal G values: ', optimal_G)

clf = svm.SVC(C = 100,gamma=0.01,kernel='rbf')
clf.fit(X_trn, y_trn)
y_tstpred=clf.predict(X_tst)
tst_error= mean_squared_error(y_tst,y_tstpred)
print("Test Accuracy: ", (1-tst_error)*100, "%")


# ## Final model selection: 
# Best C is 100 and best G is 0.01 which results in 86% accuracy on the test case.

# ---
# # 3. <font color='#556b2f'> **Breast Cancer Diagnosis with $k$-Nearest Neighbors**</font>, 25 points. 

# Use scikit-learn's [k-nearest neighbor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) classifier to learn models for Breast Cancer Diagnosis with $k \in \{1, \, 5, \, 11, \, 15, \, 21\}$, with the kd-tree algorithm.
# 
# **Plot**: For each classifier, compute **both** the **training error** and the **validation error**. Plot them together, making sure to label the axes and each curve clearly.
# 
# **Final Model Selection**: Use the validation set to select the best the classifier corresponding to the best parameter value, $k_{best}$. Report the accuracy on the **test set** for this selected best kNN model. _Note: You should report a single number, your final test set accuracy on the model corresponding to $k_{best}$_.

# In[11]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k=[1,5,11,15,21]
train_error=[]
val_error=[]
for i in k:
    neighbor = KNeighborsClassifier(n_neighbors=i,algorithm='kd_tree')
    neighbor.fit(X_trn,y_trn) 
    ytrn_pred=neighbor.predict(X_trn)
    yval_pred=neighbor.predict(X_val)
    train_error.append(mean_squared_error(y_trn,ytrn_pred))
    val_error.append(mean_squared_error(y_val,yval_pred))
    
plt.figure()
plt.plot(k, val_error, marker='o', linewidth=3)
plt.plot(k, train_error, marker='s', linewidth=3)
plt.xlabel('k values')
plt.ylabel('Error')
plt.legend(['Validation Error', 'Train Error'])


neighbor = KNeighborsClassifier(n_neighbors=11,algorithm='kd_tree')
neighbor.fit(X_trn,y_trn) 
ytst_pred=neighbor.predict(X_tst)
print('Test Accuracy: ', (1-mean_squared_error(y_tst,ytst_pred))*100, "%")


# **Discussion**: Which of these two approaches, SVMs or kNN, would you prefer for this classification task? Explain.

# ## Final Model Selection: 
# k equals 11 results in the highest accuracy for this model.
# 
# ## SVM vs kNN:
# kNN algorithm resulted in the greater accuracy overall compard to SVM. kNN performs better when given more attributes. Since this dataset has many attributes it proves that kNN results in higher accuracy.

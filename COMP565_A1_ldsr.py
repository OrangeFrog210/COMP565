"""
 COMP565 Assignment1 Question 2: Implementing LD score regression
 
 Due: 21-September, 2021.
 Yumika Shiba
 260863694
 
---------
# Input: 
#1. Marginal statistics
#2. LD matrix

# Assume:
#1. no population stratification
#2. Phenotype (y) and genotype (X) are standarized

# Task
# To implement LD score regression algorithm
# Purpose: To estimate the heritability of the phenotype
"""

import pandas as pd
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


fIn_marginal = "beta_marginal.csv"
fIn_LD = "LD.csv"


# Read in the data into pandas dataframes
df_marginal = pd.read_csv(fIn_marginal)
df_LD = pd.read_csv(fIn_LD)


"""
Used for my convenience. Can be ignored.
---------
# Check the dimension of the dataset
print(df_marginal.shape)
print(df_LD.shape)

# Check column names
print(df_marginal.columns)
print("------")  # a line for increasing readibility
print(df_LD.columns)

# Taking a look at the data
print(df_marginal.head())
print("------")  # a line for increasing readibility
print(df_LD.head())
"""


# convert the dataframes to numpy arrays
arr_mgStats = df_marginal["V1"].to_numpy()
#print(arr_mgStats)

arr_LD = df_LD.to_numpy()[:, 1:]
#print(arr_LD)


# Checking that row sums and column sums match (since the matrix is symmetric)
# print(np.sum(a=arr_LD, axis=0))
# print(np.sum(a=arr_LD, axis=1))



# Estimating heritability by implementing the basic LD score regression algorithm 
N = 1000
M = 4268
x_val = np.dot(N/M, (np.sum(a=np.square(arr_LD), axis=0))).reshape((-1, 1))
y_val = N * (np.square(arr_mgStats)) - 1
y_val2 = np.dot(N, (np.square(arr_mgStats))) - 1

model = LinearRegression(fit_intercept=False).fit(x_val, y_val, )  # fix the intercept to 1
print("Estimate of heritability ((h_^)**2) is: ", model.coef_)  # model.coef_ = slope


# Alternative
def f(h):
    return np.sum(np.square(np.dot(N, (np.square(arr_mgStats))) - np.dot(N/M, (np.square(h)) * (np.sum(a=np.square(arr_LD), axis=0)))-1))
    #return np.sum(np.square(np.dot(N, (np.square(arr_mgStats))) - np.dot((np.square(h)), (N/M) * (np.sum(a=np.square(arr_LD), axis=0)))-1))

# Declare and initialize variables
N = 1000 # Number of SNPs
M = 4268 # Number of individuals

# Optimizing the objective function derived from LD score regression
result = optimize.minimize_scalar(f)
result.success

h_min = result.x
print("Estimate of heritability ((h_^)**2) is: ", h_min**2)


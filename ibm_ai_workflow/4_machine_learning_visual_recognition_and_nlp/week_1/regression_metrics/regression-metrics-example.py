#!/usr/bin/env python
"""
linear regression example

"""

from __future__ import division
import os,sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def get_simple_regression_samples(n,b0=-0.3,b1=0.5,error=0.2,seed=None):
    if seed:
        np.random.seed(seed)

    trueX =  np.random.uniform(-1,1,n)
    trueT = b0 + (b1*trueX)
    return np.array([trueX]).T, trueT + np.random.normal(0,error,n)


if __name__ == "__main__":

    seed = 42
    n = 20
    b0_true = -0.3
    b1_true = 0.5
    X,y_true = get_simple_regression_samples(n,b0=b0_true,b1=b1_true,seed=seed)

    reg = LinearRegression().fit(X, y_true)
    reg.score(X, y_true)

    ## predict
    y_pred = reg.predict(X)

    print(round(np.linalg.norm(y_pred - y_true) / np.sqrt(n),3))
    print(round(np.sqrt(mean_squared_error(y_pred,y_true)),3))
    print(round(np.std(y_pred-y_true),3))

    ## plot the model
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.plot(X[:,0],y_true,'ko')

    ax.plot(X[:,0],y_pred,color='lightblue',linewidth=4,label='least squares')
    ax.plot(X[:,0], b0_true + X[:,0]*b1_true,linewidth=4,color='darkorange',label='truth')
    ax.legend()
    plt.show()


## print summary
#print("\n-----------------------")
#print("truth: b0=%s,b1=%s"%(b0_true,b1_true))
#print("lstsq fit: b0=%s,b1=%s"%(round(coefs_lstsq[0],3),round(coefs_lstsq[1],3)))

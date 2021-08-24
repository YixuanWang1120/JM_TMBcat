# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:51:38 2021
@author: Yixuan Wang
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import pyswarms as ps
import scipy.stats as stats

#------------------------------------------------------------------------------
# Gauss hermite coef can refer to https://github.com/chebfun/chebfun/
#------------------------------------------------------------------------------
U=np.array([-3.436159118837738,-2.532731674232790,-1.756683649299882,-1.036610829789514\
            ,-0.342901327223705,0.342901327223705,1.036610829789514,1.756683649299882\
            ,2.532731674232790,3.436159118837738])
A=np.array([0.000007640432855,0.001343645746781,0.033874394455481,0.240138611082314\
            ,0.610862633735326,0.610862633735326,0.240138611082314,0.033874394455481\
            ,0.001343645746781,0.000007640432855])
N=len(A)
#------------------------------------------------------------------------------
# Datasets input
#------------------------------------------------------------------------------
End = np.copy(pd.DataFrame(pd.read_excel(" .xlsx", sheet_name= ' ')))
Cov = np.copy(pd.DataFrame(pd.read_excel(" .xlsx", sheet_name= ' ')))
Sam = np.copy(pd.DataFrame(pd.read_excel(" .xlsx", sheet_name= ' ')))
n = len(Cov[0])-1
num = 2*n+2

#------------------------------------------------------------------------------
# Ri=wxi+ui-r+ei (logistic model)
# hi(t)=ho(t)exp(axi+ui) (Cox PH)
#------------------------------------------------------------------------------
def alpha(theta, Cov):
    alpha = np.zeros(len(Cov))
    for i in range(n):
        alpha += theta[i+1]*Cov[:,i+1]
    return alpha

def omega(theta, Cov):
    omega = np.zeros(len(Cov))
    for i in range(n):
        omega += theta[i+n+1]*Cov[:,i+1]
    return omega

def p1(theta, u, End, Cov):
    p1 = ((abs(theta[0])*(End[:,2]**(abs(theta[0])-1))\
           *np.exp(alpha(theta,Cov) + u))**End[:,3])\
         *(np.exp(-(End[:,2]**abs(theta[0]))*np.exp(alpha(theta,Cov) + u)))
    return p1

#------------------------------------------------------------------------------
# p(Ri|ui;theta)  
#------------------------------------------------------------------------------      
def p2(theta, u, End, Cov):
    p2 = ((1+np.exp(-omega(theta,Cov) - u + 0.5))**(-End[:,1]))\
        *((1+np.exp(omega(theta,Cov) + u - 0.5))**(End[:,1]-1))
    return p2

#------------------------------------------------------------------------------
# p(ui;theta) 
#------------------------------------------------------------------------------   
def p3(theta,u):
    p3 = (((2*math.pi)**(0.5)*theta[2*n+1])**(-1))*np.exp(-(1/2)*((u/theta[2*n+1])**2))
    return p3

#------------------------------------------------------------------------------
# Joint probabilty for subject i
#------------------------------------------------------------------------------  
def P(theta, End, Cov):
    f = np.zeros((len(End),N))    
    for j in range(N):
        f[:,j] = A[j]*p1(theta,U[j],End,Cov)*p2(theta,U[j],End,Cov)*p3(theta,U[j])*np.exp(U[j]**2)
    P = np.sum(f,axis=1)
    return P

def pu(theta, u, End, Cov):
    f = p1(theta,u,End,Cov)*p2(theta,u,End,Cov)*p3(theta,u)*(P(theta,End,Cov)**(-1))
    return f

#------------------------------------------------------------------------------
# Joint likelihood function  
#------------------------------------------------------------------------------  
def ML(theta, End, Cov):      
    l = np.log(P(theta, End, Cov))
    L = np.sum(l)
    return -L

#------------------------------------------------------------------------------
# Newton-Raphson
#------------------------------------------------------------------------------  
def dfun(x,End,Cov):
    df = np.zeros(num,dtype=float)
    dx = 1.0e-4
    x1 = np.copy(x)
    for i in range(num):              
        x1 = np.copy(x)
        x1[i] = x1[i]+dx           
        df[i] = (ML(x1,End,Cov)-ML(x,End,Cov))/dx
    return df

def ddfun(x,End,Cov):
    df = np.zeros((num,num),dtype=float)
    dx = 1.0e-4
    x1 = np.copy(x)
    for i in range(num):              
        for j in range(num):
            x1 = np.copy(x)
            x1[j] = x1[j]+dx         
            df[i,j] = (dfun(x1,End,Cov)[i]-dfun(x,End,Cov)[i])/dx   
    return df

def Newton(x,End,Cov):
    x1 = np.copy(x)                                                   
    i = 0
    delta = np.copy(x)
    while(np.sum(abs(delta)) > 1.e-6 and i < 500 and np.sum(abs(dfun(x,End,Cov))) > 1.e-5):
        x1 = x-np.dot(np.linalg.inv(ddfun(x,End,Cov)),dfun(x,End,Cov))        
        delta = x1-x
        x = x1    
        i = i+1
#        print(x)
#        print(ML(x,End,Cov))
    return x

#------------------------------------------------------------------------------
# Likelihood ratio tset
#------------------------------------------------------------------------------ 
def ML0(x, End, Cov):
    xx = np.zeros(num,dtype=float)
    for i in range(len(x)):
        if i < 1:
            xx[i] = x[i]
        elif i < n:
            xx[i+1] = x[i]
        else:
            xx[i+2] = x[i]
    l = np.log(P(xx, End, Cov))
    L = np.sum(l)
    return -L

def dfun0(x,End,Cov):
    df = np.zeros(num-2,dtype=float)
    dx = 1.0e-4           
    x1 = np.copy(x)
    for i in range(num-2):            
        x1 = np.copy(x)
        x1[i] = x1[i]+dx           
        df[i] = (ML0(x1,End,Cov)-ML0(x,End,Cov))/dx  
    return df

def ddfun0(x,End,Cov):
    df = np.zeros((num-2,num-2),dtype=float)
    dx = 1.0e-4                              
    x1 = np.copy(x)
    for i in range(num-2):              
        for j in range(num-2):
            x1 = np.copy(x)
            x1[j] = x1[j]+dx          
            df[i,j] = (dfun0(x1,End,Cov)[i]-dfun0(x,End,Cov)[i])/dx  
    return df

def Newton0(x,End,Cov):
    x1 = np.copy(x)                                                   
    i = 0
    delta = np.copy(x)
    while(np.sum(abs(delta)) > 1.e-6 and i < 500 and np.sum(abs(dfun0(x,End,Cov))) > 1.e-5):
        x1 = x-np.dot(np.linalg.inv(ddfun0(x,End,Cov)),dfun0(x,End,Cov))        
        delta = x1-x
        x = x1    
        i = i+1
#        print(x)
#        print(ML0(x,End,Cov))
    return x
   
def LR(theta0, theta, End, Cov):
    return 2*((-ML(theta,End,Cov))-(-ML(theta0,End,Cov)))

#------------------------------------------------------------------------------
#Random effect bayes estimator
#------------------------------------------------------------------------------ 
def mean(theta,sam,Cov):
    f = np.zeros((len(sam),N))    
    for j in range(N):
        f[:,j] = A[j]*U[j]*pu(theta,U[j],sam,Cov)*np.exp(U[j]**2)
    mean = np.sum(f,axis=1)
    return mean

#------------------------------------------------------------------------------
# ROC curves and thresholds
#------------------------------------------------------------------------------ 
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = np.zeros(len(TPR))
    for i in range(len(TPR)):
        y[i] = TPR[i] - FPR[i]
    Youden_index = np.argmax(y)
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def ROC(label, y_prob):
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return roc_auc, optimal_th, optimal_point

#------------------------------------------------------------------------------
# Particle swarm optimization
#------------------------------------------------------------------------------ 
def fun1(x):
    f = np.zeros(len(x))
    for i in range(len(x)):
        f[i] = ML(x[i],End,Cov)
    return f

def fun0(x):
    f = np.zeros(len(x))
    for i in range(len(x)):
        f[i] = ML0(x[i],End,Cov)
    return f

#%%
#----------------------- Function Definition End ------------------------------
#%%
#------------------------------------------------------------------------------
# Particle swarm optimization for H0
#------------------------------------------------------------------------------ 
lb0=np.array([0.0001,-20,-20,0.0001])
ub0=np.array([5.0,20,20,10])
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=num-2, options=options,bounds=tuple((lb0,ub0)))    
# Perform optimization
best_cost0, best_pos0 = optimizer.optimize(fun0, iters=300)
best_posn = Newton0(best_pos0, End, Cov) 

theta0 = np.zeros(num,dtype=float)
for i in range(len(best_posn)):
    if i < 1:
        theta0[i] = best_posn[i]
    elif i < n:
        theta0[i+1] = best_posn[i]
    else:
        theta0[i+2] = best_posn[i]
        
print('------------finishing fitting the null model------------')

#------------------------------------------------------------------------------
# Particle swarm optimization for H1
#------------------------------------------------------------------------------ 
lb=np.array([0.0,-0.1,-10,-0.1,-10,0])
ub=np.array([3.0,0.1,10,0.1,10,5])
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9} 
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=num, options=options,bounds=tuple((lb,ub)))
# Perform optimization
best_cost, best_pos = optimizer.optimize(fun1, iters=300)

#------------------------------------------------------------------------------
# JM parameters
#------------------------------------------------------------------------------ 
theta = Newton(best_pos,End,Cov)
theta_var = np.linalg.inv(ddfun(theta,End,Cov))
var = [theta_var[i][i] for i in range(num)]
CI_l = [theta[i]-1.96*((var[i])**(1/2)) for i in range(num)]
CI_h = [theta[i]+1.96*((var[i])**(1/2)) for i in range(num)]

print('------------finishing fitting the alternative model------------')
print('JM parameters: ', theta)

LR_sam = LR(theta0,theta,End,Cov)
p_value_chi2 = 1 - stats.chi2.cdf(LR_sam, 2)

print('Likelihood ratio statistics: ', LR_sam)
print('p-value: ', p_value_chi2)


#------------------------------------------------------------------------------
# Composite indexes calculation
#------------------------------------------------------------------------------ 
u_mean = mean(theta,End,Cov)
hazard = alpha(theta,Cov) + u_mean
latent = omega(theta,Cov) + u_mean - 0.5
tmb = Sam[:,4]
label_jmpfs = np.zeros(len(End),dtype=int)
label_jmrr = np.zeros(len(End),dtype=int)

for i in range(len(End)):
    if hazard[i] <= np.percentile(hazard,50):
        label_jmpfs[i] = 1
    else:
        label_jmpfs[i] = 0
        
for i in range(len(End)):
    if latent[i] >= -0.0:
        label_jmrr[i] = 1
    else:
        label_jmrr[i] = 0

label_jm3 = label_binarize(label_jmrr + label_jmpfs, classes=[0, 1, 2])

#------------------------------------------------------------------------------
# ROC curves
#------------------------------------------------------------------------------ 
print('------------preform the dichotomy------------')
fpr, tpr, thresholds = metrics.roc_curve(label_jmrr,tmb,drop_intermediate=False)
roc_auc, cutoff_JMR, optimal_point = ROC(label_jmrr,tmb)
print('the dichotomy threshold is: ', cutoff_JMR)

print('------------preform the trichotomy------------')
fpr_h, tpr_h, thresholds_h = metrics.roc_curve(label_jm3[:,2],tmb,drop_intermediate=False)
roc_auc_h, cutoff_h, optpoint_h = ROC(label_jm3[:,2],tmb)
fpr_l, tpr_l, thresholds_l = metrics.roc_curve(label_jm3[:,0],-tmb,drop_intermediate=False)
roc_auc_l, cutoff_l, optpoint_l = ROC(label_jm3[:,0],-tmb)
cutoff_L = min(abs(cutoff_l),abs(cutoff_h))
cutoff_H = max(abs(cutoff_l),abs(cutoff_h))
print('the trichotomy thresholds are: ', [cutoff_L,cutoff_H])

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:17:57 2022

@author: wyx
"""
import pandas as pd
import numpy as np
import math
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from scipy.stats import multivariate_normal

data = np.copy(pd.DataFrame(pd.read_excel("D:/WYX/data/simu_data.xlsx")))
itmax = 1000
epsilon = 0.001
sigma_err = 0.5
n = len(data)
num = len(data[0])-5
z1 = np.identity(n)
r = np.column_stack((data, z1))
idex = np.argsort((data[:,2]))
r = r[idex,:]
X = r[:,4:num+4]
TMB_E = r[:,num+4:num+5]
T = r[:,2:3]
delta = r[:,3:4]
z = r[:,num+4:num+4+n]

R0 = np.zeros((n,1))
R1 = np.zeros((n,1))
R2 = np.zeros((n,1))
for i in range(n):
    R0[i] = 1 if r[i,1] == 0 else 0
    R1[i] = 1 if r[i,1] == 1 else 0
    R2[i] = 1 if r[i,1] == 2 else 0

M = np.tril(np.ones((n, n), dtype=int), 0)
T_event = np.column_stack((T, delta))
for k in range(1,n):
    if(all(T_event[k] == T_event[k-1])):
        M[:,k] = M[:,k-1]
betaR1 = np.array((-1.6,0.6)).reshape(-1,1)
betaR2 = np.array((-1.0,0.3)).reshape(-1,1)
betaT  = np.array((1.2,-0.4)).reshape(-1,1)
u1 = np.zeros((n,1))
u2 = np.zeros((n,1))
v = np.zeros((n,1))
theta1 = 1
theta2 = 1
theta3 = 1
ro1 = 0.9
ro2 = 0.9
ro3 = 0.9

def P_R(betaR1, betaR2, u1, u2, X, z):
    yetaR1 = np.dot(X, betaR1) + np.dot(z, u1)
    yetaR2 = np.dot(X, betaR2) + np.dot(z, u2)
    p1 = (np.exp(yetaR1)+np.exp(yetaR2))/(1 + np.exp(yetaR1) + np.exp(yetaR2))
    return p1

def S_0(T0):
    yetaT = np.dot(X, betaT) + np.dot(z, v)
    f=0
    for i in range(n):
        if T_event[i,0] <= T0 :        
            #print(i)
            D = sum(T_event[0:i+1,1])
            #print(D)
            f += D/np.dot(M.T, np.exp(yetaT))[i]
            #print(f#%%)
    k = np.exp(-f)          
    return k

km = KaplanMeierFitter()
km.fit(T_event[:,0], event_observed=T_event[:,1])
Med = km.median_survival_time_
T0 = 1#np.mean(T_event[:,0])

def P_T(betaT, v, X, z, T0):
    yetaT = np.dot(X, betaT) + np.dot(z, v)
    p2 = (S_0(T0))**(np.exp(yetaT))
    return p2

mu = np.array([theta1, theta2, theta3])
cov = np.array([[theta1,ro1*math.sqrt(theta1*theta2),ro2*math.sqrt(theta1*theta3)],
                [ro1*math.sqrt(theta1*theta2),theta2,ro3*math.sqrt(theta2*theta3)],
                [ro2*math.sqrt(theta1*theta3),ro3*math.sqrt(theta2*theta3),theta3]])
a = np.column_stack((u1,u2,v))
var = multivariate_normal(mean=mu, cov=cov)
f1=var.pdf(a).reshape(-1,1)

def P(betaR1, betaR2, u1, u2, betaT, v):
    f = P_R(betaR1, betaR2, u1, u2, X, z)*P_T(betaT, v, X, z, T0)*f1
    return f

pro = P(betaR1, betaR2, u1, u2, betaT, v)

data_TMB = np.column_stack((pro,X[:,0:1],r[:,1:2],T_event))
idex2 = np.argsort((data_TMB[:,1]))
TMB_idx = data_TMB[idex2,:]

F = np.zeros((n,n))
P = np.ones((n,n))
for i in range(n-1):
    for j in range(i+1,n):
        print(i)   
        label = np.row_stack((np.full((i,1),1),np.full((j-i,1),2),np.full((n-j,1),3)))
        df = pd.DataFrame(np.column_stack((label, TMB_idx[:,0])))
        df.columns = ['THRE','Prob']    
        model = ols('Prob~C(THRE)',data=df).fit()
        anova_table = anova_lm(model, typ = 2)
        F[i,j] = anova_table.iloc[0,2]
        P[i,j] = anova_table.iloc[0,3]
        
plt.matshow(F,cmap=plt.get_cmap('Greens'),alpha=0.9)
plt.show()

cut1, cut2 = np.where(P == np.min(P))
print(cut1, cut2)

i = cut1[0]
j = cut2[0]
label = np.row_stack((np.full((i,1),1),np.full((j-i,1),2),np.full((n-j,1),3)))

df = pd.DataFrame(np.column_stack((label, TMB_idx[:,0])))
df.columns = ['THRE','Prob']
sns.boxplot(x='THRE',y='Prob',data = df)
model = ols('Prob~C(THRE)',data=df).fit()
anova_table = anova_lm(model, typ = 2)
print(anova_table)
sns.boxplot(x='THRE',y='Prob',data = df)

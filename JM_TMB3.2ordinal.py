# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:54:18 2021

@author: wyx
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import pyswarms as ps
import seaborn as sns
import random
import scipy.stats as stats
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
km = KaplanMeierFitter()
#gauss hermite coef
U=np.array([-3.436159118837738,-2.532731674232790,-1.756683649299882,-1.036610829789514\
            ,-0.342901327223705,0.342901327223705,1.036610829789514,1.756683649299882\
            ,2.532731674232790,3.436159118837738])
A=np.array([0.000007640432855,0.001343645746781,0.033874394455481,0.240138611082314\
            ,0.610862633735326,0.610862633735326,0.240138611082314,0.033874394455481\
            ,0.001343645746781,0.000007640432855])
#trail data
Sam = np.copy(pd.DataFrame(pd.read_excel("D:/WYX/data/GET/skcm_105.xlsx", sheet_name= 'Endpoint')))
Cov = np.copy(pd.DataFrame(pd.read_excel("D:/WYX/data/GET/skcm_105.xlsx", sheet_name= 'covariate')))
TUM = np.copy(pd.DataFrame(pd.read_excel("D:/WYX/data/GET/skcm_105.xlsx", sheet_name= 'All')))
N=len(A)
n = len(Cov[0])-1
num = 2*n+3
#%%
a=[2,3,4,6]
PFS = pd.DataFrame(TUM[:,a])
PFS.columns=['PFS','status','TMB','1']
cph = CoxPHFitter()
cph.fit(PFS, 'PFS', event_col='status',step_size=0.001)
theta_cox=cph.summary.coef
cph.print_summary()

#%%
# Ri=wxi+bui-r+ei (logistic model)
# hi(t)=ho(t)exp(axi+bui) (Cox PH)
def alpha(theta, Cov):
    alpha = np.zeros(len(Cov))
    for i in range(n):
        alpha += theta[i+1]*Cov[:,i+1]
    return alpha

def omega(theta, Cov):
    omega = np.zeros(len(Cov))
    for i in range(n):
        omega += theta[i+n+2]*Cov[:,i+1]
    return omega

def p1(theta, u, Sam, Cov):
    p1 = ((abs(theta[0])*(Sam[:,2]**(abs(theta[0])-1))\
           *np.exp(alpha(theta,Cov) + theta[n+1]*u))**Sam[:,3])\
         *(np.exp(-(Sam[:,2]**abs(theta[0]))*np.exp(alpha(theta,Cov) + theta[n+1]*u)))
    return p1

# p(Ri|ui;theta)        
def p2(theta, u, Sam, Cov):
    RR = label_binarize(Sam[:,1], classes=[0,1,2])
    p2 = ((1+np.exp(omega(theta,Cov)+theta[2*n+2]*u-0.5))**(-RR[:,0]))\
        *(((1+np.exp(omega(theta,Cov)+theta[2*n+2]*u-1.5))**(-1)\
          -(1+np.exp(omega(theta,Cov)+theta[2*n+2]*u-0.5))**(-1))**(RR[:,1]))\
        *((1+np.exp(-omega(theta,Cov)-theta[2*n+2]*u+0.5))**(-RR[:,2]))
    return p2

# p(ui;theta) 
def p3(u):
    p3 = ((2*math.pi)**(-0.5))*np.exp(-(u**2)/2)
    return p3

# joint probabilty for subject i
def P(theta, Sam, Cov):
    f = np.zeros((len(Sam),N))    
    for j in range(N):
        f[:,j] = A[j]*p1(theta,U[j],Sam,Cov)*p2(theta,U[j],Sam,Cov)*p3(U[j])*np.exp(U[j]**2)
    P = np.sum(f,axis=1)
    return P

def pu(theta, u, Sam, Cov):
    f = p1(theta,u,Sam,Cov)*p2(theta,u,Sam,Cov)*p3(u)*(P(theta,Sam,Cov)**(-1))
    return f

# joint likelihood function    
def ML(theta, Sam, Cov):      
    l = np.log(P(theta, Sam, Cov))
    L = np.sum(l)
    return -L

def dfun(x,Sam,Cov):
    df = np.zeros(num,dtype=float)
    dx = 1.0e-4
    x1 = np.copy(x)
    for i in range(num):              # differential
        x1 = np.copy(x)
        x1[i] = x1[i]+dx           #x+dx
        df[i] = (ML(x1,Sam,Cov)-ML(x,Sam,Cov))/dx  #f(x+dx)-f(x)/dx 
    return df

def ddfun(x,Sam,Cov):
    df = np.zeros((num,num),dtype=float)
    dx = 1.0e-4
    x1 = np.copy(x)
    for i in range(num):              # differential
        for j in range(num):
            x1 = np.copy(x)
            x1[j] = x1[j]+dx         #x+dx
            df[i,j] = (dfun(x1,Sam,Cov)[i]-dfun(x,Sam,Cov)[i])/dx   #f(x+dx)-f(x)/dx 
    return df

def Newton(x,Sam,Cov):
    x1 = np.copy(x)                                                   
    i = 0
    delta = np.copy(x)
    while(np.sum(abs(delta)) > 1.e-6 and i < 500 and np.sum(abs(dfun(x,Sam,Cov))) > 1.e-5):
        x1 = x-np.dot(np.linalg.inv(ddfun(x,Sam,Cov)),dfun(x,Sam,Cov))        
        delta = x1-x
        x = x1    
        i = i+1
        print(x)
        print(ML(x,Sam,Cov))
    return x

def ML0(x, Sam, Cov):
    xx = np.zeros(num,dtype=float)
    for i in range(len(x)):
        if i < 1:
            xx[i] = x[i]
        elif i < n+1:
            xx[i+1] = x[i]
        else:
            xx[i+2] = x[i]
    l = np.log(P(xx, Sam, Cov))
    L = np.sum(l)
    return -L

def dfun0(x,Sam,Cov):
    df = np.zeros(num-2,dtype=float)
    dx = 1.0e-4           
    x1 = np.copy(x)
    for i in range(num-2):              # differential
        x1 = np.copy(x)
        x1[i] = x1[i]+dx           #x+dx
        df[i] = (ML0(x1,Sam,Cov)-ML0(x,Sam,Cov))/dx  #f(x+dx)-f(x)/dx 
    return df

def ddfun0(x,Sam,Cov):
    df = np.zeros((num-2,num-2),dtype=float)
    dx = 1.0e-4                              
    x1 = np.copy(x)
    for i in range(num-2):              # differential
        for j in range(num-2):
            x1 = np.copy(x)
            x1[j] = x1[j]+dx          #x+dx
            df[i,j] = (dfun0(x1,Sam,Cov)[i]-dfun0(x,Sam,Cov)[i])/dx   #f(x+dx)-f(x)/dx 
    return df

def Newton0(x,Sam,Cov):
    x1 = np.copy(x)                                                   
    i = 0
    delta = np.copy(x)
    while(np.sum(abs(delta)) > 1.e-6 and i < 500 and np.sum(abs(dfun0(x,Sam,Cov))) > 1.e-5):
        x1 = x-np.dot(np.linalg.inv(ddfun0(x,Sam,Cov)),dfun0(x,Sam,Cov))        
        delta = x1-x
        x = x1    
        i = i+1
        print(x)
        print(ML0(x,Sam,Cov))
    return x
   
#likelihood ratio statistics
def LR(theta0, theta, Sam,Cov):
    return 2*((-ML(theta,Sam,Cov))-(-ML(theta0,Sam,Cov)))

def alpha1(theta, sam):
    alpha=0
    for i in range(n):
        alpha += theta[i+1]*sam[i+3]
    return alpha

def omega1(theta, sam):
    omega=0
    for i in range(n):
        omega += theta[i+n+2]*sam[i+3]
    return omega

def JM(theta,sam): #目标分布p中采样数据
    RR = label_binarize([sam[0]], classes=[0,1,2])
    f = np.sum(A*((abs(theta[0])*(sam[1]**(abs(theta[0])-1))*np.exp(alpha1(theta, sam)\
                          +theta[n+1]*U))**sam[2])\
                 *(np.exp(-(sam[1]**abs(theta[0]))*np.exp(alpha1(theta, sam)\
                                           +theta[n+1]*U)))\
                  *((1+np.exp(omega1(theta,sam)+theta[2*n+2]*U-0.5))**(-RR[:,0]))\
                *(((1+np.exp(omega1(theta,sam)+theta[2*n+2]*U-1.5))**(-1)\
            -(1+np.exp(omega1(theta,sam)+theta[2*n+2]*U-0.5))**(-1))**(RR[:,1]))\
            *((1+np.exp(-omega1(theta,sam)-theta[2*n+2]*U+0.5))**(-RR[:,2]))                                                                         
                 *((2*math.pi)**(-0.5))*np.exp(-(U**2)/2)*np.exp(U**2))
    return f

#random effect bayes estimator
def mean(theta,Sam,Cov):
    f = np.zeros((len(Sam),N))    
    for j in range(N):
        f[:,j] = A[j]*U[j]*pu(theta,U[j],Sam,Cov)*np.exp(U[j]**2)
    mean = np.sum(f,axis=1)
    return mean

def Find_Optimal_Cutoff(TPR, FPR, threshold,c):
    y = np.zeros(len(TPR))
    for i in range(len(TPR)):
        if TPR[i] >= c:
            y[i] = TPR[i] - FPR[i]
        else:
            y[i] = 0
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def ROC(label, y_prob, c):
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds, c=c)
    return roc_auc, optimal_th, optimal_point

def fun1(x):
    f = np.zeros(len(x))
    for i in range(len(x)):
        f[i] = ML(x[i],Sam,Cov)
    return f

def fun0(x):
    f = np.zeros(len(x))
    for i in range(len(x)):
        f[i] = ML0(x[i],Sam,Cov)
    return f
#%%
#----------------------- Function Definition End --------------------------
#%% 
lb0=np.array([0.001,-20,-0,-20,-20])
ub0=np.array([8.0,20,20,20,20])
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    # Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=200, dimensions=num-2, options=options,bounds=tuple((lb0,ub0)))    
    # Perform optimization
best_cost0, best_pos0 = optimizer.optimize(fun0, iters=100)
#%%
x = best_pos0
print(dfun0(x,Sam,Cov))
print(np.sum(abs(dfun0(x,Sam,Cov))))
print(ML0(x,Sam,Cov))  
#%%
best_posn = Newton0(best_pos0, Sam, Cov)

#%%
best_posn = [4.69116,
-3.42329,
5.89642,
-0.575545,
-1.29098]
print(dfun0(best_posn,Sam,Cov))
print(np.sum(abs(dfun0(best_posn,Sam,Cov))))
print(ML0(best_posn,Sam,Cov))    
#%%
theta0 = np.zeros(num,dtype=float)
for i in range(len(best_posn)):
    if i < 1:
        theta0[i] = best_posn[i]
    elif i < n+1:
        theta0[i+1] = best_posn[i]
    else:
        theta0[i+2] = best_posn[i]
#%%
print(dfun0(best_posn,Sam,Cov))
print(np.sum(abs(dfun0(best_posn,Sam,Cov))))
print(ML0(best_posn,Sam,Cov))
theta0 = [6.41037,0,-24.3907,18.9478,0,0.714484,-0.49435]
print(ML(theta0,Sam,Cov))
#%%
lb=np.array([0.1,-10,-10,-10,-10,-10,-10])
ub=np.array([5.0,10,10,10,10,10,10])
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9} 
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=200, dimensions=num, options=options,bounds=tuple((lb,ub)))
# Perform optimization
best_cost, best_pos = optimizer.optimize(fun1, iters=100)
x = best_pos
print(dfun(x,Sam,Cov))
print(np.sum(abs(dfun(x,Sam,Cov))))
print(ML(x,Sam,Cov))  
#%%
theta = Newton(best_pos,Sam,Cov)
#%%
x = [4.67054,-2.61722,-3.19393,5.76511,0.506622,-0.61151,-1.27787]
print(dfun(x,Sam,Cov))
print(np.sum(abs(dfun(x,Sam,Cov))))
print(ML(x,Sam,Cov)) 
theta = [2.324737496,
-17.01875479,
-9.209987153,
3.9470183,
3.305561186,
2.046808754,
-0.876293665]

#%%
theta_var = np.linalg.inv(ddfun(theta,Sam,Cov))
var = [theta_var[i][i] for i in range(num)]
CI_l = [theta[i]-1.96*((var[i])**(1/2)) for i in range(num)]
CI_h = [theta[i]+1.96*((var[i])**(1/2)) for i in range(num)]

#%%
LR_sam = LR(theta0,theta,Sam,Cov)
p_value_chi2 = 1 - stats.chi2.cdf(LR_sam, 2)
print(LR_sam)
print(p_value_chi2)

#%%
u_mean = mean(theta,Sam,Cov)
hazard = alpha(theta,Cov) + theta[n+1]*u_mean
latent = omega(theta,Cov) + theta[2*n+2]*u_mean
#%%
c = 0.0
tmb = TUM[:,4]
label_jmpfs = np.zeros(len(Sam),dtype=int)
for i in range(len(Sam)):
    if hazard[i] <= np.percentile(hazard,50):
        label_jmpfs[i] = 1
    else:
        label_jmpfs[i] = 0
        
label_jmrr = np.zeros(len(Sam),dtype=int)
for i in range(len(Sam)):
    if latent[i] >= 1.8:
        label_jmrr[i] = 1
    else:
        label_jmrr[i] = 0

label_jm3 = label_binarize(label_jmrr + label_jmpfs, classes=[0, 1, 2])
label = label_jmrr + label_jmpfs
label_RR = label_binarize(Sam[:,1], classes=[0,1,2])
   #%%
font1 = {'weight' : 'normal', 'size': 22}

fpr, tpr, thresholds = metrics.roc_curve(label_RR[:,2],tmb,drop_intermediate=False)
roc_auc, cutoff_RR, optimal_point = ROC(label_RR[:,2],tmb,c)

plt.figure(figsize=(12,10))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}",linewidth=4.0)
plt.plot([0, 1], [0, 1], linestyle="--",linewidth=4.0)
plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
plt.text(optimal_point[0], optimal_point[1], f'Threshold:{cutoff_RR:.2f}',font1)
plt.title("ROC-AUC(RR)",font1)
plt.xlabel("False Positive Rate", font1)
plt.ylabel("True Positive Rate",font1)
plt.grid(axis="y")
plt.legend(prop=font1)
plt.show()
#%%
fpr_h, tpr_h, thresholds_h = metrics.roc_curve(label_jm3[:,2],tmb,drop_intermediate=False)
roc_auc_h, cutoff_h, optpoint_h = ROC(label_jm3[:,2],tmb,c)
fpr_l, tpr_l, thresholds_l = metrics.roc_curve(label_jm3[:,0],-tmb,drop_intermediate=False)
roc_auc_l, cutoff_l, optpoint_l = ROC(label_jm3[:,0],-tmb,c)
cutoff_L = (-cutoff_l)
cutoff_H = cutoff_h

plt.figure(figsize=(12,10))
plt.plot(fpr_h, tpr_h, label=f"AUC_H = {roc_auc_h:.3f}",linestyle='--',linewidth=4.0)
plt.plot(fpr_l, tpr_l, label=f"AUC_L = {roc_auc_l:.3f}",linewidth=4.0)
plt.plot([0, 1], [0, 1], linestyle="--",linewidth=4.0)
plt.plot(optpoint_h[0], optpoint_h[1], marker='o', color='r')
plt.text(optpoint_h[0], optpoint_h[1], f'Threshold:{cutoff_H:.2f}',font1)
plt.plot(optpoint_l[0], optpoint_l[1], marker='o', color='r')
plt.text(optpoint_l[0], optpoint_l[1], f'Threshold:{cutoff_L:.2f}',font1)
plt.title("ROC-AUC(3)",font1)
plt.xlabel("False Positive Rate", font1)
plt.ylabel("True Positive Rate",font1)
plt.grid(axis="y")
plt.legend(prop=font1)
plt.show()
#%%
fpr, tpr, thresholds = metrics.roc_curve(label_jmrr,tmb,drop_intermediate=False)
roc_auc, cutoff_JMR, optimal_point = ROC(label_jmrr,tmb,c)

plt.figure(figsize=(12,10))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}",linewidth=4.0)
plt.plot([0, 1], [0, 1], linestyle="--",linewidth=4.0)
plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
plt.text(optimal_point[0], optimal_point[1], f'Threshold:{cutoff_JMR:.2f}',font1)
plt.title("ROC-AUC(2)",font1)
plt.xlabel("False Positive Rate", font1)
plt.ylabel("True Positive Rate",font1)
plt.grid(axis="y")
plt.legend(prop=font1)
plt.show()
#%%
fpr, tpr, thresholds = metrics.roc_curve(label_jmpfs,tmb,drop_intermediate=False)
roc_auc, cutoff_JMR, optimal_point = ROC(label_jmpfs,tmb,c)

plt.figure(figsize=(12,10))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}",linewidth=4.0)
plt.plot([0, 1], [0, 1], linestyle="--",linewidth=4.0)
plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
plt.text(optimal_point[0], optimal_point[1], f'Threshold:{cutoff_JMR:.2f}',font1)
plt.title("ROC-AUC(RR)",font1)
plt.xlabel("False Positive Rate", font1)
plt.ylabel("True Positive Rate",font1)
plt.grid(axis="y")
plt.legend(prop=font1)
plt.show()

#%%RR2
R_RR_H=[]
R_RR_L=[]
T_RR_H=[]
T_RR_L=[]
E_RR_H=[]
E_RR_L=[]

for i in range(len(TUM)):
    if TUM[i,4] >= cutoff_RR:
        R_RR_H.append(TUM[i,1])
        T_RR_H.append(TUM[i,2])
        E_RR_H.append(TUM[i,3])
    else:
        R_RR_L.append(TUM[i,1])
        T_RR_L.append(TUM[i,2])
        E_RR_L.append(TUM[i,3]) 

plt.figure(figsize=(12,10))
km.fit(T_RR_H, event_observed=E_RR_H, label="RR_H")
km.plot()
Med_RR_H = km.median_survival_time_
km.fit(T_RR_L, event_observed=E_RR_L, label="RR_L")
km.plot()
Med_RR_L = km.median_survival_time_

print(cutoff_RR)
print(Med_RR_L)
print(Med_RR_H)
lr = logrank_test(T_RR_L, T_RR_H, event_observed_T_RR_L=E_RR_L, event_observed_T_RR_H=E_RR_H, alpha=.95)
print(lr.p_value)
print(R_RR_L.count(2)/len(R_RR_L)*100)
print(R_RR_H.count(2)/len(R_RR_H)*100)

mw = stats.mannwhitneyu(R_RR_L, R_RR_H, use_continuity = True, alternative = None )
print(mw.pvalue)
#%%X_tile2
R_PFS_H=[]
R_PFS_L=[]
T_PFS_H=[]
T_PFS_L=[]
E_PFS_H=[]
E_PFS_L=[]

for i in range(len(TUM)):
    if TUM[i,4] >= 13.27:
        R_PFS_H.append(TUM[i,1])
        T_PFS_H.append(TUM[i,2])
        E_PFS_H.append(TUM[i,3])
    else:
        R_PFS_L.append(TUM[i,1])
        T_PFS_L.append(TUM[i,2])
        E_PFS_L.append(TUM[i,3])


km.fit(T_PFS_H, event_observed=E_PFS_H, label="PFS_H")
Med_PFS_H = km.median_survival_time_
km.fit(T_PFS_L, event_observed=E_PFS_L, label="PFS_L")
Med_PFS_L = km.median_survival_time_
print(Med_PFS_L)
print(Med_PFS_H)
lr = logrank_test(T_PFS_L, T_PFS_H, event_observed_T_PFS_L=E_PFS_L, event_observed_T_PFS_H=E_PFS_H, alpha=.95)
print(lr.p_value)
print(R_PFS_L.count(2)/len(R_PFS_L)*100)
print(R_PFS_H.count(2)/len(R_PFS_H)*100)
mw = stats.mannwhitneyu(R_PFS_L, R_PFS_H, use_continuity = True, alternative = None )
print(mw.pvalue)

#%%JM2
R_JM_H=[]
R_JM_L=[]
T_JM_H=[]
T_JM_L=[]
E_JM_H=[]
E_JM_L=[]

for i in range(len(TUM)):
    if TUM[i,4] >= cutoff_JMR:
        R_JM_H.append(TUM[i,1])
        T_JM_H.append(TUM[i,2])
        E_JM_H.append(TUM[i,3])
    else:
        R_JM_L.append(TUM[i,1])
        T_JM_L.append(TUM[i,2])
        E_JM_L.append(TUM[i,3]) 

km.fit(T_JM_H, event_observed=E_JM_H, label="JM_H")
Med_JM_H = km.median_survival_time_
km.fit(T_JM_L, event_observed=E_JM_L, label="JM_L")
Med_JM_L = km.median_survival_time_
print(cutoff_JMR)
print(Med_JM_L)
print(Med_JM_H)
lr = logrank_test(T_JM_L, T_JM_H, event_observed_T_JM_L=E_JM_L, event_observed_T_JM_H=E_JM_H, alpha=.95)
print(lr.p_value)
print(R_JM_L.count(2),R_JM_L.count(0)+R_JM_L.count(1))
print(R_JM_H.count(2),R_JM_H.count(0)+R_JM_H.count(1))
print(R_JM_L.count(2)/len(R_JM_L)*100)
print(R_JM_H.count(2)/len(R_JM_H)*100)

mw = stats.mannwhitneyu(R_JM_L, R_JM_H, use_continuity = True, alternative = None )
print(mw.pvalue)
#%%X3
R_PFS3_H=[]
R_PFS3_M=[]
R_PFS3_L=[]
T_PFS3_H=[]
T_PFS3_M=[]
T_PFS3_L=[]
E_PFS3_H=[]
E_PFS3_M=[]
E_PFS3_L=[]

for i in range(len(TUM)):
    if TUM[i,4] >= 15.09:
        R_PFS3_H.append(TUM[i,1])
        T_PFS3_H.append(TUM[i,2])
        E_PFS3_H.append(TUM[i,3])
    elif TUM[i,4] >= 3.28:
        R_PFS3_M.append(TUM[i,1])
        T_PFS3_M.append(TUM[i,2])
        E_PFS3_M.append(TUM[i,3])
    else:
        R_PFS3_L.append(TUM[i,1])
        T_PFS3_L.append(TUM[i,2])
        E_PFS3_L.append(TUM[i,3])

km.fit(T_PFS3_H, event_observed=E_PFS3_H, label="PFS3_H")
km.plot()
Med_PFS3_H = km.median_survival_time_
km.fit(T_PFS3_M, event_observed=E_PFS3_M, label="PFS3_M")
km.plot()
Med_PFS3_M = km.median_survival_time_
km.fit(T_PFS3_L, event_observed=E_PFS3_L, label="PFS3_L")
km.plot()
Med_PFS3_L = km.median_survival_time_
print(Med_PFS3_L)
print(Med_PFS3_M)
print(Med_PFS3_H)
print(R_PFS3_L.count(2)/len(R_PFS3_L)*100)
print(R_PFS3_M.count(2)/len(R_PFS3_M)*100)
print(R_PFS3_H.count(2)/len(R_PFS3_H)*100)
mw = stats.mannwhitneyu(R_PFS3_M, R_PFS3_L, use_continuity = True, alternative = None )
print(mw.pvalue)
#%%JM3
R_JM3_H=[]
R_JM3_M=[]
R_JM3_L=[]
T_JM3_H=[]
T_JM3_M=[]
T_JM3_L=[]
E_JM3_H=[]
E_JM3_M=[]
E_JM3_L=[]

for i in range(len(TUM)):
    if TUM[i,4] >= cutoff_L:
        R_JM3_H.append(TUM[i,1])
        T_JM3_H.append(TUM[i,2])
        E_JM3_H.append(TUM[i,3])
    elif TUM[i,4] >= cutoff_H:
        R_JM3_M.append(TUM[i,1])
        T_JM3_M.append(TUM[i,2])
        E_JM3_M.append(TUM[i,3])
    else:
        R_JM3_L.append(TUM[i,1])
        T_JM3_L.append(TUM[i,2])
        E_JM3_L.append(TUM[i,3])

print(cutoff_H)
print(cutoff_L)
km.fit(T_JM3_H, event_observed=E_JM3_H, label="JM3_H")
km.plot()
Med_JM3_H = km.median_survival_time_
km.fit(T_JM3_M, event_observed=E_JM3_M, label="JM3_M")
km.plot()
Med_JM3_M = km.median_survival_time_
km.fit(T_JM3_L, event_observed=E_JM3_L, label="JM3_L")
km.plot()
Med_JM3_L = km.median_survival_time_
print(Med_JM3_L)
print(Med_JM3_M)
print(Med_JM3_H)
print(R_JM3_L.count(2)/len(R_JM3_L)*100)
print(R_JM3_M.count(2)/len(R_JM3_M)*100)
print(R_JM3_H.count(2)/len(R_JM3_H)*100)
mw = stats.mannwhitneyu(R_JM3_H, R_JM3_M, use_continuity = True, alternative = None )
print(mw.pvalue)
#%%percentile
c=80
R_H=[]
R_L=[]
T_H=[]
T_L=[]
E_H=[]
E_L=[]

for i in range(len(TUM)):
    if TUM[i,4] >= np.percentile(TUM[:,4],c):
        R_H.append(TUM[i,1])
        T_H.append(TUM[i,2])
        E_H.append(TUM[i,3])
    else:
        R_L.append(TUM[i,1])
        T_L.append(TUM[i,2])
        E_L.append(TUM[i,3])

km.fit(T_H, event_observed=E_H, label="PFS_H")
Med_H = km.median_survival_time_
km.fit(T_L, event_observed=E_L, label="PFS_L")
Med_L = km.median_survival_time_
print(np.percentile(TUM[:,4],c))
print(Med_L)
print(Med_H)
lr = logrank_test(T_L, T_H, event_observed_T_L=E_L, event_observed_T_H=E_H, alpha=.95)
print(lr.p_value)
print(R_L.count(2)/len(R_L)*100)
print(R_H.count(2)/len(R_H)*100)
mw = stats.mannwhitneyu(R_L, R_H, use_continuity = True, alternative = None )
print(mw.pvalue)
#%%
idx = np.random.randint(0, len(Sam), size=len(Sam))
TUM_train = TUM[idx] 
test=[]
for i in range(len(Sam)):
    if all(i!=idx):
        test.append(i)
    
TUM_test = TUM[test] 
print(np.median(TUM_test[:,4])-np.median(TUM_train[:,4]))
#%%
Sam_tra = np.copy(pd.DataFrame(pd.read_excel("D:/WYX/data/test/skcm3.xlsx", sheet_name= 'EP_TRA')))
Cov_tra = np.copy(pd.DataFrame(pd.read_excel("D:/WYX/data/test/skcm3.xlsx", sheet_name= 'COV_TRA')))
TUM_tra = np.copy(pd.DataFrame(pd.read_excel("D:/WYX/data/test/skcm3.xlsx", sheet_name= 'TRA')))

Sam_tes = np.copy(pd.DataFrame(pd.read_excel("D:/WYX/data/test/skcm3.xlsx", sheet_name= 'EP_TES')))
Cov_tes = np.copy(pd.DataFrame(pd.read_excel("D:/WYX/data/test/skcm3.xlsx", sheet_name= 'COV_TES')))
TUM_tes = np.copy(pd.DataFrame(pd.read_excel("D:/WYX/data/test/skcm3.xlsx", sheet_name= 'TES')))

n = len(Cov_tra[0])-1
num = 2*n+3
#%%
def fun(x):
    f = np.zeros(len(x))
    for i in range(len(x)):
        f[i] = ML(x[i],Sam_tra,Cov_tra)
    return f
lb=np.array([0.1,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8])
ub=np.array([6.0,8,8,8,8,8,8,8,8,8,8])
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9} 
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=200, dimensions=num, options=options,bounds=tuple((lb,ub)))
# Perform optimization
best_cost, best_pos_tra = optimizer.optimize(fun, iters=200)
x = best_pos_tra
print(dfun(x,Sam_tra,Cov_tra))
print(np.sum(abs(dfun(x,Sam_tra,Cov_tra))))
print(ML(x,Sam_tra,Cov_tra))  
#%%
theta_tra = Newton(best_pos_tra,Sam_tra,Cov_tra)
#%%
print(dfun(theta_tra,Sam_tra,Cov_tra))
print(np.sum(abs(dfun(theta_tra,Sam_tra,Cov_tra))))
print(ML(theta_tra,Sam_tra,Cov_tra))
#%%
u_mean_tes = mean(theta_tra,Sam_tes,Cov_tes)
hazard_tes = alpha(theta_tra,Cov_tes) + theta_tra[n+1]*u_mean_tes
latent_tes = omega(theta_tra,Cov_tes) + theta_tra[2*n+2]*u_mean_tes
#%%
pre_rr = np.zeros(len(Sam_tes),dtype=int)
for i in range(len(Sam_tes)):
    if latent_tes[i] >= 1.5:
        pre_rr[i] = 2
    elif latent_tes[i] >= 0.5:
        pre_rr[i] = 1
#%%
cnf_mat = metrics.confusion_matrix(Sam_tes[:,1],pre_rr)
ACC = metrics.accuracy_score(Sam_tes[:,1],pre_rr)
PRE = metrics.precision_score(Sam_tes[:,1],pre_rr,average='macro')
REC = metrics.recall_score(Sam_tes[:,1],pre_rr,average='macro')
F1S = metrics.f1_score(Sam_tes[:,1],pre_rr,average='macro')
print(ACC)
print(PRE)
print(REC)
print(F1S)
#%%
the = [0.268,2.039,-0.478,-0.634]
logrr = np.zeros(len(Cov_tes))
for i in range(n):
    logrr += the[i]*Cov_tes[:,i+1]
#%%
logpre_rr = np.zeros(len(Sam_tes),dtype=int)
for i in range(len(Sam_tes)):
    if logrr[i] >= 2.292:
        logpre_rr[i] = 2
    elif logrr[i] >= 1.679:
        logpre_rr[i] = 1

#%%
cnf_mat2 = metrics.confusion_matrix(Sam_tes[:,1],logpre_rr)
ACC2 = metrics.accuracy_score(Sam_tes[:,1],logpre_rr)
PRE2 = metrics.precision_score(Sam_tes[:,1],logpre_rr,average='macro')
REC2 = metrics.recall_score(Sam_tes[:,1],logpre_rr,average='macro')
F1S2 = metrics.f1_score(Sam_tes[:,1],logpre_rr,average='macro')
print(ACC2)
print(PRE2)
print(REC2)
print(F1S2)
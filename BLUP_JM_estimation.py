# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:59:28 2022

@author: wyx
"""
import pandas as pd
import numpy as np
import math
import scipy.stats as stats
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from scipy.stats import multivariate_normal
from lifelines.statistics import logrank_test

# ------------------------------------------------------------------------------
# Datasets simulation
# ------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------
# Monte Carlo
# ------------------------------------------------------------------------------
K = 20
TMB_C = np.zeros((n, K), dtype=complex)
for k in range(K):
    TMB_C[:, k] = (TMB_E + np.random.normal(0,
                   sigma_err, (n, 1))*1j).reshape(-1)
#-----------------------------------------------------------------------------
def Score(betaR1,betaR2,betaT,u1,u2,v,theta1,theta2,theta3,ro1,ro2,ro3,X,z,deltaT,R1,R2,M):
    yetaR1 = np.dot(X, betaR1) + np.dot(z,u1)
    yetaR2 = np.dot(X, betaR2) + np.dot(z,u2)
    yetaT = np.dot(X, betaT) + np.dot(z,v)   
    A = deltaT / np.dot(M.T, (np.exp(yetaT)))
    FirstyetaR1 = R1 - np.exp(yetaR1) / (1 + np.exp(yetaR1) + np.exp(yetaR2))
    FirstyetaR2 = R2 - np.exp(yetaR2) / (1 + np.exp(yetaR1) + np.exp(yetaR2))
    FirstyetaT = deltaT - np.exp(yetaT) * np.dot(M, (A*np.ones((n,1))))
    
    FirstbetaR1 = np.dot(X.T, FirstyetaR1)
    FirstbetaR2 = np.dot(X.T, FirstyetaR2)
    FirstbetaT = np.dot(X.T, FirstyetaT)               	    
    Firstu1 = np.dot(z.T, FirstyetaR1)-(( u1*(1-ro3**2) * theta2 * theta3
                                         -u2*(ro1-ro2*ro3) * theta3 * math.sqrt(theta1 * theta2)
	                                     -v *(ro2-ro1*ro3) * theta2 * math.sqrt(theta1 * theta3))
	                                       /(theta1*theta2*theta3*(1-(ro1**2+ro2**2+ro3**2-2*ro1*ro2*ro3))))      	    
    Firstu2 = np.dot(z.T, FirstyetaR2)-((-u1*(ro1-ro2*ro3) * theta3 * math.sqrt(theta1 * theta2)
	                                     +u2*(1-ro2**2) * theta1 * theta3 
	                                     -v *(ro3-ro1*ro2) * theta1 * math.sqrt(theta2 * theta3))
	                                        /(theta1*theta2*theta3*(1-(ro1**2+ro2**2+ro3**2-2*ro1*ro2*ro3))))        	    
    Firstv = np.dot(z.T, FirstyetaT)-((-u1*(ro2-ro1*ro3) * theta2 * math.sqrt(theta1 * theta3)
	                                   -u2*(ro3-ro1*ro2) * theta1 * math.sqrt(theta2 * theta3)
	                                   +v *(1-ro1**2) * theta1 * theta2)
	                                   /(theta1*theta2*theta3*(1-(ro1**2+ro2**2+ro3**2-2*ro1*ro2*ro3))))
    return(np.row_stack((FirstbetaR1,FirstbetaR2,FirstbetaT,Firstu1,Firstu2,Firstv)))

def Hessian(betaR1,betaR2,betaT,u1,u2,v,theta1,theta2,theta3,ro1,ro2,ro3,X,z,deltaT,R1,R2,M):
    yetaR1 = np.dot(X, betaR1) + np.dot(z, u1)
    yetaR2 = np.dot(X, betaR2) + np.dot(z, u2)
    yetaT = np.dot(X, betaT) + np.dot(z,v)
    W1 = (np.exp(yetaR1)+np.exp(yetaR1+yetaR2)) / ((1+np.exp(yetaR1)+np.exp(yetaR2))**2)
    SecdR11 = np.diag(W1.reshape(-1))
    W2 = (np.exp(yetaR2)+np.exp(yetaR1+yetaR2)) / ((1+np.exp(yetaR1)+np.exp(yetaR2))**2)
    SecdR22 = np.diag(W2.reshape(-1))
    W3 = - np.exp(yetaR1+yetaR2) / ((1 + np.exp(yetaR1) + np.exp(yetaR2))**2)
    SecdR12 = np.diag(W3.reshape(-1))	    
    W4 = np.exp(yetaT)
    A = deltaT / np.dot(M.T, (np.exp(yetaT)))
    B = np.dot(M, (A*np.ones((n,1))))
    SecdT = np.diag((W4*B).reshape(-1))-np.dot(np.dot(np.dot(np.dot(np.diag(W4.reshape(-1)),M), np.diag((A*A).reshape(-1))), M.T), np.diag(W4.reshape(-1)))
    				        
    Info11 = np.dot(np.dot(X.T, SecdR11), X)
    Info12 = np.dot(np.dot(X.T, SecdR12), X)
    Info13 = np.zeros((num, num))
    Info14 = np.dot(np.dot(X.T, SecdR11), z)
    Info15 = np.dot(np.dot(X.T, SecdR12), z)
    Info16 = np.zeros((num,n))
	    
    Info21 = Info12.T
    Info22 = np.dot(np.dot(X.T, SecdR22), X)
    Info23 = np.zeros((num, num))
    Info24 = np.dot(np.dot(X.T, SecdR12), z)
    Info25 = np.dot(np.dot(X.T, SecdR22), z)
    Info26 = np.zeros((num,n))
    
    Info31 = Info13.T
    Info32 = Info23.T
    Info33 = np.dot(np.dot(X.T, SecdT), X)
    Info34 = np.zeros((num,n))
    Info35 = np.zeros((num,n))
    Info36 = np.dot(np.dot(X.T, SecdT), z)
      
    Info41 = Info14.T
    Info42 = Info24.T
    Info43 = Info34.T
    Info44 = np.dot(np.dot(z.T, SecdR11), z) + ((1-ro3**2)/(theta1*(1-(ro1**2+ro2**2+ro3**2-2*ro1*ro2*ro3))))*np.identity(n)
    Info45 = np.dot(np.dot(z.T, SecdR12), z) + ((ro2*ro3-ro1)/(math.sqrt(theta1*theta2)*(1-(ro1**2+ro2**2+ro3**2-2*ro1*ro2*ro3))))*np.identity(n)
    Info46 = ((ro1*ro3-ro2)/(math.sqrt(theta1*theta3)*(1-(ro1**2+ro2**2+ro3**2-2*ro1*ro2*ro3))))*np.identity(n)
      
    Info51 = Info15.T
    Info52 = Info25.T
    Info53 = Info35.T
    Info54 = Info45.T
    Info55 = np.dot(np.dot(z.T, SecdR22), z) + ((1-ro2**2)/(theta2*(1-(ro1**2+ro2**2+ro3**2-2*ro1*ro2*ro3))))*np.identity(n)
    Info56 = ((ro1*ro2-ro3)/(math.sqrt(theta2*theta3)*(1-(ro1**2+ro2**2+ro3**2-2*ro1*ro2*ro3))))*np.identity(n)
      
    Info61 = Info16.T
    Info62 = Info26.T
    Info63 = Info36.T
    Info64 = Info46.T
    Info65 = Info56.T
    Info66 = np.dot(np.dot(z.T, SecdT), z) + ((1-ro1**2)/(theta3*(1-(ro1**2+ro2**2+ro3**2-2*ro1*ro2*ro3))))*np.identity(n)
      
    Info1 = np.column_stack((Info11, Info12, Info13, Info14, Info15, Info16))
    Info2 = np.column_stack((Info21, Info22, Info23, Info24, Info25, Info26))
    Info3 = np.column_stack((Info31, Info32, Info33, Info34, Info35, Info36))
    Info4 = np.column_stack((Info41, Info42, Info43, Info44, Info45, Info46))
    Info5 = np.column_stack((Info51, Info52, Info53, Info54, Info55, Info56))
    Info6 = np.column_stack((Info61, Info62, Info63, Info64, Info65, Info66))
    Info0 = np.row_stack((Info1, Info2, Info3, Info4, Info5, Info6))
    Info = np.linalg.inv(Info0) 
    return (Info)
  
def BLUP_VAR(T44,theta1,theta2,theta3,ro1,ro2,ro3):
    KK = 1-(ro1**2+ro2**2+ro3**2-2*ro1*ro2*ro3)   
    A1 = theta1*np.identity(n)
    A2 = ro1*math.sqrt(theta1*theta2)*np.identity(n)
    A3 = ro2*math.sqrt(theta1*theta3)*np.identity(n)     
    A5 = theta2*np.identity(n)    
    A6 = ro3*math.sqrt(theta2*theta3)*np.identity(n)    
    A9 = theta3*np.identity(n)
    
    IN = []
    for i in range(n):
        for j in range(n):
            IN.append(A1[i,j])
    for i in range(n):
        for j in range(n):
            IN.append(A2[i,j])
    for i in range(n):
        for j in range(n):
            IN.append(A3[i,j])
    for i in range(n):
        for j in range(n):
            IN.append(A2[i,j])
    for i in range(n):
        for j in range(n):
            IN.append(A5[i,j])
    for i in range(n):
        for j in range(n):
            IN.append(A6[i,j])
    for i in range(n):
        for j in range(n):
            IN.append(A3[i,j])
    for i in range(n):
        for j in range(n):
            IN.append(A6[i,j])
    for i in range(n):
        for j in range(n):
            IN.append(A9[i,j])
                
    Omega = np.zeros((3*n,3*n))
    k=0
    while k < (3*n)**2:
        for i in range(3*n):
            for j in range(3*n):
                Omega[i,j]=IN[k]
                k+=1
                
    InOTh111 = -2*(1-ro3**2)*theta2*theta3*np.identity(n)
    InOTh112 = (ro1-ro2*ro3)*theta3*math.sqrt(theta1*theta2)*np.identity(n)
    InOTh113 = (ro2-ro1*ro3)*theta2*math.sqrt(theta1*theta3)*np.identity(n)
    InOTh121 = InOTh112.T
    InOTh122 = np.zeros((n,n))
    InOTh123 = np.zeros((n,n))
    InOTh131 = InOTh113.T
    InOTh132 = InOTh123.T
    InOTh133 = np.zeros((n,n))
    InOTh1 = np.row_stack((np.column_stack((InOTh111, InOTh112, InOTh113)),np.column_stack((InOTh121, InOTh122, InOTh123)),np.column_stack((InOTh131, InOTh132, InOTh133))))/(2*theta1**2*theta2*theta3*KK)
    
    InOTh211 = np.zeros((n,n))
    InOTh212 = (ro1-ro2*ro3)*theta3*math.sqrt(theta1*theta2)*np.identity(n)
    InOTh213 = np.zeros((n,n))
    InOTh221 = InOTh212.T
    InOTh222 = -2*(1-ro2**2)*theta1*theta3*np.identity(n)
    InOTh223 = (ro3-ro1*ro2)*theta1*math.sqrt(theta2*theta3)*np.identity(n)
    InOTh231 = InOTh213.T
    InOTh232 = InOTh223.T
    InOTh233 = np.zeros((n,n))
    InOTh2 = np.row_stack((np.column_stack((InOTh211, InOTh212, InOTh213)),np.column_stack((InOTh221, InOTh222, InOTh223)),np.column_stack((InOTh231, InOTh232, InOTh233))))/(2*theta1*theta2**2*theta3*KK)
    
    InOTh311 = np.zeros((n,n))
    InOTh312 = np.zeros((n,n))
    InOTh313 = (ro2-ro1*ro3)*theta2*math.sqrt(theta1*theta3)*np.identity(n)
    InOTh321 = InOTh312.T
    InOTh322 = np.zeros((n,n))
    InOTh323 = (ro3-ro1*ro2)*theta1*math.sqrt(theta2*theta3)*np.identity(n)
    InOTh331 = InOTh313.T
    InOTh332 = InOTh323.T
    InOTh333 = -2*(1-ro1**2)*theta1*theta2*np.identity(n)
    InOTh3 = np.row_stack((np.column_stack((InOTh311, InOTh312, InOTh313)),np.column_stack((InOTh321, InOTh322, InOTh323)),np.column_stack((InOTh331, InOTh332, InOTh333))))/(2*theta1*theta2*theta3**2*KK)
    
    InORo111 = 2*(ro1-ro2*ro3)*(1-ro3**2)*theta2*theta3*np.identity(n)
    InORo112 = -(KK+2*(ro1-ro2*ro3)**2)*theta3*math.sqrt(theta1*theta2)*np.identity(n)
    InORo113 = (ro3*(1+ro1**2+ro2**2-ro3**2)-2*ro1*ro2)*theta2*math.sqrt(theta1*theta3)*np.identity(n)
    InORo121 = InORo112.T
    InORo122 = 2*(ro1-ro2*ro3)*(1-ro2**2)*theta1*theta3*np.identity(n)
    InORo123 = (ro2*(1+ro1**2-ro2**2+ro3**2)-2*ro1*ro3)*theta1*math.sqrt(theta2*theta3)*np.identity(n)
    InORo131 = InORo113.T
    InORo132 = InORo123.T
    InORo133 = 2*(ro1*(ro2**2+ro3**2)-ro2*ro3*(ro1**2+1))*theta1*theta2*np.identity(n)
    InORo1 = np.row_stack((np.column_stack((InORo111, InORo112, InORo113)), np.column_stack((InORo121, InORo122, InORo123)),np.column_stack((InORo131, InORo132, InORo133))))/(theta1*theta2*theta3*KK**2)
    
    InORo211 = 2*(ro2-ro1*ro3)*(1-ro3**2)*theta2*theta3*np.identity(n)
    InORo212 = (ro3*(1+ro1**2+ro2**2-ro3**2)-2*ro1*ro2)*theta3*math.sqrt(theta1*theta2)*np.identity(n)
    InORo213 = -(KK+2*(ro2-ro1*ro3)**2)*theta2*math.sqrt(theta1*theta3)*np.identity(n)
    InORo221 = InORo212.T
    InORo222 = 2*(ro2*(ro1**2+ro3**2)-ro1*ro3*(ro2**2+1))*theta1*theta3*np.identity(n)
    InORo223 = (ro1*(1-ro1**2+ro2**2+ro3**2)-2*ro2*ro3)*theta1*math.sqrt(theta2*theta3)*np.identity(n)
    InORo231 = InORo213.T
    InORo232 = InORo223.T
    InORo233 = 2*(ro2-ro1*ro3)*(1-ro1**2)*theta1*theta2*np.identity(n)
    InORo2 = np.row_stack((np.column_stack((InORo211, InORo212, InORo213)),np.column_stack((InORo221, InORo222, InORo223)),np.column_stack((InORo231, InORo232, InORo233))))/(theta1*theta2*theta3*KK**2)
    
    InORo311 = 2*(ro3*(ro1**2+ro2**2)-ro1*ro2*(ro3**2+1))*theta2*theta3 *np.identity(n)
    InORo312 = (ro2*(1+ro1**2-ro2**2+ro3**2)-2*ro1*ro3)*theta3*math.sqrt(theta1*theta2) *np.identity(n)
    InORo313 = (ro1*(1-ro1**2+ro2**2+ro3**2)-2*ro2*ro3)*theta2*math.sqrt(theta1*theta3) *np.identity(n)
    InORo321 = InORo312.T
    InORo322 = 2*(ro3-ro1*ro2)*(1-ro2**2)*theta1*theta3 *np.identity(n)
    InORo323 = -(KK+2*(ro3-ro1*ro2)**2)*theta1*math.sqrt(theta2*theta3) *np.identity(n)
    InORo331 = InORo313.T
    InORo332 = InORo323.T
    InORo333 = 2*(ro3-ro1*ro2)*(1-ro1**2)*theta1*theta2*np.identity(n)
    InORo3 = np.row_stack((np.column_stack((InORo311, InORo312, InORo313)),
                           np.column_stack((InORo321, InORo322, InORo323)),
                           np.column_stack((InORo331, InORo332, InORo333))))/(theta1*theta2*theta3*KK**2)
      
    K1 = np.dot(T44, InOTh1)
    K2 = np.dot(Omega, InOTh1)
    K3 = np.dot(T44, InOTh2)
    K4 = np.dot(Omega, InOTh2)
    K5 = np.dot(T44, InOTh3)
    K6 = np.dot(Omega, InOTh3)
    K7 = np.dot(T44, InORo1)
    K8 = np.dot(Omega, InORo1)
    K9 = np.dot(T44, InORo2)
    K10 = np.dot(Omega, InORo2)
    K11 = np.dot(T44, InORo3)
    K12 = np.dot(Omega, InORo3)
    
    a11 = sum(np.diag(np.dot(K1, K1) + np.dot(K2, K2) - 2 * np.dot(K1, K2)))
    a12 = sum(np.diag(np.dot(K1, K3) + np.dot(K2, K4) - 2 * np.dot(K1, K4)))
    a13 = sum(np.diag(np.dot(K1, K5) + np.dot(K2, K6) - 2 * np.dot(K1, K6)))
    a14 = sum(np.diag(np.dot(K1, K7) + np.dot(K2, K8) - 2 * np.dot(K1, K8)))
    a15 = sum(np.diag(np.dot(K1, K9) + np.dot(K2, K10) - 2 * np.dot(K1, K10)))
    a16 = sum(np.diag(np.dot(K1, K11) + np.dot(K2, K12) - 2 * np.dot(K1, K12)))
    a22 = sum(np.diag(np.dot(K3, K3) + np.dot(K4, K4) - 2 * np.dot(K3, K4)))
    a23 = sum(np.diag(np.dot(K3, K5) + np.dot(K4, K6) - 2 * np.dot(K3, K6)))
    a24 = sum(np.diag(np.dot(K3, K7) + np.dot(K4, K8) - 2 * np.dot(K3, K8)))
    a25 = sum(np.diag(np.dot(K3, K9) + np.dot(K4, K10) - 2 * np.dot(K3, K10)))
    a26 = sum(np.diag(np.dot(K3, K11) + np.dot(K4, K12) - 2 * np.dot(K3, K12)))
    a33 = sum(np.diag(np.dot(K5, K5) + np.dot(K6, K6) - 2 * np.dot(K5, K6)))
    a34 = sum(np.diag(np.dot(K5, K7) + np.dot(K6, K8) - 2 * np.dot(K5, K8)))
    a35 = sum(np.diag(np.dot(K5, K9) + np.dot(K6, K10) - 2 * np.dot(K5, K10)))
    a36 = sum(np.diag(np.dot(K5, K11) + np.dot(K6, K12) - 2 * np.dot(K5, K12)))
    a44 = sum(np.diag(np.dot(K7, K7) + np.dot(K8, K8) - 2 * np.dot(K7, K8)))
    a45 = sum(np.diag(np.dot(K7, K9) + np.dot(K8, K10) - 2 * np.dot(K7, K10)))
    a46 = sum(np.diag(np.dot(K7, K11) + np.dot(K8, K12) - 2 * np.dot(K7, K12)))
    a55 = sum(np.diag(np.dot(K9, K9) + np.dot(K10, K10) - 2 * np.dot(K9, K10)))
    a56 = sum(np.diag(np.dot(K9, K11) + np.dot(K10, K12) - 2 * np.dot(K9, K12)))
    a66 = sum(np.diag(np.dot(K11, K11) + np.dot(K12, K12) - 2 * np.dot(K11, K12)))
    
    a1 = np.column_stack((a11, a12, a13, a14, a15, a16))
    a2 = np.column_stack((a12, a22, a23, a24, a25, a26))
    a3 = np.column_stack((a13, a23, a33, a34, a35, a36))
    a4 = np.column_stack((a14, a24, a34, a44, a45, a46))
    a5 = np.column_stack((a15, a25, a35, a45, a55, a56))
    a6 = np.column_stack((a16, a26, a36, a46, a56, a66))
    a = np.row_stack((a1, a2, a3, a4, a5, a6))
    var_v = 2 * np.diag(np.linalg.inv(a))
    
    se_theta1 = math.sqrt(var_v[0])
    se_theta2 = math.sqrt(var_v[1])
    se_theta3 = math.sqrt(var_v[2])
    se_ro1 = math.sqrt(var_v[3])
    se_ro2 = math.sqrt(var_v[4])
    se_ro3 = math.sqrt(var_v[5])    
    return(se_theta1,se_theta2,se_theta3,se_ro1,se_ro2,se_ro3)     

flag_var = 0
betaR1 = np.array((-1.6,0.6)).reshape(-1,1)
betaR2 = np.array((-1.0,0.3)).reshape(-1,1)
betaT  = np.array((1.2,-0.4)).reshape(-1,1)
u1 = np.zeros((n,1))
u2 = np.zeros((n,1))
v = np.zeros((n,1))
theta1 = 1
theta2 = 1
theta3 = 1
ro1 = 0.8
ro2 = 0.8
ro3 = 0.8
J1 = np.diag(np.array(([1]*n,[0]*n,[0]*n)).reshape(-1))
J2 = np.row_stack((np.column_stack((np.zeros((n,n)), np.identity(n) , np.zeros((n,n)))),
                   np.column_stack((np.identity(n) , np.zeros((n,n)), np.zeros((n,n)))),
                   np.column_stack((np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))))))
J3 = np.row_stack((np.column_stack((np.zeros((n,n)), np.zeros((n,n)), np.identity(n) )),
                   np.column_stack((np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)))),
                   np.column_stack((np.identity(n) , np.zeros((n,n)), np.zeros((n,n))))))
J4 = np.diag(np.array(([0]*n,[1]*n,[0]*n)).reshape(-1))
J5 = np.row_stack((np.column_stack((np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)))),
                   np.column_stack((np.zeros((n,n)), np.zeros((n,n)), np.identity(n) )),
                   np.column_stack((np.zeros((n,n)), np.identity(n) , np.zeros((n,n))))))
J6 = np.diag(np.array(([0]*n,[0]*n,[1]*n)).reshape(-1))

for iter0 in range(itmax):
    for iter1 in range(itmax):        
        para = np.row_stack((betaR1, betaR2, betaT, u1, u2, v))         
        para1 = para + np.dot(Hessian(betaR1,betaR2,betaT,u1,u2,v,theta1,theta2,theta3,ro1,ro2,ro3,X,z,delta,R1,R2,M),
                              Score(betaR1,betaR2,betaT,u1,u2,v,theta1,theta2,theta3,ro1,ro2,ro3,X,z,delta,R1,R2,M))	        	
        betaR1 = para1[0:num]
        betaR2 = para1[num:2*num]
        betaT = para1[2*num:3*num]
        u1 = para1[3*num:3*num+n]
        u2 = para1[3*num+n:3*num+2*n]
        v = para1[3*num+2*n:3*num+3*n]								
        if max(abs(para1 - para)) > 50:
            print('break')    
            break			
        if max(abs(para1 - para)) < epsilon:
            flag_para = 1
            break        
        para = para1    

    Info = Hessian(betaR1,betaR2,betaT,u1,u2,v,theta1,theta2,theta3,ro1,ro2,ro3,X,z,delta,R1,R2,M)
    T44 = Info[3*num:3*num+3*n, 3*num:3*num+3*n]               
    a = np.row_stack((u1, u2, v))
    L1 = sum(np.diag(np.dot(J1, (T44 + np.dot(a, a.T)))))
    L2 = sum(np.diag(np.dot(J2, (T44 + np.dot(a, a.T)))))/2
    L3 = sum(np.diag(np.dot(J3, (T44 + np.dot(a, a.T)))))/2
    L4 = sum(np.diag(np.dot(J4, (T44 + np.dot(a, a.T)))))
    L5 = sum(np.diag(np.dot(J5, (T44 + np.dot(a, a.T)))))/2
    L6 = sum(np.diag(np.dot(J6, (T44 + np.dot(a, a.T))))) 	  
    theta10 = L1/n
    theta20 = L4/n
    theta30 = L6/n	  
    ro10 = L2/math.sqrt(L1*L4)
    ro20 = L3/math.sqrt(L1*L6)
    ro30 = L5/math.sqrt(L4*L6)	  
    if max(abs(np.array([theta1-theta10,theta2-theta20,theta3-theta30,ro1-ro10,ro2-ro20,ro3-ro30])))>100:
        print('BREAK!!!')
        break		
    if max(abs(np.array([theta1-theta10,theta2-theta20,theta3-theta30,ro1-ro10,ro2-ro20,ro3-ro30])))<epsilon:
        flag_var=1
        break		
    theta1 = theta10
    theta2 = theta20
    theta3 = theta30
    ro1 = ro10
    ro2 = ro20
    ro3 = ro30		    
    print('iter0=',iter0,'theta1=',theta1,'theta2=',theta2,'theta3=',theta3,
          'ro1=',ro1,'ro2=',ro2,'ro3=',ro3)

Info = Hessian(betaR1,betaR2,betaT,u1,u2,v,theta1,theta2,theta3,ro1,ro2,ro3,X,z,delta,R1,R2,M)
T44 = Info[3*num:3*num+3*n, 3*num:3*num+3*n]               
svar = BLUP_VAR(T44,theta1,theta2,theta3,ro1,ro2,ro3)
se_betaR1 = ((np.diag(Info)[0:num])**(1/2)).reshape(-1,1)
se_betaR2 = ((np.diag(Info)[num:2*num])**(1/2)).reshape(-1,1)
se_betaT = ((np.diag(Info)[2*num:3*num])**(1/2)).reshape(-1,1)
se_theta1 = svar[0]
se_theta2 = svar[1]
se_theta3 = svar[2]
se_ro1 = svar[3]
se_ro2 = svar[4]
se_ro3 = svar[5]

theta = np.row_stack((betaR1,betaR2,betaT,theta1,theta2,theta3,ro1,ro2,ro3))
var = np.row_stack((se_betaR1,se_betaR2,se_betaT,se_theta1,se_theta2,se_theta3,se_ro1,se_ro2,se_ro3))
THETA = theta
VAR = var

print("betaR1=",betaR1,"se=",se_betaR1,"betaR1/se=",betaR1/se_betaR1,"p-value=",2*(1-stats.norm.cdf(abs(betaR1/se_betaR1))),'\n')
print("betaR2=",betaR2,"se=",se_betaR2,"betaR2/se=",betaR2/se_betaR2,"p-value=",2*(1-stats.norm.cdf(abs(betaR2/se_betaR2))),'\n')
print("betaT=",betaT,"se=",se_betaT,"betaT/se=",betaT/se_betaT,"p-value=",2*(1-stats.norm.cdf(abs(betaT/se_betaT))),'\n')
  
print("theta1=",theta1,"se=",se_theta1,"theta1/se=",theta1/se_theta1,"p-value=",2*(1-stats.norm.cdf(abs(theta1/se_theta1))),'\n')
print("theta2=",theta2,"se=",se_theta2,"theta2/se=",theta2/se_theta2,"p-value=",2*(1-stats.norm.cdf(abs(theta2/se_theta2))),'\n')
print("theta3=",theta3,"se=",se_theta3,"theta3/se=",theta3/se_theta3,"p-value=",2*(1-stats.norm.cdf(abs(theta3/se_theta3))),'\n')
  
print("ro1=",ro1,"se=",se_ro1,"ro1/se=",ro1/se_ro1,"p-value=",2*(1-stats.norm.cdf(abs(ro1/se_ro1))),'\n')
print("ro2=",ro2,"se=",se_ro2,"ro2/se=",ro2/se_ro2,"p-value=",2*(1-stats.norm.cdf(abs(ro2/se_ro2))),'\n')
print("ro3=",ro3,"se=",se_ro3,"ro3/se=",ro3/se_ro3,"p-value=",2*(1-stats.norm.cdf(abs(ro3/se_ro3))),'\n')
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 18:37:51 2021

@author: wyx
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

ROC1 = np.copy(pd.DataFrame(pd.read_excel("D:/WYX/data/ROCs.xlsx", sheet_name= '27')))
ROC2 = np.copy(pd.DataFrame(pd.read_excel("D:/WYX/data/ROCs.xlsx", sheet_name= '56')))

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

c = 0.0
font1 = {'weight' : 'normal', 'size': 22}
font2 = {'weight' : 'normal', 'size': 18}
fpr_R1, tpr_R1, thresholds_R1 = metrics.roc_curve(ROC1[:,1],ROC1[:,0],drop_intermediate=False)
roc_auc_R1, cutoff_R1, optpoint_R1 = ROC(ROC1[:,1],ROC1[:,0],c)
fpr_R2, tpr_R2, thresholds_R2 = metrics.roc_curve(ROC2[:,1],ROC2[:,0],drop_intermediate=False)
roc_auc_R2, cutoff_R2, optpoint_R2 = ROC(ROC2[:,1],ROC2[:,0],c)

plt.figure(figsize=(10,10))
plt.plot(fpr_R1, tpr_R1, label=f"AUC_bladder = {roc_auc_R1:.3f}",linewidth=3.0)
plt.plot(fpr_R2, tpr_R2, label=f"AUC_ccRCC = {roc_auc_R2:.3f}",linewidth=3.0)

plt.plot([0, 1], [0, 1], color='grey', linestyle="--",linewidth=2.0)
plt.title("ROC for single endpoint (ORR)",font1)
plt.xlabel("False Positive Rate", font1)
plt.ylabel("True Positive Rate",font1)
plt.grid(axis="y")
plt.legend(prop=font2)
plt.show()

fpr_JM1, tpr_JM1, thresholds_JM1 = metrics.roc_curve(ROC1[:,2],ROC1[:,0],drop_intermediate=False)
roc_auc_JM1, cutoff_JM1, optpoint_JM1 = ROC(ROC1[:,2],ROC1[:,0],c)
fpr_JM2, tpr_JM2, thresholds_JM2 = metrics.roc_curve(ROC2[:,4],ROC2[:,0],drop_intermediate=False)
roc_auc_JM2, cutoff_JM2, optpoint_JM2 = ROC(ROC2[:,4],ROC2[:,0],c)


plt.figure(figsize=(10,10))
plt.plot(fpr_JM1, tpr_JM1, label=f"AUC_bladder = {roc_auc_JM1:.3f}", linewidth=3.0)
plt.plot(fpr_JM2, tpr_JM2, label=f"AUC_ccRCC = {roc_auc_JM2:.3f}", linewidth=3.0)

plt.plot([0, 1], [0, 1], color='grey', linestyle="--",linewidth=2.0)
plt.title("ROC for mixed endpoint",font1)
plt.xlabel("False Positive Rate", font1)
plt.ylabel("True Positive Rate",font1)
plt.grid(axis="y")
plt.legend(prop=font2)
plt.show()

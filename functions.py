import os
import copy
import json
import itertools
import shutil as sh
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import math
from datetime import datetime

from CRYSTALpytools.crystal_io import Crystal_output, Crystal_input, Crystal_density, Crystal_gui
from CRYSTALpytools.convert import cry_gui2pmg, cry_out2pmg, cry_pmg2gui
from CRYSTALpytools.utils import view_pmg

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, PointGroupAnalyzer

from ase.visualize import view

from dscribe.descriptors import CoulombMatrix
from dscribe.descriptors import EwaldSumMatrix

from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, max_error, mean_absolute_error
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

### Error analysis
plt.rcParams["figure.figsize"] = (12,12)

def r2(real,pred):
    r2 = r2_score(real, pred)
    return r2

def mae(real,pred):
    mae = mean_absolute_error(real, pred)*1000
    return mae

#maximum error
def maxer(real,pred):
    maxer = max_error(real, pred)*1000
    return maxer

def errorgraph(real,pred, descriptor, model):
    paratesting = pd.DataFrame()
    r2_ = []
    mae_ = []
    maxer_ = []
    
    plt.figure(dpi=400)
    plt.scatter(real, pred, marker='o')
    plt.ylabel("Predicted values")
    plt.xlabel("Calculated values")
    
    vmin=min(min(real),min(pred))
    vmax=max(max(real),max(pred))
    line=np.linspace(vmin,vmax)
    plt.plot(line,line,color='green')
    
    
    plt.title('%s %s' %(descriptor, model), fontsize=16)
    
    plt.show()
    plt.close()
    r2_.append(r2(real,pred))
    mae_.append(mae(real,pred))
    maxer_.append(maxer(real,pred))
            
    paratesting['r^2 value'] = r2_
    paratesting['mean absolute error'] = mae_
    paratesting['maximum error'] = maxer_
    
    return paratesting

## simply running machine learning
''' ml_run simple way to run ml and only selecting one test size,
the random state is automatically set to 1'''
def ml_run(descriptor,energies,model,testsize):
    r2_ = []
    test = testsize
    mae_ = []
    maxer_ = []
    paratesting = pd.DataFrame()
    
    start = datetime.now()
    X_train, X_test, y_train, y_test = train_test_split(descriptor, energies, random_state=1, test_size = test)
    
    scaler = StandardScaler()  
    scaler.fit(X_train)  
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test) 

    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    
    
    r2_.append(r2(y_test,ypred))
    mae_.append(mae(y_test, ypred))
    maxer_.append(maxer(y_test,ypred))
    
    
    paratesting['r^2 value'] = r2_
    paratesting['mean absolute error'] = mae_
    paratesting['maximum error'] = maxer_
    paratesting['test size'] = float(test)
    
    print("--- %s time taken ---" % ((datetime.now() - start)))
    return paratesting,ypred,y_test,X_train,X_test

## finding the best training set
''' trainsize_ allows you to set the training size that you want, 
the random state is automatically set to 1 and uses the ml_run function'''

def trainsize_ (descriptor, energies, model, p_s, p_e, p_i):
    train_para = np.arange(p_s, p_e, p_i)
    paratesting = pd.DataFrame()
    for i in train_para:
        newrow = ml_run(descriptor,energies,model,i)
        
        paratesting = pd.concat([paratesting, newrow[0]], ignore_index=True)
    return paratesting

''' with the _trainsize, this one you can set the value for the random state,
but the test size is set to 10% till 90%'''


def train_size(descriptor, energies, model, ranstate): 
    test_para = np.arange(0.1,1,0.1)
    r2_ = []
    mae_ = []
    maxer_ = []
    paratesting = pd.DataFrame()
    paratesting['test size'] = test_para
    for i in test_para:
        X_train, X_test, y_train, y_test = train_test_split(descriptor, energies, random_state=ranstate, test_size = i)
        scaler = StandardScaler()  
        scaler.fit(X_train)  
        X_train = scaler.transform(X_train)  
        X_test = scaler.transform(X_test)
            
        model.fit(X_train, y_train)
        ypred_LR = model.predict(X_test)

        r2_.append(r2(y_test,ypred_LR))
        mae_.append(mae(y_test, ypred_LR))
        maxer_.append(maxer(y_test,ypred_LR))
            
    paratesting['r^2 value'] = r2_
    paratesting['mean absolute error'] = mae_
    paratesting['maximum error'] = maxer_
    
    print('random state =%s' %str(ranstate))
    return paratesting
          
          
          
### getting the average of the errors with 5 different random states
'''Finding the average errors of the ML run, using _trainsize,
this sets to check for test size between 10%-90%,
and only does this for random state between 1-5'''
def rancheck(descriptor, energies, model):
    descriptor = descriptor
    energies = energies
    model = model
    rn1 = train_size(descriptor, energies, model, 1)
    
    rn2 = train_size(descriptor, energies, model, 2)
    
    rn3 = train_size(descriptor, energies, model, 3)
    
    rn4 = train_size(descriptor, energies, model, 4)
    
    rn5 = train_size(descriptor, energies, model, 5)
    
    test_size = np.arange(0.1,1,0.1)
    r_2val = []
    for i in range(9):
        average = (rn1['r^2 value'][i]+rn2['r^2 value'][i]
        +rn3['r^2 value'][i]+rn4['r^2 value'][i]+rn5['r^2 value'][i])/5
        r_2val.append(average)
        
    mae_val = []
    for i in range(9):
        average = (rn1['mean absolute error'][i]+rn2['mean absolute error'][i]
        +rn3['mean absolute error'][i]+rn4['mean absolute error'][i]
        +rn5['mean absolute error'][i])/5
        mae_val.append(average)
        
    maxi_val = []
    for i in range(9):
        average = (rn1['maximum error'][i]+rn2['maximum error'][i]
        +rn3['maximum error'][i]+rn4['maximum error'][i]
        +rn5['maximum error'][i])/5
        maxi_val.append(average)
        
    ave_data = pd.DataFrame()
    ave_data['test size']=test_size
    ave_data['r^2 value']=r_2val
    ave_data['mean absolute error']=mae_val
    ave_data['maximum error']=maxi_val
    
    return ave_data


### plotting the errors with increasing training set
def sizeplot(parafile, parameter):
    r2_ = parafile['r^2 value'].tolist()
    mae_ = parafile['mean absolute error'].tolist()
    maxer_ = parafile['maximum error'].tolist()
    nn = parameter.tolist()
    
    
    fig, ax1 = plt.subplots(dpi=200)
    ax2 = ax1.twinx()
    
    ax1.plot(nn, r2_, marker='x', label='r^2 value')
    ax2.plot(nn, mae_, marker='o',color='green',label='mean absolute error')
    #ax2.plot(nn, maxer_, marker='o',color='red', label='maximum error')
    
    ax1.set_xlabel('test size')
    ax1.set_ylabel('r^2 value')
    ax2.set_ylabel('error/ meV')
    
    
    plt.show()
    plt.close()


    
    
## parallisation

#making parallisation parameter testing function
import multiprocessing as mp
core = mp.cpu_count()

def train_mp (descriptor, energies, model, p_s, p_e, p_i, core):
    test_para = np.arange(p_s, p_e, p_i)
    name = {}
    splitting = np.array_split(test_para, core)
    #worker = mp.Pool(core)
    paratesting = pd.DataFrame()
    for i in range(len(splitting)):
        testsize=splitting[i]
        for j in testsize:
            name['row_'+str(j)] = ml_run(descriptor,energies,model,j)
            paratesting = pd.concat([paratesting, name['row_%s'%str(j)][0]], ignore_index=True)
    return paratesting
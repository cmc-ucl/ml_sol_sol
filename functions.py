def sym_clus(descriptor,energies,i_pred):
    import os
    import copy
    import json
    import itertools
    import shutil as sh
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from datetime import datetime

    from CRYSTALpytools.crystal_io import Crystal_output, Crystal_input, Crystal_density, Crystal_gui
    from CRYSTALpytools.convert import cry_gui2pmg, cry_out2pmg
    from CRYSTALpytools.utils import view_pmg

    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.cif import CifWriter
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, PointGroupAnalyzer

    from ase.visualize import view

    #from dscribe.descriptors import CoulombMatrix

    from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
    from sklearn.metrics import mean_squared_error, r2_score, max_error, mean_absolute_error
    from sklearn.cluster import KMeans

    import matplotlib.pyplot as plt
    
    plt.rcParams["figure.figsize"] = (15,15)
    
        cluster_0 = pd.DataFrame()
        cluster_1 = pd.DataFrame()
        cluster_2 = pd.DataFrame()
        cluster_3 = pd.DataFrame()
        cluster_4 = pd.DataFrame()
        cluster_5 = pd.DataFrame()

        ind = []
        des = []
        en = []

        ind_1 = []
        des_1 = []
        en_1 = []

        ind_2 = []
        des_2 = []
        en_2 = []

        ind_3 = []
        des_3 = []
        en_3 = []

        ind_4 = []
        des_4 = []
        en_4 = []

        ind_5 = []
        des_5 = []
        en_5 = []
        for i in range(len(descriptor)): 
            if pred_i[i] == 0:
                en.append(energies[i])
                ind.append(str(i))
                des.append(descriptor[i])
            elif pred_i[i] == 1:
                en_1.append(energies[i])
                ind_1.append(str(i))
                des_1.append(descriptor[i])
            elif pred_i[i] == 2:
                en_2.append(energies[i])
                ind_2.append(str(i))
                des_2.append(descriptor[i])
            elif pred_i[i] == 3:
                en_3.append(energies[i])
                ind_3.append(str(i))
                des_3.append(descriptor[i])
            elif pred_i[i] == 4:
                en_4.append(energies[i])
                ind_4.append(str(i))
                des_4.append(descriptor[i])
            elif pred_i[i] == 5:
                en_5.append(energies[i])
                ind_5.append(str(i))
                des_5.append(descriptor[i])


        cluster_0['index'] = ind
        cluster_0['descriptor'] = des
        cluster_0['energies'] = en

        cluster_1['index'] = ind_1
        cluster_1['descriptor'] = des_1
        cluster_1['energies'] = en_1

        cluster_2['index'] = ind_2
        cluster_2['descriptor'] = des_2
        cluster_2['energies'] = en_2

        cluster_3['index'] = ind_3
        cluster_3['descriptor'] = des_3
        cluster_3['energies'] = en_3
        cluster_4['index'] = ind_4
        cluster_4['descriptor'] = des_4
        cluster_4['energies'] = en_4

        cluster_5['index'] = ind_5
        cluster_5['descriptor'] = des_5
        cluster_5['energies'] = en_5

        return cluster_0
        return cluster_1
        return cluster_2
        return cluster_3
        return cluster_4
        return cluster_5
    

    
    
#GBDT testing for best parameters    
def GBDT_ParaT(testset, energies, descriptor):

    #the parameters we're looking at
    estimators = np.arange(1000,30000,1000)
    maxdepth = np.arange(1,30,1)
    mssplit = np.arange(2, 15, 1)
    lrate = np.arange(0.1,3.0,0.1)
    loss_ = ['huber', 'squared_error', 'absolute']
    
    para_test = pd.DataFrame()
    n_es = []
    m_d = []
    m_s_s = []
    l_r = []
    lss = []
    r_2 = []
    m_a_e = []
    
    for e in estimators:
        for d in maxdepth:
            for s in mssplit: 
                for r in lrate:
                    for i,l in enumerate(loss_):
                        params = {
                            'n_estimators': e, 
                            'max_depth': d,  
                            'min_samples_split': s, 
                            'learning_rate': r,
                            'loss': l} 
                        model = GradientBoostingRegressor(**params)
                        X_train, X_test, y_train, y_test = train_test_split(descriptor, energies, random_state=1, test_size = testset)
                        scaler == MinMaxScaler()
                        scaler.fit(X_train)  
                        X_train = scaler.transform(X_train)  
                        X_test = scaler.transform(X_test)

                        model.fit(X_train, y_train)
                        ypred_LR = model.predict(X_test)
                        
                        r2_=(r2(y_test,ypred_LR))
                        mae_=(mae(y_test, ypred_LR))
                        
                        n_es.append(e)
                        m_d.append(d)
                        m_s_s.append(s)
                        l_r.append(r)
                        lss.append(l)
                        r_2.append(r2)
                        m_a_e.append(mae)
                                            
    para_test['n_estimators'] = n_es
    para_test['max_depth'] = m_d
    para_test['min_samples_split'] = m_s_s
    para_test['learning_rate'] = l_r
    para_test['loss'] = lss
    para_test['r2'] = r_2
    para_test['mae'] = m_a_e
    
    np.save('./data/machinelearning/GBDT_parameters.npy',para_test,allow_pickle=True)


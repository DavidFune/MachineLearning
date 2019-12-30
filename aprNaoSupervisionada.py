# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 19:25:05 2019

@author: hunter28
"""

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np



#Kmeans
#logica fuzzy
#analise de cluster, metodos para analisar agrupamento


# X = base de dados
dataBase = pd.read_table('bcw.data',delimiter=',', header=None)
targets = dataBase.iloc[:,10:11]

targets = dataBase.iloc[:,10:11]

trueFalse = targets == 4

targets.iloc[trueFalse] = 1

trueFalse = targets == 2

targets.iloc[trueFalse] = -1

dataBase = dataBase.iloc[:,1:9]

Kmeans = KMeans(n_clusters = 2, random_state = 0).fit(dataBase)




#Kmeans.predict([[0,0],[12,3]])

grup = Kmeans.cluster_centers_

labels = Kmeans.labels_

labels[labels[0:]==0] = -1






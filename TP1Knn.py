# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 22:49:12 2019

@author: hunter28
"""
import pandas as pd
import numpy as np
import matplotlib .pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#palavras chaves: disperção
# cada atributo é uma dimensão


#from prettytable import PrettyTable

#arquivo = open('bcw.data','r')
#for linha in arquivo:
#    print(linha)
#arquivo.close() 




dataBase = pd.read_table('bcw.data',delimiter=',', header=None)



dataBase = dataBase.iloc[:,1:11]


r = np.random.permutation(len(dataBase))

##aleatoriedade 
base2 = dataBase.iloc[r,:]


treinoK = int (np.floor(len(dataBase)*0.7))
#testeK = int  (np.floor(len(dataBase)*0.3))


dataTreino = dataBase[:treinoK]
dataTeste = dataBase[(treinoK+1):]

dataTreino.fillna(0)
dataTeste.fillna(0)

#cada linha é como se fosse um ponto sendo |x1 -x2| distancia euclidiana

### base de treinamento#####################
aux = dataTreino.iloc[:,9:10]

trueFalse = aux == 4

aux.iloc[trueFalse] = 1

trueFalse = aux == 2

aux.iloc[trueFalse] = -1

### base de teste#####################

aux1 = dataTeste.iloc[:,9:10]

trueFalse = aux1 == 4

aux1.iloc[trueFalse] = 1

trueFalse = aux1 == 2

aux1.iloc[trueFalse] = -1


############## Treinamento e previsão################

classifier = KNeighborsClassifier(n_neighbors=11)
classifier.fit(dataTreino.iloc[:,1:9], aux)

#---------Predição----------------

pred = classifier.predict(dataTeste.iloc[:,1:9])



comparacoN = sum(aux1.iloc[:,0]==-1) - sum(pred==-1)
comparacoP = sum(aux1.iloc[:,0]==1) - sum(pred==1)


#############Matriz de confusão #################################


print('Matriz de Confusão', '\n', confusion_matrix(aux1, pred), "\n\n\n")

print('Acuracia da classificação: ', accuracy_score(aux1, pred),'\n\n\n')

print(classification_report(aux1, pred))























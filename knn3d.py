# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 19:09:24 2019

@author: hunter28
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 19:58:16 2019

@author: hunter28
"""

import numpy as np
import matplotlib .pyplot as plt

np.random.seed(0)

print("\n teste");

#matriz
K = 3
a = np.zeros((10,2))
mean1 = [3,3]
cov1 = [[1,0], [0,1]]
#covariancia
N1 = 20

classe_1 = np.random.multivariate_normal(mean1, cov1, N1)

fig, ax = plt.subplots()

ax.scatter(classe_1[:,0], classe_1[:,1], c='gray', alpha = 0.7, edgecolors = 'none')


mean2 = [6,6]
cov2 = [[1,0], [0,1]]
N2 = 15

classe_2 = np.random.multivariate_normal(mean2, cov2, N2)

#fig, ax = plt.subplots()

ponto_teste = [1,1]

#imprime os pontos no grafico
ax.scatter(classe_2[:,0], classe_2[:,1], c='blue', alpha = 0.9, edgecolors = 'none')

#ax.scatter(4,4,c='red',alpha=0.6)

# classe_1[0,0:1] coluna 0 da linha 0:1 0 até 1
#distancia = np.linalg.norm(classe_1[0,0:1] - ponto_teste)

#concatenando classe_1 com classe_2
X = np.concatenate((classe_1,classe_2))
# len(X) = linhas 1 = colunas

labels = np.concatenate((np.repeat(1,len(classe_1)), np.repeat(-1, len(classe_2))))


#---------------------------------knn3d----------------------------------------------

#criando uma seq 1-5.5 com intervalos de 0.5
seq = np.arange(1,5.5,0.5)
# matrix de classificação
Z = np.zeros((len(seq),len(seq)))

for i_grid in range(0,len(seq)):
    for j_grid in range(0,len(seq)):
        ponto_teste = [seq[i_grid],seq[j_grid]]
        distancia = np.zeros((len(X),1))
        

        #calculo da distancia para todos pontos
        for i in range(0,N1+N2):
            distancia[i] = np.linalg.norm(X[i] - ponto_teste)
        
        
        
        # distanciasComLabels = np.array(distanciasComLabels)
        # distanciasComLabels.sort(axis=0)
        
        
        
        
        #if labelPontoTeste == 1:
        #    ax.scatter(ponto_teste[0],ponto_teste[1], c= 'gray', alpha = 0.9)
        #else:
        #    ax.scatter(ponto_teste[0],ponto_teste[1], c= 'blue', alpha = 0.6)
        
        
        distancia = np.concatenate((distancia,labels.reshape(N1+N2,1)),axis=1)
        
        ordena_distancia = distancia[distancia[:,0].argsort()]
        
        k_proximos = ordena_distancia[0:K,1]
        
        #cont1 =0
        #cont0 =0
        
        # classificando
        #for v in [x[1] for x in k_proximos]:
        #    if v == 1:
        #        cont1 += 1
        #    else:
        #        cont0 += 1
        
        if sum(k_proximos) >= 0:
            Z[i_grid, j_grid] = 1
        else:
            Z[i_grid, j_grid] = -1
        
        


            
            























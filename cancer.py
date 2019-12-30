import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import plotly.express as px
#import plotly.graph_objs as go
base = pd.read_table('breast-cancer-wisconsin.data', delimiter="," , header = None)
base.dropna(how = any)

num_train = int(0.7 * len(base))
treino = base[0:num_train]
num_train_1 = num_train + 1

teste = base[num_train_1:len(base)]
 

"""
np.random.seed(0)
k=9

a = np.zeros((10,2))
mean1 =[2,2]
cov1 = [[1,0],[0,1]]
N1 = 20
classe_1 = np.random.multivariate_normal(mean1,cov1,N1)
#fig, ax =plt.subplots()
#ax.scatter(classe_1[:,0],classe_1[:,1],c='blue',alpha=0.7,edgecolors='none')

mean2 =[5,5]
cov2 = [[1,0],[0,1]]
N2 = 20
classe_2 = np.random.multivariate_normal(mean2,cov2,N2)

#ax.scatter(classe_2[:,0],classe_2[:,1],c='yellow',alpha=0.9,edgecolors='none')

#ponto_teste = [8,6]

#ax.scatter(ponto_teste[0],ponto_teste[1],100, c='black',alpha=1.0)

#ax.scatter(classe_1[1,0],classe_1[1,1],c='green',alpha=0.4)

#distancia = np.linalg.norm(classe_1[0,0:1] - ponto_teste)


X = np.concatenate((classe_1, classe_2))

labels = np.concatenate((np.repeat(1,len(classe_1)),np.repeat(-1,len(classe_2))))
"distancia = np.zeros((10,1))"


#-----------------------------------------

#Criando um Grid

seq = np.arange(-1,6.5,1)
Z = np.zeros((len(seq),len(seq)))

for i_grid in range(0,len(seq)):
    for j_grid in range(0,len(seq)):  
        ponto_teste = [seq[i_grid],seq[j_grid]]

        #ax.scatter(ponto_teste[0],ponto_teste[1], c='black',alpha=1.0)



        distancia = np.zeros((len(X),1))


        for i in range(0,N1+N2):
            distancia[i] = np.linalg.norm(X[i] - ponto_teste)


        for i  in range(0,k):
            distancia = np.concatenate((distancia, labels.reshape(N1+N2,1)),axis=1)

        ordenado = distancia[distancia[:,0].argsort()]




#Classificando os pontos 
#classe_1 = azul = 1
#
#classe_2 = amarelo = -1

        classificadores = ordenado[0:,1]
        classe1 = 0
        classe2 = 0

        for i in range(0,k):
            if classificadores[i] == 1:
                classe1 = 1 + classe1
            else:
                classe2 = 1 + classe2
    
        if classe1 > classe2: 
            #print("Pertence a classe 1")
            Z[i_grid,j_grid] = 1
 #           ax.scatter(ponto_teste[0],ponto_teste[1], c='blue',alpha=1.0)

        else:
            #print("Pertence a classe 2")
            Z[i_grid,j_grid] = -1
 #           ax.scatter(ponto_teste[0],ponto_teste[1], c='yellow',alpha=1.0)
            

#iris = px.data.iris()
#fig = px.scatter_3d(iris, x='sepal_length', y='sepal_width', z='petal_width',
#              color='species')

#fig.show()

fig,ax1 = plt.subplots(nrows=1)

ax1.contour(seq,seq, Z ,levels = 0 ,linewidgths = 0.1, colors = 'k')
cntr1 = ax1.contourf(seq,seq,Z,levels=1,alpha=0.3,colors=['red','blue'])

ax1.scatter(classe_1[:,0], classe_1[:,1] ,c='blue',alpha=0.9)


ax1.scatter(classe_2[:,0], classe_2[:,1] ,c='red',alpha=0.9)
"""



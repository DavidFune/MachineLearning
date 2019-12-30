# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 21:17:45 2019

@author: hunter28
"""
import pandas as table
import numpy as np
import matplotlib .pyplot as plt
from sklearn import datasets
#o que svm
#palavras chave: dispesão, kernel
from sklearn import svm


iris = datasets.load_iris()
digits = datasets.load_digits()

# modelo de treino
clf = svm.SVC(gamma=0.001,C=100)

# -1 depois de : todos menoa a ultima, linha
# data = dados target = classes


#clf.fit(digits.data[:-1], digits.target[:-1])


#treinoSMV
a = int (np.floor(len(digits.data)*0.7))
#testeSVM
b = int (np.floor(len(digits.data)*0.3))

#treinando com 70% da amostras de digist
clf.fit(digits.data[:a], digits.target[:a])

 
#recebenbo predição de teste
res = clf.predict(digits.data[(a+1):])

#esperado do teste
resEsperado = digits.target[(a+1):]

#fazendo a acuracia, quantidade erro
resAcuracia = res - resEsperado

truee = sum(resAcuracia == 0)

 
total = (len(res) - truee ) / len(res)


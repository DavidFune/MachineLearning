# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:24:51 2019

@author: hunter28
"""

from sklearn import datasets
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

iris = datasets.load_iris()
digits = datasets.load_digits()

#Caracteristicas do treinamento, gamma= abertura da gluciana, C = regularização
clf = svm.SVC(gamma=0.001 , C=100.)

#classificando a ultima linha
clf.fit(digits.data,digits.target)


trainS = int(0.7*(len(digits.data)))

train = clf.fit(digits.data[:trainS], digits.target[:trainS])

pred = clf.predict(digits.data[(trainS+1):])

aux1 = digits.target[(trainS+1):]


print('Acuracia da classificação: ', accuracy_score(aux1, pred),'\n\n\n')


print('Matriz de Confusão', '\n', confusion_matrix(aux1, pred), "\n\n\n")



print(classification_report(aux1, pred))










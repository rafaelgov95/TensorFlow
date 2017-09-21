# -*- coding:UTF-8 -*-
# Autor: Rafael Viana
# Codigo: Arvore de Decision Distribuida ADD
import tensorflow as tf
import numpy as np
import graphviz
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Carrega o DataSet do Teste Iris.
iris = load_iris()
# Cria Utiliza as Cluster para processar o dado.
cluster = tf.train.ClusterSpec({"local": [ "svgov.ddns.net:3000","35.192.227.92:3001"]})
#Divido os testes em Treinamento e Teste
X_train ,X_teste , y_train , y_test = train_test_split(iris.data,iris.target,test_size=.2)
print "Data de Teste\n",X_teste
print "Data Traning\n", X_train

# Serviço sendo executado no Cluster 1
with tf.device("/job:local/task:1"):
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    a=clf.predict(X_teste)

# Serviço sendo executado no Cluster 0
with tf.device("/job:local/task:0"):
   print 'Resposta:\n', a
   print 'Correto:\n', y_test
   print 'Acertos em : ',accuracy_score(y_test, a)

# Chamada de tf.Session()
with tf.Session("grpc://svgov.ddns.net:3000") as sess:
    with sess.as_default():
        pass


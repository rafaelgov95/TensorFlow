import numpy as np
import graphviz
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
# Alguns atributos da iris

# Deep Learning
test_idx = [0,50,100]

train_target = np.delete(iris.target,test_idx)
train_data = np.delete(iris.data,test_idx,axis=0)

print "Data", train_data
print "Teste", test_idx

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

print "Resposta:",clf.predict(test_data)

# print "Resposta:",clf.predict([[5.5,2.4,3.7,1.0],[5.8,	2.7	,5.1,	1.9]])
#outra forma de criar o iris

# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("iris")

dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)

graph.render('iris')
import graphviz
from sklearn import tree

v1 = [[140,1],[130,0]]

v2 = [0,1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(v1,v2)

predictions= clf.predict([[150,0]])
print predictions
from sklearn.metrics import  accuracy_score
print accuracy_score([0],predictions)

dot_data = tree.export_graphviz(clf, out_file=None,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)

graph.render('frutas')



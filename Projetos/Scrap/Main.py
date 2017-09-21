from sklearn import tree
import graphviz

a = tree.DecisionTreeClassifier()

dot_data = tree.export_graphviz(a, out_file=None,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)

graph.render('X')
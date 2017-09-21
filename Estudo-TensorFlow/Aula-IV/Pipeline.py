from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train ,X_teste , y_train , y_test = train_test_split(x,y,test_size=.5)

# print "Dados de Treinamento", X_train
# print "Rotulos de Treinamento", y_train

# Tree of Decision
# from sklearn import  tree
# my_classifier = tree.DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train,y_train)
predictions =my_classifier.predict(X_teste)
print (predictions)

from sklearn.metrics import  accuracy_score
print (accuracy_score(y_test,predictions))
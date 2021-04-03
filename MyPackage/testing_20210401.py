
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

X, y = make_hastie_10_2(random_state=0)

X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

str_param = 'n_estimators=100' # , learning_rate=1.0, max_depth=1, random_state=0

clf = GradientBoostingClassifier(str_param).fit(X_train, y_train)

# print(clf.score(X_test, y_test))

"""


from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()

parameters = {'kernel':['linear', 'rbf'], 'C':[1, 10]}

svc = svm.SVC()

clf = GridSearchCV(svc, parameters)

clf.fit(iris.data, iris.target)

"""
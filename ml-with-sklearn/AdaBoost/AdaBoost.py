from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("X_train:",len(X_train),"; X_test:",len(X_test),"; y_train:",len(y_train),"; y_test:",len(y_test))

# Create adaboost object
Adbc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1.5)
# Train Adaboost 
model = Adbc.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#('Accuracy:', 0.8888888888888888)
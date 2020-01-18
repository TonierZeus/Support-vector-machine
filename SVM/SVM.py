import pandas as pd #import panads

#wgranie bazy danych iris
dataset = pd.read_csv("iris.csv")
print(dataset)

#usnięcie kolumny typu tekst 'Species'
X = dataset.drop('Species', axis=1)
y = dataset['Species']

#funkcja rozdzielająca na grupy testujące 20% i trenujące 80%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#wielomianowe wykorzytsanie Kenela
from sklearn.svm import SVC
svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)

#przewidywanie wyniku
y_pred = svclassifier.predict(X_test)

#ocenianie wyniku
from sklearn.metrics import classification_report, confusion_matrix

#wyświetlanie wyniku
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#wykorzystanie Kernela metoda Gaussa
from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)

#ocenianie wyniku
from sklearn.metrics import classification_report, confusion_matrix

#wyświetlanie wyniku
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#sigmoidalne klasyfikowanie
from sklearn.svm import SVC
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)

#ocenianie wyniku
from sklearn.metrics import classification_report, confusion_matrix

#wyświetlanie wyniku
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
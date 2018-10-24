import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

"""data - http://www.karlin.mff.cuni.cz/~pesta/prednasky/NMFM404/Data/binary.csv
gre - Graduate Recors Exam, gpq - grade point average, 
rank - values 1 to 4 -> 1- university with a good reputation"""
dataset = pd.read_csv("University.csv" , header = 0)
isnull = dataset.isnull().sum()
# print(isnull)

"""sb.countplot(x = "admit", data = dataset)
plt.show()"""

X = dataset.iloc[:,1:] #data
y = dataset.iloc[:,0] #target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# without scaling
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
LogReg_predict = LogReg.predict(X_test)

print("Result I: \n")
print("Confusion Matrix:\n ", confusion_matrix(y_test, LogReg_predict)) 
print("Accuracy: ", accuracy_score(y_test, LogReg_predict))
print("Precision: ", precision_score(y_test, LogReg_predict))
print("Recall: ", recall_score(y_test, LogReg_predict))

# with scaling
sc = StandardScaler()
X_train_stand = sc.fit_transform(X_train) 
X_test_stand = sc.transform(X_test)

LogReg.fit(X_train_stand, y_train)
LogReg_predictII = LogReg.predict(X_test_stand)

print("Result II: \n")
print("Confusion Matrix:\n ", confusion_matrix(y_test, LogReg_predictII)) 
print("Accuracy: ", accuracy_score(y_test, LogReg_predictII))
print("Precision: ", precision_score(y_test, LogReg_predictII))
print("Recall: ", recall_score(y_test, LogReg_predictII)) 
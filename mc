# Classification evaluation
#Select Multiclass Classifier dataset and evaluate the performance of classification model using
various evaluation matrix/matrices
#Such as accuracy,precision,recall and F1Score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
print("Daniel MscIT 011")
#load the iris dataset
iris = load_iris()
x = iris.data
y = iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=32)
from sklearn.svm import SVC
model = SVC()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print('Accuracy Score:- ',acc)
precision = precision_score(y_test,y_pred,average = 'macro')
print('Precision Score:- ',acc)
re = recall_score(y_test,y_pred,average = 'macro')
print('Recall Score:- ',re)
acc = f1_score(y_test,y_pred,average = 'macro')
print('F1 Score:- ',acc)

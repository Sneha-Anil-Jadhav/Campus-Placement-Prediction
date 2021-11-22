import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast




from joblib import dump,load
#matplotlib notebook
#matplotlib inline

import warnings
warnings.filterwarnings('ignore')



from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
#matplotlib inline

df = pd.read_csv(r"/Users/prasad/Desktop/campus placement/train.csv")
df.info()
# check null values in df
df.isnull().sum()
df = df.drop('sl_no',axis=1)
df['salary'].fillna(0,inplace=True)
df['salary'].isnull().sum()

from sklearn.preprocessing import LabelEncoder
cols = ['workex','specialisation','status','ssc_b','hsc_b']
df[cols]=df[cols].apply(LabelEncoder().fit_transform)
df.head()

dummy_hsc_s=pd.get_dummies(df['hsc_s'],prefix='hsc')
dummy_degree_t=pd.get_dummies(df['degree_t'],prefix='degree')

df1=pd.concat([df,dummy_hsc_s,dummy_degree_t],axis=1)
df1.drop(['hsc_s','degree_t','hsc_b','ssc_b','hsc_s','degree_t'],axis=1,inplace=True)
df1.head(10)

df1.to_csv(r"/Users/prasad/Desktop/campus placement/campus selection final data/preprocess_data.csv")

df1
x = df1.drop(['status'],axis = 1)
y = df1.status

X_train,x_test,Y_train,y_test = train_test_split(x,y,train_size = 0.8,random_state = 1)



#importing metrics
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score
#Logistic Regression

warnings.filter = warnings.simplefilter('ignore')
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y_pred = logreg.predict(x_test)

#Hyperparameter Tuning for Logistic Regression

param_grid=[{'penalty':['l1','l2','elasticnet','none'],
             'C':np.logspace(-4,4,20),
             'solver':['lbfgs','newton-cg','liblinear','sag','saga'],
             'max_iter':[100,200,1000,2500,5000]}]
from sklearn.model_selection import GridSearchCV
clf=GridSearchCV(logreg,param_grid,cv=3,verbose=True,n_jobs=-1)

best_clf=clf.fit(X_train,Y_train)

best_clf.best_estimator_

best_ypred=clf.predict(x_test)
print('Accuracy of logistic regression classifier(GridSearchCV) on test set: {:.3f}'.format(best_clf.score(x_test, y_test)))
print('Accuracy of logistic regression classifier(GridSearchCV) on train set: {:.3f}'.format(best_clf.score(x_train, y_train)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",confusion_matrix)
from sklearn.metrics import classification_report
print("Classification Report:\n",classification_report(y_test,y_pred))

param_grid_dt={'criterion':['gini','entropy'],'max_depth':[2,3,4,5],'max_features':('auto','sqrt','log2'),'min_samples_split':(2,4,6)}
clf_dt=GridSearchCV(dt,param_grid_dt,n_jobs=-1,verbose=True,cv=5)
clf_dt.fit(x_train,y_train)

clf_dt.best_estimator_

best_ypred_dt=clf_dt.predict(x_test)







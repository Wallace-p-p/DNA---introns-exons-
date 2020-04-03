
# coding: utf-8

# In[ ]:

import splicenum as sp
cdata, ctarg= sp.splicenum()


# In[1]:

import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


# In[ ]:

#Machine learning
clf=[]
nome=[]
cdata= x
ctarg= b
clf.append(tree.DecisionTreeClassifier())
nome.append('Decision Tree')
clf.append(SVC(gamma='auto'))
nome.append('SVC Support Vector Machines')
clf.append(KNeighborsClassifier(n_neighbors=1))
nome.append('Nearest Neighbors1')
clf.append(GaussianNB())
nome.append('GaussianNB')
clf.append(MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1))
nome.append('MLP classifier')
for i in range(len(clf)):
    scores = cross_val_score(clf[i], cdata, ctarg, cv=10)
    y_pred = cross_val_predict(clf[i], cdata, ctarg, cv=10)
    conf_mat = confusion_matrix(ctarg, y_pred)
    print(nome[i],'\nScores:',scores)
    print("Accuracy: %0.2f (+/- %0.2f) , ei: %0.2f , ie: %0.2f, n: %0.2f" % (scores.mean(), scores.std() * 2, conf_mat[0][0]/sum(conf_mat[0]), conf_mat[1][1]/sum(conf_mat[1]),conf_mat[2][2]/sum(conf_mat[2])))
    print('Confusion:\n', conf_mat, '\n')
#ENSEMBLE METHODS
#Bagging
clf=[]
nome=[]
clf.append(tree.DecisionTreeClassifier())
nome.append('Decision Tree')
clf.append(SVC(gamma='auto'))
nome.append('SVC Support Vector Machines')
clf.append(KNeighborsClassifier(n_neighbors=1))
nome.append('Nearest Neighbors')
clf.append(GaussianNB())
nome.append('GaussianNB')
clf.append(MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1))
nome.append('MLP classifier')
clf.append(SGDClassifier(loss="hinge", penalty="l2", max_iter=5, tol= None))
nome.append('SGDclassifier - Stochastic Gradient Descent')
print('BAGGING')
for i in range(len(clf)):
    bclf = BaggingClassifier(clf[i], max_samples=0.5, max_features=0.5)
    scores = cross_val_score(clf, cdata, ctarg, cv=10)
    y_pred = cross_val_predict(clf, cdata, ctarg, cv=10)
    conf_mat = confusion_matrix(ctarg, y_pred)
    print(nome[i],'\nScores:',scores)
    print("Accuracy: %0.2f (+/- %0.2f) , ei: %0.2f , ie: %0.2f, n: %0.2f" % (scores.mean(), scores.std() * 2, conf_mat[0][0]/sum(conf_mat[0]), conf_mat[1][1]/sum(conf_mat[1]),conf_mat[2][2]/sum(conf_mat[2])))
    print('Confusion:\n', conf_mat, '\n')

    
#random forest
print('random Forest')
clf = RandomForestClassifier(n_estimators=10)
scores = cross_val_score(clf, cdata, ctarg, cv=10)
y_pred = cross_val_predict(clf, cdata, ctarg, cv=10)
conf_mat = confusion_matrix(ctarg, y_pred)
print('\nScores:',scores)
print("Accuracy: %0.2f (+/- %0.2f) , ei: %0.2f , ie: %0.2f, n: %0.2f" % (scores.mean(), scores.std() * 2, conf_mat[0][0]/sum(conf_mat[0]), conf_mat[1][1]/sum(conf_mat[1]),conf_mat[2][2]/sum(conf_mat[2])))
print('Confusion:\n', conf_mat, '\n')

#Adaboost
print('AdaBoost')
clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, cdata, ctarg, cv=10)
y_pred = cross_val_predict(clf, cdata, ctarg, cv=10)
conf_mat = confusion_matrix(ctarg, y_pred)
print('\nScores:',scores)
print("Accuracy: %0.2f (+/- %0.2f) , ei: %0.2f , ie: %0.2f, n: %0.2f" % (scores.mean(), scores.std() * 2, conf_mat[0][0]/sum(conf_mat[0]), conf_mat[1][1]/sum(conf_mat[1]),conf_mat[2][2]/sum(conf_mat[2])))
print('Confusion:\n', conf_mat, '\n')

#Gradient tree boosting
print('Gradient Tree Boosting')
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
scores = cross_val_score(clf, cdata, ctarg, cv=10)
y_pred = cross_val_predict(clf, cdata, ctarg, cv=10)
conf_mat = confusion_matrix(ctarg, y_pred)
print('\nScores:',scores)
print("Accuracy: %0.2f (+/- %0.2f) , ei: %0.2f , ie: %0.2f, n: %0.2f" % (scores.mean(), scores.std() * 2, conf_mat[0][0]/sum(conf_mat[0]), conf_mat[1][1]/sum(conf_mat[1]),conf_mat[2][2]/sum(conf_mat[2])))
print('Confusion:\n', conf_mat, '\n')

#Voting Classifier
print('Voting Classifier')
clf1 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf2 = AdaBoostClassifier(n_estimators=100)
clf3 = tree.DecisionTreeClassifier()
clf4 = GaussianNB()
clf5 = SVC(gamma='auto')

clf = VotingClassifier(estimators=[('gbc', clf1), ('ada', clf2), ('DTc', clf3), ('gNB', clf4), ('svc', clf5)], voting='hard')
scores = cross_val_score(clf, cdata, ctarg, cv=10)
y_pred = cross_val_predict(clf, cdata, ctarg, cv=10)
conf_mat = confusion_matrix(ctarg, y_pred)
print('\nScores:',scores)
print("Accuracy: %0.2f (+/- %0.2f) , ei: %0.2f , ie: %0.2f, n: %0.2f" % (scores.mean(), scores.std() * 2, conf_mat[0][0]/sum(conf_mat[0]), conf_mat[1][1]/sum(conf_mat[1]),conf_mat[2][2]/sum(conf_mat[2])))
print('Confusion:\n', conf_mat, '\n')


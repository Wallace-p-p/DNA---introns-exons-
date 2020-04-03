
# coding: utf-8

# In[ ]:

#CFS base, indices selecionados atravez do algoritmo Correlation-based Feature Selection pela skfeature
import splicenum as sp
a, b= sp.splicenum()
indices= [29, 31, 28, 30, 34, 27, 32, 33, 24, 22]
CFSdata=[]
for j in range(len(a)):
    vazio1=[]
    for i in range(len(indices)):
        v= a[j][indices[i]]
        vazio1.append(v)
    CFSdata.append(vazio1)
#data armazenada em CFSdata


# In[ ]:

#base UCI numerica, filtrada com decision tree classifier da sklearn
import splicenum as sp
a, b= sp.splicenum()
import numpy as np
from sklearn import tree
clf=tree.DecisionTreeClassifier()
clf= clf.fit(a,b)
model = SelectFromModel(clf, prefit=True)
a = model.transform(a)
#data armazenada em a, ja filtrada


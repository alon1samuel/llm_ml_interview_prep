"""
Evaluate the performance of a classification model using a confusion matrix.
Implement k-fold cross-validation for a random forest classifier.
Use a pretrained word embedding model (e.g., Word2Vec) for a text similarity task.
Discuss how to handle imbalanced datasets in classification tasks.
Optimize hyperparameters of an SVM using grid search.
"""

# Evaluate the performance of a classification model using a confusion matrix.

import polars as pl
from sklearn.model_selection import train_test_split



from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred)
print(cm)



import matplotlib.pyplot as plt
import seaborn as sn

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)

plt.savefig('heatmap.png')
print()
import os
os.remove('heatmap.png')



"""
[[152  14]
 [  8 156]]

Evaluation for the above confusion matrix. We can see that there are ~150 true positives and ~150 true 
negatives compared to 14 false positives and 8 false negatives. It means the model learnt quiet well
and there are not a lot of mistakes that are crossing. We can see significantly more false positives
that the model predicts which can tell us to maybe choose a different threshold if we'd use a logistic
regression classifier. That would happen if we'd care more about false positives. for example if
we'd want a model that would alert us to a problem and we don't want a lot of alerts that are misleadings.
On the other hand a medical model would be tolerant with false positives because maybe the idea
of missing a detection is crucial, but bringin people to more screenings is fine.

 
"""


# Implement k-fold cross-validation for a random forest classifier.


X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)


clf = RandomForestClassifier(max_depth=2, random_state=0)


from sklearn.model_selection import KFold
kf = KFold(n_splits=2)

# for ind, (train, test) in enumerate(kf.split(X)):
#     print(f"fold {ind}")
#     X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

#     clf.fit(X_train, y_train)

#     y_pred = clf.predict(X_test)
#     print(accuracy_score(y_test, y_pred))
    
    
print()


# Use a pretrained word embedding model (e.g., Word2Vec) for a text similarity task.

import gensim
import gensim.downloader


# glove_model = gensim.downloader.load('glove-twitter-25')
# glove_model.save("data/glove.model")
from gensim.models import KeyedVectors

model = KeyedVectors.load("data/glove.model")

print(model.similarity('alice', 'bob'))

print(model.similarity('alice', 'wonderland'))


print()



# Discuss how to handle imbalanced datasets in classification tasks.

"""
In classification tasks there are a few ways to deal with imbalance. 
1. Subsample the majority classes to balance the dataset for the underrepresented class. 
2. Up sample the minority classes for the majority class. 
3. combination of over and under samplings. 
4. ensemble of methods and classifiers for each. 

"""

# Optimize hyperparameters of an SVM using grid search.

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()

parameters = {'kernel':('linear', 'rbf'), 'C':[1,2,3,4, 10], 'degree': [1,2,3,4,5,6]}
svc = svm.SVC()

clf = GridSearchCV(svc, parameters)

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.7, random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

print(sorted(clf.cv_results_.keys()))




#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn.feature_extraction.text
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

file = open('SemEval2018-T3-train-taskA.txt','r',encoding ="utf8")

tweets = []
classes = []
tfidf = []
buffer = []
num = 0;

line = file.readline()

while True:
#while num < 5:
    # read line
    line = file.readline()
    #print(line)
    if line:
        tweetsJustRead = line.split("\t",2)
        tweetpos = [tweetsJustRead[2]]
        category = [tweetsJustRead[1]]
        
        tweet = tweetpos[0]
        #print(tweet+" "+category[0])
        #strnum = str(num)
        #path = 'text2/tweet'+str(strnum)+'.txt';
        
        #text_file = open(path, "w")
        #n = text_file.write(tweet)
        #text_file.close()
        
        tweets.append(tweet)
        classes.append(category[0])
        
    if not line:
        break
    
    num=num+1

file.close()

#print('Total: '+str(num))

#data = {'TF-IDF':tweets,
#        'Tweets':words_vector}

classesInt = []
for i in range(1,len(classes)+1):
    classesInt.append(0)


classesUnique = np.unique(classes)
for i in range(1,len(classes)):
    for j in range(1,len(classesUnique)):
        if classes[i] == classesUnique[j]:
            classesInt[i] = j;

vectorizer = TfidfVectorizer(norm=None, stop_words="english",max_df=0.95, min_df=2)
tfidf = vectorizer.fit_transform(tweets)

#print(vectorizer.get_feature_names())
#print(tfidf.shape)

# Le informacoes sobre os dados
#numberOfInstances = len(tfidf)
#numberOfAttributes = len(tfidf.columns)

#print(str(numberOfInstances)+" "+str(numberOfAttributes))

X = tfidf;
y = classesInt;

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)

# Set the parameters by cross-validation
'''tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 2, 0.5, 0.25, 0.1, 1e-2, 1e-3, 1e-4],
                     'C': [0.5,1, 1.5, 2, 2.5, 5, 7.5, 10, 12.5, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.5, 1, 1.5, 2, 2.5, 3, 5, 10, 100, 1000]}]'''

scores = ['f1']
'''
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on C-SVM development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print("Detailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
'''

parameters = {'solver': ['sgd'], 'learning_rate' : ['constant'], 'learning_rate_init' : [0.001,0.01,0.05,0.1,0.25,0.5,0.75,1.0,1.5,2.5,5.0,10.0], 'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(10, 15), 'random_state':[0,1,2,3,4,5,6,7,8,9]}

clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)
clf.fit(X_train, y_train)
print("Best parameters set found on MLP development set:")
print(clf.best_params_)
print("Grid scores on development set:")
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))

rfc = RandomForestClassifier() 

param_grid = { 
    'n_estimators': [50,100,200,300,700],
    'max_features': ['auto', 'sqrt', 'log2']
}

print("Now, random forest grid search...")

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid)
CV_rfc.fit(X, y)
print("Best parameters set found on Randon Forest development set:")
print()
print(clf.best_params_)

print("Confusion matrix for Randon Forest: ")
confusion_matrix(y_true, y_pred)



import nltk
from sklearn import svm

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.datasets import make_classification
import sklearn.feature_extraction.text
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

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
from sklearn.naive_bayes import MultinomialNB
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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def writeDataFile(path,vectorizer,tfidf,categories):
    
    data_file = open(path, "w")
    
    [nroInstances,nroAtributtes] = tfidf.shape
    
    #print(tfidf.shape)
    
    #nroInstances = 5500
    
    line = "DY\n"+str(nroInstances)+"\n"+str(nroAtributtes-1)+"\n"
    
    n = data_file.write(line)
    
    atts = vectorizer.get_feature_names()

    #[y,nnames] = len(atts)
    line=atts[0]
    for i in range(1,len(atts)-1):
        line = line+";"+str(atts[i-1])
        
    line = line+"\n"
        #print(len(atts))
    #print(line)
    n = data_file.write(line)
    #
    for i in range(0,nroInstances):
        line = 'tweet'+str(i+1)+'.txt;';
        for j in range(0,nroAtributtes-1):
            line = line+str(round(tfidf[i,j], 5))+";"
        line = line+str(categories[i])+"\n"
        #print(line)
        n = data_file.write(line)
            
    data_file.close()
    return n

def num_there(s):
    return any(i.isdigit() for i in s)

file = open('SemEval2018-T3-train-taskA.txt','r',encoding ="utf8")

tweets = []
classes = []
tfidf = []
buffer = []
num = 0;

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
        print(tweet+" "+category[0])
        strnum = str(num)
        path = 'text2/tweet'+str(strnum)+'.txt';
        
        text_file = open(path, "w")
        n = text_file.write(tweet)
        text_file.close()
        
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

#print(classesInt)

vectorizer = TfidfVectorizer(norm=None, stop_words="english",max_df=0.95, min_df=2)
tfidf = vectorizer.fit_transform(tweets)
writeDataFile('tweet_irony_test_2018.data',vectorizer,tfidf,classesInt)

#print(df)
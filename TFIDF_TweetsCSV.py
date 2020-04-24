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

def num_there(s):
    return any(i.isdigit() for i in s)

file = open('SemEval2017-task4-test.subtask-A.english.txt','r',encoding ="utf8")
tweets = []
tfidf = []
buffer = []
while True:
    # read line
    line = file.readline()
    if line:
        tweetsJustRead= line.split("\t",2)
        buffer = [tweetsJustRead[2]] 
        
        vectorizer = TfidfVectorizer(norm=None, stop_words="english")
        
        words_vector = vectorizer.fit_transform(buffer)
        tfidf.append(words_vector)
        tweets.append(buffer)

        
    if not line:
        break

file.close()

data = {'TF-IDF':tweets,
        'Tweets':words_vector}

df = pd.DataFrame(data)

print(df)
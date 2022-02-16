import os, json
import numpy as np
import pickle , time

from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

# load data splits and combinations
import _load
# calculate measures etc
import _calculate

if not os.path.exists(os.path.join('.','Data','results')):
	os.makedirs(os.path.join('.','Data','results'))

########################################################################################
# Validation trials: testing LR, SVM, NN, RF for the 4 different GPT2 datasets

validation_results = {'small-117M':{},'xl-1542M':{},'small-117M-k40':{},'xl-1542M-k40':{}}

# datasets
sizes = ['small-117M','xl-1542M','small-117M-k40','xl-1542M-k40']

# classifiers
classifiers = [('LogisticRegression',[1/64,1/16,1/4,1/2,1,2,4,16,64]),('SVM',[1/64,1/16,1/4,1/2,1,2,4,16,64]),('NeuralNet',[0.000001,0.00001,0.0001,0.001,0.01,0.1]),('RandomForest',[1,2,5,10,15,20,25,50])]

print('Initiating validation trials...')
for size in sizes:

	data = _load._load_data('GPT2','webtext',size)

    for classifier in classifiers:
            
        start_ = time.time()
        full_results, accuracies, best_model =_calculate._evaluate_performance(data,classifier[0],classifier[1])
        validation_results[size][classifier[0]] = {}
        validation_results[size][classifier[0]]['best_model'] = best_model
        validation_results[size][classifier[0]]['accuracies'] = accuracies
        validation_results[size][classifier[0]]['full_results'] = full_results
        end_ = time.time()

        with open(os.path.join('.','Data','results','validationResults.pkl'),'wb') as pickle_out:
        	pickle.dump(validation_results, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)

        print('Data: {}\nClassifier: {}\Accuaracies:\n{}\nTime: {:8.2f}'.format(size,classifier[0],accuracies,end_-start_))
        
print('Validation trials terminated!')
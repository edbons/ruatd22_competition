import os, json, sys
import numpy as np
import pickle , time

from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from scipy import sparse

import _calculate
import _load
######################################################################################################

# collect arguments

if len(sys.argv) != 5:
	print('Expecting 4 arguments: model, size, n, save')
	sys.exit(1)

model = sys.argv[1]
size = sys.argv[2]
n = sys.argv[3]
save_dummy = sys.argv[4]

#####################################################################################################
# run (model,size,n): [('GPT2','small-117M',100000),('GPT2','small-117M-k40',100000),('GPT2','xl-1542M',100000),('GPT2','xl-1542M-k40',100000),
# 					   ('GPT3','175B',1604),('Grover','groverMega',8000)]
#####################################################################################################

data = _load._load_tfidf(model,size,int(n))
  
n_train = len(data['labels']['train'])
n_valid = len(data['labels']['valid'])

LR = LogisticRegression(solver='liblinear')
params = {'C': [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]}
split = PredefinedSplit([-1]*n_train+[0]*n_valid)
search = GridSearchCV(LR, params, cv=split)
search.fit(sparse.vstack([data['tfidf']['train'], data['tfidf']['valid']]), data['labels']['train']+data['labels']['valid'])
LR = LR.set_params(**search.best_params_)
LR.fit(data['tfidf']['train'], data['labels']['train'])
valid_accuracy = LR.score(data['tfidf']['valid'], data['labels']['valid'])
test_accuracy = LR.score(data['tfidf']['test'], data['labels']['test'])

predictions = LR.predict_proba(data['tfidf']['test'])[:,1]
groundtruth = data['labels']['test']

full_results = _calculate._calculate_auc(predictions,groundtruth,np.arange(0.05,1,0.05))

results_ = {}
results_['best_model'] = LR
results_['test_acc'] = test_accuracy
results_['full_results'] = full_results

print('Model: {}\nSize: {}\nBest C: {}\nValid Acc.: {:.4f}\nTest Acc.: {:.4f}\nAUC: {:.4f}'.format(model,size,LR.C,valid_accuracy,test_accuracy,full_results['auc']))

if save_dummy == 'True':
	with open(os.path.join('.','Data','results',size,'{}_tfidf.pkl'.format(size)),'wb') as handle:
		pickle.dump(results_, handle, protocol=pickle.HIGHEST_PROTOCOL)

 
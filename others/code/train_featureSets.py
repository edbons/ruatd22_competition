import os, json, sys
import numpy as np
import pickle, time

from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

import _calculate
import _load
###############################################################################################
# collect arguments

if len(sys.argv) != 4:
	print('Expecting 3 arguments: model, human_text, machine_text')
	sys.exit(1)

model = sys.argv[1]
human_text = sys.argv[2]
machine_text = sys.argv[3]

if not os.path.exists(os.path.join('.','Data','results','featureResults.pkl')):
	resultsFeatures = {}
	resultsFeatures[machine_text] = {}
else:
	with open(os.path.join('.','Data','results','featureResults.pkl'),'rb') as pkl_in:
		resultsFeatures = pickle.load(pkl_in)
	resultsFeatures[machine_text] = {}

###############################################################################################
# set parameters

# parameters (tuning)
layers = [(25,50,25)]
activations = ['relu']
learning_rates = [0.0001,0.001]
alphas = [0.00005,0.0001]
n_combos = len(layers) * len(activations) * len(learning_rates) * len(alphas)

# parameters (fixed)
solver = 'adam'
learning_rate = 'adaptive'
tol = 0.001
n_iter_no_change = 10
max_iter = 250

##############################################################################################
# define feature sets

feature_sets = [('basicAbs',0,6),('basicRel',7,12),('readability',13,21),
			('lexicalDiv',22,26),('formatting',27,54),('repetitiveness',55,117),
			('syntactic',118,185),('NE',186,226),('coreference',227,245),
			('entityGrid',246,261),('informationLoss',262,279),
			('empath',280,296),('Q')]

##############################################################################################
# grid search

for set in feature_sets:

	if set[0] == 'Q':
		data = _load._load_Q(model,human_text,machine_text)
	else:
		data = _load._load_features(model,human_text,machine_text,[set[1],set[2]])	

	# find best combination
	results_ = {'validation':{}, 'test':{}}
	index_ = 0
	best_acc_ = 0
	best_index_ = 0

	for lr in learning_rates:
		for alpha in alphas:
			for layer in layers:
				for activation in activations:
		  
# configure model
					model_ = MLPClassifier(hidden_layer_sizes=layer,activation=activation,
										solver=solver,alpha=alpha,learning_rate=learning_rate,
										learning_rate_init=lr,max_iter=max_iter,early_stopping=True,
										tol=tol,n_iter_no_change=n_iter_no_change,validation_fraction=0.02,
										random_state=175)

	# Train
					start_ = time.time()
					model_.fit(data['train']['features'],data['train']['labels'])
					val_acc = model_.score(data['valid']['features'],data['valid']['labels'])
					end_ = time.time()

	# Update best model
					if val_acc > best_acc_:
						best_index_ = index_
						best_acc_ = val_acc

	# Validation results results
					results_['validation'][index_] = {'val_acc': val_acc, 'Layers': layer, 'Activation': activation,
													  'LR': lr, 'Alpha': alpha, 'Model':model_}

	# Track progress
					print('{:4}/{:4} finished --- time: {:6.2f}\n'.format(index_+1,n_combos,end_-start_),results_['validation'][index_])
					index_ += 1

# results for best combination
	best_model = results_['validation'][best_index_]['Model']

	test_acc = best_model.score(data['test']['features'],data['test']['labels'])
	predictions = best_model.predict_proba(data['test']['features'])[:,1]
	groundtruth = data['test']['labels']
	results_['test']['test_acc'] = test_acc
	results_['test']['best_model'] = best_model
	results_['test']['full_results'] = _calculate._calculate_auc(predictions,groundtruth,np.arange(0.05,1,0.05))

	print('Data: {}\nFeatures: {}\nTest Acc.: {}\nBest Model:\n{}'.format(machine_text,set[0],test_acc,best_model))

# save to set-results to array
	resultsFeatures[machine_text][set[0]] = results_

# save array
with open(os.path.join('.','Data','results','featureResults.pkl'),'wb') as pkl_out:
	pickle.dump(resultsFeatures, pkl_out, protocol=pickle.HIGHEST_PROTOCOL)
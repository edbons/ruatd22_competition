import os, json, sys
import numpy as np
import pickle , time

from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

import _calculate
import _load
########################################################################################

# collect arguments

if len(sys.argv) != 3:
	print('Expected 2 arguments: combi, Q')
	sys.exit(1)

combi = sys.argv[1]
Q_dummy = sys.argv[2]

########################################################################################

data = _load._load_combi_features(combi,Q_dummy)

if Q_dummy == 'True':
	save_file = '{}_optimalClassifier.pkl'
else:
	save_file = '{}_optimalClassifier_noQ.pkl'

########################################################################################
# set parameters

# parameters (tuning)
layers = [(100),(25,50,25)]
activations = ['relu','logistic']
learning_rates = [0.001,0.01]
alphas = [0.00005,0.0001,0.0005]
n_combos = len(layers) * len(activations) * len(learning_rates) * len(alphas)

# parameters (fixed)
solver = 'adam'
learning_rate = 'adaptive'
tol = 0.001
n_iter_no_change = 10
max_iter = 250

#######################################################################################
# grid search

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
			model = MLPClassifier(hidden_layer_sizes=layer,activation=activation,
							solver=solver,alpha=alpha,learning_rate=learning_rate,
							learning_rate_init=lr,max_iter=max_iter,early_stopping=True,
							tol=tol,n_iter_no_change=n_iter_no_change,validation_fraction=0.02,
							random_state=175)

			# Train
			start_ = time.time()
			model.fit(data['train']['features'],data['train']['labels'])
			val_acc = model.score(data['valid']['features'],data['valid']['labels'])
			end_ = time.time()

			# update best model
			if val_acc > best_acc_:
				best_index_ = index_
				best_acc_ = val_acc

			# validation results results
			results_['validation'][index_] = {'val_acc': val_acc, 'Layers': layer, 'Activation': activation,
											  'LR': lr, 'Alpha': alpha, 'Model': model}

			# track
			print('{:4}/{:4} finished --- time: {:6.2f}\n'.format(index_+1,n_combos,end_-start_),results_['validation'][index_])
			index_ += 1

			# save progress
			with open(os.path.join('.','Data','results',combi,save_file.format(combi)),'wb') as handle:
				pickle.dump(results_, handle, protocol=pickle.HIGHEST_PROTOCOL)

# results for best combination
best_model = results_['validation'][best_index_]['Model']

test_acc = best_model.score(data['test']['features'],data['test']['labels'])
predictions = best_model.predict_proba(data['test']['features'])[:,1]
groundtruth = data['test']['labels']
results_['test']['test_acc'] = test_acc
results_['test']['best_model'] = best_model
results_['test']['full_results'] = _calculate._calculate_auc(predictions,groundtruth,np.arange(0.05,1,0.05))

print('Final test accuracy:\n',test_acc,'\nModel Configuration:\n',best_model)

# save
with open(os.path.join('.','Data','results',combi,save_file.format(combi)),'wb') as handle:
	pickle.dump(results_, handle, protocol=pickle.HIGHEST_PROTOCOL)		
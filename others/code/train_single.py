import os, json, sys
import numpy as np
import pickle , time

from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

import _calculate
import _load
##################################################################################################

# collect arguments

if len(sys.argv) != 6:
	print('Expecting 5 arguments: model, human_text, machine_text, filtered, Q')
	sys.exit(1)

model = sys.argv[1]
human_text = sys.argv[2]
machine_text = sys.argv[3]
filtered_dummy = sys.argv[4]
Q_dummy = sys.argv[5]

##################################################################################################
# load data: full, Q
if filtered_dummy == 'False':
	if Q_dummy == 'True':

		Q_features = _load._load_Q(model,human_text,machine_text)
		nonQ_features = _load._load_nonQ(model,human_text,machine_text)

		data = {'train': {}, 'valid': {}, 'test': {}}

		for split in ['train','valid','test']:

			data[split]['features'] = np.hstack((nonQ_features[split]['features'],Q_features[split]['features']))
			data[split]['labels'] = nonQ_features[split]['labels']

		del Q_features, nonQ_features

		save_file = '{}_optimalClassifier.pkl'

# load data: full, no Q
	elif Q_dummy == 'False':

		data = _load._load_nonQ(model,human_text,machine_text)

		save_file = '{}_optimalClassifier_noQ.pkl'

# load data: filtered, Q
elif filtered_dummy == 'True':

	# load and combine the features
	Q_features = _load._load_Q_filtered(model,human_text,machine_text)
	nonQ_features = _load._load_nonQ_filtered(model,human_text,machine_text)

	data = {'train':{}, 'valid':{}, 'test':{}}

	for split in ['train','valid','test']:

		data[split]['features'] = np.hstack((nonQ_features[split]['features'],Q_features[split]['features']))
		data[split]['labels'] = nonQ_features[split]['labels']

	del Q_features, nonQ_features

	save_file = '{}_optimalClassifier_filtered.pkl'

##################################################################################################

save_path = os.path.join('.','Data','results',machine_text)

if not os.path.exists(save_path):
	os.makedirs(save_path)

##################################################################################################
# Set grid-search parameters

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

##################################################################################################
# Grid Search

results_ = {'validation':{}, 'test':{}}
index_ = 0
best_acc_ = 0
best_index_ = 0

for lr in learning_rates:
	for alpha in alphas:
		for layer in layers:
			for activation in activations:
		
# Configure model
				modelO = MLPClassifier(learning_rate_init=lr,
									   alpha=alpha,
									   hidden_layer_sizes=layer,
									   activation=activation,
									   solver=solver,learning_rate=learning_rate,max_iter=max_iter,early_stopping=True,
									   tol=tol,n_iter_no_change=n_iter_no_change,validation_fraction=0.02,random_state=175)

# Train model
				start_ = time.time()
				modelO.fit(data['train']['features'],data['train']['labels'])
				val_acc = modelO.score(data['valid']['features'],data['valid']['labels'])
				end_ = time.time()

# Update best model
				if val_acc > best_acc_:
					best_index_ = index_
					best_acc_ = val_acc

# Save results
				results_['validation'][index_] = {'val_acc': val_acc, 'Layers': layer, 'Activation': activation,
														'LR': lr, 'Alpha': alpha, 'Model':modelO}

# Track progress
				print('{:4}/{:4} finished --- time: {:6.2f}\n'.format(index_+1,n_combos,end_-start_),results_['validation'][index_])
				index_ += 1

# Save progress
				with open(os.path.join(save_path,save_file.format(machine_text)),'wb') as handle:
					pickle.dump(results_, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Best model
best_modelO = results_['validation'][best_index_]['Model']

test_acc = best_modelO.score(data['test']['features'],data['test']['labels'])
predictions = best_modelO.predict_proba(data['test']['features'])[:,1]
groundtruth = data['test']['labels']

results_['test']['test_acc'] = test_acc
results_['test']['best_model'] = best_modelO
results_['test']['full_results'] = _calculate._calculate_auc(predictions,groundtruth,np.arange(0.05,1,0.05))

print('Final test accuracy:\n',test_acc,'\nModel Configuration:\n',best_modelO)

# Save full results

with open(os.path.join(save_path,save_file.format(machine_text)),'wb') as handle:
	pickle.dump(results_, handle, protocol=pickle.HIGHEST_PROTOCOL)
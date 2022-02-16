import os, json, sys
import numpy as np
import pickle , time

from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy import sparse

import _calculate
import _load
######################################################################################
# collect arguments

if len(sys.argv) != 7:
	print('Expected 6 arguments: model, human_text, machine_text, n, architecture, ensemble') # architecture: LR or NN, ensemble: separate or super
	sys.exit(1)

model = sys.argv[1]
human_text = sys.argv[2]
machine_text = sys.argv[3]
n = sys.argv[4]
architecture = sys.argv[5]
ensemble = sys.argv[6]

######################################################################################

# LOAD DATA
if ensemble == 'separate':

	data = _load._load_ensemble(model,human_text,machine_text,int(n),Q=False)

# load individual classifiers
	with open(os.path.join('.','Data','results',machine_text,'{}_tfidf.pkl'.format(machine_text)),'rb') as pickle_in:
		tfidf_classifier = pickle.load(pickle_in)['best_model']

	with open(os.path.join('.','Data','results',machine_text,'{}_optimalClassifier_noQ.pkl'.format(machine_text)),'rb') as pickle_in:
		feature_classifier = pickle.load(pickle_in)['test']['best_model']

# get predictions of individual classifiers
	ensemble_data = {}
	
	for split in ['train','valid','test']:

		tfidf_predictions = tfidf_classifier.predict_proba(data['tfidf'][split])[:,1].reshape((len(data['labels'][split]),1))
		feature_predictions = feature_classifier.predict_proba(data['features'][split])[:,1].reshape((len(data['labels'][split]),1))

		predictions = np.hstack((feature_predictions,tfidf_predictions)) 

		ensemble_data[split] = {}
		ensemble_data[split]['features'] = predictions
		ensemble_data[split]['labels'] = data['labels'][split]

		if split == 'train':
			scaler = StandardScaler()
			scaler.fit(ensemble_data['train']['features'])

		ensemble_data[split]['features'] = scaler.transform(ensemble_data[split]['features'])

	n_train = len(ensemble_data['train']['labels'])
	n_valid = len(ensemble_data['valid']['labels'])

elif ensemble == 'super':

	data = _load._load_ensemble(model,human_text,machine_text,int(n),Q='True')

	ixs = {'basicAbs': (0,6), 'basicRel': (7,12), 'readability': (13,21), 'lexicalDiv': (22,26), 'formatting': (27,54), 'repetitiveness': (55,117),
			'syntactic': (118,185), 'NE': (186,226), 'coreference': (227,245), 'entityGrid': (246,261), 'informationLoss': (262,279), 'empath': (280,296)}

# load classifiers
	with open(os.path.join('.','Data','results',machine_text,'{}_tfidf.pkl'.format(machine_text)),'rb') as pickle_in:
		tfidf_classifier = pickle.load(pickle_in)['best_model']

	with open(os.path.join('.','Data','results','featureResults.pkl'),'rb') as pickle_in:
		indiv_classifiers = pickle.load(pickle_in)[machine_text]

	with open(os.path.join('.','Data','results','featureResults.pkl'),'rb') as pickle_in:
		Q_classifier = pickle.load(pickle_in)[machine_text]['Q']['test']['best_model']

# initialise ensemble data
	ensemble_data = {}
	for split in ['train','valid','test']:
		ensemble_data[split] = {}
		ensemble_data[split]['features'] = np.zeros((len(data['labels'][split]),1))

# get preditctions from individual feature-classifiers
	for feature in indiv_classifiers.keys():

		if not feature == 'Q':
			feature_classifier = indiv_classifiers[feature]['test']['best_model']

			# get predictions from the indiv_classifiers
			for split in ['train','valid','test']:
				predictions_ = feature_classifier.predict_proba(data['features'][split][:,ixs[feature][0]:ixs[feature][1]+1])[:,1].reshape((len(data['labels'][split]),1))
				ensemble_data[split]['features'] = np.hstack((ensemble_data[split]['features'],predictions_))

# add Q and tf-idf features
	for split in ['train','valid','test']:

		Q_predictions = Q_classifier.predict_proba(data['Q'][split])[:,1].reshape((len(data['labels'][split]),1))
		tfidf_predictions = tfidf_classifier.predict_proba(data['tfidf'][split])[:,1].reshape((len(data['labels'][split]),1))

		ensemble_data[split]['features'] = np.hstack((ensemble_data[split]['features'][:,1:],Q_predictions,tfidf_predictions))
		ensemble_data[split]['labels'] = data['labels'][split]

		if split == 'train':
			scaler = StandardScaler()
			scaler.fit(ensemble_data['train']['features'])

		ensemble_data[split]['features'] = scaler.transform(ensemble_data[split]['features'])

	n_train = len(ensemble_data['train']['labels'])
	n_valid = len(ensemble_data['valid']['labels'])


# INITIATE MODELS
if architecture == 'NN':

	ensemble_model = MLPClassifier(solver='adam',learning_rate='adaptive',
								early_stopping=True,validation_fraction=0.02,
								random_state=175,verbose=False)

	params = {'activation': ['relu','logistic'],
			'hidden_layer_sizes': [[100], [25,50,25], [5,10,5]],
			'learning_rate_init': [0.00001,0.0001,0.001],
			'alpha': [0.00001,0.00005,0.0001,0.005]}

elif architecture == 'LR':

	ensemble_model = LogisticRegression(solver='liblinear')

	params = {'C': [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]}


# TRAIN MODELS
split = PredefinedSplit([-1]*n_train+[0]*n_valid)
search = GridSearchCV(ensemble_model, params, cv=split)
search.fit(sparse.vstack([ensemble_data['train']['features'], ensemble_data['valid']['features']]), ensemble_data['train']['labels']+ensemble_data['valid']['labels'])
ensemble_model = ensemble_model.set_params(**search.best_params_)
ensemble_model.fit(ensemble_data['train']['features'], ensemble_data['train']['labels'])


# COLLECT RESULTS
accuracies = {}
accuracies['train'] = ensemble_model.score(ensemble_data['train']['features'],ensemble_data['train']['labels'])
accuracies['valid'] = ensemble_model.score(ensemble_data['valid']['features'],ensemble_data['valid']['labels'])
accuracies['test'] = ensemble_model.score(ensemble_data['test']['features'],ensemble_data['test']['labels'])

predictions = ensemble_model.predict_proba(ensemble_data['test']['features'])[:,1]
groundtruth = ensemble_data['test']['labels']
full_results = _calculate._calculate_auc(predictions,groundtruth,np.arange(0.05,1,0.05))

results_ = {'model': ensemble_model, 'accuracies': accuracies, 'full_results': full_results}

if architecture == 'NN':
	parameters = {'alpha': ensemble_model.alpha, 'learning_rate_init': ensemble_model.learning_rate_init,
				'hidden_layer_sizes': ensemble_model.hidden_layer_sizes, 'activation': ensemble_model.activation}

elif architecture == 'LR':
	parameters = ensemble_model.C

print('Data: {}\nParameters:\n{}\nAccuracies: {}'.format(machine_text,parameters,accuracies))

# save the ensemble model
with open(os.path.join('.','Data','results',machine_text,'{}_ensemble_{}_{}.pkl'.format(machine_text,ensemble,architecture)),'wb') as pickle_out:
	pickle.dump(results_, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
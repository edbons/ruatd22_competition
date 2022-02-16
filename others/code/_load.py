
import numpy as np
import os, json, pickle
from sklearn.utils import shuffle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from scipy import sparse

############################################################################################################
# load dataset (single split)

def _load_split(data_dir, source, split, n=np.inf):
		path = os.path.join(data_dir, f'{source}.{split}.jsonl')
		texts = []
		for i, line in enumerate(open(path)):
				if i >= n:
						break
				if source in ['webtext','small-117M','small-117M-k40','xl-1542M','xl-1542M-k40']:
						texts.append(json.loads(line)['text'])
				else:
						texts.append(json.loads(line))
		if source in ['webtext','small-117M','small-117M-k40','xl-1542M','xl-1542M-k40']:
				return texts
		else:
				return texts[0]

############################################################################################################
# load combi-dataset

def _load_combi(combi,source,split):

	with open(os.path.join('.','Data','data','full_text','Combinations',combi,'{}_ix.pkl'.format(combi)),'rb') as pickle_in:
		combi_dict = pickle.load(pickle_in)

	all_texts = []

	print(combi_dict[source][split].items())

	for size, ixs in combi_dict[source][split].items():
		
		if size in ['small-117M','small-117M-k40','xl-1542M','xl-1542M-k40','webtext']:
			model = 'GPT2'
		elif size in ['groverMega','realNews']:
			model = 'Grover'
		else:
			model = 'GPT3'

		texts_ = _load_split(os.path.join('.','Data','data','full_text',model,size),size,split)
		texts = [text for i,text in enumerate(texts_) if i in ixs]
		del texts_
		all_texts = all_texts + texts
		del texts

	return all_texts

############################################################################################################
# load non-Q features of an evaluation scenario: machine text and human text; train, valid and test

def _load_nonQ(model_,human_text,machine_text):

	out_data = {'train':{}, 'valid':{}, 'test':{}}

	data_path = os.path.join('.','Data','features','full_text')

	for split in ['train','valid','test']:

		human_features = np.load(os.path.join(data_path,model_,human_text,'{}_features_{}.npy'.format(human_text,split)), allow_pickle=True)
		machine_features = np.load(os.path.join(data_path,model_,machine_text,'{}_features_{}.npy'.format(machine_text,split)), allow_pickle=True)
		
		features = np.vstack((human_features,machine_features))
		labels = np.array([0] * len(human_features) + [1] * len(machine_features))

		# shuffle
		index = np.arange(0,len(features))
		shuffled_index = shuffle(index, random_state=175)

		features = features[shuffled_index]
		labels = labels[shuffled_index]

		# add to dict
		out_data[split]['features'] = features
		out_data[split]['labels'] = labels

	return out_data

############################################################################################################
# load filtered non-Q features of an evaluation scenario: machine text and human text; train, valid and test

def _load_nonQ_filtered(model_,human_text,machine_text):

	out_data = {'train':{}, 'valid':{}, 'test':{}}

	path = os.path.join('.','Data','features','full_text',model_)
	filter_path = os.path.join('.','Data','data','filtered_text',model_)

	for split in ['train','valid','test']:
	
		filters_human = np.load(os.path.join(filter_path,human_text,'{}filtered_texts_{}'.format(human_text,split)), allow_pickle=True)
		filters_machine = np.load(os.path.join(filter_path,machine_text,'{}filtered_texts_{}'.format(machine_text,split)), allow_pickle=True)

		human_features = np.load(os.path.join(path,human_text,'{}_features_{}.npy'.format(human_text,split)), allow_pickle=True)[filters_human]
		machine_features = np.load(os.path.join(path,machine_text,'{}_features_{}.npy'.format(machine_text,split)), allow_pickle=True)[filters_machine]

		features = np.vstack((human_features,machine_features))
		labels = np.array([0] * len(human_features) + [1] * len(machine_features))

		# shuffle
		index = np.arange(0,len(features))
		shuffled_index = shuffle(index, random_state=175)

		features = features[shuffled_index]
		labels = labels[shuffled_index]

		# add to dict
		out_data[split]['features'] = features
		out_data[split]['labels'] = labels

	return out_data

############################################################################################################
# load Q features of an evaluation scenario: machine text and human text; train, valid and test 

def _load_Q(model_,human_text,machine_text):

	out_data = {'train':{}, 'valid':{}, 'test':{}}

	path_human = os.path.join('.','Data','Q','full_text')
	path_machine = os.path.join('.','Data','Q','full_text')

	for split in ['train','valid','test']:

		human_Q = np.load(os.path.join(path_human,model_,human_text,'{}_{}_QFT_{}.npy'.format(human_text,machine_text,split)), allow_pickle=True)
		machine_Q = np.load(os.path.join(path_machine,model_,machine_text,'{}_QFT_{}.npy'.format(machine_text,split)), allow_pickle=True)

		features = np.vstack((human_Q,machine_Q))
		labels = np.array([0] * len(human_Q) + [1] * len(machine_Q))

		# shuffle
		index = np.arange(0,len(features))
		shuffled_index = shuffle(index, random_state=175)

		features = features[shuffled_index]
		labels = labels[shuffled_index]

		# add to dict
		out_data[split]['features'] = features
		out_data[split]['labels'] = labels

	return out_data	

############################################################################################################
# load filtered-dataset Q features of an evaluation scenario: machine text and human text; train, valid and test

def _load_Q_filtered(model_,human_text,machine_text):

	out_data = {'train':{}, 'valid':{}, 'test':{}}

	path_human = os.path.join('.','Data','Q','filtered_text',model_)
	path_machine = os.path.join('.','Data','Q','filtered_text',model_)

	for split in ['train','valid','test']:

		human_Q = np.load(os.path.join(path_human,human_text,'{}_{}_QFT_{}.npy'.format(human_text,machine_text,split)), allow_pickle=True)
		machine_Q = np.load(os.path.join(path_machine,machine_text,'{}_QFT_{}.npy'.format(machine_text,split)), allow_pickle=True)
		
		features = np.vstack((human_Q,machine_Q))
		labels = np.array([0] * len(human_Q) + [1] * len(machine_Q))

		# shuffle
		index = np.arange(0,len(features))
		shuffled_index = shuffle(index, random_state=175)

		features = features[shuffled_index]
		labels = labels[shuffled_index]

		# add to dict
		out_data[split]['features'] = features
		out_data[split]['labels'] = labels

	return out_data	

############################################################################################################
# load array of an evaluation scenario: machine text and human text; train, valid and test
# both non-Q and Q features

def _load_data(model_,human_text,machine_text):

	nonQ_data = _load_nonQ(model_,human_text,machine_text)
	Q_data = _load_Q(model_,human_text,machine_text)

	data = {'train': {}, 'valid': {}, 'test': {}}

	for split in ['train','valid','test']:

		data[split]['features'] = np.hstack((nonQ_data[split]['features'],Q_data[split]['features']))
		data[split]['labels'] = Q_data[split]['labels']

	return data

############################################################################################################
# load and create tf-idf vectors

def _load_tfidf(model,size,n):
	
	np.random.seed(175)
	out_data = {'features':{}, 'tfidf':{}, 'Q':{}, 'labels':{}}

	if model == 'Grover':
	  human_text = 'realNews_solo'
	elif model == 'GPT3':
	  human_text = 'GPT3_webtext'
	else:
	  human_text = 'webtext'

	# tf-idf 
	human_train = _load_split(os.path.join('.','Data','data','full_text',model,human_text),human_text,'train')
	machine_train = _load_split(os.path.join('.','Data','data','full_text',model,size),size,'train')
	
	index = np.arange(0,n)
	new_index = np.random.choice(index, n, replace=False)

	human_train = [t for i,t in enumerate(human_train) if i in new_index]
	machine_train = [t for i,t in enumerate(machine_train) if i in new_index]

	out_data['labels']['train'] = [0] * len(human_train) + [1] * len(machine_train)  

	train = human_train + machine_train
	del human_train, machine_train

	vect = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=2**21)
	out_data['tfidf']['train'] = vect.fit_transform(train)
	del train

	human_valid = _load_split(os.path.join('.','Data','data','full_text',model,human_text),human_text,'valid')
	machine_valid = _load_split(os.path.join('.','Data','data','full_text',model,size),size,'valid')
	
	out_data['labels']['valid'] = [0] * len(human_valid) + [1] * len(machine_valid) 

	valid = human_valid + machine_valid
	del human_valid, machine_valid

	out_data['tfidf']['valid'] = vect.transform(valid)
	del valid

	human_test = _load_split(os.path.join('.','Data','data','full_text',model,human_text),human_text,'test')
	machine_test = _load_split(os.path.join('.','Data','data','full_text',model,size),size,'test')
	
	out_data['labels']['test'] = [0] * len(human_test) + [1] * len(machine_test) 

	test = human_test + machine_test
	del human_test, machine_test

	out_data['tfidf']['test'] = vect.transform(test)
	del test, vect
			
	return out_data

############################################################################################################
# load features

def _load_features(model_,human_text,machine_text,ixs):
	
	out_data = {'train': {}, 'valid': {}, 'test': {}}
	path = os.path.join('.','Data','features','full_text',model_)

	for split in ['train','valid','test']:
		
		human = np.load(os.path.join(path,human_text,'{}_features_{}.npy'.format(human_text,split)), allow_pickle=True)
		machine = np.load(os.path.join(path,machine_text,'{}_features_{}.npy'.format(machine_text,split)), allow_pickle=True)

		if ixs:
		  human = human[:,ixs[0]:ixs[1]+1]
		  machine = machine[:,ixs[0]:ixs[1]+1]
		 
		features = np.vstack((human,machine))
		labels = np.array([0] * len(human) + [1] * len(machine))
		
		index = np.arange(0,len(features))
		shuffled_index = shuffle(index, random_state=175)

		out_data[split]['features'] = features[shuffled_index]
		out_data[split]['labels'] = labels[shuffled_index]
		
	return out_data

############################################################################################################
# load combi-features (no Q)

def _load_combi_features(combi,Q=False):

# feature paths
	path = os.path.join('.','Data','features','full_text')
	Q_path = os.path.join('.','Data','Q','full_text','Combinations')

# load dict of required texts and indices
	with open(os.path.join(path,'Combinations',combi,'{}_ix.pkl'.format(combi)),'rb') as pickle_in:
		combi_dict = pickle.load(pickle_in)

# create empty dict
	out_data = {'train': {}, 'valid': {}, 'test': {}}

	for split in ['train','valid','test']:

		human_features = np.zeros((1,297))
		machine_features = np.zeros((1,297))

# get the individual features of the human text from the different datasets
		for size, ixs in combi_dict['human'][split].items():

			if size in ['webtext','small-117M','small-117M-k40','xl-1542M','xl-1542M-k40']:
				model_ = 'GPT2'
			elif size in ['groverMega','realNews']:
				model_ = 'Grover'
			else:
				model_ = 'GPT3'

			features_ = np.load(os.path.join(path,model_,size,'{}_features_{}.npy'.format(size,split)),allow_pickle=True)[ixs]
			human_features = np.vstack((human_features,features_))

# get the individual features of the machine text from the different datasets
		for size, ixs in combi_dict['machine'][split].items():

			if size in ['webtext','small-117M','small-117M-k40','xl-1542M','xl-1542M-k40']:
				model_ = 'GPT2'
			elif size in ['groverMega','realNews']:
				model_ = 'Grover'
			else:
				model_ = 'GPT3'

			features_ = np.load(os.path.join(path,model_,size,'{}_features_{}.npy'.format(size,split)),allow_pickle=True)[ixs]
			machine_features = np.vstack((machine_features,features_))

		# stack human and machine features
		human_features = human_features[1:]
		machine_features = machine_features[1:]

		if Q == True:

			nonQ_features = np.vstack((human_features,machine_features))

			human_Q = np.load(os.path.join(Q_path,combi,'{}_{}_QFT_{}.npy'.format('human',combi,split)),allow_pickle=True)
			machine_Q = np.load(os.path.join(Q_path,combi,'{}_{}_QFT_{}.npy'.format('machine',combi,split)),allow_pickle=True)
			Q_features = np.vstack((human_Q,machine_Q))

			features = np.hstack((nonQ_features,Q_features))
			labels = np.array([0] * len(human_features) + [1] * len(machine_features))

		else:

			features = np.vstack((human_features,machine_features))
			labels = np.array([0] * len(human_features) + [1] * len(machine_features))
			
		# shuffle
		index = np.arange(0,len(features))
		shuffled_index = shuffle(index, random_state=175)

		features = features[shuffled_index]
		labels = labels[shuffled_index]

		# add to dict
		out_data[split]['features'] = features
		out_data[split]['labels'] = labels

	return out_data		

############################################################################################################
# load features-, tf-idf- and Q-data for ensemble classifiers

def _load_ensemble(model_,human_text,machine_text,n,Q):
	
	np.random.seed(175)
	out_data = {'features':{}, 'tfidf':{}, 'Q':{}, 'labels':{}}

	text_path = os.path.join('.','Data','data','full_text')
	feature_path = os.path.join('.','Data','features','full_text')
	Q_path = os.path.join('.','Data','Q','full_text')

	for split in ['train','valid','test']:

		# tf-idf 
		human_texts = _load_split(os.path.join(text_path,model_,human_text),human_text,split)
		machine_texts = _load_split(os.path.join(text_path,model_,machine_text),machine_text,split)

		if split == 'train':
			index = np.arange(0,n)
			new_index = np.random.choice(index, n, replace=False)

			human_texts = [t for i,t in enumerate(human_texts) if i in new_index]
			machine_texts = [t for i,t in enumerate(machine_texts) if i in new_index]

		tfidf_pre = human_texts + machine_texts
		del human_texts, machine_texts

		if split == 'train':
			vect = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=2**21)
			out_data['tfidf'][split] = vect.fit_transform(tfidf_pre)
		else:
			out_data['tfidf'][split] = vect.transform(tfidf_pre)

		del tfidf_pre

		# features
		human_features = np.load(os.path.join(feature_path,model_,human_text,'{}_features_{}.npy'.format(human_text,split)), allow_pickle=True)
		machine_features = np.load(os.path.join(feature_path,model_,machine_text,'{}_features_{}.npy'.format(machine_text,split)), allow_pickle=True)

		if split == 'train':
			human_features = human_features[new_index]
			machine_features = machine_features[new_index]

		out_data['features'][split] = np.vstack((human_features,machine_features))
		
		labels = [0] * len(human_features) + [1] * len(machine_features)
		del human_features, machine_features

		out_data['labels'][split] = labels
		del labels

		# Q
		if Q == 'True':

			human_Q = np.load(os.path.join(Q_path,model_,human_text,'{}_{}_QFT_{}.npy'.format(human_text,machine_text,split)), allow_pickle=True)
			machine_Q = np.load(os.path.join(Q_path,model_,machine_text,'{}_QFT_{}.npy'.format(machine_text,split)), allow_pickle=True)

			if split == 'train':
				human_Q = human_Q[new_index]
				machine_Q = machine_Q[new_index]

			out_data['Q'][split] = np.vstack((human_Q,machine_Q))
			del human_Q, machine_Q

	del vect          
	return out_data
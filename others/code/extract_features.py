# Code to extract the single-text features
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS

import neuralcoref
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')

import textstat
import regex as re
from numpy.linalg import svd, norm

import os, json, sys
import numpy as np, pandas as pd
import time

# function to import splits of different datasets
import _load
# functions to extract indivudal feature sets
import _feature_sets

data_path = os.path.join('.','Data','data','full_text')
feature_path = os.path.join('.','Data','features','full_text')

#####################################################################################################################################
# collect arguments
if len(sys.argv) != 4:
	print('Expecting 3 arguments: model, dataset, split')
	sys.exit(1)

model = sys.argv[1]
size = sys.argv[2]
split = sys.argv[3]

#####################################################################################################################################
print('Start loading prepared data...')

# load texts
texts = _load._load_split(os.path.join(data_path,model,size),size,split)
ntexts = len(texts)
print('...texts loaded!')

# load empath
empath = np.load(os.path.join(feature_path,model,size,'{}_empath_{}.npy'.format(size,split)))
print('...empath loaded!')
print(len(empath))

# load lists
google10000_ = np.load(os.path.join('.','Data','google10000.npy'))
content_words = np.load(os.path.join('.','Data','google5000.npy'))
stop_words_ = list(STOP_WORDS)
print('... lists loaded!')

# create feature dictionary
if not os.path.exists(os.path.join(feature_path,model,size)):
	os.makedirs(os.path.join(feature_path,model,size))

print('Everything loaded!')

#####################################################################################################################################
print('Starting extraction of features for\nModel: {}\nDataset: {}\nSplit: {}'.format(model,size,split))

# take time
start_ = time.time()
checkpoint_ = time.time()

features = []

# extract features
for i_, text in enumerate(texts):
	
	# refresh dictionaries
	punctuation_dict_ = dict.fromkeys([',','.',':',';','?','!','-','\"','(',')','[',']','\n'], 0)
	pos_dict_ = dict.fromkeys(['ADJ','ADP','ADV','NOUN','VERB','AUX','CONJ','CCONJ','DET','INTJ','NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM','X','SPACE'], 0)
	ne_dict_ = dict.fromkeys(['PERSON','NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT','WORK_OF_ART','LAW','LANGUAGE','DATE','TIME','PERCENT','MONEY','QUANTITY','ORDINAL','CARDINAL'], 0)
	trans_dict_ = dict.fromkeys(['SS','SO','SX','S-','OS','OO','OX','O-','XS','XO','XX','X-','-S','-O','-X','--'],0)
	
	try: # try-except to allow for recovery from point-of-failure
	
		try:
			doc = nlp(text)
		except MemoryError: 
			features.append(np.zeros(297))
			continue

	# basic & readability & lexical diversity

		basic_abs, basic_rel, readability, lexical_div, words, sentences, punctuation_, wordlist_ = _feature_sets.get_basic(doc,text,stop_words_,google10000_)

	# formatting

		punctuation_dist, punctuation_sent, paragraph_features = _feature_sets.get_formatting(text,punctuation_,punctuation_dict_,sentences,words)     

	# repetetiveness

		lexical_rep, syntactic_rep = _feature_sets.get_rep(doc)

	# conjunction overlap

		conj_overlap = _feature_sets.get_conj(doc)

	# syntactic diversity

		syntactic_dist, syntactic_sent, syntactic_features = _feature_sets.get_syntactic(doc,pos_dict_,words,sentences)

	# named entities

		ne_dist, ne_sent, ne_features, unique_ne_, total_ne = _feature_sets.get_ne(doc,ne_dict_,words,sentences)

	# coreference

		coref_dist, coref_features = _feature_sets.get_coref(doc,words,total_ne,unique_ne_)

	# entity grid

		grid_dist = _feature_sets.get_grid(doc,trans_dict_)

	# topic redundancy

		redundancy_features = _feature_sets.get_redundancy(doc,wordlist_,lemma=False)
		redundancy_features_lemmatized = _feature_sets.get_redundancy(doc,wordlist_,lemma=True)

	# empath

		empath_features, empath_active_features, empath_tailored = _feature_sets.get_empath(empath[i_,:],words)

	# COMBINE THE FEATURES
		features.append(np.hstack((basic_abs,basic_rel,readability,lexical_div,punctuation_dist,punctuation_sent,paragraph_features,
							 lexical_rep,syntactic_rep,conj_overlap,syntactic_dist,syntactic_sent,syntactic_features,
							 ne_dist,ne_sent,ne_features,coref_dist,coref_features,grid_dist,redundancy_features,
							 redundancy_features_lemmatized,empath_features,empath_active_features,empath_tailored)))

	# progress tracker
		if (i_ % (np.round(ntexts / 100,0))) == 0:
			print('{:.2f}/100 finished - total time elapsed m: {:.2f} s: {:.2f} - time since last checkpoint m: {:.2f} s: {:.2f}'.format(np.round(100*(i_)/ntexts,0),(time.time()-start_)//60,(time.time()-start_)%60,(time.time()-checkpoint_)//60,(time.time()-checkpoint_)%60))
			checkpoint_ = time.time()

	# save results every 1250 texts
		if (i_ % 1250) == 0:
			print('Saving {} checkpoint...'.format(i_))
			np.save(os.path.join(feature_path,model,size,'BU_{}_restartWith_{}_features_{}.npy'.format(size,i_,split)),features)
			try:
				os.remove(os.path.join(feature_path,model,size,'BU_{}_restartWith_{}_features_{}.npy'.format(size,i_-1250,split)))
			except:
				pass
			  
	except Exception as e:
		
		features.append(13 * [0] + [-21.43, -15.8, 0, 206.835, 0, 0, 0, 0, 3.1291] + 275 * [0])

	except KeyboardInterrupt:

		print('Stopped execution. Index for re-start: ',i_)
		np.save(os.path.join(feature_path,model,size,'BU_{}_restartWith_{}_features_{}.npy'.format(size,i_,split)),features)
		sys.exit(1)
	            
print('Extraction of features finished for\nModel: {}\nDataset: {}\nSplit: {}'.format(model,size,split))

print(len(features))

#####################################################################################################################################
# treat nan - replace nan-/inf-/None-values with default values/zero

features = np.array(features)

replace_ = 13 * [0] + [-21.43, -15.8, 0, 206.835, 0, 0, 0, 0, 3.1291] + 275 * [0]

for i in range(features.shape[0]):
    for j in range(features.shape[1]):
        if (features[i,j] == None) or (pd.isnull(features[i,j])) or (features[i,j] == np.inf):
            features[i,j] = replace_[j]


print(len(features))
#####################################################################################################################################

# save results
np.save(os.path.join(feature_path,model,size,'{}_features_{}.npy'.format(size,split)),features)
print('... results saved!')

# remove backup-files
files = os.listdir(os.path.join(feature_path,model,size))

for file in files:
	if file[:2] == 'BU':
		os.remove(os.path.join(feature_path,model,size,file))
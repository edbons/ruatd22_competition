import os, sys
import numpy as np, pandas as pd
import pickle
import re
from tqdm import tqdm

import spacy
from spacy.lang.ru.stop_words import STOP_WORDS
nlp = spacy.load('ru_core_news_lg')
nlp.enable_pipe("senter")

from extract_Q_raw import SPLITS, DS_PATH, SAVE_PATH
import _feature_sets


def extract_feat(texts):

	print('Starting extraction of features...')

	words10000 = np.load(os.path.join('.','Data','rus10000.npy'), allow_pickle=True)

	stop_words_ = list(STOP_WORDS)
	features = []

	# extract features
	for i_, text in enumerate(tqdm(texts)):		
		# refresh dictionaries
		punctuation_dict_ = dict.fromkeys([',','.',':',';','?','!','-','\"','(',')','[',']','\n'], 0)
		pos_dict_ = dict.fromkeys(['ADJ','ADP','ADV','NOUN','VERB','AUX','CONJ','CCONJ','DET','INTJ','NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM','X','SPACE'], 0)
		ne_dict_ = dict.fromkeys(['PER','ORG','LOC'], 0)
		
		try: # try-except to allow for recovery from point-of-failure
			doc = nlp(text)

			basic_abs, basic_rel, readability, lexical_div, words, sentences, punctuation_, wordlist_ = _feature_sets.get_basic(doc, text, stop_words_, words10000) 		# basic & readability & lexical diversity
			punctuation_dist, punctuation_sent, paragraph_features = _feature_sets.get_formatting(text, punctuation_, punctuation_dict_, sentences, words)     
			lexical_rep, syntactic_rep = _feature_sets.get_rep(doc) # repetetiveness
			conj_overlap = _feature_sets.get_conj(doc) # conjunction overlap
			syntactic_dist, syntactic_sent, syntactic_features = _feature_sets.get_syntactic(doc, pos_dict_, words, sentences) # syntactic diversity
			ne_dist, ne_sent, ne_features, unique_ne_, total_ne = _feature_sets.get_ne(doc, ne_dict_, words, sentences) # named entities
			redundancy_features = _feature_sets.get_redundancy(doc, wordlist_, lemma=False) # topic redundancy
			redundancy_features_lemmatized = _feature_sets.get_redundancy(doc, wordlist_, lemma=True)

			features.append(np.hstack((basic_abs,  # size 7
										basic_rel, # size 6
										readability,  # size 9
										lexical_div,  # size 5 
										punctuation_dist,  # size 13
										punctuation_sent, # size 13
										paragraph_features, # size 13
										lexical_rep, # size 30
										syntactic_rep, # size 30
										conj_overlap, # size 3
										syntactic_dist,  # size 19 
										syntactic_sent,  # size 19 
										syntactic_features,  # size 30 
										ne_dist,  # size 3
										ne_sent,  # size 3
										ne_features, # size 5
										redundancy_features,  # size 9
										redundancy_features_lemmatized # size 9
										)))
			
		# # save results every 1250 texts
		# 	if (i_ % 1250) == 0:
		# 		print('Saving {} checkpoint...'.format(i_))
		# 		np.save(os.path.join(feature_path,model,size,'BU_{}_restartWith_{}_features_{}.npy'.format(size,i_,split)),features)
		# 		try:
		# 			os.remove(os.path.join(feature_path,model,size,'BU_{}_restartWith_{}_features_{}.npy'.format(size,i_-1250,split)))
		# 		except:
		# 			pass
				
		except Exception as e:
			print("exception", i_, e)
			features.append(np.array(13 * [0] + [-21.43, -15.8, 0, 206.835, 0, 0, 0, 0, 3.1291] + (215-13-9) * [0]))

		except KeyboardInterrupt:
			print('Stopped execution. Index for re-start: ', i_)
			# np.save(os.path.join(feature_path,model,size,'BU_{}_restartWith_{}_features_{}.npy'.format(size,i_,split)),features)
			sys.exit(1)
		
	# treat nan - replace nan-/inf-/None-values with default values/zero
	features = np.array(features)
	replace_ = 13 * [0] + [-21.43, -15.8, 0, 206.835, 0, 0, 0, 0, 3.1291] + (215-13-9) * [0]

	for i in range(features.shape[0]):
		for j in range(features.shape[1]):
			if (features[i,j] == None) or (pd.isnull(features[i,j])) or (features[i,j] == np.inf):
				print(f"Fix nan {features[i,j]} in text {i} param {j}: {repr(texts[i])}")
				features[i,j] = replace_[j]
	print(f'Extracted features for {len(features)} texts...')

	return features

if	__name__ == "__main__":
	os.chdir(os.path.dirname(__file__))
	print(os.path.dirname(__file__), __file__)

	for split in SPLITS:
		print(f"\nStart extraction for {split}")
		df = pd.read_csv(os.path.join(DS_PATH, split + '.csv')) 	
		texts = list(df['Text'])	

		feats = extract_feat(texts)	

		with open(os.path.join(SAVE_PATH, 'features', f'{split}_feats.pkl'), 'wb') as f:
			pickle.dump(feats, f)

	print('Finished!')

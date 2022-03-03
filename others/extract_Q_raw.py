import spacy
nlp = spacy.load('ru_core_news_lg', disable=['parser', 'ner', 'lemmatizer', 'morphologizer', 'attribute_ruler'])
nlp.enable_pipe("senter")

import os
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd

DS_PATH = '<PROJECT PATH>/dataset'
SPLITS = ['train', 'val', 'test']
SAVE_PATH = './Data'
LABELS = ['H', 'M']

##############################################################################################################################


def q_stats(texts: list) -> list:
	content_words = np.load(os.path.join('.','Data','rus10000.npy'), allow_pickle=True)
	content_words = np.unique(content_words)[:5000] 
	sentences = 0
	pairs_distanced_ = {}
	single_counts_ = np.zeros(5000)

	for _, text in enumerate(tqdm(texts)):

		doc = nlp(text)

		for sent in doc.sents:

			single_counts_dict_ = dict.fromkeys(content_words,0)
			# refresh the counter for the single content words before every new sentence
			words_ = [(str(x).lower(),i) for i,x in enumerate(sent) if str(x).lower() in content_words]
			# from every sentence, retrieve the content words and their index position

			if words_: # if there are content words in the sentence
				sentences += 1
				for i in range(len(words_)):
					for j in range(len(words_)):
						if j > i: # check every content word pair in a sentence once,
							if (words_[j][1] - words_[i][1]) >= 4: # whether they are 5 tokens apart

								if not (words_[i][0],words_[j][0]) in pairs_distanced_.keys():
									if not (words_[j][0],words_[i][0]) in pairs_distanced_.keys():
										pairs_distanced_[(words_[j][0],words_[i][0])] = 1
									else:
										pairs_distanced_[(words_[j][0],words_[i][0])] += 1
								else:
									pairs_distanced_[(words_[i][0],words_[j][0])] += 1
								# (add and) count the content word pair

					single_counts_dict_[words_[i][0]] = 1
					# mark the presence of every single content word in a sentence
				single_counts_ += np.array(list(single_counts_dict_.values())) 
				# add content word counts per sentence to the global state

		# ---------- Loop Finished

	single_counts_final = dict(zip(content_words, single_counts_))
	# create dictionary for word counts from content word list and global state

	Q_stats = []
	for pair, count in pairs_distanced_.items():
	# calculate the Q statistic for every content word pair found in the corpus
		c11 = count
		# occurrences of the content word pair together
		c12 = single_counts_final[pair[0]] - c11
		# occurrences of the first word of the pair, individually
		c21 = single_counts_final[pair[1]] - c11
		# occurrences of the second word of the pair, individually
		c22 = sentences - c11 - c12 - c21
		# occurrences of neither word in a sentence
		Q = ((c11 * c22) - (c12 * c21)) / ((c11 * c22) + (c12 * c21))
		# calculate Q follwing the formula
		Q_stats.append([pair,Q])

	return Q_stats

def main():
	os.chdir(os.path.dirname(__file__))
	print(os.path.dirname(__file__))

	os.makedirs(os.path.join(SAVE_PATH, 'features'), exist_ok=True)
	print('Starting extraction of Q-stats...')
	for label in LABELS:
		for split in SPLITS:
			df = pd.read_csv(os.path.join(DS_PATH, split + '.csv'))
			if split != 'test':
				df = df[df['Class'] == label]    	
			texts = list(df['Text'])			
			q_stat = q_stats(texts)
			with open(os.path.join(SAVE_PATH, 'features', f'{label}_{split}_QStats.pkl'), 'wb') as f:
				pickle.dump(q_stat, f)


if __name__=="__main__":
	main()
import spacy
nlp = spacy.load('ru_core_news_lg', disable=['parser', 'ner', 'lemmatizer', 'morphologizer', 'attribute_ruler'])
nlp.enable_pipe("senter")

import os
import numpy as np
import pickle
import pandas as pd
from extract_Q_raw import SPLITS, DS_PATH, SAVE_PATH
from tqdm import tqdm


def extract_q(texts, Q_human, Q_machine):
	# prepare lists
	features = []
	content_words = np.load(os.path.join('.','Data','rus10000.npy'), allow_pickle=True)
	content_words = np.unique(content_words)[:5000] 

	# extraction
	print('Extracting Q...')

	for _, text in enumerate(tqdm(texts)):

		doc = nlp(text)

		Q_human_score = 0
		Q_machine_score = 0
		Q_human_count = 0
		Q_machine_count = 0
		Q_human_novel = 0
		Q_machine_novel = 0

		for sent in doc.sents:

			words_ = [(str(x).lower(),i) for i,x in enumerate(sent) if str(x).lower() in content_words]
			# extract content words and their index for every sentence

			if words_:
				for i in range(len(words_)):
					for j in range(len(words_)):
						if j > i:
							if (words_[j][1] - words_[i][1]) >= 4:
							# if content words are present, check that their distance is at least 4

								if (words_[i][0],words_[j][0]) in Q_human.keys():
									Q_human_score += Q_human[(words_[i][0],words_[j][0])]
									Q_human_count += 1
								elif (words_[j][0],words_[i][0]) in Q_human.keys():
									Q_human_score += Q_human[(words_[j][0],words_[i][0])]
									Q_human_count += 1
								# if content word pair is known, add its score to the human state
								else:
									Q_human_novel += 1
								# if content word pair is unknown, add to novel count

								if (words_[i][0],words_[j][0]) in Q_machine.keys():
									Q_machine_score += Q_machine[(words_[i][0],words_[j][0])]
									Q_machine_count += 1
								elif (words_[j][0],words_[i][0]) in Q_machine.keys():
									Q_machine_score += Q_machine[(words_[j][0],words_[i][0])]
									Q_machine_count += 1
								# if content word pair is known, add its score to machine state
								else:
									Q_machine_novel += 1
								# if content word pair is unknown, add to novel count

		if (Q_human_count == 0) or (Q_machine_count == 0):
			features_ = [0,0,0,0]
		else:
			human_Q = Q_human_score / Q_human_count
			machine_Q = Q_machine_score / Q_machine_count
			human_novelty = Q_human_novel / (Q_human_count + Q_human_novel)
			machine_novelty = Q_machine_novel / (Q_machine_count + Q_machine_novel)
			features_ = [human_Q,machine_Q,human_novelty,machine_novelty]


		features.append(features_)
	
	return features

def main():
	os.chdir(os.path.dirname(__file__))
	print(os.path.dirname(__file__), __file__)

	for split in SPLITS:
		df = pd.read_csv(os.path.join(DS_PATH, split + '.csv')) 	
		texts = list(df['Text'])
		
		with open(os.path.join(SAVE_PATH, 'features', f'H_{split}_QStats.pkl'), 'rb') as f:
			q_human = dict(pickle.load(f))
		
		with open(os.path.join(SAVE_PATH, 'features', f'M_{split}_QStats.pkl'), 'rb') as f:
			q_machine = dict(pickle.load(f))
		
		q_feat = extract_q(texts, Q_human=q_human, Q_machine=q_machine)	
		with open(os.path.join(SAVE_PATH, 'features', f'{split}_QFT.pkl'), 'wb') as f:
			pickle.dump(q_feat, f)


if __name__ == "__main__":
	main()
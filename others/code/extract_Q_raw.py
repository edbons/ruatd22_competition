import spacy
nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))  # not in ru_core_news_lg

import regex as re
import os, json, time, sys
import numpy as np
import pickle

# function to import splits of different datasets
import _load

##############################################################################################################################

# collect arguments
if sys.argv[1] == 'single':

	if len(sys.argv) != 5:
		print('Expecting 4 arguments: single, model, dataset, filtered')
		sys.exit(1)

	model = sys.argv[2]
	size = sys.argv[3] # the features which are extracted (either equal human- or machine-text)
	filtered = sys.argv[4]

elif sys.argv[1] == 'multi':

	if len(sys.argv) != 4:
		print('Expecting 3 arguments: single, combi, source')
		sys.exit(1)

	combi = sys.argv[2]
	source = sys.argv[3]

if sys.argv[1] == 'single':

	if filtered == 'False':

		data_path = os.path.join('.','Data','data','full_text')
		save_path = os.path.join('.','Data','Q','full_text')
  
		texts = _load._load_split(os.path.join(data_path,model,size),size,'train')

	elif filtered == 'True':

		data_path = os.path.join('.','Data','data','full_text')
		filter_path = os.path.join('.','Data','data','filtered_text')
		save_path = os.path.join('.','Data','Q','filtered_text')

		filters = np.load(os.path.join(filter_path,model,size,'{}filtered_texts_{}.npy'.format(size,'train')))
		texts_ = _load._load_split(os.path.join(text_path,model,size),size,'train')
		texts = [text for i,text in enumerate(texts_) if filters[i]==True]

elif sys.argv[1] == 'multi':

	save_path = os.path.join('.','Data','Q','full_text','Combinations')
	texts = _load._load_combi(combi,source,'train')


##############################################################################################################################
print('Starting extraction of Q-stats...')

content_words = np.load(os.path.join('.','Data','google5000.npy'))  

sentences = 0
pairs_distanced_ = {}
single_counts_dict = dict.fromkeys(content_words,0) # THIS
single_counts_ = np.zeros(5000)

start_ = time.time()
checkpoint_ = time.time()

for i_, text in enumerate(texts):

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

	if (i_ % (np.round(len(texts)/100,0))) == 0:
		print('{:4d}% finished --- Total Time: {:8.2f} --- Checkpoint Time: {:8.2f}'.format(int(100*(i_/len(texts))),time.time()-start_,time.time()-checkpoint_))
		checkpoint_ = time.time()

	# ---------- Loop Finished

single_counts_final = dict(zip(content_words,single_counts_))
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

if sys.argv[1] == 'single':

	if not os.path.exists(os.path.join(save_path,model,size)):
		os.makedirs(os.path.join(save_path,model,size))
	
	np.save(os.path.join(save_path,model,size,'{}_QStats.npy'.format(size)),Q_stats)
	
elif sys.argv[2] == 'multi':

	if not os.path.exists(os.path.join(save_path,combi)):
		os.makedirs(os.path.join(save_path,combi))

	np.save(os.path.join(save_path,combi,'{}_{}_QStats.npy'.format(source,combi)),Q_stats)
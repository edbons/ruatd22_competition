import spacy
nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))

import regex as re
import os, json, time, sys
import numpy as np
import pickle

# function to import splits of different datasets
import _load
#######################################################################################################################################

# collect arguments
if sys.argv[1] == 'single':

	if len(sys.argv) != 8:
		print('Expecting 7 arguments: single, model, size, human_text, machine_text, split, filtered')
		sys.exit(1)

	model = sys.argv[2]
	size = sys.argv[3] # the features which are extracted (either equal human- or machine-text)
	human_text = sys.argv[4]
	machine_text =  sys.argv[5]
	split = sys.argv[6]
	filtered = sys.argv[7]

elif sys.argv[1] == 'multi':

	if len(sys.argv) != 5:
		print('Expecting 4 arguments: single, combi, source, split')
		sys.exit(1)

	combi = sys.argv[2]
	source = sys.argv[3]
	split = sys.argv[4]

else:
	print('Please make sure to enter either single or multi as your first argument')
	sys.exit(1)

# set paths, load files
if sys.argv[1] == 'single':

	if filtered == 'False':

		data_path = os.path.join('.','Data','data','full_text')
		Q_path = os.path.join('.','Data','Q','full_text')

		# load texts
		texts = _load._load_split(os.path.join(data_path,model,size),size,split)

		# load Q statistics			
		Q_human = dict(np.load(os.path.join(Q_path,model,human_text,'{}_QStats.npy'.format(human_text)),allow_pickle=True))
		Q_machine = dict(np.load(os.path.join(Q_path,model,machine_text,'{}_QStats.npy'.format(machine_text)),allow_pickle=True))

	elif filtered == 'True':

		data_path = os.path.join('.','Data','data','full_text')
		Q_path = os.path.join('.','Data','Q','filtered_text')
		filter_path = os.path.join('.','Data','data','filtered_text')

		# load texts
		filters = np.load(os.path.join(filter_path,model,size,'{}filtered_texts_{}.npy'.format(size,split)), allow_pickle=True)
		texts_ = _load._load_split(os.path.join(data_path,model,size),size,split)
		texts = [text for i,text in enumerate(texts_) if filters[i]==True]

		# load Q statistics
		Q_human = dict(np.load(os.path.join(Q_path,model,human_text,'{}_QStats.npy'.format(human_text)),allow_pickle=True))
		Q_machine = dict(np.load(os.path.join(Q_path,model,machine_text,'{}_QStats.npy'.format(machine_text)),allow_pickle=True))

elif sys.argv[1] == 'multi':

	Q_path = os.path.join('.','Data','Q','full_text','Combinations')

	# load texts
	texts = _load._load_combi(combi,source,split)

	# load Q statistics
	Q_human = dict(np.load(os.path.join(Q_path,combi,'human_{}_QStats.npy'.format(combi)),allow_pickle=True))
	Q_machine = dict(np.load(os.path.join(Q_path,combi,'machine_{}_QStats.npy'.format(combi)),allow_pickle=True))


#######################################################################################################################################

# prepare lists
features = []
content_words = np.load(os.path.join('.','Data','google5000.npy'))

# extraction
print('Extracting Q...')
start_ = time.time()
checkpoint_ = time.time()

for i_, text in enumerate(texts):

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

# append Q-stats of given text
	features.append(features_)

# callback
	if (i_ % (np.round(len(texts)/100,0))) == 0:
		print('{:4d}% finished --- Total Time: {:8.2f} --- Checkpoint Time: {:8.2f}'.format(int(100*(i_/len(texts))),time.time()-start_,time.time()-checkpoint_))
		checkpoint_ = time.time()

# save according to extracted dataset
if sys.argv[1] == 'single':

	if size == machine_text:
		np.save(os.path.join(Q_path,model,machine_text,'{}_QFT_{}.npy'.format(machine_text,split)),features)
	elif size == human_text:
		np.save(os.path.join(Q_path,model,human_text,'{}_{}_QFT_{}.npy'.format(human_text,machine_text,split)),features)

elif sys.argv[1] == 'multi':

	np.save(os.path.join(Q_path,combi,'{}_{}_QFT_{}.npy'.format(source,combi,split)),features)
# Code to download (and prepare) GPT-2, GPT-3 and Grover samples
# download.py creates single-datasets
# download.py True creates single-datasets and multi-datasets
# download.py True True creates single-datasets, multi-datasets and filtered versions

import os, sys
import requests
from tqdm import tqdm
import json
import numpy as np
import regex as re

# SINGLE DATASETS

# Function to import splits of different datasets
import _load

# Collect arguments
if len(sys.argv) == 1:
	create_multi = 'True'
	create_filter = 'True'
elif len(sys.argv) == 2:
	create_multi = sys.argv[1]
elif len(sys.argv) == 3:
	create_multi = sys.argv[1]
	create_filter = sys.argv[2]
elif len(sys.argv) > 3:
	print('Too many arguments. Expected 2: multi, filter')
	sys.exit(1)

if create_multi == 'False' and create_filter == 'False':
	print('Downloading single-datasets')
elif create_multi == 'True' and create_filter == 'True':
	print('Downloading single-datasets, creating multi- and filtered-versions')
elif create_multi == 'True' and create_filter == 'False':
	print('Downloading single-datasets and creating multi-version')
elif create_multi == 'False' and create_filter == 'True':
	print('Downloading single-datasets and creating filtered-versions')

# Data URLs
data_url = {'gpt2': 'https://storage.googleapis.com/gpt-2/output-dataset/v1/', 
		'gpt3': 'https://raw.githubusercontent.com/openai/gpt-3/master/175b_samples.jsonl',
		'grover': 'https://storage.googleapis.com/grover-models/generation_examples/generator=mega~dataset=p0.94.jsonl'}

# Create folder for data
save_path = os.path.join('.','Data','data','full_text')

if not os.path.exists(save_path):
	os.makedirs(save_path)

# Get GPT-2 data

print('Getting GPT-2 data...')

for ds in ['webtext','small-117M','small-117M-k40','xl-1542M','xl-1542M-k40']:
	if not os.path.exists(os.path.join(save_path,'GPT2',ds)):
		os.makedirs(os.path.join(save_path,'GPT2',ds))
	for split in ['train', 'valid', 'test']:
		filename = ds + "." + split + '.jsonl'
		r = requests.get(data_url['gpt2'] + filename, stream=True)

		with open(os.path.join(save_path,'GPT2',ds,filename), 'wb') as f:
			file_size = int(r.headers["content-length"])
			chunk_size = 1000
			with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
			# 1k for chunk_size, since Ethernet packet size is around 1500 bytes
				for chunk in r.iter_content(chunk_size=chunk_size):
					f.write(chunk)
					pbar.update(chunk_size)

print('...done getting GPT-2 data!')

# Get GPT-3 data

print('Getting GPT-3 data...')

if not os.path.exists(os.path.join(save_path,'GPT3')):
	os.makedirs(os.path.join(save_path,'GPT3'))
	
g = requests.get(data_url['gpt3'])

with open(os.path.join(save_path,'GPT3','gpt3Raw.jsonl'), 'wb') as f:
	chunk_size = 1000
	for chunk in g.iter_content(chunk_size=chunk_size):
		f.write(chunk)

# Prepare GPT-3 data

print('...processing GPT-3 data...')

texts = []

for line in open(os.path.join(save_path,'GPT3','gpt3Raw.jsonl'), encoding='utf-8'):
	texts.append(json.loads(line))
	
gpt3_machine = []

for text in texts:
	splits = re.split(r'<\|endoftext\|>',text)
	for split in splits:
		gpt3_machine.append(split)

# split dataset into train, test and valid (80%train,10%test,10%valid)

	np.random.seed(175)

	ixs = np.arange(0,len(gpt3_machine))
	n_train = int(len(gpt3_machine) * 0.8)

	_train = np.random.choice(ixs,n_train,replace=False)
	_remainder = [ix for ix in ixs if not ix in _train]
	_valid = np.random.choice(_remainder,int(len(_remainder)*0.5),replace=False)
	_test = [ix for ix in _remainder if not ix in _valid]

	gpt3_train = []
	gpt3_valid = []
	gpt3_test  = []

	for i, text in enumerate(gpt3_machine):
		if len(text) > 5:
			if i in _train:
				gpt3_train.append(text)
			elif i in _valid:
				gpt3_valid.append(text)
			else:
				gpt3_test.append(text)

# load webtext samples to create GPT3_webtext, matching webtext samples with length of GPT3 datasets

path = os.path.join(save_path,'GPT2','webtext')

webtext_train = _load._load_split(path,'webtext','train')
webtext_valid = _load._load_split(path,'webtext','valid')
webtext_test  = _load._load_split(path,'webtext','test')

# gpt3 train: 1604, valid: 201, test: 201

train_ix = np.random.choice(np.arange(0,len(webtext_train)), len(gpt3_train), replace=False)
webtext_train_new = [text for i,text in enumerate(webtext_train) if i in train_ix]

valid_ix = np.random.choice(np.arange(0,len(webtext_valid)), len(gpt3_valid), replace=False)
webtext_valid_new = [text for i,text in enumerate(webtext_valid) if i in valid_ix]

test_ix = np.random.choice(np.arange(0,len(webtext_test)), len(gpt3_test), replace=False)
webtext_test_new = [text for i,text in enumerate(webtext_test) if i in test_ix]

# save gpt3 and gpt3_webtext 
def save_gpt3(name,split,file,save_path=save_path):
	gpt3_path = os.path.join(save_path,'GPT3')
	if not os.path.exists(os.path.join(gpt3_path,name)):
		os.makedirs(os.path.join(gpt3_path,name))
	with open(os.path.join(gpt3_path,name,name+'.'+split+'.jsonl'),'w') as outfile:
		json.dump(file,outfile)
		
save_gpt3('175B','train',gpt3_train)
save_gpt3('175B','valid',gpt3_valid)
save_gpt3('175B','test',gpt3_test)

print('test 1')

save_gpt3('GPT3_webtext','train',list(webtext_train_new))
save_gpt3('GPT3_webtext','valid',list(webtext_valid_new))
print('test 2')
save_gpt3('GPT3_webtext','test',list(webtext_test_new))

print('test 3')

os.remove(os.path.join(save_path,'GPT3','gpt3Raw.jsonl'))

print('...done getting GPT-3 data!')

# Get Grover data

print('Getting Grover data...')

if not os.path.exists(os.path.join(save_path,'Grover')):
	os.makedirs(os.path.join(save_path,'Grover'))
	
g = requests.get(data_url['grover'])
	
with open(os.path.join(save_path,'Grover','groverRaw.jsonl'), 'wb') as f:
	chunk_size = 1000
	for chunk in g.iter_content(chunk_size=chunk_size):
		f.write(chunk)

# Prepare Grover data

print('...processing Grover data...')

texts = []

for line in open(os.path.join(save_path,'Grover','groverRaw.jsonl')):
	texts.append(json.loads(line))
	
grover_human = []
grover_machine = []

for text in texts:
	if text['label'] == 'human':
		grover_human.append(text['article'])
	elif text['label'] == 'machine':
		grover_machine.append(text['article'])

# split datasets into train, test and valid (80/10/10)

np.random.seed(175)

ixs = np.arange(0,len(grover_human))
n_train = int(len(grover_human) * 0.8)

_train = np.random.choice(ixs,n_train,replace=False)
_remainder = [ix for ix in ixs if not ix in _train]
_valid = np.random.choice(_remainder,int(len(_remainder)*0.5),replace=False)
_test = [ix for ix in _remainder if not ix in _valid]

realNews_train = []
realNews_valid = []
realNews_test  = []

for i, text in enumerate(grover_human):
	if i in _train:
		realNews_train.append(text)
	elif i in _valid:
		realNews_valid.append(text)
	else:
		realNews_test.append(text)

# groverMega (grover_machine) - introduce splits (80%train,10%test,10%valid)
ixs = np.arange(0,len(grover_machine))
n_train = int(len(grover_machine) * 0.8)

_train = np.random.choice(ixs,n_train,replace=False)
_remainder = [ix for ix in ixs if not ix in _train]
_valid = np.random.choice(_remainder,int(len(_remainder)*0.5),replace=False)
_test = [ix for ix in _remainder if not ix in _valid]

groverMega_train = []
groverMega_valid = []
groverMega_test  = []

for i, text in enumerate(grover_machine):
	if len(text) > 5:
		if i in _train:
			groverMega_train.append(text)
		elif i in _valid:
			groverMega_valid.append(text)
		else:
			groverMega_test.append(text)


# separate realNews_solo data (matching number of human samples and machine generations for Grover experiments)

train_index = np.arange(0,len(realNews_train))
new_train_index = np.random.choice(train_index, 8000, replace=False)
realNews_solo_train = [text for i, text in enumerate(realNews_train) if i in new_train_index]

valid_index = np.arange(0,len(realNews_valid))
new_valid_index = np.random.choice(valid_index, 1000, replace=False)
realNews_solo_valid = [text for i, text in enumerate(realNews_valid) if i in new_valid_index]

test_index = np.arange(0,len(realNews_test))
new_test_index = np.random.choice(test_index, 1000, replace=False)
realNews_solo_test = [text for i, text in enumerate(realNews_test) if i in new_test_index]

# save grover texts

def save_grover(name,split,file,save_path=save_path):
	grover_path = os.path.join(save_path,'Grover')
	if not os.path.exists(os.path.join(grover_path,name)):
		os.makedirs(os.path.join(grover_path,name))
	with open(os.path.join(grover_path,name,name+'.'+split+'.jsonl'),'w') as outfile:
		json.dump(file,outfile)
		
save_grover('realNews','train',realNews_train)
save_grover('realNews','valid',realNews_valid)
save_grover('realNews','test',realNews_test)

save_grover('realNews_solo','train',realNews_solo_train)
save_grover('realNews_solo','valid',realNews_solo_valid)
save_grover('realNews_solo','test',realNews_solo_test)

save_grover('groverMega','train',groverMega_train)
save_grover('groverMega','valid',groverMega_valid)
save_grover('groverMega','test',groverMega_test)

if os.path.exists(os.path.join(save_path,'Grover','groverRaw.jsonl')):
	os.remove(os.path.join(save_path,'Grover','groverRaw.jsonl'))

print('...done getting Grover data!')


# MULTI DATASETS
if create_multi == 'True':

	print('Creating multi-datasets...')

	import spacy
	nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
	nlp.add_pipe(nlp.create_pipe('sentencizer'))

	import time, pickle

	combi_path = './Data/data/full_text/Combinations'

	# create folders
	for combi in ['superGPT2','superGPT2_un','superGPT2_k','superAll']:
		if not os.path.exists(os.path.join(combi_path,combi)):
			os.makedirs(os.path.join(combi_path,combi))

	np.random.seed(175)

# super GPT2: 62500 (train), 1250 (valid,test) per GPT2 dataset
	superGPT2 = {'human': {'train': {}, 'valid': {}, 'test': {}}, 'machine':{'train': {}, 'valid': {}, 'test': {}}}

	superGPT2['human']['train']['webtext'] = np.arange(0,250000)
	superGPT2['human']['valid']['webtext'] = np.arange(0,5000)
	superGPT2['human']['test']['webtext'] = np.arange(0,5000)

	superGPT2['machine']['train']['small-117M'] = np.random.choice(np.arange(0,250000),62500,replace=False)
	superGPT2['machine']['train']['small-117M-k40'] = np.random.choice(np.arange(0,250000),62500,replace=False)
	superGPT2['machine']['train']['xl-1542M'] = np.random.choice(np.arange(0,250000),62500,replace=False)
	superGPT2['machine']['train']['xl-1542M-k40'] = np.random.choice(np.arange(0,250000),62500,replace=False)

	superGPT2['machine']['valid']['small-117M'] = np.random.choice(np.arange(0,5000),1250,replace=False)
	superGPT2['machine']['valid']['small-117M-k40'] = np.random.choice(np.arange(0,5000),1250,replace=False)
	superGPT2['machine']['valid']['xl-1542M'] = np.random.choice(np.arange(0,5000),1250,replace=False)
	superGPT2['machine']['valid']['xl-1542M-k40'] = np.random.choice(np.arange(0,5000),1250,replace=False)

	superGPT2['machine']['test']['small-117M'] = np.random.choice(np.arange(0,5000),1250,replace=False)
	superGPT2['machine']['test']['small-117M-k40'] = np.random.choice(np.arange(0,5000),1250,replace=False)
	superGPT2['machine']['test']['xl-1542M'] = np.random.choice(np.arange(0,5000),1250,replace=False)
	superGPT2['machine']['test']['xl-1542M-k40'] = np.random.choice(np.arange(0,5000),1250,replace=False)

	with open(os.path.join(combi_path,'superGPT2','superGPT2_ix.pkl'),'wb') as pickle_out:
	  pickle.dump(superGPT2, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)

# super GPT2 un: 250000 (train), 2500 (valid,test) per GPT2 untruncated dataset
	superGPT2_un = {'human': {'train': {}, 'valid': {}, 'test': {}}, 'machine':{'train': {}, 'valid': {}, 'test': {}}}

	superGPT2_un['human']['train']['webtext'] = np.arange(0,250000)
	superGPT2_un['human']['valid']['webtext'] = np.arange(0,5000)
	superGPT2_un['human']['test']['webtext'] = np.arange(0,5000)

	superGPT2_un['machine']['train']['small-117M'] = np.random.choice(np.arange(0,250000),125000,replace=False)
	superGPT2_un['machine']['train']['xl-1542M'] = np.random.choice(np.arange(0,250000),125000,replace=False)

	superGPT2_un['machine']['valid']['small-117M'] = np.random.choice(np.arange(0,5000),2500,replace=False)
	superGPT2_un['machine']['valid']['xl-1542M'] = np.random.choice(np.arange(0,5000),2500,replace=False)

	superGPT2_un['machine']['test']['small-117M'] = np.random.choice(np.arange(0,5000),2500,replace=False)
	superGPT2_un['machine']['test']['xl-1542M'] = np.random.choice(np.arange(0,5000),2500,replace=False)

	with open(os.path.join(combi_path,'superGPT2_un','superGPT2_un_ix.pkl'),'wb') as pickle_out:
	  pickle.dump(superGPT2_un, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)

# super GPT2 k: 250000 (train), 2500 (valid,test) per GPT2 top-k dataset
	superGPT2_k = {'human': {'train': {}, 'valid': {}, 'test': {}}, 'machine':{'train': {}, 'valid': {}, 'test': {}}}

	superGPT2_k['human']['train']['webtext'] = np.arange(0,250000)
	superGPT2_k['human']['valid']['webtext'] = np.arange(0,5000)
	superGPT2_k['human']['test']['webtext'] = np.arange(0,5000)

	superGPT2_k['machine']['train']['small-117M-k40'] = np.random.choice(np.arange(0,250000),125000,replace=False)
	superGPT2_k['machine']['train']['xl-1542M-k40'] = np.random.choice(np.arange(0,250000),125000,replace=False)

	superGPT2_k['machine']['valid']['small-117M-k40'] = np.random.choice(np.arange(0,5000),2500,replace=False)
	superGPT2_k['machine']['valid']['xl-1542M-k40'] = np.random.choice(np.arange(0,5000),2500,replace=False)

	superGPT2_k['machine']['test']['small-117M-k40'] = np.random.choice(np.arange(0,5000),2500,replace=False)
	superGPT2_k['machine']['test']['xl-1542M-k40'] = np.random.choice(np.arange(0,5000),2500,replace=False)

	with open(os.path.join(combi_path,'superGPT2_k','superGPT2_k_ix.pkl'),'wb') as pickle_out:
	  pickle.dump(superGPT2_k, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)

# super All: 
# human: 236936 (train), 3299 (valid,test) per GPT2 dataset, 1604 (train), 201 (valid,test) GPT3, 12000 (train), 1500 (valid,test) Grover
# machine: 600099 (train), 950 (valid,test) per GPT2 dataset, 1604 (train), 201 (valid,test) GPT3, 8000 (train), 1000 (valid,test) Grover
	superALL = {'human': {'train': {}, 'valid': {}, 'test': {}}, 'machine': {'train': {}, 'valid': {}, 'test': {}}}

	superALL['human']['train']['webtext'] = np.random.choice(np.arange(0,250000),236396,replace=False)
	superALL['human']['train']['realNews'] = np.arange(0,12000)
	superALL['human']['train']['GPT3_webtext'] = np.arange(0,1604)
	superALL['human']['valid']['webtext'] = np.random.choice(np.arange(0,5000),3299,replace=False)
	superALL['human']['valid']['realNews'] = np.arange(0,1500)
	superALL['human']['valid']['GPT3_webtext'] = np.arange(0,201)
	superALL['human']['test']['webtext'] = np.random.choice(np.arange(0,5000),3299,replace=False)
	superALL['human']['test']['realNews'] = np.arange(0,1500)
	superALL['human']['test']['GPT3_webtext'] = np.arange(0,201)

	superALL['machine']['train']['small-117M'] = np.random.choice(np.arange(0,250000),60099,replace=False)
	superALL['machine']['train']['small-117M-k40'] = np.random.choice(np.arange(0,250000),60099,replace=False)
	superALL['machine']['train']['xl-1542M'] = np.random.choice(np.arange(0,250000),60099,replace=False)
	superALL['machine']['train']['xl-1542M-k40'] = np.random.choice(np.arange(0,250000),60099,replace=False)
	superALL['machine']['train']['175B'] = np.arange(0,1604)
	superALL['machine']['train']['groverMega'] = np.arange(0,8000)

	superALL['machine']['valid']['small-117M'] = np.random.choice(np.arange(0,5000),950,replace=False)
	superALL['machine']['valid']['small-117M-k40'] = np.random.choice(np.arange(0,5000),950,replace=False)
	superALL['machine']['valid']['xl-1542M'] = np.random.choice(np.arange(0,5000),950,replace=False)
	superALL['machine']['valid']['xl-1542M-k40'] = np.random.choice(np.arange(0,5000),949,replace=False)
	superALL['machine']['valid']['175B'] = np.arange(0,201)
	superALL['machine']['valid']['groverMega'] = np.arange(0,1000)

	superALL['machine']['test']['small-117M'] = np.random.choice(np.arange(0,5000),950,replace=False)
	superALL['machine']['test']['small-117M-k40'] = np.random.choice(np.arange(0,5000),950,replace=False)
	superALL['machine']['test']['xl-1542M'] = np.random.choice(np.arange(0,5000),950,replace=False)
	superALL['machine']['test']['xl-1542M-k40'] = np.random.choice(np.arange(0,5000),949,replace=False)
	superALL['machine']['test']['175B'] = np.arange(0,201)
	superALL['machine']['test']['groverMega'] = np.arange(0,1000)

	with open(os.path.join(combi_path,'superAll','superAll_ix.pkl'),'wb') as pickle_out:
	  pickle.dump(superALL, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)

	print('...done creating multi-datasets!')


# FILTER DATASETS
if create_filter == 'True':

	print('Creating filtered datasets...')

	import tokenizers
	from tokenizers import BertWordPieceTokenizer
	tokenizer = BertWordPieceTokenizer(os.path.join('.','Data','bert-base-uncased-vocab.txt'), lowercase=True)

	from langdetect import detect_langs

	import os, json
	import pandas as pd
	import time

	# function that returns vector of filter-information
	def get_filters(text):
		
		tokens = 1 if len(tokenizer.encode(text)) <= 192 else 0 # number of wordpiece tokens
		curly = 1 if (text.find('{') > -1) or (text.find('}') > -1) else 0 # curly brackets
		cookies = 1 if (text.find('cookies') > -1) or (text.find('Cookies') > -1) or (text.find('COOKIES') > -1) else 0 # 'cookies' or 'Cookies' or 'COOKIES'
		javascript = 1 if (text.find('javascript') > -1) or (text.find('Javascript') > -1) else 0 # 'javascript' or 'Javascript'
		try:
			language = 1 if (detect_langs(text)[0].lang != 'en') or (detect_langs(text)[0].prob < 0.99) else 0 # language not english with reasonable doubt
		except:
			language = 0
		numbers = 1 if len(re.findall(r'\d+',text))/len(text) > 0.03 else 0 # excessive use of numbers
		punctuation = 1 if len(re.findall(r'[.,:;?!-\"\(\)\[\]]',text))/len(text) > 0.07 else 0 # excessive use of punctuation
		formatting = 1 if len(re.findall(r'\n',text))/len(text) > 0.08 else 0 # excessive use of formatting
		
		return [tokens,curly,cookies,javascript,language,numbers,punctuation,formatting]


	datas = [('GPT2','webtext'),('GPT2','small-117M'),('GPT2','small-117M-k40'),('GPT2','xl-1542M'),('GPT2','xl-1542M-k40'),
			 ('GPT3','GPT3_webtext'),('GPT3','175B'),
			 ('Grover','realNews'),('Grover','realNews_solo'),('Grover','groverMega')]

	splits = ['train','valid','test']

	data_path = os.path.join('.','Data','data','full_text')
	save_path = os.path.join('.','Data','data','filtered_text')

	for data in datas:
		for split in splits:
			
			# load texts
			texts = _load._load_split(os.path.join(data_path,data[0],data[1]),data[1],split)
			
			# create empty table
			filter_table = np.zeros((len(texts),8))
			
			# fill table
			for i, text in enumerate(texts):
				filter_table[i] = get_filters(text)
				
			# create vector of 'good' texts
			filtered = np.sum(filter_table, axis=1) == 0
			
			# save results
			if not os.path.exists(os.path.join(save_path,data[0],data[1])):
				os.makedirs(os.path.join(save_path,data[0],data[1]))
			np.save(os.path.join(save_path,data[0],data[1]),'{}filtered_texts_{}.npy'.format(data[1],split),filtered)

	print('...done creating filtered datasets!')
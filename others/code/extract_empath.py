import os, json, sys
import numpy as np
import time
import regex as re

import _load

# feature: Empath word categories

from empath import Empath
lexicon = Empath()

# add new tailored categories to empath

lexicon.create_category('spatial',['short','long','circular','small','big','large','huge','gigantic','tiny','rectangular','rectangle','massive',
                                  'giant','enormous','smallish','rounded','middle','oval','sized','size','miniature','circle',
                                  'colossal','center','triangular','shape','boxy','round','shaped','dimensioned'])

lexicon.create_category('sentiment',['good','bad','nice','annoying','formidable','superb','abysmal','mean','wonderful',
                                    'great','terrible','horrendous','awful','dreadful','neat','fantastic','terrific'])

lexicon.create_category('opinion',['think','feel','mean','argue','reason','say','state','mind','believe','suggest','proves',
                                   'although','find','opinion','guess','deem','consider'])

lexicon.create_category('logic',['logical','rational','reasonable','justified','reasoned','obvious','coherent','consistent',
                                'coherent','legitimate','valid','plausible'])

lexicon.create_category('ethic',['ethical','right','morally','decent','indecent','unethical','honorable','honourable',
                                'value','virtue','honest','legal','legitimate','illegitimate','right','wrong','vice',
                                'norm','immoral'])

# take arguments
if len(sys.argv) != 4:
    print('Expecting 3 arguments: model, dataset, split')
    sys.exit(1)

model = sys.argv[1]
size = sys.argv[2]
split = sys.argv[3]

################################################################################################################################################

def extract_empath(text):
    
    wordlist = [str(x).lower() for x in re.findall(r'\'*[a-zA-Z]\w*',text)]
    empath_ = list(lexicon.analyze(text).values())
    
    return empath_

data_path = os.path.join('.','Data','data','full_text')
save_path = os.path.join('.','Data','features','full_text')

texts = _load._load_split(os.path.join(data_path,model,size),size,split)

empath = []
print('Starting extraction of empath stats...')
        
for i, text in enumerate(texts):
    
    empath.append(extract_empath(text))
            

print('... empath extracted!')
if not os.path.exists(os.path.join(save_path,model,size)):
    os.makedirs(os.path.join(save_path,model,size))
    
np.save(os.path.join(save_path,model,size,'{}_empath_{}.npy'.format(size,split)),empath)
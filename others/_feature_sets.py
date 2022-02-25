import pandas as pd
import numpy as np
from numpy.linalg import svd, norm
import re
import textstat
textstat.set_lang('ru')

RU_WORDS_PATTERN = re.compile(r'[a-zA-Zа-яА-Я]+')

def difficult_words(text, syllable_threshold=2):

    def is_difficult_word(word, syllable_threshold=2):
        if textstat.syllable_count(word) < syllable_threshold:
            return False
        return True

    words = set(re.findall(r"[\w\='‘’]+", text.lower()))
    diff_words = [word for word in words
                  if is_difficult_word(word, syllable_threshold)]

    return len(diff_words)


def get_basic(doc, raw, stop_words, google10000):

    # basic abs and basic rel

    wordlist = [str(x).lower() for x in re.findall(RU_WORDS_PATTERN, raw)]
    lenlist = np.array([len(x) for x in re.findall(RU_WORDS_PATTERN, raw)])
    raw = raw.replace("—", "-")
    punctuation_ = re.findall(r'[.,:;?!-\"\(\)\[\]\n]', raw)

    sentlist_ = []
    for sent in doc.sents:
        sentlist_.append(
            len([x for x in re.findall(RU_WORDS_PATTERN, str(sent))]))

    # number of (non-empty) sentences
    sentences = np.sum(np.array(sentlist_) > 0)
    words = len(wordlist)
    sylls = textstat.syllable_count(raw)
    chars = np.sum(lenlist) + len(punctuation_)
    # dwords = textstat.difficult_words(raw)  # TODO: There is no easy words vocabulary for ru, use en
    # number of difficult words (>3 syllables)
    dwords = difficult_words(raw, syllable_threshold=4)

    swords = np.sum(lenlist < 4)
    lwords = np.sum(lenlist > 6)

    stopwords_ = 0
    for word in stop_words:
        stopwords_ += wordlist.count(word)

    uniquewords_ = 0
    uniquewords_ = len(np.unique(wordlist))

    basic_abs = np.array(
        [chars, sylls, words, sentences, dwords, swords, lwords])

    if words == 0:
        return basic_abs, np.zeros(6), [-21.43, -15.8, 0, 206.835, 0, 0, 0, 0, 3.1291], np.zeros(5), words, sentences, punctuation_, wordlist

    basic_rel = np.array([chars/words, sylls/words, words /
                         sentences, dwords/words, swords/words, lwords/words])

    # readability

    ARI = 4.71 * (chars/words) + 0.5 * (words/sentences) - 21.43
    CLI = 0.0588 * (chars/words) * 100 - 0.296 * (sentences/words) * 100 - 15.8
    FKGL = 0.39 * (words/sentences) + 11.8 * (sylls/words)
    FKRE = 206.835 - 1.015 * (words/sentences) - 84.6 * (sylls/words)
    GFOG = 0.4 * ((words/sentences) + 100 * (dwords/words))
    LIX = (words/sentences) + ((lwords * 100)/words)
    MCAL = (words+swords)/sentences
    RIX = (lwords/sentences)
    SMOG = 1.0430 * np.sqrt((dwords * (30/sentences))) + 3.1291

    readability = np.array([ARI, CLI, FKGL, FKRE, GFOG, LIX, MCAL, RIX, SMOG])

    # lexical diversity

    frequencies = []
    # for every word on the top-list, count the appearances in the document
    for word in google10000:
        frequencies.append(wordlist.count(word))

    # share of total words in top100, top1000 and top10000 list
    top100 = np.sum(frequencies[:100]) / words
    top1000 = np.sum(frequencies[:1000]) / words
    top10000 = np.sum(frequencies) / words

    lexical_div = np.array(
        [stopwords_/words, uniquewords_/words, top100, top1000, top10000])

    # return values: basic_abs (7,), basic_rel (6,), readability (9,) and lexical_div (5,) are feature-vectors,
    #                punctuation_ is a list for subsequent use,
    #                words and sentences are integers for subsequent use.
    return basic_abs, basic_rel, readability, lexical_div, words, sentences, punctuation_, wordlist


# function to extract formatting features
def get_formatting(raw, punctuation_, dict_, sentences, words):

    for p in punctuation_:
        dict_[p] += 1

    punctuation_dist = np.array(list(dict_.values())) / (np.sum(list(dict_.values())) + 1e-8)  # probability distribution over punctuation        
    punctuation_sent = np.array(list(dict_.values())) / (sentences + 1e-8) # punctuation-tokens per sentence with russian words

    pars = re.split(r'\n+', raw)
    paragraphs_ = [len([x for x in re.findall(RU_WORDS_PATTERN, par)]) for par in pars]  # paragraphs only with words (no pars with numerics only)
    del pars

    num_paragraphs = np.sum(np.array(paragraphs_) > 0)
    avglen_paragraphs = np.mean([x for x in paragraphs_])
    paragraphs = np.array([num_paragraphs, avglen_paragraphs])

    # return values: feature-vectors distribution over punctuation (11,), punctuation per sentence (11,),
    #                paragraph-features (2,)
    return punctuation_dist, punctuation_sent, paragraphs


# function to extract lexical and syntactic repetetiveness features

def get_rep(doc):

    wordlist_ = []  # word-list of previous sentence (empty for first)
    poslist_ = []

    wordlist_bi_ = []
    poslist_bi_ = []

    wordlist_tri_ = []
    poslist_tri_ = []

    overlap_word = []  # overlap
    overlap_pos = []

    overlap_word_bi = []
    overlap_pos_bi = []

    overlap_word_tri = []
    overlap_pos_tri = []

    for sent in doc.sents:

        overlap_word_ = 0
        overlap_pos_ = 0

        overlap_word_bi_ = 0
        overlap_pos_bi_ = 0

        overlap_word_tri_ = 0
        overlap_pos_tri_ = 0

        # word-list & pos-list of current sentence
        wordlist = [str(x).lower() for x in re.findall(
            RU_WORDS_PATTERN, sent.text)]  # consistent definition of word
        poslist = [str(token.pos_) for token in sent]

        # jump to next sentence if sentence is empty
        if not wordlist:
            continue

        # unigram
        for word in wordlist:
            if word in wordlist_:  # if word already in previous sentence
                overlap_word_ += 1

        for pos in poslist:
            if pos in poslist_:
                overlap_pos_ += 1

        if len(wordlist) != 0:
            # share of words that have already been in prev. sentence
            overlap_word.append(overlap_word_ / len(wordlist))

        if len(poslist) != 0:
            # share of words that have already been in prev. sentence
            overlap_pos.append(overlap_pos_ / len(poslist))

        # bigram
        wordlist_bi = []
        for i in range(len(wordlist)-1):
            wordlist_bi.append((wordlist[i], wordlist[i+1]))
            if (wordlist[i], wordlist[i+1]) in wordlist_bi_:
                overlap_word_bi_ += 1

        poslist_bi = []
        for i in range(len(poslist) - 1):
            poslist_bi.append((poslist[i], poslist[i+1]))
            if (poslist[i], poslist[i+1]) in poslist_bi_:
                overlap_pos_bi_ += 1

        if len(wordlist_bi) != 0:
            overlap_word_bi.append(overlap_word_bi_ / len(wordlist_bi))

        if len(poslist_bi) != 0:
            overlap_pos_bi.append(overlap_pos_bi_ / len(poslist_bi))

        # trigram
        wordlist_tri = []
        for i in range(len(wordlist)-2):
            wordlist_tri.append((wordlist[i], wordlist[i+1], wordlist[i+2]))
            if (wordlist[i], wordlist[i+1], wordlist[i+2]) in wordlist_tri_:
                overlap_word_tri_ += 1

        poslist_tri = []
        for i in range(len(poslist)-2):
            poslist_tri.append((poslist[i], poslist[i+1], poslist[i+2]))
            if (poslist[i], poslist[i+1], poslist[i+2]) in poslist_tri_:
                overlap_pos_tri_ += 1

        if len(wordlist_tri) != 0:
            overlap_word_tri.append(overlap_word_tri_ / len(wordlist_tri))

        if len(poslist_tri) != 0:
            overlap_pos_tri.append(overlap_pos_tri_ / len(poslist_tri))

        wordlist_ = wordlist
        poslist_ = poslist

        wordlist_bi_ = wordlist_bi
        poslist_bi_ = poslist_bi

        wordlist_tri_ = wordlist_tri
        poslist_tri_ = poslist_tri

    overlap_hist_word_ = np.histogram(
        overlap_word[1:], bins=np.arange(0, 1.1, 0.1))[0]
    overlap_hist_word = overlap_hist_word_ / (np.sum(overlap_hist_word_) + 1e-8)  # для текста с одним предложением нет пересечений слов с прошлым предложением, поэтому 0

 
    overlap_hist_pos_ = np.histogram(
        overlap_pos[1:], bins=np.arange(0, 1.1, 0.1))[0]
    overlap_hist_pos = overlap_hist_pos_ / (np.sum(overlap_hist_pos_)  + 1e-8 )

    overlap_hist_word_bi_ = np.histogram(
        overlap_word_bi[1:], bins=np.arange(0, 1.1, 0.1))[0]
    overlap_hist_word_bi = overlap_hist_word_bi_ / (np.sum(overlap_hist_word_bi_) + 1e-8)
    overlap_hist_pos_bi_ = np.histogram(
        overlap_pos_bi[1:], bins=np.arange(0, 1.1, 0.1))[0]
    overlap_hist_pos_bi = overlap_hist_pos_bi_ / (np.sum(overlap_hist_pos_bi_) + 1e-8)

    overlap_hist_word_tri_ = np.histogram(
        overlap_word_tri[1:], bins=np.arange(0, 1.1, 0.1))[0]
    overlap_hist_word_tri = overlap_hist_word_tri_ / (np.sum(overlap_hist_word_tri_) + 1e-8) 
    overlap_hist_pos_tri_ = np.histogram(
        overlap_pos_tri[1:], bins=np.arange(0, 1.1, 0.1))[0]
    overlap_hist_pos_tri = overlap_hist_pos_tri_ / (np.sum(overlap_hist_pos_tri_) + 1e-8)

    lexical_rep = np.concatenate(
        (overlap_hist_word, overlap_hist_word_bi, overlap_hist_word_tri))
    syntactic_rep = np.concatenate(
        (overlap_hist_pos, overlap_hist_pos_bi, overlap_hist_pos_tri))

    # return values: lexical overlap histograms of uni-,bi- and tri-grams (words) between consecutive sentences (30,) and
    #               syntactic overlap histograms of uni-,bi- and tri-grams (pos-tags) between consecutive sentences (30,)
    return lexical_rep, syntactic_rep


# function to extract overlap around conjunctions features
def get_conj(doc):

    conj_overlap = {'1': 0, '2': 0, '3': 0}

    for token in doc:
        if (str(token) == 'and') or (str(token) == 'And'):  # find and-conjunctions
            if not doc[token.i-1].is_punct:  # make sure direct context is no punctuation
                if not str(doc[token.i-1]) == '':  # make sure direct context is non-empty
                    if not doc[token.i-1].is_digit:  # make sure direct context is no number
                        if not str(doc[token.i-1]) == '\n':
                            # make sure direct context is no established figure-of-speech
                            if not str(doc[token.i-1]) in ['more', 'around', 'ever', 'on', 'fewer', 'over', 'lots', 'so']:

                                context_left = doc[token.i-3:token.i]
                                context_right = doc[token.i+1:token.i+4]

                                for n in [1, 2, 3]:
                                    if str(context_left[-n:]) == str(context_right[:n]):
                                        conj_overlap[str(n)] += 1

    conj_features = list(conj_overlap.values())
    # return values: number of times a [1,2,3]-word repetition occurred around a conjunction
    return np.array(conj_features)


# function to extract (POS-based) syntactic features
def get_syntactic(doc, dict_, words, sentences):

    if words == 0:
        return np.zeros(19), np.zeros(19), np.zeros(30)

    unique_dict_ = dict_.copy()
    unique_words_ = []

    for token in doc:
        dict_[token.pos_] += 1
        if str(token) not in unique_words_:
            unique_words_.append(str(token))
            unique_dict_[token.pos_] += 1

    dist_full_ = np.array(list(dict_.values()))

    dist_lite_ = np.array(list(dict_.values())[:5])  # [ADJ,ADP,ADV,NOUN,VERB]-tags
    dist_lite_unique_ = np.array(list(unique_dict_.values())[:5])

    # POS features
    type_per_doc = dist_lite_ / words
    unique_type_per_doc = dist_lite_unique_ / words
    type_per_types = dist_lite_ / (np.sum(dist_lite_) + 1e-8)
    unique_type_per_types = dist_lite_unique_ / (np.sum(dist_lite_unique_) + 1e-8)
    type_per_sent = dist_lite_ / sentences
    unique_type_per_sent = dist_lite_unique_ / sentences

    pos_features = np.concatenate((type_per_doc, unique_type_per_doc, type_per_types,
                                  unique_type_per_types, type_per_sent, unique_type_per_sent))

    # Probability distribution over POS tags
    dist_full = dist_full_ / np.sum(dist_full_)
    # POS tags per sentence
    sent_full = dist_full_ / sentences

    # return values: Probability distribution over POS-tags (19,), POS tags per sentence (19,) and POS features (30,)
    return dist_full, sent_full, pos_features


# function to extract NE-based features
def get_ne(doc, dict_, words, sentences):

    unique_ne_ = []

    # test whether text has any NEs
    if (len(doc.ents) == 0) or (words == 0):
        return np.zeros(3), np.zeros(3), np.zeros(5), [], 0

    for ent in doc.ents:
        dict_[str(ent.label_)] += 1
        if str(ent) not in unique_ne_:
            unique_ne_.append(str(ent))

    dist_full_ = np.array(list(dict_.values()))

    # Probability distribution over NEs
    ne_dist = dist_full_ / np.sum(dist_full_)
    # NEs per sentence
    ne_sent = dist_full_ / sentences

    # Number of NEs
    total_ne = np.sum(list(dict_.values()))
    # Number of unique NEs
    unique_ne = len(unique_ne_)

    # NE-features
    ne_features = np.array([unique_ne / total_ne, total_ne / words, unique_ne / words, total_ne / sentences, unique_ne / sentences])

    # return values: probability distribution over NEs (18,), NEs per sentence (18,) and NE-features (5,)
    #                unique_nes_ is a list of unique NEs in the text, total_ne is the total number of entities
    return ne_dist, ne_sent, ne_features, unique_ne_, total_ne


# function to extract coreference-based features
def get_coref(doc, words, total_ne, unique_ne_):

    clusters = doc._.coref_clusters

    # number of coreference-chains in the text
    num_chains = len(clusters)

    if num_chains == 0:
        return np.zeros(10), np.zeros(9)

    # list of all references used throughout the text
    all_references = []

    # list of unique-references-share per cluster
    unique_shares = []

    # total length of coreference-chains in the text
    total_length = 0

    # number of long chains (chains that are longer than half the document length)
    total_long_chains = 0

    # number of short inferences (inference distance <20), shorter inferences (<10), shortest inferences (5)
    short_inferences = 0
    shorter_inferences = 0
    shortest_inferences = 0
    total_inferences = 0

    for cluster in clusters:

        # span-length
        first = cluster.mentions[0].start
        last = cluster.mentions[-1].end
        length = last - first
        total_length += length
        if length > (words/2):
            total_long_chains += 1

        # number of references
        references = len(cluster)

        # number of unique references | inference distances
        unique_references_ = []
        for i in range(len(cluster.mentions)):
            # collect inference distances of mentions in cluster
            if i < (len(cluster.mentions)-1):
                inference_distance = cluster.mentions[i +
                                                      1].start - cluster.mentions[i].start
                total_inferences += 1
                if inference_distance <= 5:
                    shortest_inferences += 1
                if inference_distance <= 10:
                    shorter_inferences += 1
                if inference_distance <= 20:
                    short_inferences += 1
            # collect reference
            all_references.append(str(cluster.mentions[i]))
            # collect unique references
            if not str(cluster.mentions[i]) in unique_references_:
                unique_references_.append(str(cluster.mentions[i]))
        # number of unique references in cluster
        unique_references = len(unique_references_)

        # append share of unique references
        unique_shares.append(unique_references/references)

    # average number of coreferences per coreference-chain
    average_references = len(all_references) / num_chains

    # average span (distance between first and last coreference) of coreference-chains
    average_span = total_length / num_chains

    # share of long chains in total chains
    share_long_chains = total_long_chains / num_chains

    # share of short, shorter and shortest inferences in total inferences
    share_short_inferences = short_inferences / total_inferences
    share_shorter_inferences = shorter_inferences / total_inferences
    share_shortest_inferences = shortest_inferences / total_inferences

    # share of NEs in all references
    overlap = 0
    for ne in unique_ne_:
        for reference in all_references:
            if ne in reference:
                overlap += 1
    share_ne = overlap / len(all_references)

    # active chains per word (total span of chains / number of words)
    if words != 0:
        active_word = total_length / words
    else:
        active_word = 0

    # active chains per NE (total span of chains / number of words)
    if total_ne != 0:
        active_ne = total_length / total_ne
    else:
        active_ne = 0

    # histogram (10 bins between 0 and 1) over the per-cluster shares of unique to total references
    coref_dist_ = np.histogram(unique_shares, bins=np.arange(0, 1.1, 0.1))[0]
    coref_dist = coref_dist_ / np.sum(coref_dist_)

    # features-vector
    coref_features = np.array([average_references, average_span, share_long_chains, share_short_inferences, share_shorter_inferences,
                               share_shortest_inferences, share_ne, active_word, active_ne])

    # return values: histogram over the per-cluster shares of unique to total coreferences (10,), coref-features (9,)
    return coref_dist, coref_features


# function to extract probability distribution over entity-grid transitions
def get_grid(doc, dict_):

    clusters = doc._.coref_clusters

    # Get index-tokens spanned by every individual, non-empty sentence
    sent_ix = []
    ix = 0
    for sent in doc.sents:
        # test that sentence is not empty
        wordlist = [str(x).lower()
                    for x in re.findall(RU_WORDS_PATTERN, str(sent))]
        if wordlist:
            sent_ix.append([ix, np.arange(sent.start, sent.end)])
            ix += 1

    # ENTITIES FROM COREF-CLUSTERS
    corefs = []
    entities_tokens = []
    for cluster in clusters:

        roles = []  # role of the mention
        sentx = []  # sentence-id of the mention

        for mention in cluster.mentions:

            # add inidices of all the tokens that belong to the coreference
            entities_tokens.append(np.arange(mention.start, mention.end))

            # get role of the mention - first role that is found in multi-token mention decides about role of whole mention
            ctrl = 0
            for token in mention:
                if token.dep_ in ['csubj', 'nsubj', 'nusbjpass']:
                    mrole = 'S'
                    ctrl = 1
                    break
                elif token.dep_ in ['pobj', 'dobj']:
                    mrole = 'O'
                    ctrl = 1
                    break
            if ctrl != 1:
                mrole = 'X'
            roles.append(mrole)

            # get sentence-id of the mention
            for sent in sent_ix:
                if mention.start in sent[1]:
                    sentx.append(sent[0])

        # collect 'name' of the mention, role and sentence-id
        corefs.append([str(cluster.main), roles, sentx])
    # collect list of token-ids that have already been considered
    entities_tokens = np.array(entities_tokens).ravel()

    # ENTITIES FROM REMAINDER (extracting entities identity-based from noun chunks)

    unused = []

    for chunk in doc.noun_chunks:
        jump = 0
        for token in chunk:
            # skip if some part of the noun-chunk has already been extracted in coreference-clusters
            if token.i in entities_tokens:
                jump = 1
                continue
        if jump == 1:
            continue
        else:

            # get role of noun-chunk
            ctrl = 0
            for token in chunk:
                if token.dep_ in ['csubj', 'nsubj', 'nusbjpass']:
                    mrole = 'S'
                    ctrl = 1
                    break
                elif token.dep_ in ['pobj', 'dobj']:
                    mrole = 'O'
                    ctrl = 1
                    break
            if ctrl != 1:
                mrole = 'X'

            # get sentence-id of the noun-chunk
            for sent in sent_ix:
                if chunk.start in sent[1]:
                    sentx = sent[0]
                else:
                    sentx = None

            # problem: sent_ix empty, so that no sentx assigned in for-loop
            try:
                if sentx:
                    unused.append([str(chunk), [mrole, sentx]])
            except:
                pass

    # DE-DUPLICATE REMAINDER

    # identity-based entity-identification
    df = pd.DataFrame(unused, columns=['entity', 'info'])
    g = df.groupby('entity')

    remainder = []
    for e in g.groups.items():
        entity = e[0]
        roles = []
        sentx = []
        ixs = list(e[1].values)
        for ix in ixs:
            roles.append(df.iloc[ix, 1][0])
            sentx.append(df.iloc[ix, 1][1])
        remainder.append([entity, roles, sentx])

    # COMBINE (format: 'name of entity', list of roles, list of sentences); use only entities if no cluster has been found

    if not corefs:
        entities = remainder
    elif not remainder:
        entities = corefs
    else:
        entities = np.vstack([corefs, remainder])

    # CONSTRUCT ENTITY-GRID (rows represent sentences, columns represent entities, entries represent roles)
    # S: Subject, O: Object, X: Other, -: Not in Sentence
    sents = len([1 for sent in doc.sents])
    entities = [entity for entity in entities if len(
        entity[1]) == len(entity[2])]
    ents = len(entities)
    entity_grid = np.full((sents, ents), '-').astype(str)

    for i_y, e in enumerate(entities):
        for i_x in range(len(e[1])):
            try:
                x = int(e[2][i_x])
                y = i_y
                entity_grid[x, y] = e[1][i_x]
            except TypeError:
                continue

    # EXTRACT TRANSITION-PROBABILITY-DISTRIBUTION

    for i_x in range(entity_grid.shape[0]-1):
        for i_y in range(entity_grid.shape[1]):
            trans = str(entity_grid[i_x, i_y])+str(entity_grid[i_x+1, i_y])
            dict_[trans] += 1

    trans_dist = np.array(list(dict_.values())) / np.sum(list(dict_.values()))

    # return value: relative frequencies of the entity transitions throughout the text (16,)
    return trans_dist


# function to extract Topic Redundancy (Information Loss)
def get_redundancy(doc, wordlist, lemma=False):

    if not wordlist:  # check if text has words
        return np.zeros(9)

    if lemma == False:

        # get vocabulary (unique, excluding punctuation and formatting)
        vocabulary = np.unique(wordlist)
        vocabulary_dict = dict.fromkeys(vocabulary, 0)

        # create sentence x vocabulary frequencies-matrix ||| exclude empty sentences
        D = []
        for sent in doc.sents:
            vocabulary_dict_ = vocabulary_dict.copy()
            wordlist_sent = [str(x).lower()
                             for x in re.findall(RU_WORDS_PATTERN, str(sent))]
            # filter out empty / short sentences
            if len(wordlist_sent) < 5:
                continue
            for word in wordlist_sent:
                try:
                    vocabulary_dict_[word] += 1
                except KeyError:
                    pass
            D.append(list(vocabulary_dict_.values()))
        D = np.array(D)

    else:

        # get lemmatized vocabulary
        wordlist = []
        for token in doc:
            str_lemma = re.findall(RU_WORDS_PATTERN, str(token.lemma_))
            if not str_lemma:
                continue
            elif str_lemma[0] == 'PRON':
                word = str(token).lower()
            else:
                word = str_lemma[0]
            wordlist.append(word)

        vocabulary = np.unique(wordlist)
        vocabulary_dict = dict.fromkeys(vocabulary, 0)

        # create sentence x vocabulary frequencies-matrix for lemmatized vocabulary
        D = []
        for sent in doc.sents:
            vocabulary_dict_ = vocabulary_dict.copy()
            wordlist_sent = []
            for token in sent:
                str_lemma = re.findall(RU_WORDS_PATTERN, str(token.lemma_))
                if not str_lemma:
                    continue
                elif str_lemma[0] == 'PRON':
                    word = str(token).lower()
                else:
                    word = str_lemma[0]
                wordlist_sent.append(word)
            # filter out empty sentences
            if len(wordlist_sent) < 5:
                continue
            for word in wordlist_sent:
                try:
                    vocabulary_dict_[word] += 1
                except KeyError:
                    pass
            D.append(list(vocabulary_dict_.values()))
        D = np.array(D)

    # check if sentences in text
    if D.size == 0:
        return np.zeros(9)

    # transpose to vocabulary x sentence frequencies-matrix
    D_T = np.transpose(D)

    # calculate A; sentence x sentence matrix
    A = np.dot(D, D_T)

    # perform svd on A
    U, S, k = svd(A)
    n = len(S)
    k = int(n * 0.25)  # determine k (25% of values)
    S[n-k:] = 0  # set k least-important components (eigen-values) to 0

    # re-construct truncated A
    U_T = np.transpose(U)
    S_ = np.diag(S)
    A_ = np.dot(np.dot(U, S_), U_T)

    # calculate the information loss between A and its reconstruction A_
    A_diff = A - A_
    information_loss = norm(A_diff) ** 2

    # calculate mean, median, max and min of the reconstructed matrix A_
    A_mean = np.mean(A_)
    A_median = np.median(A_)
    A_max = np.max(A_)
    A_min = np.min(A_)

    # calculate difference in these statistics between A and A_
    A_mean_diff = np.mean(A) - A_mean
    A_median_diff = np.median(A) - A_median
    A_max_diff = np.max(A) - A_max
    A_min_diff = np.min(A) - A_min

    redundancy_features = np.array([information_loss, A_mean, A_median,
                                   A_max, A_min, A_mean_diff, A_median_diff, A_max_diff, A_min_diff])

    # return values: information loss feature-vector (9,)
    return redundancy_features


# function to extract empath-features, based on the 200 individual category-scores
def get_empath(empath_, words):

    empath = np.array(empath_)

    # share of topical words in total words
    empath_score = np.sum(empath) / words

    # mean, median, min, max and variance of topicality score
    empath_mean = np.mean(empath)
    empath_median = np.median(empath)
    empath_min = np.min(empath)
    empath_max = np.max(empath)
    empath_var = np.var(empath)

    # empath scores for tailored categories - normalised to text-length
    empath_tailored_ = empath[-5:]
    empath_tailored = empath_tailored_ / words

    # empath_active -> empath categories that are > 0
    empath_active = empath[empath > 0]

    # number of active categories
    empath_act = len(empath_active)

    # empath features
    empath_features = np.array(
        [empath_score, empath_mean, empath_median, empath_min, empath_max, empath_var, empath_act])

    if not empath_active.any():
        return empath_features, np.zeros(5), empath_tailored

    # mean, median, min, max and variance of active categories
    empath_active_mean = np.mean(empath_active)
    empath_active_median = np.median(empath_active)
    empath_active_min = np.min(empath_active)
    empath_active_max = np.max(empath_active)
    empath_active_var = np.var(empath_active)

    # empath-active features
    empath_active_features = np.array([empath_active_mean, empath_active_median, empath_active_min, empath_active_max,
                                       empath_active_var])

    # return values: empath-features (7,), empath-features for active categories (5,), empath values for tailored
    #                categories (5,)
    return empath_features, empath_active_features, empath_tailored

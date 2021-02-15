import sys

sys.path.append('../')
from collections import defaultdict
import re
from gensim.models.keyedvectors import KeyedVectors
#from gensim.models.wrappers import FastText
import numpy
from nltk.tokenize import TweetTokenizer
import data_preprocessing.glove2Word2vec2Loader as glove
import itertools


# loading the emoji dataset
def load_unicode_mapping(path):
    emoji_dict = defaultdict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split('\t')
            emoji_dict[tokens[0]] = tokens[1]
    return emoji_dict


#def load_word2vec(path=None):
#    word2vecmodel = KeyedVectors.load_word2vec_format(path, binary=True)
#    return word2vecmodel


#def load_fasttext(path=None):
#    word2vecmodel = FastText.load_fasttext_format(path)
#    return word2vecmodel

def normalize_word(word):
    temp = word
    while True:
        w = re.sub(r"([a-zA-Z])\1\1", r"\1\1", temp)
        if (w == temp):
            break
        else:
            temp = w
    return w

def load_split_word(split_word_file_path):
    split_word_dictionary = defaultdict()
    with open(split_word_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.lower().strip().split('\t')
            if (len(tokens) >= 2):
                split_word_dictionary[tokens[0]] = tokens[1]

    #print('split entry found:', len(split_word_dictionary.keys()))
    return split_word_dictionary


def load_abbreviation(path='../TEXT_FILES/abbreviations.txt'):
    abbreviation_dict = defaultdict()
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            token = line.lower().strip().split('\t')
            abbreviation_dict[token[0]] = token[1]
    return abbreviation_dict


def filter_text(text,split_word_list, emoji_dict, abbreviation_dict, normalize_text=False,
                ignore_profiles=False,
                replace_emoji=True):
    filtered_text = []

    filter_list = ['/', '-', '=', '+', '…', '\\', '(', ')', '&', ':']

    for t in text:
        word_tokens = None

        # discarding symbols
        # if (str(t).lower() in filter_list):
        #     continue

        # ignoring profile information if ignore_profiles is set
        if (ignore_profiles and str(t).startswith("@")):
            continue

        # ignoring links
        if (str(t).startswith('http')):
            continue
        # replacing emoji with its unicode description
        if (replace_emoji):
            if (t in emoji_dict):
                t = emoji_dict.get(t).split('_')
                filtered_text.extend(t)
                continue

        # splitting hastags
        #if (split_hashtag and str(t).startswith("#")):
        #    splits = split_hashtags(t, word_list, split_word_list, dump_file='../TEXT_FILES/hastash_split_dump.txt')
        #    # adding the hashtags
        #    if (splits != None):
        #        filtered_text.extend([s for s in splits if (not filtered_text.__contains__(s))])
        #        continue

        # removes repeatation of letters
        if (normalize_text):
            t = normalize_word(t)

        # expands the abbreviation
        if (t in abbreviation_dict):
            tokens = abbreviation_dict.get(t).split(' ')
            filtered_text.extend(tokens)
            continue

        # appends the text
        filtered_text.append(t)

    return filtered_text


def parsedata(lines,split_word_list, emoji_dict, abbreviation_dict, normalize_text=False,
              ignore_profiles=False,
              lowercase=False, replace_emoji=True, n_grams=None, at_character=False):
    data = []
    out_dict={}
    ini_point = 0
    names= (lines[0].split('\t'))[0]
    for i, line in enumerate(lines):
        if (i % 100 == 0):
            print(str(i) + '...', end='', flush=True)

        try:

            # convert the line to lowercase
            if (lowercase):
                line = line.lower()

            # split into token
            token = line.split('\t')
            
            # ID
            id = token[0]
            if line[0]=='\n':
              out_dict[names] = [ini_point,i-1]
              if i+1<=len(lines)-1:
                names= (lines[i+1].split('\t'))[0] 
                ini_point = i+1  
             
              
            #print(id)
            # label
            #label = (token[1].strip())
            #print(label)
            # tweet text
            #print(token[1])
            if line[0]!='\n':
              target_text = TweetTokenizer().tokenize(token[1].strip())
              if (at_character):
                  target_text = [c for c in token[1].strip()]

              if (n_grams != None):
                  n_grams_list = list(create_ngram_set(target_text, ngram_value=n_grams))
                  target_text.extend(['_'.join(n) for n in n_grams_list])

            # filter text
              target_text = filter_text(target_text,split_word_list, emoji_dict, abbreviation_dict,
                                      normalize_text,
                                      ignore_profiles, replace_emoji=replace_emoji)

            # awc dimensions
            #dimensions = []
            #if (len(token) > 3 and token[3].strip() != 'NA'):
            #    dimensions = [dimension.split('@@')[1] for dimension in token[3].strip().split('|')]

            # context tweet
            #context = []
            #if (len(token) > 4):
            #    if (token[4] != 'NA'):
            #        context = TweetTokenizer().tokenize(token[4].strip())
            #        context = filter_text(context, word_list, split_word_list, emoji_dict, abbreviation_dict,
            #                              normalize_text,
            #                              split_hashtag,
            #                              ignore_profiles, replace_emoji=replace_emoji)

            # author
            #author = 'NA'
            #if (len(token) > 5):
            #    author = token[5]

              if (len(target_text) != 0):
                # print((label, target_text, dimensions, context, author))
                  data.append((id, target_text))
        except:
            raise
    #print(out_dict)
    print('')
    return data,out_dict


def load_resources(split_word_path, emoji_file_path, replace_emoji=True):
    #word_list = None
    emoji_dict = None

    # load split files
    split_word_list = load_split_word(split_word_path)

    if (replace_emoji):
        emoji_dict = load_unicode_mapping(emoji_file_path)

    abbreviation_dict = load_abbreviation()

    return emoji_dict, split_word_list, abbreviation_dict


def loaddata(filename,split_word_path, emoji_file_path, normalize_text=False,
             ignore_profiles=False,
             lowercase=True, replace_emoji=True, n_grams=None, at_character=False):

    emoji_dict, split_word_list, abbreviation_dict = load_resources(split_word_path,
                                                                               emoji_file_path,
                                                                               replace_emoji=replace_emoji)
    lines = open(filename, 'r').readlines()

    data,dictionary = parsedata(lines,split_word_list, emoji_dict, abbreviation_dict, normalize_text=normalize_text,
                     ignore_profiles=ignore_profiles, lowercase=lowercase, replace_emoji=replace_emoji,
                     n_grams=n_grams, at_character=at_character)
    return data,dictionary


def build_vocab(data, ignore_context=False, min_freq=0):
    vocab = defaultdict(int)
    vocab_freq = defaultdict(int)

    total_words = 1
    #if (not without_dimension):
    #    for i in range(1, 101):
    #        vocab_freq[str(i)] = 0
            # vocab[str(i)] = total_words
            # total_words = total_words + 1

    #for sentence_no, token in enumerate(data):
    #    for word in token[2]:
    #        if (word not in vocab_freq):
                # vocab[word] = total_words
                # total_words = total_words + 1
    #            vocab_freq[word] = 0
    #        vocab_freq[word] = vocab_freq.get(word) + 1

    #    if (not without_dimension):
    #        for word in token[3]:
                # if (word not in vocab_freq):
                #     vocab[word] = total_words
                #     total_words = total_words + 1
    #            vocab_freq[word] = vocab_freq.get(word) + 1

    #    if (ignore_context == False):
    #        for word in token[4]:
    #            if (not word in vocab):
                    # vocab[word] = total_words
                    # total_words = total_words + 1
    #                vocab_freq[word] = 0
    #            vocab_freq[word] = vocab_freq.get(word) + 1

    for k, v in vocab_freq.items():
        if (v >= min_freq):
            vocab[k] = total_words
            total_words = total_words + 1

    return vocab


#def build_reverse_vocab(vocab):
#    rev_vocab = defaultdict(str)
#    for k, v in vocab.items():
#        rev_vocab[v] = k
#    return rev_vocab



def vectorize_word_dimension(data, vocab, verbose=False):
    X = []
    #Y = []
    #D = []
    #C = []
    #A = []

    known_words_set = set()
    unknown_words_set = set()

    tokens = 0
    token_coverage = 0

    for id, line in data:
        vec = []
        #context_vec = []
        #if (len(dimensions) != 0):
        #    dvec = [vocab.get(d) for d in dimensions]
        #else:
        #    dvec = [vocab.get('unk')] * 11

        #if drop_dimension_index != None:
        #    dvec.pop(drop_dimension_index)

        # tweet
        for words in line:
            tokens = tokens + 1
            if (words in vocab):
                vec.append(vocab[words])
                token_coverage = token_coverage + 1
                known_words_set.add(words)
            else:
                vec.append(vocab['unk'])
                unknown_words_set.add(words)
        # context_tweet
        #if (len(context) != 0):
        #    for words in line:
        #        tokens = tokens + 1
        #        if (words in vocab):
        #            context_vec.append(vocab[words])
        #            token_coverage = token_coverage + 1
        #            known_words_set.add(words)
        #        else:
        #            context_vec.append(vocab['unk'])
        #            unknown_words_set.add(words)
        #else:
        #    context_vec = [vocab['unk']]

        X.append(vec)
        #Y.append(label)
        #D.append(dvec)
        #C.append(context_vec)
        #A.append(author)

    if verbose:
        print('Token coverage:', token_coverage / float(tokens))
        print('Word coverage:', len(known_words_set) / float(len(vocab.keys())))

    return numpy.asarray(X)


def pad_sequence_1d(sequences, maxlen=None, dtype='float32', padding='pre', truncating='pre', value=0.):
    X = [vectors for vectors in sequences]

    nb_samples = len(X)

    x = (numpy.zeros((nb_samples, maxlen)) * value).astype(dtype)

    for idx, s in enumerate(X):
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)

    return x


def write_vocab(filepath, vocab):
    with open(filepath, 'w') as fw:
        for key, value in vocab.items():
            fw.write(str(key) + '\t' + str(value) + '\n')


#def get_fasttext_weight(vocab, n=300, path=None):
#    word2vecmodel = load_word2vec(path=path)
#    emb_weights = numpy.zeros((len(vocab.keys()) + 1, n))
#    for k, v in vocab.items():
#        if (word2vecmodel.__contains__(k)):
#            emb_weights[v, :] = word2vecmodel[k][:n]

#    return emb_weights


#def get_word2vec_weight(vocab, n=300, path=None):
#    word2vecmodel = load_word2vec(path=path)
#    emb_weights = numpy.zeros((len(vocab.keys()) + 1, n))
#    for k, v in vocab.items():
#        if (word2vecmodel.__contains__(k)):
#            emb_weights[v, :] = word2vecmodel[k][:n]

#    return emb_weights


def load_glove_model(vocab, n=400, glove_path='/home/glove/glove.twitter.27B/glove.twitter.27B.200d.txt'):
    word2vecmodel = glove.load_glove_word2vec(glove_path)

    embedding_matrix = numpy.zeros((len(vocab.keys()) + 1, n))
    for k, v in vocab.items():
        embedding_vector = word2vecmodel.get(k)
        if embedding_vector is not None:
            embedding_matrix[v] = embedding_vector

    return embedding_matrix


#def add_ngram(sequences, token_indice, ngram_range=2):
#    """
#    Augment the input list of list (sequences) by appending n-grams values.
#    Example: adding bi-gram
#    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
#    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
#    >>> add_ngram(sequences, token_indice, ngram_range=2)
#    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
#    Example: adding tri-gram
#    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
#    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
#    >>> add_ngram(sequences, token_indice, ngram_range=3)
#    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
#    """
#    new_sequences = []
#   for input_list in sequences:
#        new_list = input_list[:]
#        for i in range(len(new_list) - ngram_range + 1):
#            for ngram_value in range(2, ngram_range + 1):
#                ngram = tuple(new_list[i:i + ngram_value])
#                if ngram in token_indice:
#                    new_list.append(token_indice[ngram])
#        new_sequences.append(new_list)

#    return new_sequences


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))



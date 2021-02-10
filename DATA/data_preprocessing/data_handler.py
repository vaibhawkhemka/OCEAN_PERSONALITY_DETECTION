import sys

sys.path.append('../')
from collections import defaultdict
import re
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.wrappers import FastText
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


def load_word2vec(path=None):
    word2vecmodel = KeyedVectors.load_word2vec_format(path, binary=True)
    return word2vecmodel


def load_fasttext(path=None):
    word2vecmodel = FastText.load_fasttext_format(path)
    return word2vecmodel


def InitializeWords(word_file_path):
    word_dictionary = defaultdict()

    with open(word_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.lower().strip().split('\t')
            word_dictionary[tokens[0]] = int(tokens[1])

    for alphabet in "bcdefghjklmnopqrstuvwxyz":
        if (alphabet in word_dictionary):
            word_dictionary.__delitem__(alphabet)

    for word in ['ann', 'assis',
                 'bz',
                 'ch', 'cre', 'ct',
                 'di',
                 'ed', 'ee',
                 'ic',
                 'le',
                 'ng', 'ns',
                 'pr', 'picon',
                 'th', 'tle', 'tl', 'tr',
                 'um',
                 've',
                 'yi'
                 ]:
        if (word in word_dictionary):
            word_dictionary.__delitem__(word)

    return word_dictionary


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


def split_hashtags(term, wordlist, split_word_list, dump_file=''):
    # print('term::',term)

    if (len(term.strip()) == 1):
        return ['']

    if (split_word_list != None and term.lower() in split_word_list):
        # print('found')
        return split_word_list.get(term.lower()).split(' ')
    #else:
        #print(term)

    # discarding # if exists
    if (term.startswith('#')):
        term = term[1:]

    if (wordlist != None and term.lower() in wordlist):
        return [term.lower()]

    words = []
    # max freq
    penalty = -69971
    max_coverage = penalty

    split_words_count = 6
    # checking camel cases
    term = re.sub(r'([0-9]+)', r' \1', term)
    term = re.sub(r'(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|0th)', r'\1 ', term)
    term = re.sub(r'([A-Z][^A-Z ]+)', r' \1', term.strip())
    term = re.sub(r'([A-Z]{2,})+', r' \1', term)
    words = term.strip().split(' ')

    n_splits = 0

    if (len(words) < 3):
        # splitting lower case and uppercase words upto 5 words
        chars = [c for c in term.lower()]

        found_all_words = False

        while (n_splits < split_words_count and not found_all_words):
            for idx in itertools.combinations(range(0, len(chars)), n_splits):
                output = numpy.split(chars, idx)
                line = [''.join(o) for o in output]

                score = (1. / len(line)) * sum(
                    [wordlist.get(
                        word.strip()) if word.strip() in wordlist else 0. if word.strip().isnumeric() else penalty for
                     word in line])

                if (score > max_coverage):
                    words = line
                    max_coverage = score

                    line_is_valid_word = [word.strip() in wordlist if not word.isnumeric() else True for word in line]

                    if (all(line_is_valid_word)):
                        found_all_words = True

                    # uncomment to debug hashtag splitting
                    # print(line, score, line_is_valid_word)

            n_splits = n_splits + 1

    # removing hashtag sign
    words = [str(s) for s in words]

    # dumping splits for debug
    with open(dump_file, 'a') as f:
        if (term != '' and len(words) > 0):
            f.write('#' + str(term).strip() + '\t' + ' '.join(words) + '\t' + str(n_splits) + '\n')

    return words


def load_abbreviation(path='../TEXT_FILES/abbreviations.txt'):
    abbreviation_dict = defaultdict()
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            token = line.lower().strip().split('\t')
            abbreviation_dict[token[0]] = token[1]
    return abbreviation_dict


def filter_text(text, word_list, split_word_list, emoji_dict, abbreviation_dict, normalize_text=False,
                split_hashtag=False,
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

        # ignoring sarcastic marker
        # uncomment the following line for Fracking sarcasm using neural network
        # if (str(t).lower() in ['#sarcasm','#sarcastic', '#yeahright','#not']):
        #     continue

        # for onlinesarcasm
        # comment if you are running the code for Fracking sarcasm using neural network
        if (str(t).lower() in ['#sarcasm']):
            continue

        # replacing emoji with its unicode description
        if (replace_emoji):
            if (t in emoji_dict):
                t = emoji_dict.get(t).split('_')
                filtered_text.extend(t)
                continue

        # splitting hastags
        if (split_hashtag and str(t).startswith("#")):
            splits = split_hashtags(t, word_list, split_word_list, dump_file='../TEXT_FILES/hastash_split_dump.txt')
            # adding the hashtags
            if (splits != None):
                filtered_text.extend([s for s in splits if (not filtered_text.__contains__(s))])
                continue

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


def parsedata(lines, word_list, split_word_list, emoji_dict, abbreviation_dict, normalize_text=False,
              split_hashtag=False,
              ignore_profiles=False,
              lowercase=False, replace_emoji=True, n_grams=None, at_character=False):
    data = []
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

            # label
            label = (token[1].strip())
            #print(token[2])
            # tweet text
            target_text = TweetTokenizer().tokenize(token[2].strip())
            if (at_character):
                target_text = [c for c in token[2].strip()]

            if (n_grams != None):
                n_grams_list = list(create_ngram_set(target_text, ngram_value=n_grams))
                target_text.extend(['_'.join(n) for n in n_grams_list])

            # filter text
            target_text = filter_text(target_text, word_list, split_word_list, emoji_dict, abbreviation_dict,
                                      normalize_text,
                                      split_hashtag,
                                      ignore_profiles, replace_emoji=replace_emoji)

            # awc dimensions
            dimensions = []
            if (len(token) > 3 and token[3].strip() != 'NA'):
                dimensions = [dimension.split('@@')[1] for dimension in token[3].strip().split('|')]

            # context tweet
            context = []
            if (len(token) > 4):
                if (token[4] != 'NA'):
                    context = TweetTokenizer().tokenize(token[4].strip())
                    context = filter_text(context, word_list, split_word_list, emoji_dict, abbreviation_dict,
                                          normalize_text,
                                          split_hashtag,
                                          ignore_profiles, replace_emoji=replace_emoji)

            # author
            author = 'NA'
            if (len(token) > 5):
                author = token[5]

            if (len(target_text) != 0):
                # print((label, target_text, dimensions, context, author))
                data.append((id, label, target_text, dimensions, context, author))
        except:
            raise
    print('')
    return data


def load_resources(word_file_path, split_word_path, emoji_file_path, split_hashtag=False, replace_emoji=True):
    word_list = None
    emoji_dict = None

    # load split files
    split_word_list = load_split_word(split_word_path)

    # load word dictionary
    if (split_hashtag):
        word_list = InitializeWords(word_file_path)

    if (replace_emoji):
        emoji_dict = load_unicode_mapping(emoji_file_path)

    abbreviation_dict = load_abbreviation()

    return word_list, emoji_dict, split_word_list, abbreviation_dict


def loaddata(filename, word_file_path, split_word_path, emoji_file_path, normalize_text=False, split_hashtag=False,
             ignore_profiles=False,
             lowercase=True, replace_emoji=True, n_grams=None, at_character=False):

    word_list, emoji_dict, split_word_list, abbreviation_dict = load_resources(word_file_path, split_word_path,
                                                                               emoji_file_path,
                                                                               split_hashtag=split_hashtag,
                                                                               replace_emoji=replace_emoji)
    lines = open(filename, 'r').readlines()

    data = parsedata(lines, word_list, split_word_list, emoji_dict, abbreviation_dict, normalize_text=normalize_text,
                     split_hashtag=split_hashtag,
                     ignore_profiles=ignore_profiles, lowercase=lowercase, replace_emoji=replace_emoji,
                     n_grams=n_grams, at_character=at_character)
    return data


def build_vocab(data, without_dimension=True, ignore_context=False, min_freq=0):
    vocab = defaultdict(int)
    vocab_freq = defaultdict(int)

    total_words = 1
    if (not without_dimension):
        for i in range(1, 101):
            vocab_freq[str(i)] = 0
            # vocab[str(i)] = total_words
            # total_words = total_words + 1

    for sentence_no, token in enumerate(data):
        for word in token[2]:
            if (word not in vocab_freq):
                # vocab[word] = total_words
                # total_words = total_words + 1
                vocab_freq[word] = 0
            vocab_freq[word] = vocab_freq.get(word) + 1

        if (not without_dimension):
            for word in token[3]:
                # if (word not in vocab_freq):
                #     vocab[word] = total_words
                #     total_words = total_words + 1
                vocab_freq[word] = vocab_freq.get(word) + 1

        if (ignore_context == False):
            for word in token[4]:
                if (not word in vocab):
                    # vocab[word] = total_words
                    # total_words = total_words + 1
                    vocab_freq[word] = 0
                vocab_freq[word] = vocab_freq.get(word) + 1

    for k, v in vocab_freq.items():
        if (v >= min_freq):
            vocab[k] = total_words
            total_words = total_words + 1

    return vocab


def build_reverse_vocab(vocab):
    rev_vocab = defaultdict(str)
    for k, v in vocab.items():
        rev_vocab[v] = k
    return rev_vocab



def vectorize_word_dimension(data, vocab, drop_dimension_index=None, verbose=False):
    X = []
    Y = []
    D = []
    C = []
    A = []

    known_words_set = set()
    unknown_words_set = set()

    tokens = 0
    token_coverage = 0

    for id, label, line, dimensions, context, author in data:
        vec = []
        context_vec = []
        if (len(dimensions) != 0):
            dvec = [vocab.get(d) for d in dimensions]
        else:
            dvec = [vocab.get('unk')] * 11

        if drop_dimension_index != None:
            dvec.pop(drop_dimension_index)

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
        if (len(context) != 0):
            for words in line:
                tokens = tokens + 1
                if (words in vocab):
                    context_vec.append(vocab[words])
                    token_coverage = token_coverage + 1
                    known_words_set.add(words)
                else:
                    context_vec.append(vocab['unk'])
                    unknown_words_set.add(words)
        else:
            context_vec = [vocab['unk']]

        X.append(vec)
        Y.append(label)
        D.append(dvec)
        C.append(context_vec)
        A.append(author)

    if verbose:
        print('Token coverage:', token_coverage / float(tokens))
        print('Word coverage:', len(known_words_set) / float(len(vocab.keys())))

    return numpy.asarray(X), numpy.asarray(Y), numpy.asarray(D), numpy.asarray(C), numpy.asarray(A)


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


def get_fasttext_weight(vocab, n=300, path=None):
    word2vecmodel = load_word2vec(path=path)
    emb_weights = numpy.zeros((len(vocab.keys()) + 1, n))
    for k, v in vocab.items():
        if (word2vecmodel.__contains__(k)):
            emb_weights[v, :] = word2vecmodel[k][:n]

    return emb_weights


def get_word2vec_weight(vocab, n=300, path=None):
    word2vecmodel = load_word2vec(path=path)
    emb_weights = numpy.zeros((len(vocab.keys()) + 1, n))
    for k, v in vocab.items():
        if (word2vecmodel.__contains__(k)):
            emb_weights[v, :] = word2vecmodel[k][:n]

    return emb_weights


def load_glove_model(vocab, n=200, glove_path='/home/glove/glove.twitter.27B/glove.twitter.27B.200d.txt'):
    word2vecmodel = glove.load_glove_word2vec(glove_path)

    embedding_matrix = numpy.zeros((len(vocab.keys()) + 1, n))
    for k, v in vocab.items():
        embedding_vector = word2vecmodel.get(k)
        if embedding_vector is not None:
            embedding_matrix[v] = embedding_vector

    return embedding_matrix


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))



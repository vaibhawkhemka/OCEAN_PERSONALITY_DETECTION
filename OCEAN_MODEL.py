import tensorflow as tf
import os
import sys
import numpy as np
from data_processing.data_handler import load_glove_model

sys.path.append('../')

import collections
import time
import numpy
import matplotlib.pyplot as plt
from keras import backend as K

from keras import backend as K, regularizers
from sklearn import metrics
from keras.models import model_from_json, load_model
from keras.layers.core import Dropout, Dense, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.models import model_from_json
from keras.layers.merge import concatenate, multiply
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Input, Reshape, Permute, RepeatVector, Lambda, merge
import src.data_processing.data_handler as dh
from collections import defaultdict


class sarcasm_model():
    _train_file = None
    _test_file = None
    _tweet_file = None
    _output_file = None
    _model_file_path = None
    _word_file_path = None
    _split_word_file_path = None
    _emoji_file_path = None
    _vocab_file_path = None
    _input_weight_file_path = None
    _vocab = None
    _line_maxlen = None

    def __init__(self):
        self._line_maxlen = 400
    def attention_3d_block(self, inputs, SINGLE_ATTENTION_VECTOR=False):
        # inputs.shape = (batch_size, time_steps, input_dim)
        input_dim = int(inputs.shape[2])
        a = Permute((2, 1))(inputs)
        a = Reshape((input_dim, self._line_maxlen))(a)
        # this line is not useful. It's just to know which dimension is what.
        a = Dense(self._line_maxlen, activation='softmax')(a)
        if SINGLE_ATTENTION_VECTOR:
            a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
        return output_attention_mul

    def _build_network(self, vocab_size, maxlen, emb_weights=[], embedding_dimension=30, hidden_units=256,
                       batch_size=1):
        print('Build model...')

        text_input = Input(name='text', shape=(maxlen,))

        if (len(emb_weights) == 0):
            emb = Embedding(vocab_size, embedding_dimension, input_length=maxlen,
                            embeddings_initializer='glorot_normal',
                            trainable=True)(text_input)
        else:
            emb = Embedding(vocab_size, emb_weights.shape[1], input_length=maxlen, weights=[emb_weights],
                            trainable=False)(text_input)
        
        emb_dropout = Dropout(0.5)(emb)
       
        lstm_bwd = LSTM(hidden_units, kernel_initializer='he_normal', activation='sigmoid', dropout=0.4,
                        go_backwards=True, return_sequences=True)(emb_dropout)
        lstm_fwd = LSTM(hidden_units, kernel_initializer='he_normal', activation='sigmoid', dropout=0.4,
                        return_sequences=True)(emb_dropout)

        lstm_merged = concatenate([lstm_bwd, lstm_fwd])

        attention_mul = self.attention_3d_block(lstm_merged)

        flat_attention = Flatten()(attention_mul)

        reshaped = Reshape((-1, 1))(flat_attention)

        cnn1 = Convolution1D(hidden_units, 3, kernel_initializer='he_normal', padding='valid', activation='relu')(
            reshaped)
        pool1 = MaxPooling1D(pool_size=3)(cnn1)
        

        cnn2 = Convolution1D(2 * hidden_units, 3, kernel_initializer='he_normal', padding='valid', activation='relu')(
            pool1)
        pool2 = MaxPooling1D(pool_size=3)(cnn2)
        

        flat_cnn = Flatten()(pool2)

        dnn_1 = Dense(hidden_units)(flat_cnn)
        dropout_1 = Dropout(0.25)(dnn_1)
    
        dnn_2 = Dense(5)(dropout_1)
        

        softmax = Activation('softmax')(dnn_2)

        
        model = Model(inputs=text_input, outputs=softmax)

        adam = tf.keras.optimizers.Adam(learning_rate=0.1)

        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        print('No of parameter:', model.count_params())

        print(model.summary())

        return model


class train_model(sarcasm_model):
    train = None
    print("Loading resource...")

    def __init__(self, train_file, word_file_path, split_word_path, emoji_file_path, model_file,
                 vocab_file,
                 output_file,
                 input_weight_file_path=None):
        sarcasm_model.__init__(self)

        self._train_file = train_file
        self._word_file_path = word_file_path
        self._split_word_file_path = split_word_path
        self._emoji_file_path = emoji_file_path
        self._model_file = model_file
        self._vocab_file_path = vocab_file
        self._output_file = output_file
        self._input_weight_file_path = input_weight_file_path

        self.load_train_data()

        
        batch_size = 32

        # build vocabulary
        # truncates words with min freq=1
        
        self._vocab = dh.build_vocab(self.train, min_freq=1)
        if ('unk' not in self._vocab):
            self._vocab['unk'] = len(self._vocab.keys()) + 1

        print(len(self._vocab.keys()) + 1)
        print('unk::', self._vocab['unk'])

        dh.write_vocab(self._vocab_file_path, self._vocab)

        self.train = self.train[:-(len(self.train) % batch_size)]
        
        # prepares input
        X, Y, D, C, A = dh.vectorize_word_dimension(self.train, self._vocab)
        X = dh.pad_sequence_1d(X, maxlen=self._line_maxlen)
        # embedding dimension
        dimension_size = 300
        emb_weights = load_glove_model(self._vocab, n=dimension_size,
                                       glove_path='DATA/glove.6B.300d.txt')

        LABEL = []
        for l in Y:
          m = []
          for b in str(l):
               m.append(int(b))
          if len(m)!=5:
            o = 5 - len(m)
            m= [0]*o + m       
          LABEL.append(m)
        Y = np.asarray(LABEL)

        # trainable true if you want word2vec weights to be updated
        # Not applicable in this code
        model = self._build_network(len(self._vocab.keys()) + 1, self._line_maxlen, emb_weights, hidden_units=32,
                                    embedding_dimension=dimension_size, batch_size=batch_size)
        model.fit(X, Y, batch_size=batch_size, epochs=5,
                  shuffle=True)
        model_json = model.to_json()
        with open(model_file + 'model.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(model_file + 'model.json.hdf5')
        print("Saved model to disk")          


    def load_train_data(self):
        self.train = dh.loaddata(self._train_file, self._word_file_path, self._split_word_file_path,
                                 self._emoji_file_path, normalize_text=True,
                                 split_hashtag=True,
                                 ignore_profiles=False)
        print('Training data loading finished...')


    def get_maxlen(self):
        return max(map(len, (x for _, x in self.train + self.validation)))

    def write_vocab(self):
        with open(self._vocab_file_path, 'w') as fw:
            for key, value in self._vocab.iteritems():
                fw.write(str(key) + '\t' + str(value) + '\n')

    def calculate_label_ratio(self, labels):
        return collections.Counter(labels)


class test_model(sarcasm_model):
    test = None
    model = None

    def __init__(self,test_file,model_file, word_file_path, split_word_path, emoji_file_path, vocab_file_path, output_file,
                 input_weight_file_path=None):
        print('initializing...')
        sarcasm_model.__init__(self)
        self._test_file = test_file
        self._model_file_path = model_file
        self._word_file_path = word_file_path
        self._split_word_file_path = split_word_path
        self._emoji_file_path = emoji_file_path
        self._vocab_file_path = vocab_file_path
        self._output_file = output_file
        self._input_weight_file_path = input_weight_file_path

        print('test_maxlen', self._line_maxlen)

    def load_trained_model(self, model_file='model.json', weight_file='model.json.hdf5'):
        start = time.time()
        self.__load_model(self._model_file_path + model_file)   
        end = time.time()
        print('model loading time::', (end - start))

    def __load_model(self, model_path):
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.summary()    
        print('model loaded from file...')
        self._load_weights(self._model_file_path,'model.json.hdf5')
    
    def _load_weights(self,weight_path,weight_file='model.json.hdf5'):
        self.model.load_weights(weight_path+weight_file)
        print("weights loaded") 

    def load_vocab(self):
        vocab = defaultdict()
        with open(self._vocab_file_path, 'r') as f:
            for line in f.readlines():
                key, value = line.split('\t')
                vocab[key] = value

        return vocab

    def test_predict(self,verbose=False):
        start = time.time()
        self.test = dh.loaddata(self._test_file, self._word_file_path, self._split_word_file_path, self._emoji_file_path,
                                    normalize_text=True, split_hashtag=True,
                                    ignore_profiles=False)
        end = time.time()
        if (verbose == True):
            print('test resource loading time::', (end - start))
 
        self._vocab = dh.build_vocab(self.test, min_freq=1)
        if ('unk' not in self._vocab):
            self._vocab['unk'] = len(self._vocab.keys()) + 1

        dh.write_vocab(self._vocab_file_path, self._vocab)
        
        tX, tY, D, C, A = dh.vectorize_word_dimension(self.test, self._vocab)
        tX = dh.pad_sequence_1d(tX, maxlen=self._line_maxlen)
      
        dimension_size = 300
        emb_weights = load_glove_model(self._vocab, n=dimension_size,
                                       glove_path='/content/SarcasmDetection/src/glove.6B.300d.txt')

        label_dict = {0:'EXTRAVERSION',1:'NEUROTICISM',2:'AGREEABLENESS',3:'CONSCIENTIOUSNESS',4:'OPENNESS'}
        predictions = self.model.predict(tX)
        total_pred = np.array([0,0,0,0,0])
        for i in predictions:
            total_pred = np.add(total_pred,np.array(i))    
        pos = np.where(total_pred==max(total_pred))
        l_pos = pos[0].tolist()
        RESULT= l_pos[0]
        print("THE RESULT IS " + str(label_dict[RESULT]))    

       
if __name__ == "__main__":
    basepath = os.getcwd()[:os.getcwd().rfind('/')]
    train_file = basepath + '/resource/train/train_data.txt'
    test_file = basepath + '/resource/test/text_data.txt'
    word_file_path = basepath + '/resource/word_list_freq.txt'
    split_word_path = basepath + '/resource/word_split.txt'
    emoji_file_path = basepath + '/resource/emoji_unicode_names_final.txt'

    output_file = basepath + '/resource/text_model/TestResults.txt'
    model_file = basepath + '/resource/text_model/weights/'
    vocab_file_path = basepath + '/resource/text_model/vocab_list.txt'

    # uncomment for training
    tr = train_model(train_file,word_file_path, split_word_path, emoji_file_path, model_file,
                     vocab_file_path, output_file)

    #t = test_model(test_file,model_file, word_file_path, split_word_path, emoji_file_path, vocab_file_path, output_file)
    #t.load_trained_model(weight_file='model.json.hdf5')
    #t.test_predict()

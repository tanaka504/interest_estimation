import CaboCha
import MeCab
import sys
from gensim.models import word2vec
import keras
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,LSTM,Conv1D,core,Merge,InputLayer,LocallyConnected1D, Bidirectional
from keras.layers.pooling import GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import Flatten, Lambda, TimeDistributed, Permute, RepeatVector, Input,Reshape
from keras.preprocessing import sequence
from keras.backend import tensorflow_backend as backend
from keras import backend as K
from keras.layers.merge import Concatenate, Average, Multiply, Add
from keras.regularizers import l2
import numpy as np
import re
from pprint import pprint
from numpy.random import *
from Embedding import *
from tensorflow import expand_dims
import tensorflow as tf
from setting import *


wv_model = word2vec.Word2Vec.load("./model_weight/w2v_jawiki_alltag.model")

def sigmoid(z):
    return ((1 / (1 + np.exp(-z))) * 2) -1

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        y_true = y_true[:,0]
        y_pred = y_pred[:,0]
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        y_true = y_true[:, 0]
        y_pred = y_pred[:, 0]
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

np.random.seed(5)
random_array = sigmoid(normal(size=200))
padding_vec = np.zeros(200,dtype=np.float32)

class LSTM_NN:
    def __init__(self,pad_size=pad_size,wv_dim=input_neurons):
        #前の文ベクトル
        e_model = Sequential()
        e_model.add(Embedding_Slice(batch_input_shape=(None, pad_size, wv_dim)))
        e_model.add(LSTM(hidden_neurons, batch_input_shape=(None, pad_size, wv_dim), return_sequences=False,dropout=0.3))
        #名詞ベクトル
        n_model = Sequential()
        n_model.add(Embedding_Slice(batch_input_shape=(None, 1, wv_dim)))
        n_model.add(Flatten())
        #後の文ベクトル
        l_model = Sequential()
        l_model.add(Embedding_Slice(batch_input_shape=(None, pad_size, wv_dim)))
        l_model.add(LSTM(hidden_neurons, batch_input_shape=(None, pad_size, wv_dim), return_sequences=False,dropout=0.3))
        #興味推定のNN
        estimate_model = Sequential()
        estimate_model.add(Merge([e_model, n_model, l_model], mode='concat',name='layer1'))
        estimate_model.add(core.Dropout(0.3))
        estimate_model.add(Dense(hidden_neurons * 2 + 250, activation='relu'))
        estimate_model.add(core.Dropout(0.3))
        estimate_model.add(Dense(100,activation='relu'))
        estimate_model.add(core.Dropout(0.3))
        estimate_model.add(Dense(output_neurons, activation='softmax'))
        adam = keras.optimizers.adam(lr=0.0001)
        estimate_model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy',f1])
        self.estimate_model = estimate_model

class Char_embed_LSTM:
    def __init__(self, pad_size=pad_size, wv_dim=1 + max_char_length, char_dict={}):
        #前の文ベクトル
        e_model = Sequential()
        e_model.add(Char_Word_Embedding(batch_input_shape=(None, pad_size, wv_dim), char_dict=char_dict))
        e_model.add(LSTM(hidden_neurons, batch_input_shape=(None, pad_size, 300), return_sequences=False,dropout=0.3))
        #名詞ベクトル
        n_model = Sequential()
        n_model.add(Char_Word_Embedding(batch_input_shape=(None, 1, wv_dim), char_dict=char_dict))
        n_model.add(Flatten())
        #後の文ベクトル
        l_model = Sequential()
        l_model.add(Char_Word_Embedding(batch_input_shape=(None, pad_size, wv_dim), char_dict=char_dict))
        l_model.add(LSTM(hidden_neurons, batch_input_shape=(None, pad_size, 300), return_sequences=False,dropout=0.3))
        #興味推定のNN
        estimate_model = Sequential()
        estimate_model.add(Merge([e_model, n_model, l_model], mode='concat',name='layer1'))
        estimate_model.add(core.Dropout(0.3))
        estimate_model.add(Dense(1000, activation='relu'))
        estimate_model.add(core.Dropout(0.3))
        estimate_model.add(Dense(200,activation='relu'))
        estimate_model.add(core.Dropout(0.3))
        estimate_model.add(Dense(output_neurons, activation='softmax'))
        adam = keras.optimizers.adam(lr=0.0001)
        estimate_model.compile(loss='mse', optimizer=adam,metrics=['accuracy',f1])
        self.estimate_model = estimate_model

class CNN:
    def __init__(self, pad_size=pad_size * 2 + 1, wv_dim=input_neurons):
        model = Sequential()
        model.add(Embedding_Slice(batch_input_shape=(None, pad_size, wv_dim)))
        model.add(Conv1D(filters=filters, kernel_size=3, batch_input_shape=(None, pad_size, 250)))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(50,activation='relu'))
        model.add(Dense(output_neurons, activation='softmax'))
        adam = keras.optimizers.adam(lr=0.0005)
        model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy',f1])
        self.model = model

class CNN_NN:
    def __init__(self,pad_size=pad_size,wv_dim=input_neurons):

        #前の文ベクトル
        e_model = Sequential(name='e_model')
        e_model.add(Embedding_Slice(batch_input_shape=(None, pad_size, wv_dim)))
        #e_model.add(Conv1D(filters=filters,kernel_size=3,batch_input_shape=(None, pad_size, wv_dim),activation='relu',padding='causal'))
        e_model.add(conv_attn(filters=filters,kernel_sizes=[3],batch_input_shape=(None, pad_size, 250)))
        #e_model.add(GlobalMaxPooling1D())

        #名詞ベクトル
        n_model = Sequential(name='n_model')
        n_model.add(Embedding_Slice(batch_input_shape=(None,1,wv_dim)))
        n_model.add(Flatten())
        #n_model.add(InputLayer(batch_input_shape=(None,wv_dim),name='nounlayer'))

        #後の文ベクトル
        l_model = Sequential(name='l_model')
        l_model.add(Embedding_Slice(batch_input_shape=(None, pad_size, wv_dim)))
        #l_model.add(Conv1D(filters=filters,kernel_size=3,batch_input_shape=(None, pad_size, wv_dim),activation='relu',padding='causal'))
        l_model.add(conv_attn(filters=filters, kernel_sizes=[3], batch_input_shape=(None, pad_size, 250)))
        #l_model.add(GlobalMaxPooling1D())

        #興味推定のNN
        estimate_model = Sequential()
        estimate_model.add(Merge([e_model, n_model, l_model], mode='concat',name='layer1'))
        estimate_model.add(core.Dropout(0.3))
        estimate_model.add(Dense(filters * 2 + 250, activation='relu'))
        estimate_model.add(core.Dropout(0.3))
        estimate_model.add(Dense(100,activation='relu'))
        estimate_model.add(core.Dropout(0.3))
        estimate_model.add(Dense(output_neurons, activation='softmax'))
        adam = keras.optimizers.adam(lr=0.0001)
        estimate_model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy',f1])
        self.estimate_model = estimate_model
        self.e_model = e_model

class context_only:
    def __init__(self,pad_size=pad_size,wv_dim=input_neurons):
        #前の文ベクトル
        e_model = Sequential()
        e_model.add(Embedding_Slice(batch_input_shape=(None, pad_size, wv_dim)))
        e_model.add(Conv1D(filters=filters,kernel_size=3,batch_input_shape=(None, pad_size, wv_dim)))
        e_model.add(GlobalAveragePooling1D())
        #後の文ベクトル
        l_model = Sequential()
        l_model.add(Embedding_Slice(batch_input_shape=(None, pad_size, wv_dim)))
        l_model.add(Conv1D(filters=filters,kernel_size=3,batch_input_shape=(None, pad_size, wv_dim)))
        l_model.add(GlobalAveragePooling1D())
        #興味推定のNN
        estimate_model = Sequential()
        estimate_model.add(Merge([e_model, l_model], mode='concat',name='layer1'))
        estimate_model.add(Dense(filters * 2, activation='relu'))
        estimate_model.add(core.Dropout(0.3))
        estimate_model.add(Dense(100,activation='relu'))
        estimate_model.add(Dense(output_neurons, activation='softmax'))
        adam = keras.optimizers.adam(lr=0.0005)
        estimate_model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=["accuracy",f1])
        self.estimate_model = estimate_model

class target_only:
    def __init__(self, pad_size=1, wv_dim=input_neurons):
        model = Sequential()
        model.add(Embedding(len(wv_model.wv.vocab)+1,200,weights=[np.vstack((wv_model.wv.syn0,random_array))],trainable=True,name='word_embedding'))
        #model.add(InputLayer(batch_input_shape=(None,200)))
        model.add(Reshape((200,)))
        model.add(Dense(200,activation='relu',name='1',batch_input_shape=(None,200)))
        model.add(core.Dropout(0.3))
        model.add(Dense(output_neurons,activation='softmax',name='2'))
        adam = keras.optimizers.adam(lr=0.0001)
        model.compile(loss='mean_squared_error', optimizer=adam,metrics=['accuracy',f1])
        self.model = model


def conv_attn(filters,kernel_sizes, batch_input_shape):
    inp = Input(shape=batch_input_shape[1:])
    convs = []
    for kernel_size in kernel_sizes:
        conv1 = Conv1D(
            filters,
            kernel_size,
            padding='causal',
            activation='relu',
            dilation_rate=1,
            #strides=1,
            batch_input_shape=batch_input_shape,
            init='uniform',
            kernel_regularizer=l2(0.00005)
        )(inp)

        unitsize = filters
        attention = TimeDistributed(Dense(1, activation='relu'))(conv1)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(unitsize)(attention)
        attention = Permute([2, 1])(attention)
        attention = Multiply()([conv1, attention])
        attention = Lambda(lambda xin: K.sum(xin, axis=-2))(attention)
        convs.append(attention)

    if len(kernel_sizes) > 1:
        out = Concatenate()(convs)
    else:
        out = convs[0]
    unitsize = len(kernel_sizes) * filters
    # out = Flatten()(out)

    return Model(input=inp, output=out)

def Embedding_Slice(batch_input_shape):

    inp = Input(shape=batch_input_shape[1:],name='embedding_input')
    word = Lambda(lambda x:x[:,:,:1])(inp)
    position = Lambda(lambda x:x[:,:,1:])(inp)

    word = Embedding(len(wv_model.wv.vocab)+2,200,weights=[np.vstack((np.vstack((wv_model.wv.syn0,random_array)),padding_vec))],trainable=True,name='word_embedding')(word)
    word = Reshape((batch_input_shape[1],200))(word)

    position = Embedding(11,50,embeddings_initializer='uniform',input_length=batch_input_shape[1],trainable=True,name='position_embedding')(position)
    position = Reshape((batch_input_shape[1],50))(position)
    out = Concatenate()([word,position])

    # out = Flatten()(out)

    return Model(input=inp, output=out)

def Char_Word_Embedding(batch_input_shape, char_dict):
    # ideal input_shape = (None, 5 or 1, 1+25)

    inp = Input(shape=batch_input_shape[1:])
    word_ids = Lambda(lambda x:x[:,:,:1])(inp)
    char_ids = Lambda(lambda x:x[:,:,1:])(inp)

    word_embeddings = Embedding(len(wv_model.wv.vocab) + 2, 200,
                     weights=[np.vstack((np.vstack((wv_model.wv.syn0, random_array)), padding_vec))], trainable=True,
                     name='word_embedding')(word_ids)
    word_embeddings = Reshape((batch_input_shape[1], 200))(word_embeddings)
    # ideal word_embedding shape = (None, 5, 200)

    char_embeddings = Embedding(len(char_dict),50,embeddings_initializer='uniform',input_length=batch_input_shape[1],trainable=True)(char_ids)
    s = K.shape(char_embeddings)
    char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, max_char_length, 50)))(char_embeddings)
    #char_embeddings = Reshape((batch_input_shape[1],50))(char_embeddings)

    fwd_state = LSTM(50, return_state=True)(char_embeddings)[-2]
    bwd_state = LSTM(50, return_state=True, go_backwards=True)(char_embeddings)[-2]
    char_embeddings = Concatenate(axis=-1)([fwd_state, bwd_state])

    char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, batch_input_shape[1], 2 * 50)))(char_embeddings)
    #char_embeddings = Reshape((batch_input_shape[1],100))(char_embeddings)

    # ideal char_embedding shape = (None, 5, 100)

    embeddings = Concatenate(axis=-1)([word_embeddings, char_embeddings])

    # ideal embeddings shape = (None, 5, 300)


    return Model(input=inp, output=embeddings)

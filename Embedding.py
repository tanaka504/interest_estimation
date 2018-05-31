import CaboCha
import MeCab
import sys
from gensim.models import word2vec
import keras
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,LSTM,Conv1D,core,Merge,InputLayer
from keras.layers.pooling import GlobalAveragePooling1D
from keras.preprocessing import sequence
from keras.backend import tensorflow_backend as backend
import numpy as np
import re
from pprint import pprint
from numpy.random import *
from setting import *
from Function import *
import pickle
import sentencepiece as spm


model = word2vec.Word2Vec.load("./model_weight/w2v_jawiki_alltag.model")
#tokenize = model.wv
index2word = model.wv.index2word
index2word.append('<unk>')
tokenize = {key:value for value,key in enumerate(index2word)}
tokenize['<pad>'] = -1
index2word.append('<pad>')

def pos_tag(pos):
    pos_dic = {'名詞': 1,
               '動詞': 2,
               '形容詞': 3,
               '副詞': 4,
               '助詞': 5,
               '接続詞': 6,
               '助動詞': 7,
               '連体詞': 8,
               '感動詞': 9,}
    if pos[0] in pos_dic.keys():
        tag = pos_dic[pos[0]]
    else:
        tag = 0

    return tag

def chk_pos(pos):
    negative_pos = set(['固有名詞','代名詞','一般'])
    if pos[0] == '名詞' and pos[1] in negative_pos:
        return 0
    else:
        return 1

def freq_tag(word):
    freq = model.wv.vocab[word].count
    if freq > 15000:
        cls = 4
    elif freq > 10000 and freq <= 15000:
        cls = 3
    elif freq > 5000 and freq <= 10000:
        cls = 2
    elif freq > 1000 and freq <= 5000:
        cls = 1
    else:
        cls = 0

    return cls

def sigmoid(z):
    return ((1 / (1 + np.exp(-z))) * 2) -1

#np.random.seed(1)
#normal_array = sigmoid(np.array(normal(size=embedding_dim), dtype=np.float32))
normal_array = tokenize['<unk>']
padding_array = tokenize['<pad>']

#target only
def W2V(text_list):
    context_vector = []
    for text in text_list:
        try:
            text_vector = tokenize[text[1][1]]
        except:
            text_vector = normal_array
        context_vector.append([np.array(text_vector,dtype=np.float32)])
    return context_vector

# Related work Char Embedding Model
def W2V2(text_list, char_dict):
    #char_dict = Char_Corpus(create_flg=True)
    sentence_vector = []
    for text in text_list:
        text_vector = []
        context_vector = []
        for earlier_word in text[0]:
            char_ids = char_dict.char2id(earlier_word[1])
            if earlier_word[1] in tokenize.keys():
                tmp_vector = np.hstack((tokenize[earlier_word[1]], np.array(char_ids, dtype=np.float32)))
            else:
                tmp_vector = np.hstack((normal_array, np.array(char_ids, dtype=np.float32)))
            text_vector.append(tmp_vector)

        if len(text_vector) == 0:
            text_vector.append(np.zeros(embedding_dim + max_char_length,dtype=np.float32))
        context_vector.append(text_vector)
        text_vector = []

        for noun in text[1]:
            char_ids = char_dict.char2id(noun)
            if noun in tokenize.keys():
                text_vector = np.hstack((tokenize[noun], np.array(char_ids, dtype=np.float32)))
            else:
                text_vector = np.hstack((normal_array, np.array(char_ids, dtype=np.float32)))
        context_vector.append([text_vector])
        text_vector = []

        for later_word in text[2]:
            char_ids = char_dict.char2id(later_word[1])
            if later_word[1] in tokenize.keys():
                tmp_vector = np.hstack((tokenize[later_word[1]], np.array(char_ids, dtype=np.float32)))
            else:
                tmp_vector = np.hstack((normal_array, np.array(char_ids, dtype=np.float32)))
            text_vector.append(tmp_vector)
        if len(text_vector) == 0:
            text_vector.append(np.zeros(embedding_dim + max_char_length,dtype=np.float32))
        context_vector.append(text_vector)
        sentence_vector.append(context_vector)
    return sentence_vector

#Tag model
def W2V3(text_list):
    #normal_array = sigmoid(np.array(normal(size=embedding_dim), dtype=np.float32))
    #print(normal_array)
    high_freq = get_freq_dict()
    sentence_vector = []
    for text in text_list:
        text_vector = []
        context_vector = []
        #前文脈
        for earlier_word in text[0]:
            if earlier_word[1] in high_freq:
                try:
                    # tmp_vector = np.hstack((np.hstack((tokenize[earlier_word[0]], np.array([position-5], dtype=np.float32))), tag))
                    tmp_vector = np.hstack((tokenize[earlier_word[1]], np.array([0], dtype=np.float32)))
                except:
                    # tmp_vector = np.hstack((np.hstack((normal_array, np.array([position - 5], dtype=np.float32))), tag))
                    tmp_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
            else:
                #PF = np.hstack(np.array([1],dtype=np.float32),np.array(list(map(int,list(str(format(position,'03b'))))),dtype=np.float32))
                tag = pos_tag(earlier_word[2])
                try:
                    #tmp_vector = np.hstack((np.hstack((tokenize[earlier_word[0]], np.array([position-5], dtype=np.float32))), tag))
                    tmp_vector = np.hstack((tokenize[earlier_word[0]], np.array([0], dtype=np.float32)))
                except:
                    #tmp_vector = np.hstack((np.hstack((normal_array, np.array([position - 5], dtype=np.float32))), tag))
                    tmp_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
            text_vector.append(tmp_vector)
        if len(text_vector) == 0:
            text_vector.append(np.zeros(embedding_dim + 1,dtype=np.float32))
        context_vector.append(text_vector)

        #対象名詞
        if text[1][1] in high_freq:
            try:
                #text_vector = np.hstack((np.hstack((tokenize[text[1][0]], np.array([0], dtype=np.float32))), tag))
                text_vector = np.hstack((tokenize[text[1][1]], np.array([0], dtype=np.float32)))
            except:
                #text_vector = np.hstack((np.hstack((normal_array, np.array([0], dtype=np.float32))),tag))
                text_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
        else:
            try:
                #text_vector = np.hstack((np.hstack((tokenize[text[1][0]], np.array([0], dtype=np.float32))), tag))
                text_vector = np.hstack((tokenize[text[1][0]], np.array([0], dtype=np.float32)))
            except:
                #text_vector = np.hstack((np.hstack((normal_array, np.array([0], dtype=np.float32))),tag))
                text_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))

        context_vector.append([text_vector])
        text_vector = []

        #後文脈
        for later_word in text[2]:
            if later_word[1] in high_freq:
                #PF = np.hstack(np.array([0], dtype=np.float32),np.array(list(map(int, list(str(format(position, '03b'))))), dtype=np.float32))
                tag = pos_tag(later_word[2])
                try:
                    #tmp_vector = np.hstack((np.hstack((tokenize[later_word[0]], np.array([position], dtype=np.float32))), tag))
                    tmp_vector = np.hstack((tokenize[later_word[1]], np.array([0], dtype=np.float32)))
                except:
                    #tmp_vector = np.hstack((np.hstack((normal_array, np.array([position], dtype=np.float32))), tag))
                    tmp_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
            else:
                try:
                    #tmp_vector = np.hstack((np.hstack((tokenize[later_word[0]], np.array([position], dtype=np.float32))), tag))
                    tmp_vector = np.hstack((tokenize[later_word[0]], np.array([0], dtype=np.float32)))
                except:
                    #tmp_vector = np.hstack((np.hstack((normal_array, np.array([position], dtype=np.float32))), tag))
                    tmp_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
            text_vector.append(tmp_vector)
        if len(text_vector) == 0:
            text_vector.append(np.zeros(embedding_dim + 1,dtype=np.float32))
        context_vector.append(text_vector)
        sentence_vector.append(context_vector)
    return sentence_vector

#
def W2V4(text_list):
    model = word2vec.Word2Vec.load("./model_weight/word2vec_jawiki_alltag.model")
    high_freq = get_freq_dict()
    sentence_vector = []
    for text in text_list:
        text_vector = []
        context_vector = []
        for earlier_word in text[0]:
            tmp_vector = []
            try:
                tmp_vector = np.hstack((model.wv[earlier_word],np.array([0],dtype=np.float32)))
            except:
                tmp_vector = np.hstack((np.zeros(embedding_dim, dtype=np.float32),np.array([0],dtype=np.float32)))
            text_vector.append(tmp_vector)
            tmp_vector = []
        if len(text_vector) == 0:
            text_vector.append(np.zeros(embedding_dim + 1,dtype=np.float32))
        context_vector.append(text_vector)
        text_vector = []

        for noun in text[1]:
            text_vector = np.hstack((np.zeros(embedding_dim, dtype=np.float32),np.array([1],dtype=np.float32)))
        context_vector.append(text_vector)
        text_vector = []

        for later_word in text[2]:
            try:
                tmp_vector = np.hstack((model.wv[later_word],np.array([0],dtype=np.float32)))
            except:
                tmp_vector = np.hstack((np.zeros(embedding_dim, dtype=np.float32),np.array([0],dtype=np.float32)))
            text_vector.append(tmp_vector)
            tmp_vector = []
        if len(text_vector) == 0:
            text_vector.append(np.zeros(embedding_dim + 1,dtype=np.float32))
        context_vector.append(text_vector)
        sentence_vector.append(context_vector)
    return sentence_vector

#Position Features -5 -4 -3 -2 -1 0 1 2 3 4 5
#Pos tag 0:名詞 1:動詞 2:形容詞 3:副詞 4:助詞 5:接続詞 6:助動詞 7:連体詞 8:感動詞 9:*
def W2VPF(text_list):
    #print(normal_array)
    #normal_array = sigmoid(np.array(normal(size=embedding_dim), dtype=np.float32))
    #vocab_model = word2vec.Word2Vec.load('./model_weight/word2vec_jawiki.model')
    sentence_vector = []
    for text in text_list:
        text_vector = []
        context_vector = []
        #前文脈
        for position,earlier_word in enumerate(text[0]):
            #PF = np.hstack(np.array([1],dtype=np.float32),np.array(list(map(int,list(str(format(position,'03b'))))),dtype=np.float32))
            tag = pos_tag(earlier_word[2])
            if not earlier_word[1] in tokenize.keys():
                try:
                    #tmp_vector = np.hstack((np.hstack((tokenize[earlier_word[0]], np.array([position-5], dtype=np.float32))), tag))
                    tmp_vector = np.hstack((tokenize[earlier_word[0]], np.array([position - len(text[0])], dtype=np.float32)))
                except:
                    #tmp_vector = np.hstack((np.hstack((normal_array, np.array([position - 5], dtype=np.float32))), tag))
                    tmp_vector = np.hstack((normal_array, np.array([position - len(text[0])], dtype=np.float32)))
            else:
                #tmp_vector = np.hstack((np.hstack((tokenize[earlier_word[1]], np.array([position - 5], dtype=np.float32))), tag))
                tmp_vector = np.hstack((tokenize[earlier_word[1]], np.array([position - len(text[0])], dtype=np.float32)))
            text_vector.append(tmp_vector)
        if len(text_vector) == 0:
            text_vector.append(np.zeros(embedding_dim + 1,dtype=np.float32))
        context_vector.append(text_vector)

        #対象名詞
        tag = pos_tag(text[1][2])
        if not text[1][1] in tokenize.keys():
            try:
                #text_vector = np.hstack((np.hstack((tokenize[text[1][0]], np.array([0], dtype=np.float32))), tag))
                text_vector = np.hstack((tokenize[text[1][0]], np.array([0], dtype=np.float32)))
            except:
                #text_vector = np.hstack((np.hstack((normal_array, np.array([0], dtype=np.float32))),tag))
                text_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
        else:
            #text_vector = np.hstack((np.hstack((tokenize[text[1][1]], np.array([0], dtype=np.float32))), tag))
            text_vector = np.hstack((tokenize[text[1][1]], np.array([0], dtype=np.float32)))

        context_vector.append([text_vector])
        text_vector = []

        #後文脈
        for position,later_word in enumerate(text[2],1):
            #PF = np.hstack(np.array([0], dtype=np.float32),np.array(list(map(int, list(str(format(position, '03b'))))), dtype=np.float32))
            tag = pos_tag(later_word[2])
            if not later_word[1] in tokenize.keys():
                try:
                    #tmp_vector = np.hstack((np.hstack((tokenize[later_word[0]], np.array([position], dtype=np.float32))), tag))
                    tmp_vector = np.hstack((tokenize[later_word[0]], np.array([position], dtype=np.float32)))
                except:
                    #tmp_vector = np.hstack((np.hstack((normal_array, np.array([position], dtype=np.float32))), tag))
                    tmp_vector = np.hstack((normal_array, np.array([position], dtype=np.float32)))
            else:
                #tmp_vector = np.hstack((np.hstack((tokenize[later_word[1]], np.array([position], dtype=np.float32))), tag))
                tmp_vector = np.hstack((tokenize[later_word[1]], np.array([position], dtype=np.float32)))
            text_vector.append(tmp_vector)
        if len(text_vector) == 0:
            text_vector.append(np.zeros(embedding_dim + 1,dtype=np.float32))
        context_vector.append(text_vector)
        sentence_vector.append(context_vector)

    return sentence_vector

# Proposal Model
def W2Valltag(text_list):
    cnt = 0
    #print(normal_array)
    #normal_array = sigmoid(np.array(normal(size=embedding_dim), dtype=np.float32))
    #vocab_model = word2vec.Word2Vec.load('./model_weight/word2vec_jawiki.model')
    high_freq = get_freq_dict()
    polar_dic = get_polar_dict()
    print('vocab size:',len(index2word))
    print('high freq word:',len(high_freq))
    sentence_vector = []
    for text in text_list:
        text_vector = []
        context_vector = []
        #前文脈
        for position,earlier_word in enumerate(text[0]):
            tag = pos_tag(earlier_word[2])
            if earlier_word[1] in polar_dic.keys():
                 polar = polar_dic[earlier_word[1]]
            else:
                 polar = 3
            if earlier_word[0] == '<名詞>':
                #freq = freq_tag(earlier_word[1])
                if earlier_word[1] in high_freq:
                #if chk_pos(earlier_word[2]) or earlier_word[1] in high_freq:
                    cnt += 1
                    #tmp_vector = np.hstack((np.hstack((tokenize[earlier_word[0]], np.array([position-5], dtype=np.float32))), tag))
                    #tmp_vector = np.hstack((tokenize[earlier_word[1]], np.array([position - len(text[0])], dtype=np.float32)))
                    #tmp_vector = np.hstack((tokenize[earlier_word[1]], np.array([polar], dtype=np.int32)))
                    #tmp_vector = np.hstack((tokenize[earlier_word[1]], np.array([freq], dtype=np.float32)))
                    #tmp_vector = np.hstack((tokenize[earlier_word[1]], tag))
                    tmp_vector = np.hstack((tokenize[earlier_word[1]], np.array([0], dtype=np.float32)))

                else:
                    #tmp_vector = np.hstack((np.hstack((tokenize[earlier_word[0]], np.array([position-5], dtype=np.float32))), tag))
                    #tmp_vector = np.hstack((tokenize[earlier_word[0]], np.array([position - len(text[0])], dtype=np.float32)))
                    #tmp_vector = np.hstack((tokenize[earlier_word[0]], np.array([polar], dtype=np.int32)))
                    #tmp_vector = np.hstack((tokenize[earlier_word[0]], np.array([freq], dtype=np.float32)))
                    #tmp_vector = np.hstack((tokenize[earlier_word[0]], tag))
                    tmp_vector = np.hstack((tokenize[earlier_word[0]], np.array([0], dtype=np.float32)))
            else:
                if earlier_word[1] in tokenize.keys():
                    #tmp_vector = np.hstack((np.hstack((tokenize[earlier_word[1]], np.array([position - 5], dtype=np.float32))), tag))
                    #tmp_vector = np.hstack((tokenize[earlier_word[1]], np.array([position - len(text[0])], dtype=np.float32)))
                    #tmp_vector = np.hstack((tokenize[earlier_word[1]], np.array([polar], dtype=np.int32)))
                    #tmp_vector = np.hstack((tokenize[earlier_word[1]], np.array([0], dtype=np.float32)))
                    #tmp_vector = np.hstack((tokenize[earlier_word[1]], tag))
                    tmp_vector = np.hstack((tokenize[earlier_word[1]], np.array([0], dtype=np.float32)))
                else:
                    #tmp_vector = np.hstack((np.hstack((normal_array, np.array([position - 5], dtype=np.float32))), tag))
                    #tmp_vector = np.hstack((normal_array, np.array([position - len(text[0])], dtype=np.float32)))
                    #tmp_vector = np.hstack((normal_array, np.array([polar], dtype=np.int32)))
                    #tmp_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
                    #tmp_vector = np.hstack((normal_array, tag))
                    tmp_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
            text_vector.append(tmp_vector)
        if len(text_vector) == 0:
            #text_vector.append(np.hstack((padding_array,np.array([-1],dtype=np.float32))))
            text_vector.append(np.hstack((padding_array,np.array([-1],dtype=np.float32))))
        context_vector.append(text_vector)

        #対象名詞
        #tag = pos_tag(text[1][2])
        if text[1][1] in polar_dic.keys():
            polar = polar_dic[text[1][1]]
        else:
            polar = 2
        tag = pos_tag(text[1][2])
        #freq = freq_tag(text[1][1])
        if text[1][1] in high_freq:
        #if chk_pos(text[1][2]) or text[1][1] in high_freq:
            #if chk_pos(text[1][2]) == 1: cnt += 1
            #text_vector = np.hstack((np.hstack((tokenize[text[1][0]], np.array([0], dtype=np.float32))), tag))
            text_vector = np.hstack((tokenize[text[1][1]], np.array([0], dtype=np.float32)))
            #text_vector = np.hstack((tokenize[text[1][1]], np.array([polar], dtype=np.int32)))
            #text_vector = np.hstack((tokenize[text[1][1]], np.array([freq], dtype=np.float32)))
            #text_vector = np.hstack((tokenize[text[1][1]], tag))
        else:
            #text_vector = np.hstack((np.hstack((tokenize[text[1][0]], np.array([0], dtype=np.float32))), tag))
            text_vector = np.hstack((tokenize[text[1][0]], np.array([0], dtype=np.float32)))
            #text_vector = np.hstack((tokenize[text[1][0]], np.array([polar], dtype=np.float32)))
            #text_vector = np.hstack((tokenize[text[1][0]], np.array([freq], dtype=np.float32)))
            #text_vector = np.hstack((tokenize[text[1][0]], tag))
        context_vector.append([text_vector])
        text_vector = []

        #後文脈
        for position,later_word in enumerate(text[2],1):
            #PF = np.hstack(np.array([0], dtype=np.float32),np.array(list(map(int, list(str(format(position, '03b'))))), dtype=np.float32))
            tag = pos_tag(later_word[2])
            if later_word[1] in polar_dic.keys():
                polar = polar_dic[later_word[1]]
            else:
                polar = 2
            if later_word[0] == '<名詞>':
                #freq = freq_tag(later_word[1])
                if later_word[1] in high_freq:
                #if chk_pos(later_word[2]) or later_word[1] in high_freq:
                    cnt += 1
                    #tmp_vector = np.hstack((np.hstack((tokenize[later_word[0]], np.array([position], dtype=np.float32))), tag))
                    #tmp_vector = np.hstack((tokenize[later_word[1]], np.array([position], dtype=np.float32)))
                    #tmp_vector = np.hstack((tokenize[later_word[1]], np.array([polar], dtype=np.int32)))
                    #tmp_vector = np.hstack((tokenize[later_word[1]], np.array([freq], dtype=np.float32)))
                    #tmp_vector = np.hstack((tokenize[later_word[1]], tag))
                    tmp_vector = np.hstack((tokenize[later_word[1]], np.array([0], dtype=np.float32)))
                else:
                    #tmp_vector = np.hstack((np.hstack((tokenize[later_word[0]], np.array([position], dtype=np.float32))), tag))
                    #tmp_vector = np.hstack((tokenize[later_word[0]], np.array([position], dtype=np.float32)))
                    #tmp_vector = np.hstack((tokenize[later_word[0]], np.array([polar], dtype=np.int32)))
                    #tmp_vector = np.hstack((tokenize[later_word[0]], np.array([freq], dtype=np.float32)))
                    #tmp_vector = np.hstack((tokenize[later_word[0]], tag))
                    tmp_vector = np.hstack((tokenize[later_word[0]], np.array([0], dtype=np.float32)))
            else:
                if later_word[1] in tokenize.keys():
                    # tmp_vector = np.hstack((np.hstack((later_word[1], np.array([position], dtype=np.float32))), tag))
                    #tmp_vector = np.hstack((tokenize[later_word[1]], np.array([position], dtype=np.float32)))
                    # tmp_vector = np.hstack((later_word[1], np.array([polar], dtype=np.int32)))
                    # tmp_vector = np.hstack((later_word[1], np.array([0], dtype=np.float32)))
                    # tmp_vector = np.hstack((later_word[1], tag))
                    tmp_vector = np.hstack((tokenize[later_word[1]], np.array([0], dtype=np.float32)))
                else:
                    #tmp_vector = np.hstack((np.hstack((normal_array, np.array([position], dtype=np.float32))), tag))
                    #tmp_vector = np.hstack((normal_array, np.array([position], dtype=np.float32)))
                    #tmp_vector = np.hstack((normal_array, np.array([polar], dtype=np.int32)))
                    #tmp_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
                    #tmp_vector = np.hstack((normal_array, tag))
                    tmp_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
            text_vector.append(tmp_vector)
        if len(text_vector) == 0:
            #text_vector.append(np.hstack((padding_array,np.array([-1],dtype=np.float32))))
            text_vector.append(np.hstack((padding_array,np.array([-1],dtype=np.float32))))
        context_vector.append(text_vector)
        sentence_vector.append(context_vector)

    #print('positive polar:',cnt_p)
    #print('negative polar:',cnt_n)
    #print('neutral polar:',cnt_neu)
    #print('cannot match:',cnt_v)
    print('used noun:',cnt)
    return sentence_vector

#Position model
def W2VPF2(text_list):
    #print(normal_array)
    #normal_array = sigmoid(np.array(normal(size=embedding_dim), dtype=np.float32))
    sentence_vector = []
    for text in text_list:
        text_vector = []
        context_vector = []
        #前文脈
        for position,earlier_word in enumerate(text[0],1):
            #PF = np.hstack(np.array([1],dtype=np.float32),np.array(list(map(int,list(str(format(position,'03b'))))),dtype=np.float32))
            if not earlier_word[1] in tokenize.keys():
                try:
                    tmp_vector = np.hstack((normal_array, np.array([position-len(text[0])], dtype=np.float32)))
                    #tmp_vector = np.hstack((np.zeros(embedding_dim,dtype=np.float32), np.array([position - 5], dtype=np.float32)))
                except:
                    tmp_vector = np.hstack((normal_array, np.array([position-len(text[0])], dtype=np.float32)))
                    #tmp_vector = np.hstack((np.zeros(embedding_dim, dtype=np.float32), np.array([position - 5], dtype=np.float32)))
            else:
                tmp_vector = np.hstack((tokenize[earlier_word[1]], np.array([position-len(text[0])], dtype=np.float32)))
            text_vector.append(tmp_vector)
        if len(text_vector) == 0:
            text_vector.append(np.zeros(embedding_dim + 1,dtype=np.float32))
        context_vector.append(text_vector)

        #対象名詞
        try:
            text_vector = np.hstack((tokenize[text[1][1]],np.array([0],dtype=np.float32)))
        except:
            text_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
            #text_vector = np.hstack((np.zeros(embedding_dim, dtype=np.float32), np.array([0], dtype=np.float32)))

        context_vector.append([text_vector])
        text_vector = []

        #後文脈
        for position,later_word in enumerate(text[2],1):
            #PF = np.hstack(np.array([0], dtype=np.float32),np.array(list(map(int, list(str(format(position, '03b'))))), dtype=np.float32))
            if not later_word[1] in tokenize.keys():
                try:
                    tmp_vector = np.hstack((normal_array, np.array([position], dtype=np.float32)))
                    #tmp_vector = np.hstack((np.zeros(embedding_dim, dtype=np.float32), np.array([position], dtype=np.float32)))
                except:
                    tmp_vector = np.hstack((normal_array, np.array([position], dtype=np.float32)))
                    #tmp_vector = np.hstack((np.zeros(embedding_dim, dtype=np.float32), np.array([position], dtype=np.float32)))
            else:
                tmp_vector = np.hstack((tokenize[later_word[1]], np.array([position], dtype=np.float32)))
            text_vector.append(tmp_vector)
        if len(text_vector) == 0:
            text_vector.append(np.zeros(embedding_dim + 1,dtype=np.float32))
        context_vector.append(text_vector)
        sentence_vector.append(context_vector)
    return sentence_vector


# existing model
def W2VR(text_list):
    #print(normal_array)
    #normal_array = sigmoid(np.array(normal(size=embedding_dim), dtype=np.float32))
    sentence_vector = []
    for text in text_list:
        text_vector = []
        context_vector = []
        for earlier_word in text[0]:
            if earlier_word == '<EOS>':
                tmp_vector = np.hstack((np.zeros(embedding_dim,dtype=np.float32),np.array([1],dtype=np.float32)))
            else:
                if not earlier_word[1] in tokenize.keys():
                    try:
                        tmp_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
                        #tmp_vector = np.hstack((np.zeros(embedding_dim, dtype=np.float32), np.array([0], dtype=np.float32)))
                    except:
                        tmp_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
                        #tmp_vector = np.hstack((np.zeros(embedding_dim, dtype=np.float32), np.array([0], dtype=np.float32)))
                else:
                    tmp_vector = np.hstack((tokenize[earlier_word[1]], np.array([0], dtype=np.float32)))
            text_vector.append(tmp_vector)
        if len(text_vector) == 0:
            text_vector.append(np.zeros(embedding_dim+1,dtype=np.float32))
        context_vector.append(text_vector)

        if not text[1][1] in tokenize.keys():
            try:
                text_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
                #text_vector = np.hstack((np.zeros(embedding_dim, dtype=np.float32), np.array([0], dtype=np.float32)))
            except:
                text_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
                #text_vector = np.hstack((np.zeros(embedding_dim, dtype=np.float32), np.array([0], dtype=np.float32)))
        else:
            text_vector = np.hstack((tokenize[text[1][1]], np.array([0], dtype=np.float32)))
        context_vector.append([text_vector])
        text_vector = []

        for later_word in text[2]:
            if later_word == '<EOS>':
                tmp_vector = np.hstack((np.zeros(embedding_dim,dtype=np.float32),np.array([1],dtype=np.float32)))
            else:
                if not later_word[1] in tokenize.keys():
                    try:
                        tmp_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
                        #tmp_vector = np.hstack((np.zeros(embedding_dim, dtype=np.float32), np.array([0], dtype=np.float32)))
                    except:
                        tmp_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
                        #tmp_vector = np.hstack((np.zeros(embedding_dim, dtype=np.float32), np.array([0], dtype=np.float32)))
                else:
                    tmp_vector = np.hstack((tokenize[later_word[1]], np.array([0], dtype=np.float32)))
            text_vector.append(tmp_vector)
        if len(text_vector) == 0:
            text_vector.append(np.zeros(embedding_dim + 1,dtype=np.float32))
        context_vector.append(text_vector)
        sentence_vector.append(context_vector)
    return sentence_vector

#Dependency model
def W2Vnotagtrain(text_list):
    #print(normal_array)
    #normal_array = sigmoid(np.array(normal(size=embedding_dim), dtype=np.float32))
    sentence_vector = []
    for text in text_list:
        text_vector = []
        context_vector = []
        for earlier_word in text[0]:
            if not earlier_word[1] in tokenize.keys():
                try:
                    tmp_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
                    #tmp_vector = np.zeros(embedding_dim+1,dtype=np.float32)
                except:
                    tmp_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
                    #tmp_vector = np.zeros(embedding_dim + 1, dtype=np.float32)
            else:
                tmp_vector = np.hstack((tokenize[earlier_word[1]], np.array([0], dtype=np.float32)))
            text_vector.append(tmp_vector)
        if len(text_vector) == 0:
            text_vector.append(np.zeros(embedding_dim+1,dtype=np.float32))
        context_vector.append(text_vector)

        if not text[1][1] in tokenize.keys():
            try:
                text_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
                #text_vector = np.zeros(embedding_dim + 1, dtype=np.float32)
            except:
                text_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
                #text_vector = np.zeros(embedding_dim + 1, dtype=np.float32)
        else:
            text_vector = np.hstack((tokenize[text[1][1]], np.array([0], dtype=np.float32)))
        context_vector.append([text_vector])
        text_vector = []

        for later_word in text[2]:
            if not later_word[1] in tokenize.keys():
                try:
                    tmp_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
                    #tmp_vector = np.zeros(embedding_dim + 1, dtype=np.float32)
                except:
                    tmp_vector = np.hstack((normal_array, np.array([0], dtype=np.float32)))
                    #tmp_vector = np.zeros(embedding_dim + 1, dtype=np.float32)
            else:
                tmp_vector = np.hstack((tokenize[later_word[1]], np.array([0], dtype=np.float32)))
            text_vector.append(tmp_vector)
        if len(text_vector) == 0:
            text_vector.append(np.zeros(embedding_dim + 1,dtype=np.float32))
        context_vector.append(text_vector)
        sentence_vector.append(context_vector)
    return sentence_vector

def W2Vcontext_only(text_list):
    sentence_vector = []
    for text in text_list:
        text_vector = []
        context_vector = []
        #前文脈
        for position,earlier_word in enumerate(text[0]):
            #PF = np.hstack(np.array([1],dtype=np.float32),np.array(list(map(int,list(str(format(position,'03b'))))),dtype=np.float32))
            try:
                #tmp_vector = np.hstack((np.hstack((tokenize[earlier_word[0]], np.array([position-5], dtype=np.float32))), tag))
                tmp_vector = np.hstack((tokenize[earlier_word[0]], np.array([position - len(text[0])], dtype=np.float32)))
            except:
                #tmp_vector = np.hstack((np.hstack((normal_array, np.array([position - 5], dtype=np.float32))), tag))
                tmp_vector = np.hstack((normal_array, np.array([position - len(text[0])], dtype=np.float32)))
            text_vector.append(tmp_vector)
        if len(text_vector) == 0:
            text_vector.append(np.hstack((normal_array,np.array([0],dtype=np.float32))))
        context_vector.append(text_vector)
        text_vector = []

        #後文脈
        for position,later_word in enumerate(text[2],1):
            #PF = np.hstack(np.array([0], dtype=np.float32),np.array(list(map(int, list(str(format(position, '03b'))))), dtype=np.float32))
            try:
                #tmp_vector = np.hstack((np.hstack((tokenize[later_word[0]], np.array([position], dtype=np.float32))), tag))
                tmp_vector = np.hstack((tokenize[later_word[0]], np.array([position], dtype=np.float32)))
            except:
                #tmp_vector = np.hstack((np.hstack((normal_array, np.array([position], dtype=np.float32))), tag))
                tmp_vector = np.hstack((normal_array, np.array([position], dtype=np.float32)))
            text_vector.append(tmp_vector)
        if len(text_vector) == 0:
            text_vector.append(np.hstack((normal_array,np.array([0],dtype=np.float32))))
        context_vector.append(text_vector)
        sentence_vector.append(context_vector)
    return sentence_vector
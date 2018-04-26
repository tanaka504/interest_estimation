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
from Embedding import *
from Model import *
from Function import *

def reshape_data(filename):
    train_data = []
    with open(filename,'r') as f:
        with open('./data/ans_data.txt','w') as ansf:
            for idx,line in enumerate(f):
                col = line.split('\t')
                train_data.append(col[0])
                try:
                    ansf.write(col[1])
                except:
                    print('reshape_data:',idx)
                    sys.exit()
    return train_data

def ans_reshape(filename):
    with open(filename,'r') as f:
        ans_list = []
        for idx,line in enumerate(f,1):
            line = line.replace('\n',',')
            tmp_list = line.split(',')
            try:
                tmp_list.remove('')
            except:
                pass
            for ans in tmp_list:
                ans = re.sub(r'[\t \s]','',ans)
                if ans == '1':
                    ans_list.append([1,0])
                elif ans == '0':
                    ans_list.append([0,1])
                else:
                    print('ans_reshape:',idx,ans)
                    raise StopIteration
        ans_array = np.array(ans_list,dtype=np.float32)
    return ans_array

def get_train_vector(train_file,mode=4,embed=10,relating=True, char_dict={}):
    c = CaboCha.Parser('-f1')
    train_data = reshape_data(train_file)
    text_list = []
    if relating:
        for line in train_data:
            analyzed_sentence = c.parseToString(line)
            analyzed_list = analyzed_sentence.split('\n')
            relation = relation_index(analyzed_list)
            relations = relation_shape(relation)
            text = noun_exploit(relations)
            text_list.extend(text)
    else:
        text_list = noun_exploit2(train_data, embed=embed)

    # select embedding method
    if embed == 1:
        sentence_vector = W2V(text_list)
    elif embed == 2:
        sentence_vector = W2V2(text_list, char_dict)
    elif embed == 3:
        sentence_vector = W2V3(text_list)
    elif embed == 4:
        sentence_vector = W2V4(text_list)
    elif embed == 5:
        sentence_vector = W2VPF(text_list)
    elif embed == 6:
        sentence_vector = W2VR(text_list)
    elif embed == 7:
        sentence_vector = W2Vnotagtrain(text_list)
    elif embed == 8:
        sentence_vector = W2VPF2(text_list)
    elif embed == 9:
        sentence_vector = W2Vcontext_only(text_list)
    elif embed == 10:
        sentence_vector = W2Valltag(text_list)
    else:
        print('Usage:(weight name,train_file path,model mode,embed method'
              '(1:tag 2:tag(no target) 3:tagtrain 4:alltag 5:Position Featuring))')
        sys.exit()

    label_y = ans_reshape('./data/ans_data.txt')
    if mode == 2:
        sent_v = padding(sentence_vector)
        sent_v = np.array(sent_v)
    elif mode == 1:
        sent_v = pad_seq2(sentence_vector)
    elif embed == 1:
        sent_v = sentence_vector
    else:
        sent_v = pad_seq(sentence_vector)

    return sent_v, label_y

def minibatch_generator(data_x, label_y, batch_size, shuffle=True):
    num_batches_per_epoch = int((len(data_x) - 1) / batch_size) + 1
    def data_generator():
        data_size = len(data_x)
        while True:
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data_x[shuffle_indices]
                shuffled_labels = label_y[shuffle_indices]
            else:
                shuffled_data = data_x
                shuffled_labels = label_y

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                x = shuffled_data[start_index:end_index]
                y = shuffled_labels[start_index:end_index]
                x = pad_seq(x)
                yield x, y

    return num_batches_per_epoch, data_generator()

#--------------------------------------------

def main(filename,train_file,valid_file,mode=1,embed=5,relating=True):
    char_dict = Char_Corpus(create_flg=True)
    train_x, train_y = get_train_vector(train_file, mode=mode, embed=embed, relating=relating,char_dict=char_dict)
    valid_x, valid_y = get_train_vector(valid_file, mode=mode, embed=embed, relating=relating,char_dict=char_dict)

    # generate model
    if mode == 1:
        estimation = context_only().estimate_model
    elif mode == 2:
        estimation = CNN().model
    elif mode == 3:
        estimation = LSTM_NN().estimate_model
    elif mode == 4:
        estimation = CNN_NN().estimate_model
    elif mode == 5:
        estimation = target_only(wv_dim=1).model
    elif mode == 6:
        estimation = Char_embed_LSTM(char_dict=char_dict.char_vocab).estimate_model
    else:
        print('Usage:(weight name,train_file path,model mode(1:CNN&NN 2:CNN 3:LSTM&NN))')
        sys.exit()
    keras.callbacks.EarlyStopping(monitor='val_f1', patience=5, verbose=0, mode='max')
    ms_cb = keras.callbacks.ModelCheckpoint(filename,verbose=1,monitor='val_f1', save_best_only=True,save_weights_only=True, mode='max')
    # training
    #train_steps, train_batches = minibatch_generator(train_x, train_y,batch_size=batch_size)
    #valid_steps, valid_batches = minibatch_generator(valid_x, valid_y, batch_size=batch_size)
    #fit = estimation.fit_generator(train_batches, train_steps, epochs=30, verbose=1, callbacks=[ms_cb], validation_data=valid_batches, validation_steps=valid_steps)

    fit = estimation.fit(train_x, train_y, epochs=30, batch_size=64,verbose=1, callbacks=[ms_cb],validation_data=(valid_x, valid_y))
    #plot_training_loss(fit)
    #estimation.save_weights(filename)
    print('save model as ',filename)
    backend.clear_session()

if __name__ == '__main__':
    #############################################################################################
    ## Usage : main(model_weight path , train_file path , mode , embed)                        ##
    ## mode >> 1:CNNCO 2:CNN 3:LSTM&NN 4:CNN&NN 5:targetonly                                   ##
    ## embed >> 1:tag 2:tag(no target) 3:tagtrain 4:alltag 5:Position Featuring 6:related work ##
    #############################################################################################

    train_file = './train_data/train_data_o10/train_data'
    valid_file = './train_data/train_data_o10/validation_data'
    seed = 5


    #main('./model_weight/LSTM&NN{}_related_workmodel_weights.h5'.format(seed), train_file, valid_file, mode=3, embed=6, relating=False)
    #main('./model_weight/CNN&NN{}_alltagmodel_weights.h5'.format(seed), train_file, valid_file, mode=4, embed=10, relating=True)
    #main('./model_weight/LSTM&NN{}_alltagmodel_weights.h5'.format(seed), train_file, valid_file, mode=3, embed=10, relating=True)
    main('./model_weight/LSTM_Char{}_related_workmodel_weights.h5'.format(seed),train_file,valid_file,mode=6,embed=2,relating=False)


    #main('./model_weight/LSTM&NN_tagtrainmodel_weights.h5', train_file, mode=3, embed=3, relating=False)

    #main('./model_weight/CNN&NN_PFmodel_weights.h5', train_file, mode=4, embed=5, relating=True)
    #main('./model_weight/CNN&NN_tagtrainmodel_weights.h5',train_file,mode=4,embed=3,relating=True)

    #main('./model_weight/CNN_PFmodel_weights.h5', train_file, mode=2, embed=5, relating=True)

    #main('./model_weight/CO_context_onlymodel_weights.h5', train_file, mode=1, embed=9, relating=True)
    #main('./model_weight/TO{}_target_onlymodel_weights.h5'.format(seed), train_file, mode=5, embed=1, relating=True)

    #main('./model_weight/CNN&NN{}_related_workmodel_weights.h5'.format(seed),train_file,mode=4,embed=7,relating=True)
    #main('./model_weight/CNN&NN{}_tagtrainmodel_weights.h5'.format(seed), train_file, mode=4, embed=3, relating=False)
    #main('./model_weight/CNN&NN{}_PF2model_weights.h5'.format(seed), train_file, mode=4, embed=8, relating=False)

    #main('./model_weight/CNN&NN{}_wotmodel_weights.h5'.format(seed), train_file, mode=4, embed=8, relating=True)
    #main('./model_weight/CNN&NN{}_wodmodel_weights.h5'.format(seed), train_file, mode=4, embed=10, relating=False)
    #main('./model_weight/CNN&NN{}_wopmodel_weights.h5'.format(seed), train_file, mode=4, embed=3, relating=True)


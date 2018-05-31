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
import matplotlib.pyplot as plt
from setting import *
import pickle
import sentencepiece as spm

class Char_Corpus:
    def __init__(self, create_flg):
        self.char_vocab = None
        if create_flg:
            self._construct_dict()

    def _construct_dict(self):
        char_vocab = {'<pad>': -1,
                      '<unk>': 0}
        with open('./train_data/train_data_o10/train_data', 'r') as f:
            data = f.read().split('\n')
            for line in data:
                tweet_label = line.split('\t')
                for c in tweet_label[0]:
                    if not c in char_vocab.keys():
                        char_vocab[c] = len(char_vocab)
            print('character vocab size:',len(char_vocab))
            self.save()

            self.char_vocab = char_vocab

    def char2id(self, word):
        char_ids = []
        for c in word:
            if c in self.char_vocab:
                char_ids.append(self.char_vocab[c])
            else:
                char_ids.append(self.char_vocab['<unk>'])

        padded_ids = char_ids + [-1] * (25 - len(char_ids))

        return padded_ids

    def save(self):
        with open('./model_weight/char_vocab.dict','wb') as f:
            pickle.dump(self.char_vocab, f)

    def load(self):
        with open('./model_weight/char_vocab.dict', 'rb') as f:
            self.char_vocab = pickle.load(f)



def get_freq_dict():
    high_freq = set()
    add = high_freq.add
    with open('./model_weight/freq_dict.pkl', 'rb') as f:
        freq_dict = pickle.load(f)
        for word, freq in freq_dict.items():
            if freq > 5000:
                add(word)
    return high_freq

def get_polar_dict():
    polar_dic = {}
    with open('./data/pn_ja.dic') as f:
        data = f.read().split('\n')
        data.remove('')
        for line in data:
            polar = line.split(':')
            if float(polar[3]) > 0.3:
                polar_dic[polar[0]] = 1
            elif float(polar[3]) < -0.3:
                polar_dic[polar[0]] = 2
            else:
                polar_dic[polar[0]] = 0
    return polar_dic

#for CNN model
def padding(sentence_vector):
    result = []
    a,b,c = zip(*sentence_vector)
    aa = sequence.pad_sequences(np.array(a),pad_size,dtype=np.float32)
    cc = sequence.pad_sequences(np.array(c),pad_size,dtype=np.float32,padding='post')
    sent_v = [aa,np.array(b),cc]
    for i in range(len(sent_v[0])):
        mae = sent_v[0][i]
        meisi = sent_v[1][i]
        usiro = sent_v[2][i]
        context = np.append(mae,meisi,axis=0)
        context = np.append(context,usiro,axis=0)

        result.append(context)
        context = []

    return result

def pad_seq(sentence_vector):
    a,b,c = zip(*sentence_vector)
    aa = sequence.pad_sequences(np.array(a),pad_size,dtype=np.float32,value=np.array([-1]))
    cc = sequence.pad_sequences(np.array(c),pad_size,dtype=np.float32,padding='post',value=np.array([-1]))
    sent_v = [aa,np.array(b),cc]
    return sent_v

def pad_seq2(sentence_vector):
    a,c = zip(*sentence_vector)
    aa = sequence.pad_sequences(np.array(a),pad_size,dtype=np.float32,value=np.hstack((normal_array,np.array([0],dtype=np.float32))))
    cc = sequence.pad_sequences(np.array(c),pad_size,dtype=np.float32,value=np.hstack((normal_array,np.array([0],dtype=np.float32))),padding='post')
    sent_v = [aa,cc]
    return sent_v

def relation_index(analyzed):
    relation = {}
    tmp = []
    id = 1
    for line in analyzed:
        if line[0] == '*':
            if len(tmp) > 0:
                relation[idx] = (dst,tmp)
                tmp = []
            line_list = line.split(' ')
            dst = int(re.search(r'(.*?)D',line_list[2]).group(1))
            idx = int(line_list[1])
        elif line == 'EOS':
            relation[idx] = (dst,tmp)
            return relation
        else:
            col1 = line.split('\t')
            col2 = col1[1].split(',')
            tmp.append((col1[0],(col2[0],col2[1]),id))
            id +=1

def relation_shape(relation):
    result = []
    text_list = []
    for idx,k in relation.items():
        dst = k[0]
        sent = k[1]
        for word in sent:
            text_list.append(word)
        while dst != -1:
            idx = dst
            sent = relation[dst][1]
            dst = relation[dst][0]
            for word in sent:
                text_list.append(word)
        if len(text_list) > 0:
            result.append(text_list)
        text_list = []
    return result

def noun_exploit(relations):
    text_list = []
    chk_id = []
    for sent in relations:
        for idx,word in enumerate(sent):
            result = []
            tmp_list = []
            num_word = len(sent)
            if word[1][0] == '名詞' and word[2] not in chk_id:
                chk_id.append(word[2])
                # 名詞より前の文脈を見る
                if idx != 0:  # 前の文がない場合を除く
                    if idx - window_size < 0:
                        for i in range(0, idx):
                            if sent[i][1][0] == '名詞':
                                tmp_list.append(['<名詞>',sent[i][0],sent[i][1]])
                            else:
                                tmp_list.append([sent[i][0],sent[i][0],sent[i][1]])
                    else:
                        for i in range(idx - window_size, idx):
                            if sent[i][1][0] == '名詞':
                                tmp_list.append(['<名詞>',sent[i][0],sent[i][1]])
                            else:
                                tmp_list.append([sent[i][0],sent[i][0],sent[i][1]])
                # tmp_list.append('<EOS>')
                result.append(tmp_list)
                result.append([word[2],'<名詞>',word[0],word[1]])
                tmp_list = []
                # 名詞より後の文脈を見る
                if idx != (num_word - 1):  # 後に文がない場合を除く
                    if idx + window_size > num_word - 1:
                        for i in range(idx + 1, num_word):
                            if sent[i][1][0] == '名詞':
                                tmp_list.append(['<名詞>',sent[i][0],sent[i][1]])
                            else:
                                tmp_list.append([sent[i][0],sent[i][0],sent[i][1]])
                    else:
                        for i in range(idx + 1, idx + 1 + window_size):
                            if sent[i][1][0] == '名詞':
                                tmp_list.append(['<名詞>',sent[i][0],sent[i][1]])
                            else:
                                tmp_list.append([sent[i][0],sent[i][0],sent[i][1]])
                # tmp_list.append('<EOS>')
                result.append(tmp_list)
                text_list.append(result)
    text_list = sorted(text_list,key=lambda x:x[1])
    for contxt in text_list:
        del contxt[1][0]
    return text_list

def noun_exploit2(f,embed=1):
    text_list = []
    tagger = MeCab.Tagger('mecabrc')
    for line in f:
        tmp_text = []
        # コーパスによって対話の区切りが違うから適宜コードを足す必要あり
        analysis = tagger.parse(line)
        analysis = analysis.split('\n')
        analysis.remove('EOS')
        analysis.remove('')
        for text in analysis:
            col1 = text.split('\t')
            if len(col1) < 2:
                raise StopIteration
            col2 = col1[1].split(',')
            tmp_text.append((col1[0], col2[0]))
        for idx, word in enumerate(tmp_text):
            result = []
            tmp_list = []
            num_word = len(tmp_text)
            if word[1] == '名詞':
                # 名詞より前の文脈を見る
                if idx != 0:  # 前の文がない場合を除く
                    if idx - window_size < 0:
                        for i in range(0, idx):
                            if tmp_text[i][1] == '名詞':
                                tmp_list.append(['<名詞>',tmp_text[i][0],tmp_text[i][1]])
                            else:
                                tmp_list.append([tmp_text[i][0],tmp_text[i][0],tmp_text[i][1]])
                    else:
                        for i in range(idx - window_size, idx):
                            if tmp_text[i][1] == '名詞':
                                tmp_list.append(['<名詞>', tmp_text[i][0],tmp_text[i][1]])
                            else:
                                tmp_list.append([tmp_text[i][0], tmp_text[i][0],tmp_text[i][1]])
                #if embed == 6:
                    #tmp_list.append('<EOS>')
                result.append(tmp_list)
                result.append(['<名詞>',word[0],word[1]])
                tmp_list = []
                # 名詞より後の文脈を見る
                if idx != (num_word - 1):  # 後に文がない場合を除く
                    if idx + window_size > num_word - 1:
                        for i in range(idx + 1, num_word):
                            if tmp_text[i][1] == '名詞':
                                tmp_list.append(['<名詞>',tmp_text[i][0],tmp_text[i][1]])
                            else:
                                tmp_list.append([tmp_text[i][0],tmp_text[i][0],tmp_text[i][1]])
                    else:
                        for i in range(idx + 1, idx + 1 + window_size):
                            if tmp_text[i][1] == '名詞':
                                tmp_list.append(['<名詞>',tmp_text[i][0],tmp_text[i][1]])
                            else:
                                tmp_list.append([tmp_text[i][0],tmp_text[i][0],tmp_text[i][1]])
                #if embed == 6:
                    #tmp_list.append('<EOS>')
                result.append(tmp_list)
                text_list.append(result)
    return text_list

def plot_training_loss(fit):
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
    # Plot the loss in the history
    axL.plot(fit.history['loss'], label="loss for training")
    axL.plot(fit.history['val_loss'], label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')
    axR.plot(fit.history['acc'], label="loss for training")
    axR.plot(fit.history['val_acc'], label="loss for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')
    fig.savefig('./image/loss_graph.png')
    plt.close()

def data_modify(tweet, labels):
    pos_dic = {'名詞': 'NOU',
               '動詞': 'VER',
               '形容詞': 'ADJ',
               '副詞': 'ADV',
               '助詞': 'PAR',
               '接続詞': 'CON',
               '助動詞': 'AUX',
               '連体詞': 'REN',
               '感動詞': 'INT',
               '記号':'TOK',
               '接頭詞':'HEA',
               'フィラー':'FIL'}
    crf_label = []
    app = crf_label.append
    tagger = MeCab.Tagger('mecabrc')
    data = tagger.parse(tweet).split('\n')
    data.remove('EOS')
    data.remove('')
    for word in data:
        col1 = word.split('\t')
        col2 = col1[1].split(',')
        if col2[0] == '名詞':
            label = labels.pop(0)
            if label == '0':
                label = 'NEG'
            elif label == '1':
                label = 'POS'
            else:
                print('error')
                sys.exit()
        else:
            if col2[0] in pos_dic.keys():
                label = pos_dic[col2[0]]
            else:
                label = 'OTH'
        app((col1[0], label))
    assert len(labels) == 0
    return crf_label

def main():
    with open('./train_data/train_data_o10/real_test_data', 'r') as f:
        with open('./train_data/train_data_pos/real_test_data', 'w') as out_f:
            data = f.read().split('\n')
            #data.remove('')
            for line in data:
                tweet_label = line.split('\t')
                labels = tweet_label[1].split(',')
                crf_labels = data_modify(tweet_label[0], labels)

                for iter in crf_labels:
                    out_f.write(iter[0] + '\t' + iter[1] + '\n')
                out_f.write('\n')

def train_spm():
    spm.SentencePieceTrainer.Train('--input=data/w2v_data.txt --model_prefix=m --vocab_size=120000')

if __name__ == '__main__':
    main()
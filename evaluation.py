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
from sklearn.metrics import precision_recall_curve,auc,f1_score,precision_score,recall_score,accuracy_score
import math
from Embedding import *
from Model import *
from Function import *
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from setting import *
import random

model = word2vec.Word2Vec.load("./model_weight/w2v_jawiki_alltag.model")
random.seed(1)

def reshape_data(filename):
    train_data = []
    with open(filename,'r') as f:
        with open('./data/eval_ans_data.txt','w') as ansf:
            for idx,line in enumerate(f):
                col = line.split('\t')
                train_data.append(col[0])
                try:
                    ansf.write(col[1])
                except:
                    pass
    return train_data

def ans_reshape(filename):
    with open(filename,'r') as f:
        ans_list = []
        for idx,line in enumerate(f):
            line = line.replace('\n',',')
            tmp_list = line.split(',')
            try:
                tmp_list.remove('')
            except:
                pass
            for ans in tmp_list:
                if ans == '1':
                    ans_list.append([1,0])
                elif ans == '0':
                    ans_list.append([0,1])
                else:
                    pass
        ans_array = np.array(ans_list,dtype=np.float32)
    return ans_array

def del_vocab(text_list,percent):
    high_freq = get_freq_dict()
    per_dict = {20:0,
                30:473,
                40:1351,
                50:2230,
                60:3108,
                70:3986,
                80:4864,
                90:5742}
    remove_sum = per_dict[percent]

    fleq = []
    for contxt in text_list:
        if contxt[1][1] in model.wv.vocab.keys():
            fleq.append(contxt[1][1])
    counter = Counter(fleq)
    f_dict = {k:v for k,v in counter.items()}

    word_list = []
    for x in fleq:
        if not x in word_list:
            word_list.append(x)

    while del_vocab.remove_total < remove_sum:
        del_word = random.choice(word_list)
        if del_word in high_freq:
            continue
        word_list.remove(del_word)
        del_vocab.remove_total += f_dict[del_word]
        del tokenize[del_word]
    print('current vocab :',len(tokenize))
del_vocab.remove_total = 0

def draw_heatmap(data, row_labels, column_labels):
    # 描画する
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    plt.show()
    plt.savefig('image.png')

    return heatmap

#--------------------------------------------

def predicting(filename,nw_test = 0,mode=1,embed=5,relating = True,p=20,seed=1):
    polar_dic = get_polar_dict()
    if embed == 1:
        title2 = 'target_only'
    elif embed == 2:
        title2 = 'wod'
    elif embed == 3:
        title2 = 'tagtrain'
    elif embed == 4:
        title2 = 'alltag'
    elif embed == 5:
        title2 = 'PF'
    elif embed == 6:
        title2 = 'related_work'
    elif embed == 7:
        title2 = 'related_work'
    elif embed == 8:
        title2 = 'PF2'
    elif embed == 9:
        title2 = 'context_only'
    elif embed == 10:
        title2 = 'alltag'
    elif embed == 11:
        title2 = 'wop'
    elif embed == 12:
        title2 = 'wot'
    elif embed == 13:
        title2 = 'related_work'
    else:
        print('Usage:(weight name,train_file path,model mode,embed method'
              '(1:tag 2:tag(no target) 3:tagtrain 4:alltag 5:Position Featuring))')
        sys.exit()
    # loading model
    if mode == 1:
        estimation = context_only().estimate_model
        title1 = 'CO'
    elif mode == 2:
        estimation = CNN().model
        title1 = 'CNN'
    elif mode == 3:
        estimation = LSTM_NN().estimate_model
        title1 = 'LSTM&NN'
    elif mode == 4:
        estimation = CNN_NN().estimate_model
        title1 = 'CNN&NN'
    elif mode == 5:
        estimation = target_only().model
        title1 = 'TO'
        seed = 1
    elif mode == 6:
        char_dict = Char_Corpus(create_flg=True)
        # char_dict.load()
        # print(char_dict.char_vocab)
        estimation = Char_embed_LSTM(char_dict=char_dict.char_vocab).estimate_model
        title1 = 'LSTM_Char'
    else:
        print('Usage:(weight name,train_file path,model mode(1:CNN&NN 2:CNN 3:LSTM&NN))')
        sys.exit()
    estimation.load_weights('./model_weight/{}{}_{}model_weights.h5'.format(title1,seed,title2))

    c = CaboCha.Parser('-f1')
    train_data = reshape_data(filename)
    label_y = ans_reshape('./data/eval_ans_data.txt')
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
        text_list = noun_exploit2(train_data,embed=embed)

    # テストデータ中の未知語の割合をWord2Vecの辞書を減らすことで変化
    #del_vocab(text_list,percent=p)


    if nw_test == 1:
        NW_text_list = []
        NW_label = []
        for (label,contxt) in zip(label_y,text_list):
            if not contxt[1][1] in tokenize.keys():
                NW_text_list.append(contxt)
                NW_label.append(label)
        label_y = NW_label
        text_list = NW_text_list
    elif nw_test == 2:
        NW_text_list = []
        NW_label = []
        for (label,contxt) in zip(label_y,text_list):
            if contxt[1][1] in tokenize.keys():
                NW_text_list.append(contxt)
                NW_label.append(label)
        label_y = NW_label
        text_list = NW_text_list
    else:
        #print('CANDIDATES:', len(text_list))
        pass

    if embed == 1:
        sentence_vector = W2V(text_list)
    elif embed == 2 or embed == 10:
        sentence_vector = W2Valltag(text_list)
    elif embed == 3 or embed == 11:
        sentence_vector = W2V3(text_list)
    elif embed == 4:
        sentence_vector = W2V4(text_list)
    elif embed == 5:
        sentence_vector = W2VPF(text_list)
    elif embed == 6:
        sentence_vector = W2VR(text_list)
    elif embed == 7:
        sentence_vector = W2Vnotagtrain(text_list)
    elif embed == 8 or embed == 12:
        sentence_vector = W2VPF2(text_list)
    elif embed == 9:
        sentence_vector = W2Vcontext_only(text_list)
    elif embed == 13:
        sentence_vector = W2V2(text_list, char_dict)
    else:
        print('Usage:(weight name,train_file path,model mode,embed method'
              '(1:tag 2:tag(no target) 3:tagtrain 4:alltag 5:Position Featuring))')
        sys.exit()

    if len(sentence_vector) == 0:
        return 0

    if mode == 2:
        sent_v = padding(sentence_vector)
        sent_v = np.array(sent_v)
    elif mode == 1:
        sent_v = pad_seq2(sentence_vector)
    elif embed == 1:
        sent_v = sentence_vector
    else:
        sent_v = pad_seq(sentence_vector)
    predict = estimation.predict(sent_v)

    true_y = []
    score_y = []
    for (a,b) in zip(label_y,predict):
        true_y.append(a[0])
        score_y.append(b[0])
    backend.clear_session()
    return true_y, score_y, '{}_{}'.format(title1,title2), text_list

def main(nw=1,mode=1,embed=5,relating=True,p=20,seed=1):
    valname = './train_data/train_data_o10/validation_data'
    testname = './train_data/train_data_o10/test_data'

    # validation
    true_y, score_y, title,list = predicting(valname,nw_test=0,mode=mode,embed=embed,relating=relating,seed=seed)
    precision, recall, thresholds = precision_recall_curve(true_y, score_y)
    f_list = np.array(2*recall*precision / (recall + precision))
    f_list = np.array([i if not math.isnan(i) else 0 for i in f_list])
    max_id = f_list.argmax()
    threshold = thresholds[max_id]
    #threshold = 0.5
    print('threshold:',threshold)


    true_y, score_y, title, list = predicting(testname,nw_test=0,mode=mode,embed=embed,relating=relating,p=p,seed=seed)
    print('all word CANDIDATE:',len(true_y))
    #print(title,'all')
    predicts0 = []
    for predict in score_y:
        if predict > threshold:
            predicts0.append(1.0)
        else:
            predicts0.append(0.0)

    # with open('./example/exist_example_all.txt', 'w') as f:
    #     score2tag = {1.0:'POS',
    #                  0.0:'NEG'}
    #     ex_list = [(target[1][1], ans, label) for (target, ans, label) in zip(list, predicts0, true_y)]
    #     for (word, pred, true) in ex_list:
    #         f.write('{} | {} | {} \n'.format(word, score2tag[pred], score2tag[true]))

    p1 = precision_score(true_y,predicts0,pos_label=1)
    r1 = recall_score(true_y,predicts0,pos_label=1)
    f1 = f1_score(true_y,predicts0,pos_label=1)
    a1 = accuracy_score(true_y, predicts0)


    
    print('precision:',p1)
    print('recall:',r1)
    print('F-measure:',f1)
    print('accuracy:',a1)



    true_y, score_y, title,list = predicting(testname,nw_test=1,mode=mode,embed=embed,relating=relating,p=p,seed=seed)
    print('unk word CANDIDATE:',len(true_y))
    print('num of 1:',len([a for a in true_y if a == 1]))
    #print(title,'nw')
    predicts1 = []
    for predict in score_y:
        if predict > threshold:
            predicts1.append(1.0)
        else:
            predicts1.append(0.0)
    #print([(target[1][1], label, ans) for (target, label, ans) in zip(list, true_y, predicts1)])
    # with open('./example/exist_example_unk.txt', 'w') as f:
    #     score2tag = {1.0:'POS',
    #                  0.0:'NEG'}
    #     ex_list = [(target[1][1], ans, label) for (target, ans, label) in zip(list, predicts1, true_y)]
    #     for (word, pred, true) in ex_list:
    #         f.write('{} | {} | {} \n'.format(word, score2tag[pred], score2tag[true]))

    p2 = precision_score(true_y,predicts1,pos_label=1)
    r2 = recall_score(true_y,predicts1,pos_label=1)
    f2 = f1_score(true_y,predicts1,pos_label=1)
    a2 = accuracy_score(true_y,predicts1)

    print('precision:',p2)
    print('recall:',r2)
    print('F-measure:',f2)
    print('accuracy:',a2)

    if p != 100:
        true_y, score_y, title, list = predicting(testname,nw_test=2,mode=mode,embed=embed,relating=relating,p=p,seed=seed)
        print('known word CANDIDATE:', len(true_y))
        print('num of 1:', len([a for a in true_y if a == 1]))
        #print(title,'nonw')
        predicts2 = []
        for predict in score_y:
            if predict > threshold:
                predicts2.append(1.0)
            else:
                predicts2.append(0.0)
        '''
        tmp00 = []
        tmp01 = []
        tmp10 = []
        tmp11 = []
        for (target,pred,true) in zip(list,predicts2,true_y):
            if pred == 1.0 and true == 0.0:
                tmp10.append(target[1][1])
                #print('{}:{} | {}'.format(target[1][1],pred,true))
            elif pred == 0.0 and true == 1.0:
                tmp01.append(target[1][1])
            elif pred == 0.0 and true == 0.0:
                tmp00.append(target[1][1])
            else:
                tmp11.append(target[1][1])
        data = np.array([[len(tmp00),len(tmp01)],[len(tmp10),len(tmp11)]])
        plt.rcParams['font.size'] = 16
        sns.heatmap(data,cmap='Blues',annot=True,fmt='d',
                    xticklabels=['興味なし','興味あり'],
                    yticklabels=['興味なし','興味あり'])
        plt.xlabel('正解ラベル')
        plt.ylabel('推定ラベル')
        plt.tight_layout()
        plt.savefig('./image/heatmap.png')
        #counter01 = Counter(tmp01)
        #counter10 = Counter(tmp10)
        #print([(k,v) for k,v in counter01.most_common()])
        #print([(k, v) for k, v in counter10.most_common()])
        '''

        # with open('./example/exist_example_known.txt', 'w') as f:
        #     score2tag = {1.0: 'POS',
        #                  0.0: 'NEG'}
        #     ex_list = [(target[1][1], ans, label) for (target, ans, label) in zip(list, predicts2, true_y)]
        #     for (word, pred, true) in ex_list:
        #         f.write('{} | {} | {} \n'.format(word, score2tag[pred], score2tag[true]))

        p3 = precision_score(true_y,predicts2,pos_label=1)
        r3 = recall_score(true_y,predicts2,pos_label=1)
        f3 = f1_score(true_y,predicts2,pos_label=1)
        a3 = accuracy_score(true_y, predicts2)
    else:
        p3 = 0
        r3 = 0
        f3 = 0
        a3 = 0


    print('precision:',p3)
    print('recall:',r3)
    print('F-measure:',f3)
    print('accuracy:', a3)
    '''
    precision, recall, thresholds = precision_recall_curve(true_y, score_y)
    area = auc(recall, precision)
    print('AUC:', area)
    fig_title = title + ' AUC:' + str(area) + 'CANDIDATES:' + str(len(true_y))
    file_title = './image/' + title + '.png'

    plt.plot(recall, precision)
    plt.title(fig_title)
    plt.savefig(file_title)
    '''

    #return predicts0,predicts1,predicts2,true_y
    return (p1,r1,f1),(p2,r2,f2),(p3,r3,f3)


def ensemble():
    valname = './train_data/train_data_o10/validation_data'
    testname = './train_data/train_data_o10/test_data'

    # validation
    true_y, score_y1, title,list = predicting(valname, nw_test=0, mode=4, embed=7,relating=True)
    true_y, score_y2, title,list = predicting(valname, nw_test=0, mode=4, embed=3,relating=False)
    true_y, score_y3, title,list = predicting(valname, nw_test=0, mode=4, embed=8, relating=False)
    pos_y = np.amax(np.array([score_y1,score_y2,score_y3],dtype=np.float32),axis=0)
    neg_y = np.amax(np.array([1-np.array(score_y1),1-np.array(score_y2),1-np.array(score_y3)],dtype=np.float32),axis=0)
    score_y = np.array([a if a >= b else 1-b for a,b in zip(pos_y,neg_y)])

    precision, recall, thresholds = precision_recall_curve(true_y, score_y)
    f_list = np.array(2 * recall * precision / (recall + precision))
    f_list = np.array([i if not math.isnan(i) else 0 for i in f_list])
    max_id = f_list.argmax()
    threshold = thresholds[max_id]
    # print('threshold:',threshold)

    true_y, score_y1, title,list = predicting(testname, nw_test=0, mode=4, embed=7, relating=True)
    true_y, score_y2, title,list = predicting(testname, nw_test=0, mode=4, embed=3, relating=False)
    true_y, score_y3, title,list = predicting(testname, nw_test=0, mode=4, embed=8, relating=False)
    pos_y = np.amax(np.array([score_y1, score_y2, score_y3], dtype=np.float32), axis=0)
    neg_y = np.amax(np.array([1 - np.array(score_y1), 1 - np.array(score_y2), 1 - np.array(score_y3)], dtype=np.float32), axis=0)
    score_y = np.array([a if a >= b else 1 - b for a, b in zip(pos_y, neg_y)])
    # print(title,'all')
    predicts0 = []
    for predict in score_y:
        if predict > threshold:
            predicts0.append(1.0)
        else:
            predicts0.append(0.0)
    p1 = precision_score(true_y, predicts0, pos_label=1)
    r1 = recall_score(true_y, predicts0, pos_label=1)
    f1 = f1_score(true_y, predicts0, pos_label=1)
    a1 = accuracy_score(true_y, predicts0)

    true_y, score_y1, title,list = predicting(testname, nw_test=1, mode=4, embed=7, relating=True)
    true_y, score_y2, title,list = predicting(testname, nw_test=1, mode=4, embed=3, relating=False)
    true_y, score_y3, title,list = predicting(testname, nw_test=1, mode=4, embed=8, relating=False)
    pos_y = np.amax(np.array([score_y1, score_y2, score_y3], dtype=np.float32), axis=0)
    neg_y = np.amax(np.array([1 - np.array(score_y1), 1 - np.array(score_y2), 1 - np.array(score_y3)], dtype=np.float32), axis=0)
    score_y = np.array([a if a >= b else 1 - b for a, b in zip(pos_y, neg_y)])
    # print(title,'nw')
    predicts1 = []
    for predict in score_y:
        if predict > threshold:
            predicts1.append(1.0)
        else:
            predicts1.append(0.0)
    # for i,(aa,bb,cc) in enumerate(zip(target_list,predicts,true_y),1):
    #   if i==9:
    #      break
    # print('{}:{}:{}'.format(aa[1],bb,cc))
    p2 = precision_score(true_y, predicts1, pos_label=1)
    r2 = recall_score(true_y, predicts1, pos_label=1)
    f2 = f1_score(true_y, predicts1, pos_label=1)
    a2 = accuracy_score(true_y, predicts1)

    true_y, score_y1, title,list = predicting(testname, nw_test=2, mode=4, embed=7, relating=True)
    true_y, score_y2, title,list = predicting(testname, nw_test=2, mode=4, embed=3, relating=False)
    true_y, score_y3, title,list = predicting(testname, nw_test=2, mode=4, embed=8, relating=False)
    pos_y = np.amax(np.array([score_y1, score_y2, score_y3], dtype=np.float32), axis=0)
    neg_y = np.amax(np.array([1 - np.array(score_y1), 1 - np.array(score_y2), 1 - np.array(score_y3)], dtype=np.float32), axis=0)
    score_y = np.array([a if a >= b else 1 - b for a, b in zip(pos_y, neg_y)])
    # print(title,'nonw')
    predicts2 = []
    for predict in score_y:
        if predict > threshold:
            predicts2.append(1.0)
        else:
            predicts2.append(0.0)

    p3 = precision_score(true_y, predicts2, pos_label=1)
    r3 = recall_score(true_y, predicts2, pos_label=1)
    f3 = f1_score(true_y, predicts2, pos_label=1)
    a3 = accuracy_score(true_y, predicts2)

    return (p1,r1,f1),(p2,r2,f2),(p3,r3,f3)

if __name__ == '__main__':
    #############################################################################################
    ## Usage : main(testing new word , mode , embed)                                           ##
    ## testing new word >> 0:ALL 1:YES 2:NEVER                                                 ##
    ## mode >> 1:CNNCO 2:CNN 3:LSTM&NN 4:CNN&NN                                                ##
    ## embed >> 1:tag 2:tag(no target) 3:tagtrain 4:alltag 5:Position Featuring 6:related_work ##
    #############################################################################################
    seed = 1
    nw_percent = 20

    #for nw_percent in range(20,70,10):

    print('proposal LSTM')
    result1, result2, result3 = main(nw=0, mode=3, embed=10, relating=False, p=nw_percent, seed=seed)
    table = pd.DataFrame([[result1[0], result1[1], result1[2]],
                          [result2[0], result2[1], result2[2]],
                          [result3[0], result3[1], result3[2]]],
                         index=['all words', 'unknown words', 'known words'],
                         columns=['Precision', 'Recall', 'F-measure'])
    table.to_csv('./result/alltaglstm1_{}_{}.csv'.format(seed, nw_percent), index=False)
    '''

    # All tag replace
    print('proposal')
    result1, result2, result3 = main(nw=0, mode=4, embed=10, relating=True, p=nw_percent,seed=seed)
    table = pd.DataFrame([[result1[0], result1[1], result1[2]],
                          [result2[0], result2[1], result2[2]],
                          [result3[0], result3[1], result3[2]]],
                         index=['all words', 'unknown words', 'known words'],
                         columns=['Precision', 'Recall', 'F-measure'])
    print(table)
    table.to_csv('./result/alltag1_{}_{}.csv'.format(seed, nw_percent), index=False)
    

    # Exisitng Model
    print('exist')
    result1, result2, result3 = main(nw=0, mode=3, embed=6, relating=False,p=nw_percent,seed=seed)
    exs = pd.DataFrame([[result1[0], result1[1], result1[2]],
                        [result2[0], result2[1], result2[2]],
                        [result3[0], result3[1], result3[2]]],
                       index=['all words', 'unknown words', 'known words'],
                       columns=['Precision', 'Recall', 'F-measure'])
    print(exs)
    exs.to_csv('./result/real_exist1_{}_{}.csv'.format(seed, nw_percent), index=False)

    #pprint([ (proposal[0],proposal[2],exist[2],proposal[1]) for proposal,exist in zip(list1,list2) if proposal[1] == proposal[2] and exist[1] != exist[2]])

    
    result1, result2, result3 = main(nw=0, mode=6, embed=13, relating=False, p=nw_percent, seed=seed)
    table = pd.DataFrame([[result1[0], result1[1], result1[2]],
                        [result2[0], result2[1], result2[2]],
                        [result3[0], result3[1], result3[2]]],
                       index=['all words', 'unknown words', 'known words'],
                       columns=['Precision', 'Recall', 'F-measure'])
    table.to_csv('./result/exist_char1_{}_{}.csv'.format(seed, nw_percent), index=False)
    
    # Tag Model
    print('tag')
    result1, result2, result3 = main(nw=0, mode=4, embed=3, relating=False, p=nw_percent, seed=seed)
    tag = pd.DataFrame([[result1[0], result1[1], result1[2]],
                        [result2[0], result2[1], result2[2]],
                        [result3[0], result3[1], result3[2]]],
                       index=['all words', 'unknown words', 'known words'],
                       columns=['Precision', 'Recall', 'F-measure'])
    tag.to_csv('./result/tag2_{}_{}.csv'.format(seed, nw_percent), index=False)
    
    # Dependency Model
    print('dep')
    result1,result2,result3 = main(nw=0, mode=4, embed=7, relating=True,p=nw_percent,seed=seed)
    dep = pd.DataFrame([[result1[0], result1[1], result1[2]],
                        [result2[0], result2[1], result2[2]],
                        [result3[0], result3[1], result3[2]]],
                       index=['all words', 'unknown words', 'known words'],
                       columns=['Precision', 'Recall', 'F-measure'])
    
    # Position Model
    print('posi')
    result1,result2,result3 = main(nw=0, mode=4, embed=8, relating=False,p=nw_percent,seed=seed)
    pos = pd.DataFrame([[result1[0], result1[1], result1[2]],
                        [result2[0], result2[1], result2[2]],
                        [result3[0], result3[1], result3[2]]],
                       index=['all words', 'unknown words', 'known words'],
                       columns=['Precision', 'Recall', 'F-measure'])
                       
    dep.to_csv('./result/dependency_{}_{}.csv'.format(seed,nw_percent),index=False)
    
    pos.to_csv('./result/position_{}_{}.csv'.format(seed,nw_percent),index=False)

    
    # Context Only
    result1,result2,result3 = main(nw=0, mode=1, embed=9, relating=True,p=nw_percent)
    print(pd.DataFrame([[result1[0], result1[1], result1[2]],
                        [result2[0], result2[1], result2[2]],
                        [result3[0], result3[1], result3[2]]],
                       index=['all words', 'unknown words', 'known words'],
                       columns=['Precision', 'Recall', 'F-measure']))
    
    # Target Only
    print('tar')
    result1,result2,result3 = main(nw=0, mode=5, embed=1, relating=True,p=nw_percent,seed=seed)
    tar = pd.DataFrame([[result1[0], result1[1], result1[2]],
                        [result2[0], result2[1], result2[2]],
                        [result3[0], result3[1], result3[2]]],
                       index=['all words', 'unknown words', 'known words'],
                       columns=['Precision', 'Recall', 'F-measure'])
    tar.to_csv('./result/target2_{}_{}.csv'.format(seed, nw_percent), index=False)
    
    
    # Proposal w/o Dep
    print('Proposal w/o Dep')
    result1, result2, result3 = main(nw=0, mode=4, embed=2, relating=False,p=nw_percent,seed=seed)
    wod = pd.DataFrame([[result1[0], result1[1], result1[2]],
                        [result2[0], result2[1], result2[2]],
                        [result3[0], result3[1], result3[2]]],
                       index=['all words', 'unknown words', 'known words'],
                       columns=['Precision', 'Recall', 'F-measure'])
    print(wod)
    wod.to_csv('./result/wod1_{}_{}.csv'.format(seed, nw_percent), index=False)
    
    # Proposal w/o posi
    print('Proposal w/o posi')
    result1, result2, result3 = main(nw=0, mode=4, embed=11, relating=True,p=nw_percent,seed=seed)
    wop = pd.DataFrame([[result1[0], result1[1], result1[2]],
                        [result2[0], result2[1], result2[2]],
                        [result3[0], result3[1], result3[2]]],
                       index=['all words', 'unknown words', 'known words'],
                       columns=['Precision', 'Recall', 'F-measure'])
    print(wop)
    wop.to_csv('./result/wop1_{}_{}.csv'.format(seed, nw_percent), index=False)
    
    #w/o Tag
    print('w/o Tag')
    result1, result2, result3 = main(nw=0, mode=4, embed=12, relating=True, p=nw_percent, seed=seed)
    wot = pd.DataFrame([[result1[0], result1[1], result1[2]],
                        [result2[0], result2[1], result2[2]],
                        [result3[0], result3[1], result3[2]]],
                       index=['all words', 'unknown words', 'known words'],
                       columns=['Precision', 'Recall', 'F-measure'])
    print(wot)
    wot.to_csv('./result/wot1_{}_{}.csv'.format(seed, nw_percent), index=False)
    '''


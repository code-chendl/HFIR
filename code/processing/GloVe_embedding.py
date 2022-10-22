#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : GloVe_embedding.py
@Time    : 2022/09/11 19:48
@Author  : Danlei Chen, Wang Su
@Version : 1.0
'''

import pickle
import xlrd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import models


#--------Load data--------------------#
def load_f(path):
    with open(path, 'rb')as f:
        data = pickle.load(f)
    return data

#-------Read excel--------------------#
def readXls(path):
    xl=xlrd.open_workbook(path)
    sheet=xl.sheets()[0]
    data=[]
    for i in range(0,sheet.ncols): # ncols represents: read by column
        data.append(list(sheet.col_values(i)))
    return data

#-----Preprocessing: tokenize, remove the stop words, lemmatize-------------
def dealTxt(txt):
    txt=txt.lower()   #Convert uppercase letters in text to lowercase
    wnl = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    #Symbols=[',','.','<','>',':',')','(','!','?','"','#','/',';','&','_','*','《','》','%','~','|','+','@','^','“','”','…','...','......','`','``','•']
    word_tokens=word_tokenize(txt)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]   #remove the stop words
    #noSymbol_sentence = [w for w in filtered_sentence if not w in Symbols]   #remove the punctuations
    final_solved=[]
    for item in filtered_sentence:
        final_solved.append(wnl.lemmatize(item))
    return final_solved

#--------------Count the words-----------------
def getFileLineNums(filename):
	f = open(filename, 'r',encoding='utf-8')
	count = 0
	for line in f:
		count += 1
	return count

#-----If you open the GloVe file under Windows, add a line at the beginning of it: The number of words; Word embedding dimension
def prepend_slow(infile, outfile, line):
	with open(infile, 'r',encoding='utf-8') as fin:
		with open(outfile, 'w',encoding='utf-8') as fout:
			fout.write(line + "\n")
			for line in fin:
				fout.write(line)

#--------embed words into a vector space----------
def turn2Vector(final_txts):
    num_lines = getFileLineNums('glove.twitter.27B.200d.txt')
    gensim_file = 'glove_model.txt'
    gensim_first_line = "{} {}".format(num_lines, 200)
    prepend_slow('glove.twitter.27B.200d.txt', gensim_file, gensim_first_line)
    model = models.KeyedVectors.load_word2vec_format(gensim_file,binary=False)
    max_len = 50 #Maximum sentence length
    txts_arr = np.zeros((len(final_txts), max_len, 200)).astype(np.float32)
    empty = np.zeros(200).astype(np.float32)
    for idx, txt in enumerate(final_txts):
        for jdx, word in enumerate(txt):
            if jdx == max_len:
                break
            else:
                if word in model:
                    txts_arr[idx, jdx, :] = model[word]
                else:
                    txts_arr[idx, jdx, :] = empty
    return txts_arr

def main():
    excel=readXls(r'uncor_txt_arr_train_new.xlsx')
    txts=excel[0]
    final_txts=[]
    for t in txts:
        final_solved=dealTxt(t)
        final_txts.append(final_solved)
    #print(final_txts)
    len_txts = []
    for i in final_txts:
        len_txts.append(len(i))
    #print(max(len_txts))   #Maximum sentence length
    txts_arr=turn2Vector(final_txts)
    #print(txts_arr)
    with open(r'uncor_txt_arr_train_new.pickle', 'wb')as f:     #save results in the form of pickles
        pickle.dump(txts_arr, f)
    print(len(txts_arr))
    #print(load_f(r'txt.pickle'))

if __name__ == '__main__':
     main()


 

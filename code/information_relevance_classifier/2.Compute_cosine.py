
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : Compute_cosine.py
@Time    : 2022/09/09 15:05
@Author  : Danlei Chen, Wang Su
@Version : 1.0
'''

import pickle
import numpy as np
import xlwt
import xlrd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import models
from scipy.spatial.distance import cosine


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

#-------save results in the form of pickles-----------
def save_data(data, path):
    with open(path, 'wb')as f:
        pickle.dump(data, f)

#-----Preprocessing: tokenize, remove the stop words, lemmatize-------------
def dealTxt(txt):
    txt=txt.lower()  #Convert uppercase letters in text to lowercase
    wnl = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    Symbols=['#','@','<','>']
    #Symbols=[',','.','<','>',':',')','(','!','?','"','#','/',';','&','_','*','《','》','%','~','|','+','@','^','“','”','…','...','......','`','``','•']
    word_tokens=word_tokenize(txt)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]   #remove the stop words
    noSymbol_sentence = [w for w in filtered_sentence if not w in Symbols] #remove the punctuations
    final_solved=[]
    for item in noSymbol_sentence:
        final_solved.append(wnl.lemmatize(item))
    return final_solved

#---Count the words-----------------
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

#-------------Word vectors are aggregated with an unsupervised method of average to form the sentence embedding----------
def turn2Vector(final_txts):
    num_lines = getFileLineNums('glove.twitter.27B.200d.txt')
    gensim_file = 'glove_model.txt'
    gensim_first_line = "{} {}".format(num_lines, 200)
    prepend_slow('glove.twitter.27B.200d.txt', gensim_file, gensim_first_line)
    model = models.KeyedVectors.load_word2vec_format(gensim_file,binary=False)
    sent_arr=np.zeros((len(final_txts),200)).astype(np.float32)
    for idx, txt in enumerate(final_txts):
        txt_arr=[]
        count=0
        for jdx, word in enumerate(txt):
            if word in model: 
                txt_arr.append(model[word])
                count=count+1
            #else:
                #continue
        txt_arr=np.array(txt_arr).astype(np.float32)
        #print(txt_arr.shape)
        v=txt_arr.sum(axis=0) #axis=1: add by row, axis=0: add by column
        if count==0: #If all words of a short text are not included in the pre-trained GloVe:
            sentence=np.zeros(200).astype(np.float32)
        else:
            sentence=v/count #The sentence-level embedding is calculated using the average method
        sent_arr[idx,:]=sentence
        #print(sent_arr.shape)
    return sent_arr

#-------Calculate the cosine similarity-------------------------
def cos_compute(text_arr,tag_arr):
    cos_arr=[]
    for i in range(len(text_arr)):
        empty = np.zeros(200).astype(np.float32)
        if (text_arr[i,:]==empty).all() or (tag_arr[i,:]== empty).all():
            result=0
        else:
            cos_distance=cosine(text_arr[i,:],tag_arr[i,:])    #cos_distance=1-cos_similarity
            result=0.5+ 0.5*(1-cos_distance)   #map similarity scores to the interval [0,1]
        cos_arr.append(result)
    return cos_arr

def main():
    excel_txt=readXls(r'train_txt.xlsx')
    excel_tag=readXls(r'train_tag_uni.xlsx')
    txts=excel_txt[0]
    tags=excel_tag[0]
    '''
    tags_vgg16=excel_tag[0]
    tags_vgg19=excel_tag[1]
    tags_resnet=excel_tag[2]
    tags_inception=excel_tag[3]
    tags_xception=excel_tag[4]
    tags=[]
    for i in range(0,len(tags_vgg16)):
        tags.append(tags_vgg16[i]+tags_vgg19[i]+tags_resnet[i]+tags_inception[i]+tags_xception[i])
    print(len(tags))
    print(tags[0])  
    '''
    #--------sentence-level textual features--------
    final_txts=[]
    for t in txts:
        final_solvedtxt=dealTxt(t)
        final_txts.append(final_solvedtxt)
    txts_arr=turn2Vector(final_txts)
    #save_data(txts_arr, r'all_doc_arr_test.pickle') 
    #--------the combined tag-ANP feature (sentence-level)----------
    final_tags=[]
    for g in tags:
        final_solvedtag=dealTxt(g)
        final_tags.append(final_solvedtag)
    tags_arr=turn2Vector(final_tags)
    save_data(tags_arr, r'all_tag_arr_train.pickle')
    print(tags_arr.shape)
    #------compute the image-text similarity------------
    cos_cor=cos_compute(txts_arr,tags_arr)
    #--------write the result to excel-----------------
    workbook = xlwt.Workbook(encoding="ascii")
    worksheet = workbook.add_sheet('Cos')
    for i in range(0,len(cos_cor)):
        worksheet.write(i,0,label =cos_cor[i])
    workbook.save('cos_txtTag_uni_train.xls')
    
if __name__ == '__main__':
     main()

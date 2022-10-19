from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import xlrd
from gensim import models
import pickle
import numpy as np
import codecs

#-----读取pickle-------
def load_f(path):
    with open(path, 'rb')as f:
        data = pickle.load(f)
    return data

#-----读取excel--------
def readXls(path):
    xl=xlrd.open_workbook(path)
    sheet=xl.sheets()[0]
    data=[]
    for i in range(0,sheet.ncols): # ncols 表示按列读取
        data.append(list(sheet.col_values(i)))
    return data

#-----分词，去停用词，词形还原--------------
def dealTxt(txt):
    txt=txt.lower()#将文本中的大写字母转化为小写
    wnl = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    #Symbols=[',','.','<','>',':',')','(','!','?','"','#','/',';','&','_','*','《','》','%','~','|','+','@','^','“','”','…','...','......','`','``','•']
    word_tokens=word_tokenize(txt)#分词
    filtered_sentence = [w for w in word_tokens if not w in stop_words]#去停用词
    #noSymbol_sentence = [w for w in filtered_sentence if not w in Symbols]#去标点
    final_solved=[]
    for item in filtered_sentence:
        final_solved.append(wnl.lemmatize(item))#词形还原
    return final_solved

#---计算单词数
def getFileLineNums(filename):
	f = open(filename, 'r',encoding='utf-8')
	count = 0
	for line in f:
		count += 1
	return count

#-----Windows下打开词向量文件，在开始增加一行：所有的单词数 词向量的维度
def prepend_slow(infile, outfile, line):
	with open(infile, 'r',encoding='utf-8') as fin:
		with open(outfile, 'w',encoding='utf-8') as fout:
			fout.write(line + "\n")
			for line in fin:
				fout.write(line)

#--------生成词向量----------
def turn2Vector(final_txts):
    num_lines = getFileLineNums('glove.twitter.27B.200d.txt')
    gensim_file = 'glove_model.txt'
    gensim_first_line = "{} {}".format(num_lines, 200)
    prepend_slow('glove.twitter.27B.200d.txt', gensim_file, gensim_first_line)
    model = models.KeyedVectors.load_word2vec_format(gensim_file,binary=False)
    max_len = 50 #最大文本长度（实际本语料中最大文本长度为32）
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
    #print(max(len_txts))#最大文本长度
    txts_arr=turn2Vector(final_txts)
    #print(txts_arr)
    with open(r'uncor_txt_arr_train_new.pickle', 'wb')as f:#保存为pickle文件
        pickle.dump(txts_arr, f)
    print(len(txts_arr))
    #print(load_f(r'txt.pickle'))

if __name__ == '__main__':
     main()


 

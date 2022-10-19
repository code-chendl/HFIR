from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import xlrd
from gensim import models
import pickle
import numpy as np
import codecs
from scipy.spatial.distance import cosine
import xlwt

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

#-------存储pickle-----------
def save_data(data, path):
    with open(path, 'wb')as f:
        pickle.dump(data, f)

#-----分词，去停用词，词形还原--------------
def dealTxt(txt):
    txt=txt.lower()#将文本中的大写字母转化为小写
    wnl = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    Symbols=['#','@','<','>']
    #Symbols=[',','.','<','>',':',')','(','!','?','"','#','/',';','&','_','*','《','》','%','~','|','+','@','^','“','”','…','...','......','`','``','•']
    word_tokens=word_tokenize(txt)#分词
    filtered_sentence = [w for w in word_tokens if not w in stop_words]#去停用词
    noSymbol_sentence = [w for w in filtered_sentence if not w in Symbols]#去标点
    final_solved=[]
    for item in noSymbol_sentence:
        final_solved.append(wnl.lemmatize(item))#词形还原
    return final_solved

#---计算单词数-----------------
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

#--------生成句子级向量----------
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
        v=txt_arr.sum(axis=0) #axis=1表示按行相加 , axis=0表示按列相加
        if count==0: #若数据集中某短文本的所有word不在model内
            sentence=np.zeros(200).astype(np.float32)
        else:
            sentence=v/count #采用平均法求句向量
        sent_arr[idx,:]=sentence
        #print(sent_arr.shape)
    return sent_arr

#-------计算余弦相似度----------------
def cos_compute(text_arr,tag_arr):
    cos_arr=[]
    for i in range(len(text_arr)):
        empty = np.zeros(200).astype(np.float32)
        if (text_arr[i,:]==empty).all() or (tag_arr[i,:]== empty).all():
            result=0
        else:
            cos_distance=cosine(text_arr[i,:],tag_arr[i,:])#余弦距离=1-余弦相似度
            result=0.5+ 0.5*(1-cos_distance)#归一化至[0,1]
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
    #--------文本句子级特征--------
    final_txts=[]
    for t in txts:
        final_solvedtxt=dealTxt(t)
        final_txts.append(final_solvedtxt)
    txts_arr=turn2Vector(final_txts)
    #save_data(txts_arr, r'all_doc_arr_test.pickle') 
    #--------Tag句子级特征----------
    final_tags=[]#法1：ANP标签特征进行分词去停用词等处理
    for g in tags:
        final_solvedtag=dealTxt(g)
        final_tags.append(final_solvedtag)
    #法2：Tag标签特征直接生成词向量(在excel“Boston_bull”的词组内"_"替换为" "）
    #tags=[tag.split(' ') for tag in tags]
    tags_arr=turn2Vector(final_tags)
    save_data(tags_arr, r'all_tag_arr_train.pickle')
    print(tags_arr.shape)


    cos_cor=cos_compute(txts_arr,tags_arr)
    #将结果写入excel文件中--------
    workbook = xlwt.Workbook(encoding="ascii")
    worksheet = workbook.add_sheet('Cos')
    for i in range(0,len(cos_cor)):
        worksheet.write(i,0,label =cos_cor[i])
    workbook.save('cos_txtTag_uni_train.xls')
    
if __name__ == '__main__':
     main()

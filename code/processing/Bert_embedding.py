import os
import codecs
from keras_bert import load_trained_model_from_checkpoint
import numpy as np
from keras_bert import Tokenizer
import pickle
import numpy as np
import xlrd

#首先pip install keras_bert(升级tensorflow版本）
#-----读取excel--------
def readXls(path):
    xl=xlrd.open_workbook(path)
    sheet=xl.sheets()[0]
    data=[]
    for i in range(0,sheet.ncols): # ncols 表示按列读取
        data.append(list(sheet.col_values(i)))
    return data

def bert(texts):
    #接受输入['', ''];
    #加载模型；
    
    pretrained_path = 'uncased_L-12_H-768_A-12'
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
    vocab_path = os.path.join(pretrained_path, 'vocab.txt')
    print(config_path)
    print( checkpoint_path)
    model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    
    #加载分词词典；
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
            
    #抽取特征；
    tokenizer = Tokenizer(token_dict)
    count=0
    features = []
    for text in texts:
        tokens = tokenizer.tokenize(text)
        #num=len(tokens)
        #number.append(num)-最大长度98
        indices, segments = tokenizer.encode(first=text, max_len=512)#max_len要求设置为512
        predicts = model.predict([np.array([indices]), np.array([segments])])[0]
        feature = predicts
        features.append(feature)
        print(count)
        count=count+1
    return features

def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def main():
    excel=readXls(r'txt.xlsx')
    txts=excel[0]
    vectors=bert(txts)
    with open(r'bert_txt.pickle', 'wb')as f:#保存为pickle文件
        pickle.dump(vectors, f)
    print(np.array(read_pickle(r'bert_txt.pickle')).shape)

if __name__ == '__main__':
     main()

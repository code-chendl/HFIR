import pickle
import time
from attention.layers import AttentionLayer
from keras.callbacks import EarlyStopping, ModelCheckpoint,CSVLogger
from keras.layers import (LSTM,  Bidirectional, Dense,Dropout)
from keras.models import Input, Model,load_model
from sklearn.metrics import classification_report
from sklearn.model_selection import  train_test_split
from keras.utils import CustomObjectScope


#---------------------读取数据集--------------------------------------------------#
def load_data(path):
    with open(path, 'rb')as f:
        data = pickle.load(f)
    return data
#---------------------保存文件----------------------------------------------------#
def save_data(data, path):
    with open(path, 'wb')as f:
        pickle.dump(data, f)

#----------------构造基于BiLSTM SelfAttention的文本情感识别模型---------------------#
def text_model(train_X,train_Y,test_X,test_Y,taskSavePath):

    #设置模型参数
    unitParameter=128    # the number of hidden neurons in each cell
    dropoutParameter=0.3   # dropout
    bachsizeParameter=32   # batch size
    #构建模型
    inputs = Input(shape=(50, 200))
    vect = Bidirectional(LSTM(unitParameter, return_sequences=True))(inputs) #BiLSTM模型
    vects = Dropout(dropoutParameter)(vect)
    atten_vect = AttentionLayer(name='attention')(vects)

    outputs = Dense(3, activation = 'softmax')(atten_vect)
    model = Model(inputs= inputs, outputs= outputs)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['accuracy'])
    model.summary()#打印网络结构和参数状况

    #EarlyStopping机制设置,并将符合早停条件最佳模型保存
    early_stopping = EarlyStopping(monitor='val_acc', patience=10,mode='max')
    best_checkpoint = taskSavePath+'BiLSTM_SelfAttention_Text.h5'
    modelcheckpoint = ModelCheckpoint(best_checkpoint, monitor= 'val_acc', save_best_only= True, 
                                        mode= 'max', verbose= 1)
    
    #记录每轮训练后的acc loss val_acc val_loss
    csv_path = taskSavePath+'BiLSTM_SelfAttention_Text.csv'
    csv_log = CSVLogger(csv_path)

    #训练模型
    start_train =time.perf_counter() #记录训练开始时间
    model.fit(train_X, train_Y, epochs = 100, validation_data = (test_X, test_Y), 
            batch_size = bachsizeParameter, verbose = 1,callbacks=[early_stopping,modelcheckpoint,csv_log])
    end_train =time.perf_counter() #记录训练结束时间

        
    #读取保存在best_checkpoint中的最佳模型文件,并在其上运行测试集
    with CustomObjectScope({'AttentionLayer': AttentionLayer}):
        Best_model = load_model(best_checkpoint)
    start_test =time.perf_counter() #记录测试开始时间
    predictions = Best_model.predict(test_X, batch_size = 32)
    end_test =time.perf_counter() #记录测试结束时间

    save_data(predictions,taskSavePath+"predictions.pickle") #保存模型预测结果
    print(classification_report(test_Y.argmax(axis= 1), predictions.argmax(axis= 1), digits= 5))
    
    # 打印输出模型运行的时间
    print("时间打印\n")
    print(start_train)
    print("\n")
    print(end_train)
    print("\n")
    print(start_test)
    print("\n")
    print(end_test)
    print("\n")
    print('Running time: %s Seconds'%(end_train-start_train))
    print('test time: %s Seconds'%(end_test-start_test))

if __name__ == '__main__':
    #------------------加载数据----------------------------------------------

    taskSavePath="Our_tw1517/" #taskSavePath为模型训练过程中输出文件的保存位置
    basePath="/gpfs/share/home/2101211725/HFIR/DataSets/twitter1517/"
    X_text=load_data(basePath+"text_over/"+"twitter1517_all_text_arr.pickle")
    Y=load_data(basePath+"label_over/"+'all_text_label.pickle')
    train_X, test_X, train_Y, test_Y = train_test_split(X_text, Y, test_size=0.2, random_state=1) 
    #----------------训练并测试模型--------------------
    text_model(train_X,train_Y,test_X,test_Y,taskSavePath)

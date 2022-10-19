import pickle
import time
from keras import regularizers
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Input
from keras.models import Model, load_model
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


#---------------------读取数据集----------------------------------------------------#
def load_data(path):
    with open(path, 'rb')as f:
        data = pickle.load(f)
    return data
#---------------------保存文件----------------------------------------------------#
def save_data(data, path):
    with open(path, 'wb')as f:
        pickle.dump(data, f)
# -------------------------构造DeepSentiBank情感识别模型---------------------------#
def ANP_model(train_X,train_Y,test_X,test_Y,taskSavePath):
    #构建模型,由三层全连层组成，最后一层全连层用于分类;
    anp_input = Input(shape= (4096,))
    anp_model=Dense(512, activation="relu", W_regularizer=regularizers.l2(0.01))(anp_input)
    anp_model = Dropout(0.5)(anp_model)
    anp_model=Dense(256, activation="relu", W_regularizer=regularizers.l2(0.01))(anp_input)
    anp_model = Dropout(0.5)(anp_model)
    outputs = Dense(3, activation = 'softmax')(anp_model)
    #设置模型的相关参数并构建
    model = Model(inputs= anp_input, outputs= outputs)
    opt = SGD(lr=0.001, momentum= 0.9)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt , metrics= ['accuracy'])
    model.summary()#打印网络结构和参数状况
    
    #EarlyStopping机制设置,并将符合早停条件最佳模型保存
    earlystopping = EarlyStopping(monitor= 'loss', patience=20, verbose= 1)
    best_checkpoint = taskSavePath+'ANPDNN.h5'
    modelcheckpoint = ModelCheckpoint(best_checkpoint, monitor= 'val_acc', save_best_only= True, 
                                        mode= 'max', verbose= 1)
    #记录每轮训练后的acc loss val_acc val_loss
    csv_path = taskSavePath+'ANPDNN.csv'
    csv_log = CSVLogger(csv_path)

    #训练模型
    start_train =time.perf_counter() #记录训练开始时间
    model.fit(train_X, train_Y, epochs=500,
                validation_data=(test_X, test_Y), batch_size=128, 
                verbose=1, callbacks= [earlystopping, modelcheckpoint,csv_log])
    end_train =time.perf_counter() #记录训练结束时间


    #读取保存在best_checkpoint中的最佳模型文件,并在其上运行测试集
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
    taskSavePath="tw1517_Over/" #taskSavePath为模型训练过程中输出文件的保存位置
    basePath="/gpfs/share/home/2101211725/HFIR/DataSets/twitter1517/"
    X_anp=load_data(basePath+"anp_over/"+"twitter1517_OS_anp_arr.pickle")
    Y=load_data(basePath+"label_over/"+'all_image_label.pickle')
    train_X, test_X, train_Y, test_Y = train_test_split(X_anp, Y, test_size=0.2, random_state=1) 
    #----------------训练并测试模型--------------------
    ANP_model(train_X,train_Y,test_X,test_Y,taskSavePath)


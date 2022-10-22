#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : IntermediateFusion.py
@Time    : 2022/09/10 18:33
@Author  : Danlei Chen, Wang Su
@Version : 1.0
'''

import pickle
import time
from attention.layers import AttentionLayer
from keras import regularizers
from keras.applications import VGG19
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.layers import (LSTM, Bidirectional, Concatenate, Dense, Dropout, Input, Reshape)
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import CustomObjectScope
from sklearn.metrics import classification_report


#---------------------Load data--------------------------------------------------------#
def load_data(path):
    with open(path, 'rb')as f:
        data = pickle.load(f)
    return data

#---------------------Save the file----------------------------------------------------#
def save_data(data, path):
    with open(path, 'wb')as f:
        pickle.dump(data, f)

def dnn(taskSavePath,train_X_txt,train_X_img,train_X_anp,train_Y,test_X_txt,test_X_img,test_X_anp,test_Y):
  
    #txt-model
    print('-----txt_model-----')
    txt_input = Input(shape = (50, 200),)
    txt_model = Bidirectional(LSTM(128, input_shape=(50, 200), return_sequences=True))(txt_input)
    txt_model= AttentionLayer(name='attention_txt')(txt_model)
    txt_model = Dense(256, W_regularizer=regularizers.l2(0.01),activation= 'relu')(txt_model)
    txt_model = Dropout(0.15)(txt_model) 

    
    #image-model
    print('-----image_model-----')
    img_input = Input(shape= (224, 224, 3))
    baseModel = VGG19(weights="imagenet", include_top=False, input_tensor= img_input)
    img_model = baseModel.get_layer('block5_conv4').output
    img_model = Reshape((196, 512))(img_model)
    img_model = AttentionLayer(name='attention_img')(img_model)
    img_model = Dense(256,W_regularizer=regularizers.l2(0.01), activation="relu")(img_model)
    img_model = Dropout(0.15)(img_model)

    #anp-model
    print('-----anp_model-----')
    anp_input = Input(shape= (4096,))
    anp_model=Dense(256, activation="relu")(anp_input)
    anp_model = Dropout(0.5)(anp_model)


    #Freeze Conv layers and FC layers of VGG19
    for layer in baseModel.layers:
        layer.trainable = False


    
    print('-----concatenate-----')
    conca = Concatenate()([txt_model,img_model])
    conca = Dense(128,W_regularizer=regularizers.l2(0.01), activation = 'relu')(conca)
    conca = Dense(128,W_regularizer=regularizers.l2(0.01),activation = 'relu')(conca)
    conca_outputs = Dense(3, activation = 'softmax')(conca)
    
    print('----model compile-----')
    opt = Adam(lr=0.001)
    earlystopping = EarlyStopping(monitor= 'loss', patience= 10, verbose= 1)
    best_checkpoint = str(taskSavePath) + 'inter_all_511.h5'
    modelcheckpoint = ModelCheckpoint(best_checkpoint, monitor= 'val_acc', save_best_only= True, 
                                      mode= 'max', verbose= 1)

    csv_path = str(taskSavePath) + 'inter_all_511.csv'
    csv_log = CSVLogger(csv_path)
    
    model = Model(inputs= [txt_input, img_input], outputs = conca_outputs)
    model.compile(loss='categorical_crossentropy', optimizer= opt, metrics=['accuracy'])
    model.summary()
    start_train =time.perf_counter()
    model.fit([train_X_txt,train_X_img],train_Y, epochs=100,
              validation_data=([test_X_txt,test_X_img],test_Y), batch_size=128, 
              verbose=1, callbacks= [earlystopping, modelcheckpoint,csv_log])
    end_train =time.perf_counter()
    #load the best_checkpoint, and conduct evaluation
    with CustomObjectScope({'AttentionLayer': AttentionLayer}):
        IF_model = load_model(best_checkpoint)
    start_test =time.perf_counter()
    predictions = IF_model.predict([test_X_txt,test_X_img], batch_size = 32)
    end_test =time.perf_counter()
    print(classification_report(test_Y.argmax(axis= 1), predictions.argmax(axis= 1), digits= 5))

    save_data(predictions,str(taskSavePath)+"predictions.pickle")
    
    print("print the computational efficiency: \n")
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

    print("print predictions: ")
    print(predictions.argmax(axis= 1))



if __name__ == '__main__':

    basePath="/gpfs/share/home/2101211725/HFIR/DataSets/Our/"
    taskSavePath="Our_SP1/" #taskSavePath is the location where output files are saved during the training process
    print('----load data----')
    train_X_txt_Path=basePath+"text/"+"all_txt_arr_train.pickle"
    train_X_img_Path=basePath+"img/"+"all_img_arr_train.pickle"
    train_X_anp_Path=basePath+"anp/"+"all_anp_arr_train.pickle"
    train_Y_Path=basePath+"label/"+"all_multi_label_train.pickle"

    test_X_txt_Path=basePath+"text/"+"all_txt_arr_test.pickle"
    test_X_img_Path=basePath+"img/"+"all_img_arr_test.pickle"
    test_X_anp_Path=basePath+"anp/"+"all_anp_arr_test.pickle"
    test_Y_Path=basePath+"label/"+"all_multi_label_test.pickle"



    train_X_txt=load_data(train_X_txt_Path)
  
    train_X_img=load_data(train_X_img_Path)
    train_X_anp=load_data(train_X_anp_Path)
    train_Y=load_data(train_Y_Path)

    test_X_txt=load_data(test_X_txt_Path)
    test_X_img=load_data(test_X_img_Path)
    test_X_anp=load_data(test_X_anp_Path)
    test_Y=load_data(test_Y_Path)

    dnn(taskSavePath,train_X_txt,train_X_img,train_X_anp,train_Y,test_X_txt,test_X_img,test_X_anp,test_Y)


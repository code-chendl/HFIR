#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : Attention-BiLSTM.py
@Time    : 2022/09/09 16:55
@Author  : Danlei Chen, Wang Su
@Version : 1.0
'''

import pickle
import time
from attention.layers import AttentionLayer
from keras.callbacks import EarlyStopping, ModelCheckpoint,CSVLogger
from keras.layers import (LSTM,  Bidirectional, Dense,Dropout)
from keras.models import Input, Model,load_model
from sklearn.metrics import classification_report
from sklearn.model_selection import  train_test_split
from keras.utils import CustomObjectScope


#---------------------Load data--------------------------------------------------------#
def load_data(path):
    with open(path, 'rb')as f:
        data = pickle.load(f)
    return data
#---------------------Save the file----------------------------------------------------#
def save_data(data, path):
    with open(path, 'wb')as f:
        pickle.dump(data, f)

#---Construct self attention-based BiLSTM for textual sentiment classification----------#
def text_model(train_X,train_Y,test_X,test_Y,taskSavePath):

    #hyper-parameter settings
    unitParameter=128      # the number of hidden neurons in each cell
    dropoutParameter=0.3   # dropout
    bachsizeParameter=32   # batch size
    #Construct the model
    inputs = Input(shape=(50, 200))
    vect = Bidirectional(LSTM(unitParameter, return_sequences=True))(inputs) #BiLSTM
    vects = Dropout(dropoutParameter)(vect)
    atten_vect = AttentionLayer(name='attention')(vects)

    outputs = Dense(3, activation = 'softmax')(atten_vect)
    model = Model(inputs= inputs, outputs= outputs)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['accuracy'])
    model.summary()  #Print the structure and parameters of the network

    # Employ early stopping to avoid overfitting, and save the best-performing model
    early_stopping = EarlyStopping(monitor='val_acc', patience=10,mode='max')
    best_checkpoint = taskSavePath+'BiLSTM_SelfAttention_Text.h5'
    modelcheckpoint = ModelCheckpoint(best_checkpoint, monitor= 'val_acc', save_best_only= True, 
                                        mode= 'max', verbose= 1)
    
    #record acc loss val_acc val_loss over epochs
    csv_path = taskSavePath+'BiLSTM_SelfAttention_Text.csv'
    csv_log = CSVLogger(csv_path)

    #Training process
    start_train =time.perf_counter() #The start time of the training process
    model.fit(train_X, train_Y, epochs = 100, validation_data = (test_X, test_Y), 
            batch_size = bachsizeParameter, verbose = 1,callbacks=[early_stopping,modelcheckpoint,csv_log])
    end_train =time.perf_counter() #The end time of the training process

        
    #load the best_checkpoint, and conduct evaluation
    with CustomObjectScope({'AttentionLayer': AttentionLayer}):
        Best_model = load_model(best_checkpoint)
    start_test =time.perf_counter() #The start time of the testing process
    predictions = Best_model.predict(test_X, batch_size = 32)
    end_test =time.perf_counter() #The end time of the testing process

    save_data(predictions,taskSavePath+"predictions.pickle") #Save predictions
    print(classification_report(test_Y.argmax(axis= 1), predictions.argmax(axis= 1), digits= 5))
    

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

if __name__ == '__main__':
    #load the training set and testing set
    taskSavePath="Our_tw1517/" #taskSavePath is the location where output files are saved during the training process
    basePath="/gpfs/share/home/HFIR/DataSets/twitter1517/"
    X_text=load_data(basePath+"text_over/"+"twitter1517_all_text_arr.pickle")
    Y=load_data(basePath+"label_over/"+'all_text_label.pickle')
    train_X, test_X, train_Y, test_Y = train_test_split(X_text, Y, test_size=0.2, random_state=1) 
    #Conduct the training process and evaluation
    text_model(train_X,train_Y,test_X,test_Y,taskSavePath)

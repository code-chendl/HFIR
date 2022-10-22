#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : DeepSentiBank-DNN.py
@Time    : 2022/09/09 16:55
@Author  : Danlei Chen, Wang Su
@Version : 1.0
'''

import pickle
import time
from keras import regularizers
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Input
from keras.models import Model, load_model
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


#---------------------Load data--------------------------------------------------------#
def load_data(path):
    with open(path, 'rb')as f:
        data = pickle.load(f)
    return data
#---------------------Save the file----------------------------------------------------#
def save_data(data, path):
    with open(path, 'wb')as f:
        pickle.dump(data, f)
# ------Construct DeepSentiBank for visual sentiment classification--------------------#
def ANP_model(train_X,train_Y,test_X,test_Y,taskSavePath):
    #The model consists of 3 FC layers, of which the last one is used for classification;
    anp_input = Input(shape= (4096,))
    anp_model=Dense(512, activation="relu", W_regularizer=regularizers.l2(0.01))(anp_input)
    anp_model = Dropout(0.5)(anp_model)
    anp_model=Dense(256, activation="relu", W_regularizer=regularizers.l2(0.01))(anp_input)
    anp_model = Dropout(0.5)(anp_model)
    outputs = Dense(3, activation = 'softmax')(anp_model)
    #set hyper-parameters and compile
    model = Model(inputs= anp_input, outputs= outputs)
    opt = SGD(lr=0.001, momentum= 0.9)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt , metrics= ['accuracy'])
    model.summary()    #Print the structure and parameters of the network
    
    # Employ early stopping to avoid overfitting, and save the best-performing model
    earlystopping = EarlyStopping(monitor= 'loss', patience=20, verbose= 1)
    best_checkpoint = taskSavePath+'ANPDNN.h5'
    modelcheckpoint = ModelCheckpoint(best_checkpoint, monitor= 'val_acc', save_best_only= True, 
                                        mode= 'max', verbose= 1)
    #record acc loss val_acc val_loss over epochs
    csv_path = taskSavePath+'ANPDNN.csv'
    csv_log = CSVLogger(csv_path)

    #Training process
    start_train =time.perf_counter() #The start time of the training process
    model.fit(train_X, train_Y, epochs=500,
                validation_data=(test_X, test_Y), batch_size=128, 
                verbose=1, callbacks= [earlystopping, modelcheckpoint,csv_log])
    end_train =time.perf_counter() #The end time of the training process


    #load the best_checkpoint, and conduct evaluation
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
    taskSavePath="tw1517_Over/" #taskSavePath is the location where output files are saved during the training process
    basePath="/gpfs/share/home/HFIR/DataSets/twitter1517/"
    X_anp=load_data(basePath+"anp_over/"+"twitter1517_OS_anp_arr.pickle")
    Y=load_data(basePath+"label_over/"+'all_image_label.pickle')
    train_X, test_X, train_Y, test_Y = train_test_split(X_anp, Y, test_size=0.2, random_state=1) 
    #Conduct the training process and evaluation
    ANP_model(train_X,train_Y,test_X,test_Y,taskSavePath)


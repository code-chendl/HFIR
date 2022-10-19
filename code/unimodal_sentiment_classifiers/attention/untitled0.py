import pickle
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, Activation, Bidirectional, LSTM, GRU, RNN
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from attention.layers import AttentionLayer
from sklearn.model_selection import train_test_split

def load_f(path):
    with open(path, 'rb')as f:
        data = pickle.load(f)
    return data

txt = load_f(r'w2v_weibo_data.pickle')
labels = load_f(r'labels.pickle')

train_X, test_X, train_Y, test_Y = train_test_split(txt, labels, test_size = 0.2, random_state = 42)


model = Sequential()
model.add(Bidirectional(LSTM(128,return_sequences=True, input_shape = (50, 100))))
#model.add(Bidirectional(GRU(128,return_sequences=True, input_shape = (50, 100))))
model.add(Dropout(0.5))
model.add(AttentionLayer(name='attention'))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['accuracy'])
model.fit(train_X, train_Y, nb_epoch = 5, validation_data = (test_X, test_Y),
          batch_size = 128, verbose = 1)

#score, acc = model.evaluate(test_X, test_Y, batch_size = 128)
y_pre = model.predict(test_X).argmax(axis=1)

#print('Accuracy:%.4f'%acc)

print(classification_report(test_Y.argmax(axis= 1), y_pre, digits= 5))









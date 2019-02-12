# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 18:36:04 2018

@author: Роман
"""

from __future__ import print_function
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
np.random.seed(1671)     #для воспроизведения результатов

#сеть и ее обучение
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = RMSprop()
N_HIDDEN = 128
VALIDATION_SPLIT=0.2
DROPOUT = 0.3
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

#данные
(X_train,y_train),(X_test,y_test) = cifar10.load_data()
RESHAPED = 28*28
#X_train = X_train.reshape(60000, RESHAPED)
#X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#нормировать
X_train /= 255
X_test /= 255
print(X_train.shape[0],'train samples') 
print(X_test.shape[0],'test samples') 

#преобразовать векторы классов в бинарные матрицы классов
Y_train = np_utils.to_categorical(y_train,NB_CLASSES)
Y_test = np_utils.to_categorical(y_test,NB_CLASSES)

#скрытый слой
#10 выходов + в конце softmax (обобщение сигмоиды)
model = Sequential()
model.add(Conv2D(32,(3,3),padding = 'same',
                 input_shape = (IMG_ROWS,IMG_COLS,IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3),padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(DROPOUT))
model.add(Conv2D(64,(3,3),padding = 'same'))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3),padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(DROPOUT))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

#выбор оптимизатора, функцию потерь, оценить качество
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

#обучение
history = model.fit(X_train,Y_train,batch_size=BATCH_SIZE,epochs=NB_EPOCH,
                    verbose=VERBOSE,validation_split=VALIDATION_SPLIT)

#проверка
score = model.evaluate(X_test,Y_test,verbose=VERBOSE)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])

#сохранение модель
model_json = model.to_json()
open('cifar10_architecture.json','w').write(model_json)
#сохранить веса
model.save_weights('cifar10_weights.h5', overwrite=True)

#график изменения потери
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc = 'upper left')
plt.show()
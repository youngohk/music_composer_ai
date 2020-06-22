import numpy as np
import pandas as pd

from os import path
import os
import glob
from IPython.display import Audio

from keras.layers import Dense, LSTM, LeakyReLU
from keras.models import Sequential, load_model
from scipy.io.wavfile import read, write
from time import sleep

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping


import pydub


# mp3_files = glob.glob('music_data/Jazz/*.mp3')


# ### mp3 파일 --> wav 파일로 변환.
# for mp3_file  in mp3_files:
#     wav_file = os.path.splitext(mp3_file)[0] + '.wav'
#     sound = pydub.AudioSegment.from_mp3(mp3_file)
#     sound.export(wav_file, format="wav")
# print("Conversion Complete")

# mp3_files = glob.glob('music_data/hiphop/*.mp3')


# ### mp3 파일 --> wav 파일로 변환.
# for mp3_file  in mp3_files:
#     wav_file = os.path.splitext(mp3_file)[0] + '.wav'
#     sound = pydub.AudioSegment.from_mp3(mp3_file)
#     sound.export(wav_file, format="wav")
# print("Conversion Complete")


# mp3_files = glob.glob('music_data/EDM/*.mp3')


# ### mp3 파일 --> wav 파일로 변환.
# for mp3_file  in mp3_files:
#     wav_file = os.path.splitext(mp3_file)[0] + '.wav'
#     sound = pydub.AudioSegment.from_mp3(mp3_file)
#     sound.export(wav_file, format="wav")
# print("Conversion Complete")


# mp3_files = glob.glob('music_data/Rock/*.mp3')


# ### mp3 파일 --> wav 파일로 변환.
# for mp3_file  in mp3_files:
#     wav_file = os.path.splitext(mp3_file)[0] + '.wav'
#     sound = pydub.AudioSegment.from_mp3(mp3_file)
#     sound.export(wav_file, format="wav")
# print("Conversion Complete")

#================================================================================


### wav 파일 읽기 
rate, music1 = read('music_data/Jazz/Blue Bossa -Dexter Gordon.wav')
rate, music2 = read('music_data/EDM/Darude - Sandstorm.wav')
rate, music4 = read('music_data/Rock/AC DC Highway to hell Letra.wav')

# ### 파일 나뉘어서 읽기 
music1 = pd.DataFrame(music1[0:400000, :])
music2 = pd.DataFrame(music2[0:400000, :])
music4 = pd.DataFrame(music4[0:400000, :])

# ### 파일 헤더 읽기 
# print(music1.head())

# ### 파일 꼬리 읽기
# print(music2.tail())


music1 = np.array(music1)
music2 = np.array(music2)
music4 = np.array(music4)

print("music1:",music1.shape)
print("music2:",music2.shape)
print("music4:",music4.shape)


# # function to create train data by shifting the music data
# def create_test_data(df, look_back):
#     dataX1, dataX2 = [], []
#     for i in range(len(df)-look_back-1):
#         dataX1.append(df.iloc[i : i + look_back, 0].values)
#         dataX2.append(df.iloc[i : i + look_back, 1].values)
        
#     return np.array(dataX1), np.array(dataX2)



# test1, test2 = create_test_data(pd.concat([music1.iloc[160001 : 400000, :],music2.iloc[160001 : 400000, :]], axis=0), 3)

music1 = music1.reshape((-1, 1, 3))
music2 = music2.reshape((-1, 1, 3))
music4 = music4.reshape((-1, 1, 3))

train_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.7,
                                   zoom_range=[0.9, 2.2],
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')


train_generator = train_datagen.flow_from_directory(
        'music_data/',
        target_size=(24, 24),
        batch_size=12,
        class_mode='categorical')


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(-1,-1,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(12, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', verbose=1)

model.fit_generator(
        music1,
        steps_per_epoch=15 * 100,
        epochs=50,
        validation_data=music1,
        validation_steps=5,
        callbacks=[early_stopping])



# print("X1:",X1.shape)
# print("X2:",X2.shape)

# # LSTM Model for channel 1 of the music data
# rnn1 = Sequential()
# rnn1.add(LSTM(units=100, activation='linear', input_shape=(None, 3)))
# rnn1.add(LeakyReLU())
# rnn1.add(Dense(units=50, activation='linear'))
# rnn1.add(LeakyReLU())
# rnn1.add(Dense(units=25, activation='linear'))
# rnn1.add(LeakyReLU())
# rnn1.add(Dense(units=12, activation='linear'))
# rnn1.add(LeakyReLU())
# rnn1.add(Dense(units=1, activation='linear'))
# rnn1.add(LeakyReLU())

# rnn1.compile(optimizer='adam', loss='mean_squared_error')
# rnn1.fit(X1, y1, epochs=20, batch_size=100)
# print("RNN1 completed fit")

# # LSTM Model for channel 2 of the music data
# rnn2 = Sequential()
# rnn2.add(LSTM(units=100, activation='linear', input_shape=(None, 3)))
# rnn2.add(LeakyReLU())
# rnn2.add(Dense(units=50, activation='linear'))
# rnn2.add(LeakyReLU())
# rnn2.add(Dense(units=25, activation='linear'))
# rnn2.add(LeakyReLU())
# rnn2.add(Dense(units=12, activation='linear'))
# rnn2.add(LeakyReLU())
# rnn2.add(Dense(units=1, activation='linear'))
# rnn2.add(LeakyReLU())

# rnn2.compile(optimizer='adam', loss='mean_squared_error')
# rnn2.fit(X2, y2, epochs=20, batch_size=100)

# print("RNN2 completed fit")

# # saving LSTM models
# rnn1.model.save('rnn1.h5')
# rnn2.model.save('rnn2.h5')
# print("saved rnn1.h5")

# # # loading the saved models
# # rnn1 = load_model('rnn1.h5')

# # making predictions for channel 1 and channel 2
# pred_rnn1 = rnn1.predict(test1.reshape(-1, 1, 3))
# pred_rnn2 = rnn2.predict(test2.reshape(-1, 1, 3))

# # reshaping the data for input to ANN
# X1 = X1.reshape((-1, 3))
# X2 = X2.reshape((-1, 3))

# print("X1:",X1.shape)
# print("X2:",X2.shape)


# ###################### ANN Model  ##########################

# # ANN Model for channel 1 of the music data
# ann1 = Sequential()
# ann1.add(Dense(units=100, activation='linear', input_dim=3))
# ann1.add(LeakyReLU())
# ann1.add(Dense(units=50, activation='linear'))
# ann1.add(LeakyReLU())
# ann1.add(Dense(units=25, activation='linear'))
# ann1.add(LeakyReLU())
# ann1.add(Dense(units=12, activation='linear'))
# ann1.add(LeakyReLU())
# ann1.add(Dense(units=1, activation='linear'))
# ann1.add(LeakyReLU())

# ann1.compile(optimizer='adam', loss='mean_squared_error')
# ann1.fit(X1, y1, epochs = 20, batch_size=100)
# print("ANN1 completed fit")


# # ANN Model for channel 2 of the music data
# ann2 = Sequential()
# ann2.add(Dense(units=100, activation='linear', input_dim=3))
# ann2.add(LeakyReLU())
# ann2.add(Dense(units=50, activation='linear'))
# ann2.add(LeakyReLU())
# ann2.add(Dense(units=25, activation='linear'))
# ann2.add(LeakyReLU())
# ann2.add(Dense(units=12, activation='linear'))
# ann2.add(LeakyReLU())
# ann2.add(Dense(units=1, activation='linear'))
# ann2.add(LeakyReLU())

# ann2.compile(optimizer='adam', loss='mean_squared_error')
# ann2.fit(X2, y2, epochs=20, batch_size=100)
# print("ANN2 completed fit")

# # saving ANN models
# ann1.model.save('ann1.h5')
# ann2.model.save('ann2.h5')
# print("saved rnn1.h5")

# # # loading saved ANN models
# ann1 = load_model('ann1.h5')
# ann2 = load_model('ann2.h5')

# # making predictions for channel 1 and channel 2
# pred_ann1 = ann1.predict(test1)
# pred_ann2 = ann2.predict(test2)


# # saving the ANN predicitons in wav format
# write('pred_ann.wav', rate, pd.concat([pd.DataFrame(pred_ann1.astype('int16')), pd.DataFrame(pred_ann2.astype('int16'))], axis=1).values)

# # saving the LSTM predicitons in wav format
# write('pred_rnn.wav', rate, pd.concat([pd.DataFrame(pred_rnn1.astype('int16')), pd.DataFrame(pred_rnn2.astype('int16'))], axis=1).values)

# # saving the original music in wav format
# write('original.wav',rate, pd.concat([music1.iloc[160001 : 400000, :], music2.iloc[160001 : 400000, :]], axis=0).values)

# print("pred_ann.wav play")
# Audio("pred_ann.wav")
# sleep(3)

# print("pred_rnn.wav play")
# Audio("pred_rnn.wav")
# sleep(3)

# print("original.wav play")
# Audio("original.wav")
# sleep(3)

# Using TensorFlow backend.
# Conversion Complete
#    0  1
# 0  0  0
# 1  0  0
# 2  0  0
# 3  0  0
# 4  0  0
#            0    1
# 399995 -1325 -138
# 399996 -1310 -152
# 399997 -1280 -163
# 399998 -1245 -188
# 399999 -1205 -223
# X1: (319996, 3)
# X2: (319996, 3)
# y1: (319996,)
# y2: (319996,)
# X1: (319996, 1, 3)
# X2: (319996, 1, 3)
# 2020-02-04 18:03:19.649430: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# 2020-02-04 18:03:20.106257: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7fbabf4662b0 executing computations on platform Host. Devices:
# 2020-02-04 18:03:20.106314: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
# Epoch 1/20
# 319996/319996 [==============================] - 9s 27us/step - loss: 75307.4892  
# Epoch 2/20
# 319996/319996 [==============================] - 7s 21us/step - loss: 10957.0077
# Epoch 3/20
# 319996/319996 [==============================] - 6s 20us/step - loss: 8895.0337
# Epoch 4/20
# 319996/319996 [==============================] - 7s 21us/step - loss: 8359.9622
# Epoch 5/20
# 319996/319996 [==============================] - 6s 20us/step - loss: 8262.2493
# Epoch 6/20
# 319996/319996 [==============================] - 7s 21us/step - loss: 8337.4925
# Epoch 7/20
# 319996/319996 [==============================] - 7s 21us/step - loss: 8414.1704
# Epoch 8/20
# 319996/319996 [==============================] - 7s 20us/step - loss: 8053.7079
# Epoch 9/20
# 319996/319996 [==============================] - 7s 21us/step - loss: 7969.2446
# Epoch 10/20
# 319996/319996 [==============================] - 6s 20us/step - loss: 7998.7555
# Epoch 11/20
# 319996/319996 [==============================] - 6s 19us/step - loss: 7845.1603
# Epoch 12/20
# 319996/319996 [==============================] - 7s 22us/step - loss: 7719.4384
# Epoch 13/20
# 319996/319996 [==============================] - 7s 21us/step - loss: 7825.3728
# Epoch 14/20
# 319996/319996 [==============================] - 7s 21us/step - loss: 7799.3673
# Epoch 15/20
# 319996/319996 [==============================] - 7s 21us/step - loss: 7760.9297
# Epoch 16/20
# 319996/319996 [==============================] - 7s 21us/step - loss: 7601.0233
# Epoch 17/20
# 319996/319996 [==============================] - 7s 22us/step - loss: 7637.1692
# Epoch 18/20
# 319996/319996 [==============================] - 7s 23us/step - loss: 7570.1789
# Epoch 19/20
# 319996/319996 [==============================] - 7s 20us/step - loss: 7611.3275
# Epoch 20/20
# 319996/319996 [==============================] - 7s 22us/step - loss: 7555.0586
# RNN1 completed fit
# Epoch 1/20
# 319996/319996 [==============================] - 7s 23us/step - loss: 262885.8432
# Epoch 2/20
# 319996/319996 [==============================] - 8s 25us/step - loss: 27439.7523
# Epoch 3/20
# 319996/319996 [==============================] - 7s 23us/step - loss: 16868.6763
# Epoch 4/20
# 319996/319996 [==============================] - 7s 23us/step - loss: 16167.4631
# Epoch 5/20
# 319996/319996 [==============================] - 7s 22us/step - loss: 15873.1104
# Epoch 6/20
# 319996/319996 [==============================] - 7s 23us/step - loss: 14807.2122
# Epoch 7/20
# 319996/319996 [==============================] - 7s 23us/step - loss: 14213.7397
# Epoch 8/20
# 319996/319996 [==============================] - 7s 21us/step - loss: 14884.5233
# Epoch 9/20
# 319996/319996 [==============================] - 7s 22us/step - loss: 14591.9311
# Epoch 10/20
# 319996/319996 [==============================] - 7s 23us/step - loss: 14288.5247
# Epoch 11/20
# 319996/319996 [==============================] - 7s 23us/step - loss: 14235.8462
# Epoch 12/20
# 319996/319996 [==============================] - 7s 22us/step - loss: 14268.2147
# Epoch 13/20
# 319996/319996 [==============================] - 7s 21us/step - loss: 14297.0686
# Epoch 14/20
# 319996/319996 [==============================] - 8s 24us/step - loss: 14007.5915
# Epoch 15/20
# 319996/319996 [==============================] - 7s 21us/step - loss: 13731.2466
# Epoch 16/20
# 319996/319996 [==============================] - 7s 21us/step - loss: 13679.9735
# Epoch 17/20
# 319996/319996 [==============================] - 8s 24us/step - loss: 14051.1262
# Epoch 18/20
# 319996/319996 [==============================] - 9s 29us/step - loss: 13354.8731
# Epoch 19/20
# 319996/319996 [==============================] - 9s 28us/step - loss: 13618.9558
# Epoch 20/20
# 319996/319996 [==============================] - 8s 25us/step - loss: 13425.7119
# RNN2 completed fit
# /Users/jongphilkim/anaconda3/lib/python3.7/site-packages/keras/engine/sequential.py:111: UserWarning: `Sequential.model` is deprecated. `Sequential` is a subclass of `Model`, you can just use your `Sequential` instance directly.
#   warnings.warn('`Sequential.model` is deprecated. '
# saved rnn1.h5
# X1: (319996, 3)
# X2: (319996, 3)
# Epoch 1/20
# 319996/319996 [==============================] - 5s 16us/step - loss: 49615.7229
# Epoch 2/20
# 319996/319996 [==============================] - 6s 19us/step - loss: 8716.7020
# Epoch 3/20
# 319996/319996 [==============================] - 4s 12us/step - loss: 8013.9452
# Epoch 4/20
# 319996/319996 [==============================] - 4s 12us/step - loss: 7830.6136
# Epoch 5/20
# 319996/319996 [==============================] - 4s 12us/step - loss: 7878.6092
# Epoch 6/20
# 319996/319996 [==============================] - 3s 9us/step - loss: 7847.1912
# Epoch 7/20
# 319996/319996 [==============================] - 3s 9us/step - loss: 7709.6675
# Epoch 8/20
# 319996/319996 [==============================] - 3s 9us/step - loss: 7601.9642
# Epoch 9/20
# 319996/319996 [==============================] - 3s 10us/step - loss: 7575.7453
# Epoch 10/20
# 319996/319996 [==============================] - 4s 13us/step - loss: 7567.6611
# Epoch 11/20
# 319996/319996 [==============================] - 4s 13us/step - loss: 7480.4437
# Epoch 12/20
# 319996/319996 [==============================] - 3s 10us/step - loss: 7314.7532
# Epoch 13/20
# 319996/319996 [==============================] - 3s 11us/step - loss: 7368.9721
# Epoch 14/20
# 319996/319996 [==============================] - 4s 12us/step - loss: 7406.2244
# Epoch 15/20
# 319996/319996 [==============================] - 3s 10us/step - loss: 7393.5803
# Epoch 16/20
# 319996/319996 [==============================] - 3s 10us/step - loss: 7332.8874
# Epoch 17/20
# 319996/319996 [==============================] - 3s 10us/step - loss: 7312.5008
# Epoch 18/20
# 319996/319996 [==============================] - 3s 10us/step - loss: 7270.6459
# Epoch 19/20
# 319996/319996 [==============================] - 4s 11us/step - loss: 7283.3276
# Epoch 20/20
# 319996/319996 [==============================] - 3s 10us/step - loss: 7267.0052
# ANN1 completed fit
# Epoch 1/20
# 319996/319996 [==============================] - 4s 13us/step - loss: 170391.8027
# Epoch 2/20
# 319996/319996 [==============================] - 4s 11us/step - loss: 17183.5802
# Epoch 3/20
# 319996/319996 [==============================] - 3s 10us/step - loss: 16295.2502
# Epoch 4/20
# 319996/319996 [==============================] - 3s 10us/step - loss: 15167.2421
# Epoch 5/20
# 319996/319996 [==============================] - 4s 11us/step - loss: 14871.3242
# Epoch 6/20
# 319996/319996 [==============================] - 4s 14us/step - loss: 14688.2833
# Epoch 7/20
# 319996/319996 [==============================] - 4s 13us/step - loss: 13844.5934
# Epoch 8/20
# 319996/319996 [==============================] - 3s 10us/step - loss: 13954.6628
# Epoch 9/20
# 319996/319996 [==============================] - 3s 9us/step - loss: 13963.0031
# Epoch 10/20
# 319996/319996 [==============================] - 3s 9us/step - loss: 14069.2670
# Epoch 11/20
# 319996/319996 [==============================] - 3s 9us/step - loss: 13313.3629
# Epoch 12/20
# 319996/319996 [==============================] - 3s 9us/step - loss: 13531.0801
# Epoch 13/20
# 319996/319996 [==============================] - 3s 9us/step - loss: 13510.5788
# Epoch 14/20
# 319996/319996 [==============================] - 3s 9us/step - loss: 13153.2675
# Epoch 15/20
# 319996/319996 [==============================] - 3s 9us/step - loss: 13592.9853
# Epoch 16/20
# 319996/319996 [==============================] - 3s 9us/step - loss: 13153.5326
# Epoch 17/20
# 319996/319996 [==============================] - 3s 9us/step - loss: 12968.0556
# Epoch 18/20
# 319996/319996 [==============================] - 3s 9us/step - loss: 13004.7918
# Epoch 19/20
# 319996/319996 [==============================] - 3s 9us/step - loss: 12989.1596
# Epoch 20/20
# 319996/319996 [==============================] - 3s 9us/step - loss: 12871.8088
# ANN2 completed fit
# saved rnn1.h5
# pred_ann.wav play
# pred_rnn.wav play
# original.wav play
# 실행  python3 all_composer.py

from pydub.playback import play
from pydub import AudioSegment
import numpy as np
import pandas as pd
import pydub


from keras.layers import Dense, LSTM, LeakyReLU
from keras.models import Sequential, load_model
from scipy.io.wavfile import read, write
from time import sleep


TEST_WAV = "music_data/test.wav"  ## test.wav (custmoer WAV)

JAZZ_WAV_FILE = "music_data/Jazz/sample.wav"  ## 샘플
EDM_WAV_FILE = "music_data/EDM/sample.wav"
ROCK_WAV_FILE = "music_data/Rock/sample.wav"
HIPHOP_WAV_FILE = "music_data/Hiphop/sample.wav"

JAZZ_RNN_FILE = "music_data/Jazz/jazz_rnn.wav"
EDM_RNN_FILE = "music_data/EDM/edm_rnn.wav"
HIHOP_RNN_FILE = "music_data/Hiphop/hiphop_rnn.wav"
ROCK_RNN_FILE = "music_data/Rock/rock_rnn.wav"


JAZZ_MODEL_1_FILE = "Brain/rnn1_jazz.h5"
EDM_MODEL_1_FILE = "Brain/rnn1_EDM.h5"
HIPHOP_MODEL_1_FILE = "Brain/rnn1_hiphop.h5"
ROCK_MODEL_1_FILE = "Brain/rnn1_Rock.h5"

JAZZ_MODEL_2_FILE = "Brain/rnn2_jazz.h5"
EDM_MODEL_2_FILE = "Brain/rnn2_EDM.h5"
HIPHOP_MODEL_2_FILE = "Brain/rnn2_hiphop.h5"
ROCK_MODEL_2_FILE = "Brain/rnn2_Rock.h5"


# mp3 ---> WAV로 변환
sound = pydub.AudioSegment.from_mp3("music_data/test.mp3")
sound.export(TEST_WAV, format="wav")


print("WAV Conversion Complete")
print(sound)



# LSTM Model for channel 1 of the music data
rnn1 = Sequential()
rnn1.add(LSTM(units=100, activation='linear', input_shape=(None, 3)))
rnn1.add(LeakyReLU())
rnn1.add(Dense(units=50, activation='linear'))
rnn1.add(LeakyReLU())
rnn1.add(Dense(units=25, activation='linear'))
rnn1.add(LeakyReLU())
rnn1.add(Dense(units=12, activation='linear'))
rnn1.add(LeakyReLU())
rnn1.add(Dense(units=1, activation='linear'))
rnn1.add(LeakyReLU())

rnn1.compile(optimizer='adam', loss='mean_squared_error')

# LSTM Model for channel 2 of the music data
rnn2 = Sequential()
rnn2.add(LSTM(units=100, activation='linear', input_shape=(None, 3)))
rnn2.add(LeakyReLU())
rnn2.add(Dense(units=50, activation='linear'))
rnn2.add(LeakyReLU())
rnn2.add(Dense(units=25, activation='linear'))
rnn2.add(LeakyReLU())
rnn2.add(Dense(units=12, activation='linear'))
rnn2.add(LeakyReLU())
rnn2.add(Dense(units=1, activation='linear'))
rnn2.add(LeakyReLU())

rnn2.compile(optimizer='adam', loss='mean_squared_error')



# function to create train data by shifting the music data
def create_test_data(df, look_back):
    dataX1, dataX2 = [], []
    for i in range(len(df)-look_back-1):
        dataX1.append(df.iloc[i: i + look_back, 0].values)
        dataX2.append(df.iloc[i: i + look_back, 1].values)

    return np.array(dataX1), np.array(dataX2)


def composer(kind_music_path, model_rnn1_path,model_rnn2_path, kind_rnn_path):
    
    rate, music1 = read(TEST_WAV)
    music1 = pd.DataFrame(music1[0:800000, :])


    rate, music2 = read(kind_music_path)
    music2 = pd.DataFrame(music2[0:800000, :])


    # 시간이 걸림...
    print("test1,test2 data creating....{}".format(kind_music_path))

    test1, test2 = create_test_data(pd.concat(
        [music1.iloc[160001: 800000, :], music2.iloc[160001: 800000, :]], axis=0), 3)

    test1 = test1.reshape((-1,1,3))
    test2 = test2.reshape((-1,1,3))
    
    # loading the saved models
    print("Loading...{},{}".format(model_rnn1_path,model_rnn2_path ))
    rnn1 = load_model(model_rnn1_path)
    rnn2 = load_model(model_rnn2_path)

    # making predictions for channel 1 and channel 2
    print("Predicting....{}".format(kind_music_path))  # 바로 됨.

    pred_rnn1 = rnn1.predict(test1)
    pred_rnn2 = rnn2.predict(test2)

    print("{} writing....".format(kind_rnn_path))
    # saving the LSTM predicitons in wav format
    write(kind_rnn_path, rate, pd.concat([pd.DataFrame(pred_rnn1.astype(
        'int16')), pd.DataFrame(pred_rnn2.astype('int16'))], axis=1).values)

    # sleep(2)
    print("{} sound play".format(kind_rnn_path))
    sound = AudioSegment.from_wav(kind_rnn_path)
    play(sound)


composer(JAZZ_WAV_FILE,JAZZ_MODEL_1_FILE,JAZZ_MODEL_2_FILE,JAZZ_RNN_FILE)
composer(EDM_WAV_FILE,EDM_MODEL_1_FILE,EDM_MODEL_2_FILE,EDM_RNN_FILE)
composer(HIPHOP_WAV_FILE,HIPHOP_MODEL_1_FILE,HIPHOP_MODEL_2_FILE,HIHOP_RNN_FILE) 
composer(ROCK_WAV_FILE, ROCK_MODEL_1_FILE, ROCK_MODEL_2_FILE, ROCK_RNN_FILE) 

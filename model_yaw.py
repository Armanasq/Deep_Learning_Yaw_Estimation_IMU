import tensorflow_addons as tfa
import keras.backend as K
from tensorflow.keras import backend as k

from tcn import TCN
import tensorflow as tf

from keras.layers import *
from tensorflow import keras

from keras.initializers import *
from keras.utils import *
from keras.callbacks import *
from keras.regularizers import *
from keras.applications import *
from keras.losses import *
from keras.models import *
from keras.optimizers import *


import matplotlib.pyplot as plt
from fileinput import filename
from numba import cuda
from sklearn.utils import shuffle
import pandas as pd
import random as rn
import numpy as np
import math
import time
import os
from symbol import import_from
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'


def yaw_arch_1(window_size):
    ''' take Acc and Gyro and fed into CNN and LSTM, then concat with fs and then
    concat with Mag and the feed into cnn and lstm then fed into dense layer
    '''


def CNN_LSTM_yaw_mag(window_size=200):
    Mag = Input((window_size, 3), name='Mag')
    fs = Input((1,), name='fs')

    fsD = Dense(128, activation='relu', name='fsD')(fs)
    # mag = Lambda(lambda x: K.l2_normalize(
    #    x, axis=1), name='NormalizeMag')(Mag)

    #mag = Reshape((Mag.shape[1], Mag.shape[2], 1), name='ReshapeMag')(Mag)
    mag = Conv1D(128, 11, activation='relu', name='MagConv1')(Mag)

    mag = Bidirectional(
        CuDNNLSTM(128, return_sequences=False, name='MagLSTM2'))(mag)
    mag = Dropout(0.2, name='DropoutMag2')(mag)
    mag = Dense(256, activation="relu", name='DenseConc')(mag)
    # mag = Dense(128, activation='relu', name='DenseMag')(mag)

    conc = concatenate([mag, fsD], name='Conc')

    yaw = Dense(4, activation='linear', name='Yaw')(conc)
    model = Model(inputs=[Mag, fs], outputs=[yaw])
    model.summary()
    return model


def CNN_LSTM_yaw_gyro_mag(window_size=200):
    Mag = Input((window_size, 3), name='Mag')

    mag = CuDNNLSTM(50, return_sequences=True, name='MagLSTM')(Mag)
    mag = Dropout(0.2, name='DropoutMagLSTM')(mag)
    mag = Flatten(name='FlattenMag')(mag)

    Gyro = Input((window_size, 3), name='Gyro')
    gyro = Lambda(lambda x: K.l2_normalize(
        x, axis=1), name='NormalizeGyro')(Gyro)
    gyro = CuDNNLSTM(50, return_sequences=True, name='GyroLSTM')(Gyro)
    gyro = Dropout(0.2, name='DropoutGyroLSTM')(gyro)
    gyro = Flatten(name='FlattenGyro')(gyro)

    conc = concatenate([mag, gyro], name='Conc')
    conc = Dense(256, activation="relu", name='DenseConc')(conc)

    fs = Input((1,), name='Fs')
    fsDense = Dense(256, activation="relu", name='FsDense')(fs)
    conc = concatenate([conc, fsDense], name='Conc2')
    conc = Dense(128, activation="relu", name='DenseConc2')(conc)
    yaw = Dense(4, activation='linear', name='Yaw')(conc)
    model = Model(inputs=[Mag, Gyro, fs], outputs=[yaw])
    model.summary()
    return model


def LSTM_yaw_mag_gyro(window_size=200):
    Mag = Input((window_size, 3), name='Mag')
    Gyro = Input((window_size, 3), name='Gyro')

    mag = CuDNNLSTM(50, return_sequences=True, name='MagLSTM')(Mag)
    mag = Dropout(0.2, name='DropoutMagLSTM')(mag)
    mag = Flatten(name='FlattenMag')(mag)

    gyro = CuDNNLSTM(50, return_sequences=True, name='GyroLSTM')(Gyro)
    gyro = Dropout(0.2, name='DropoutGyroLSTM')(gyro)
    gyro = Flatten(name='FlattenGyro')(gyro)

    conc = concatenate([mag, gyro], name='Conc')
    conc = Dense(256, activation="relu", name='DenseConc')(conc)

    quat = Dense(4, activation='linear', name='Quat')(conc)
    model = Model(inputs=[Mag, Gyro], outputs=[quat])
    model.summary()
    return model


def LSTM_yaw_mag_gyro_acc(window_size=200):
    Mag = Input((window_size, 3), name='Mag')
    Gyro = Input((window_size, 3), name='Gyro')
    Acc = Input((window_size, 3), name='Acc')

    mag = CuDNNLSTM(50, return_sequences=True, name='MagLSTM')(Mag)
    mag = Dropout(0.2, name='DropoutMagLSTM')(mag)
    mag = Flatten(name='FlattenMag')(mag)

    gyro = CuDNNLSTM(50, return_sequences=True, name='GyroLSTM')(Gyro)
    gyr


def CNN_LSTM_yaw_mag_gyro(window_size=200):
    Mag = Input((window_size, 3), name='Mag')
    Gyro = Input((window_size, 3), name='Gyro')

    cnnM = Conv1D(128, 5, activation='relu', name='Conv1DMag')(Mag)
    cnnM = Conv1D(128, 5, activation='relu', name='Conv1DMag2')(Mag)
    cnnM = MaxPooling1D(2, name='MaxPool1DMag')(cnnM)

    cnnG = Conv1D(128, 5, activation='relu', name='Conv1DGyro')(Gyro)
    cnnG = Conv1D(128, 5, activation='relu', name='Conv1DGyro2')(Gyro)
    cnnG = MaxPooling1D(2, name='MaxPool1DGyro')(cnnG)

    conc = concatenate([cnnM, cnnG], name='Conc')
    conc = Flatten(name='FlattenConc')(conc)

    quat = Dense(4, activation='linear', name='Quat')(conc)
    model = Model(inputs=[Mag, Gyro], outputs=[quat])
    model.summary()
    return model


def CNN_LSTM_yaw_mag2(window_size=200):
    Mag = Input((window_size, 3), name='Mag')
    # calibrate mag
    # mag = Lambda(lambda x: K.l2_normalize(
    #    x, axis=1), name='NormalizeMag')(Mag)

    # split data
    Mag1 = Lambda(lambda x: x[:, :, 0], name='Mag1')(Mag)
    Mag1 = Reshape((Mag1.shape[1], 1), name='ReshapeMag1')(Mag1)
    M1LSTM = CuDNNLSTM(100, return_sequences=True, name='M1LSTM')(Mag1)
    M1LSTM = Dropout(0.2, name='DropoutM1LSTM')(M1LSTM)

    Mag2 = Lambda(lambda x: x[:, :, 1], name='Mag2')(Mag)
    Mag2 = Reshape((Mag2.shape[1], 1), name='ReshapeMag2')(Mag2)
    M2LSTM = CuDNNLSTM(100, return_sequences=True, name='M2LSTM')(Mag2)
    M2LSTM = Dropout(0.2, name='DropoutM2LSTM')(M2LSTM)

    Mag3 = Lambda(lambda x: x[:, :, 2], name='Mag3')(Mag)
    Mag3 = Reshape((Mag3.shape[1], 1), name='ReshapeMag3')(Mag3)
    M3LSTM = CuDNNLSTM(100, return_sequences=True, name='M3LSTM')(Mag3)
    M3LSTM = Dropout(0.2, name='DropoutM3LSTM')(M3LSTM)

    Mconc = concatenate([M1LSTM, M2LSTM, M3LSTM], name='Mconc')
    Mconc = CuDNNLSTM(100, return_sequences=True, name='MconcLSTM')(Mconc)
    Mconc = CuDNNLSTM(50, return_sequences=True, name='MconcLSTM2')(Mconc)
    Mconc = Flatten(name='FlattenMconc')(Mconc)

    yaw = Dense(4, activation='linear', name='Yaw')(Mconc)
    model = Model(inputs=[Mag], outputs=[yaw])
    model.summary()
    return model


def CNN_LSTM_yaw_mag_Fs(window_size=200):
    Mag = Input((window_size, 3), name='Mag')
    # Mag = GaussianNoise(0.1, name='MagNoise')(mag)
    # calibrate mag
    # mag = Lambda(lambda x: K.l2_normalize(
    #    x, axis=1), name='NormalizeMag')(Mag)

    # split data
    Mag1 = Lambda(lambda x: x[:, :, 0], name='Mag1')(Mag)
    Mag1 = Reshape((Mag1.shape[1], 1), name='ReshapeMag1')(Mag1)
    M1LSTM = CuDNNLSTM(100, return_sequences=True, name='M1LSTM')(Mag1)
    M1LSTM = Dropout(0.2, name='DropoutM1LSTM')(M1LSTM)

    Mag2 = Lambda(lambda x: x[:, :, 1], name='Mag2')(Mag)
    Mag2 = Reshape((Mag2.shape[1], 1), name='ReshapeMag2')(Mag2)
    M2LSTM = CuDNNLSTM(100, return_sequences=True, name='M2LSTM')(Mag2)
    M2LSTM = Dropout(0.2, name='DropoutM2LSTM')(M2LSTM)

    Mag3 = Lambda(lambda x: x[:, :, 2], name='Mag3')(Mag)
    Mag3 = Reshape((Mag3.shape[1], 1), name='ReshapeMag3')(Mag3)
    M3LSTM = CuDNNLSTM(100, return_sequences=True, name='M3LSTM')(Mag3)
    M3LSTM = Dropout(0.2, name='DropoutM3LSTM')(M3LSTM)

    Mconc = concatenate([M1LSTM, M2LSTM, M3LSTM], name='Mconc')
    # Mconc = CuDNNLSTM(100, return_sequences=True, name='MconcLSTM')(Mconc)
    Mconc = Dense(256, activation='relu', name='MconcDense')(Mconc)
    # Mconc = CuDNNLSTM(50, return_sequences=True, name='MconcLSTM2')(Mconc)
    Mconc = Flatten(name='FlattenMconc')(Mconc)

    fs = Input((1), name='fs')
    Fs = Dense(128, activation='linear', name='Fs')(fs)

    Mconc = concatenate([Mconc, Fs], name='MconcFs')
    yaw = Dense(4, activation='linear', name='Yaw')(Mconc)
    model = Model(inputs=[Mag, fs], outputs=[yaw])
    model.summary()
    return model


def CNN_LSTM_yaw_mag_acc_gyro_fs(window_size=200):
    mag = Input((window_size, 3), name='Mag')
    gyro = Input((window_size, 3), name='Gyro')
    acc = Input((window_size, 3), name='Acc')
    # normalize data
    Mag = mag
    Gyro = Lambda(lambda x: K.l2_normalize(
        x, axis=2), name='NormalizeGyro')(gyro)
    Acc = Lambda(lambda x: K.l2_normalize(x, axis=2), name='NormalizeAcc')(acc)

    Fs = Input((1), name='Fs')

    Mag1 = Lambda(lambda x: x[:, :, 0], name='Mag1')(Mag)
    Mag1 = Reshape((Mag1.shape[1], 1), name='ReshapeMag1')(Mag1)
    M1LSTM = CuDNNLSTM(100, return_sequences=True, name='M1LSTM')(Mag1)
    M1LSTM = Dropout(0.2, name='DropoutM1LSTM')(M1LSTM)

    Mag2 = Lambda(lambda x: x[:, :, 1], name='Mag2')(Mag)
    Mag2 = Reshape((Mag2.shape[1], 1), name='ReshapeMag2')(Mag2)
    M2LSTM = CuDNNLSTM(100, return_sequences=True, name='M2LSTM')(Mag2)
    M2LSTM = Dropout(0.2, name='DropoutM2LSTM')(M2LSTM)

    Mag3 = Lambda(lambda x: x[:, :, 2], name='Mag3')(Mag)
    Mag3 = Reshape((Mag3.shape[1], 1), name='ReshapeMag3')(Mag3)
    M3LSTM = CuDNNLSTM(100, return_sequences=True, name='M3LSTM')(Mag3)
    M3LSTM = Dropout(0.2, name='DropoutM3LSTM')(M3LSTM)

    Mconc = concatenate([M1LSTM, M2LSTM, M3LSTM], name='Mconc')
    # MCuDNNLSTM = CuDNNLSTM(100, return_sequences=True,
    #                       name='MCuDNNLSTM')(Mconc)
    MCuDNNLSTM = Conv1D(100, 4, activation='relu', name='Mconv1')(Mconc)
    MCuDNNLSTM = MaxPooling1D(3, name='Mmaxpool1')(MCuDNNLSTM)

    GCNN = Conv1D(128, 3, activation='relu', name='GCNN')(Gyro)
    GCNN = Conv1D(128, 3, activation='relu', name='GCNN2')(GCNN)
    GCNN = MaxPooling1D(3, name='GMaxPool')(GCNN)

    ACNN = Conv1D(128, 3, activation='relu', name='ACNN')(Acc)
    ACNN = Conv1D(128, 3, activation='relu', name='ACNN2')(ACNN)
    ACNN = MaxPooling1D(3, name='AMaxPool')(ACNN)

    AGconc = concatenate([GCNN, ACNN], name='AGconc')
    AGCuDNNLSTM = CuDNNLSTM(100, return_sequences=True,
                            name='AGCuDNNLSTM')(AGconc)

    conc = concatenate([MCuDNNLSTM, AGCuDNNLSTM], name='conc')
    conc = CuDNNLSTM(100, return_sequences=True, name='concLSTM')(conc)
    conc = Flatten(name='FlattenConc')(conc)

    Fdense = Dense(128, activation='linear', name='Fdense')(Fs)

    conc = concatenate([conc, Fdense], name='concFs')
    quat = Dense(4, activation='linear', name='Quat')(conc)
    model = Model(inputs=[acc, gyro, mag, Fs], outputs=[quat])
    model.summary()
    return model




def CNN_LSTM_yaw_mag_gyro(window_size=200):
    Mag = Input((window_size, 3), name='Mag')
    Mag1 = Lambda(lambda x: x[:, :, 0], name='Mag1')(Mag)
    Mag1 = Reshape((Mag1.shape[1], 1), name='ReshapeMag1')(Mag1)
    M1LSTM = CuDNNLSTM(100, return_sequences=True, name='M1LSTM')(Mag1)
    M1LSTM = Dropout(0.2, name='DropoutM1LSTM')(M1LSTM)

    Mag2 = Lambda(lambda x: x[:, :, 1], name='Mag2')(Mag)
    Mag2 = Reshape((Mag2.shape[1], 1), name='ReshapeMag2')(Mag2)
    M2LSTM = CuDNNLSTM(100, return_sequences=True, name='M2LSTM')(Mag2)
    M2LSTM = Dropout(0.2, name='DropoutM2LSTM')(M2LSTM)

    Mag3 = Lambda(lambda x: x[:, :, 2], name='Mag3')(Mag)
    Mag3 = Reshape((Mag3.shape[1], 1), name='ReshapeMag3')(Mag3)
    M3LSTM = CuDNNLSTM(100, return_sequences=True, name='M3LSTM')(Mag3)
    M3LSTM = Dropout(0.2, name='DropoutM3LSTM')(M3LSTM)

    Mconc = concatenate([M1LSTM, M2LSTM, M3LSTM], name='Mconc')

    Gyro = Input((window_size, 3), name='Gyro')
    Gyro1 = Lambda(lambda x: x[:, :, 0], name='Gyro1')(Gyro)
    Gyro1 = Reshape((Gyro1.shape[1], 1), name='ReshapeGyro1')(Gyro1)
    G1LSTM = CuDNNLSTM(100, return_sequences=True, name='G1LSTM')(Gyro1)
    G1LSTM = Dropout(0.2, name='DropoutG1LSTM')(G1LSTM)

    Gyro2 = Lambda(lambda x: x[:, :, 1], name='Gyro2')(Gyro)
    Gyro2 = Reshape((Gyro2.shape[1], 1), name='ReshapeGyro2')(Gyro2)
    G2LSTM = CuDNNLSTM(100, return_sequences=True, name='G2LSTM')(Gyro2)
    G2LSTM = Dropout(0.2, name='DropoutG2LSTM')(G2LSTM)

    Gyro3 = Lambda(lambda x: x[:, :, 2], name='Gyro3')(Gyro)
    Gyro3 = Reshape((Gyro3.shape[1], 1), name='ReshapeGyro3')(Gyro3)
    G3LSTM = CuDNNLSTM(100, return_sequences=True, name='G3LSTM')(Gyro3)
    G3LSTM = Dropout(0.2, name='DropoutG3LSTM')(G3LSTM)

    Gconc = concatenate([G1LSTM, G2LSTM, G3LSTM], name='Gconc')

    GM = concatenate([Mconc, Gconc], name='GMconc')
    GM = CuDNNLSTM(100, return_sequences=True, name='GMconcLSTM')(GM)
    yaw = Dense(4, activation='linear', name='Yaw')(GM)
    model = Model(inputs=[Mag, Gyro], outputs=[yaw])
    model.summary()
    return model

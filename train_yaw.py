from sklearn.ensemble import RandomForestRegressor
import tensorboard
# import libraries
import keras.backend as K
from keras.callbacks import LambdaCallback
from matplotlib import pyplot as plt
import numpy as np
# import tendorboard
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
import matplotlib.pyplot as plt
import os
import random as rn
import numpy as np
import argparse
import math
import time
import os


import tensorflow_addons as tfa
from tqdm.keras import TqdmCallback
from wandb.keras import WandbCallback
import wandb
from sklearn.model_selection import train_test_split

from data import *
from util import *
from dataset_loader import *
from model_yaw import *
from learning import *

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
# time
timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
# To  Get Reproducible Results
np.random.seed(0)
rn.seed(1254)
tf.random.set_seed(89)
imu_data = []

epochs = 120
batch_size = 500
lr = 0.002  # 0.00283559  # 0.00234375  # 0.0005  ## 00075
window_size = 100
stride = 4
if window_size == 1:
    stride = 1
version = 2


def learningRate(model):
    [acc, gyro, _, fs], [quat] = data_train(window_size, stride)
    acc, x_acc_val, gyro, x_gyro_val, fs, x_fs_val, quat, Head_quat_val = train_test_split(
        acc, gyro, fs, quat, test_size=0.2, random_state=42, shuffle=True)
    model = model(window_size)
    model.compile(optimizer=keras.optimizers.Adam(
        learning_rate=lr,), loss=QQuat_mult, metrics=[Quat_error_angle])
    # find best learning rate
    lr_finder = LRFinder(model)
    lr_finder.find([acc, gyro, fs], quat,
                   start_lr=1e-5,
                   end_lr=10, batch_size=500, epochs=1)
    lr_finder.plot_loss()
    print("Learning rate finder complete")


def data_broad(window_size, stride):
    # broad data
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))

    #
    broad_set = [1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 16,
                 17, 18, 20, 21, 22, 23, 26, 28, 29, 30, 38, 39]
    for i in broad_set:
        acc_temp, gyro_temp, mag_temp, quat_temp, fs = BROAD_data(
            BROAD_path(i)[0], BROAD_path(i)[1])
        acc = np.concatenate((acc, acc_temp), axis=0)
        gyro = np.concatenate((gyro, gyro_temp), axis=0)
        mag = np.concatenate((mag, mag_temp), axis=0)
        quat = np.concatenate((quat, quat_temp), axis=0)
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    return [gyro, acc, mag, fs], [quat]


def data_oxiod(window_size, stride):
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    for i in range(10):
        acc_temp, gyro_temp, mag_temp, quat_temp, fs = OxIOD_data(
            OxIOD_path().values[i][0])
        acc = np.concatenate((acc, acc_temp), axis=0)
        gyro = np.concatenate((gyro, gyro_temp), axis=0)
        mag = np.concatenate((mag, mag_temp), axis=0)
        quat = np.concatenate((quat, quat_temp), axis=0)
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    return [gyro, acc, mag, fs], [quat]


def data_repoIMU_TStick(window_size, stride):
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    df_TStick, df_Pendulum = repoIMU_path()
    for i in range(len(df_TStick.values)//2):
        acc_temp, gyro_temp, mag_temp, quat_temp, fs = repoIMU_data(
            df_TStick.values[i][0])
        acc = np.concatenate((acc, acc_temp), axis=0)
        gyro = np.concatenate((gyro, gyro_temp), axis=0)
        mag = np.concatenate((mag, mag_temp), axis=0)
        quat = np.concatenate((quat, quat_temp), axis=0)
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    return [gyro, acc, mag, fs], [quat]


def data_repoIMU_Pendulum(window_size, stride):
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    df_TStick, df_Pendulum = repoIMU_path()
    for i in range(len(df_Pendulum.values)//2):
        acc_temp, gyro_temp, mag_temp, quat_temp, fs = repoIMU_data(
            df_Pendulum.values[i][0])
        acc = np.concatenate((acc, acc_temp), axis=0)
        gyro = np.concatenate((gyro, gyro_temp), axis=0)
        mag = np.concatenate((mag, mag_temp), axis=0)
        quat = np.concatenate((quat, quat_temp), axis=0)
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    return [gyro, acc, mag, fs], [quat]


def data_sassari(window_size, stride):
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    MIMU = ["XS1", "AP2", "SH1", "XS2", "AP1", "SH2"]
    file_list = sassari_path()
    for i in range(len(file_list)):
        for j in range(len(MIMU)//2):
            acc_temp, gyro_temp, mag_temp, quat_temp, fs = sassari_data(
                file_list[i], MIMU[j])
            acc = np.concatenate((acc, acc_temp), axis=0)
            gyro = np.concatenate((gyro, gyro_temp), axis=0)
            mag = np.concatenate((mag, mag_temp), axis=0)
            quat = np.concatenate((quat, quat_temp), axis=0)
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    return [gyro, acc, mag, fs], [quat]


def data_RoNIN(window_size, stride):
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    file_list = RoNIN_path()[0].values
    for i in range(15):
        acc_temp, gyro_temp, mag_temp, quat_temp, fs = RoNIN_data(
            file_list[i, 0])
        acc = np.concatenate((acc, acc_temp), axis=0)
        gyro = np.concatenate((gyro, gyro_temp), axis=0)
        mag = np.concatenate((mag, mag_temp), axis=0)
        quat = np.concatenate((quat, quat_temp), axis=0)
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    return [gyro, acc, mag, fs], [quat]


def data_ridi(window_size, stride):
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    file_list = RIDI_path()
    for i in range(20):
        acc_temp, gyro_temp, mag_temp, quat_temp, fs = RIDI_data(
            file_list[i])
        acc = np.concatenate((acc, acc_temp), axis=0)
        gyro = np.concatenate((gyro, gyro_temp), axis=0)
        mag = np.concatenate((mag, mag_temp), axis=0)
        quat = np.concatenate((quat, quat_temp), axis=0)
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    return [gyro, acc, mag, fs], [quat]


def data_train(window_size, stride):
    [gyro_broad, acc_broad, mag_broad, fs_broad], [
        quat_broad] = data_broad(window_size, stride)
    print("broad done", gyro_broad.shape, acc_broad.shape,
          mag_broad.shape, fs_broad.shape, quat_broad.shape)

    [gyro_sassari, acc_sassari, mag_sassari, fs_sassari], [
        quat_sassari] = data_sassari(window_size, stride)
    print("sassari done", gyro_sassari.shape, acc_sassari.shape,
          mag_sassari.shape, fs_sassari.shape, quat_sassari.shape)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    [gyro_RoNIN, acc_RoNIN, mag_RoNIN, fs_RoNIN], [
        quat_RoNIN] = data_RoNIN(window_size, stride)
    print("RoNIN done", gyro_RoNIN.shape, acc_RoNIN.shape,
          mag_RoNIN.shape, fs_RoNIN.shape, quat_RoNIN.shape)

    [gyro_ridi, acc_ridi, mag_ridi, fs_ridi], [
        quat_ridi] = data_ridi(window_size, stride)
    print("ridi done", gyro_ridi.shape, acc_ridi.shape,
          mag_ridi.shape, fs_ridi.shape, quat_ridi.shape)
    Acc = np.concatenate((acc_broad,
                         acc_sassari, acc_RoNIN, acc_ridi), axis=0)
    Gyro = np.concatenate((gyro_broad,
                           gyro_sassari, gyro_RoNIN, gyro_ridi), axis=0)
    Mag = np.concatenate((mag_broad,
                          mag_sassari, mag_RoNIN, mag_ridi), axis=0)
    Fs = np.concatenate((fs_broad,
                         fs_sassari, fs_RoNIN, fs_ridi), axis=0)
    Quat = np.concatenate((quat_broad,
                           quat_sassari, quat_RoNIN, quat_ridi), axis=0)
    # shuffle
    #Quat = force_quaternion_uniqueness(Quat)
    #Quat = Head(Quat)
    print(Quat.shape)
    return [Acc, Gyro, Mag, Fs, ], [Quat]


def train(pred_model):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # training callbacks
    # Checkpoint
    [x_acc, x_gyro, x_mag, x_fs], [quat] = data_train(window_size, stride)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    quat = Head(quat)
    x_acc, x_acc_val, x_gyro, x_gyro_val, x_mag, x_mag_val, x_fs, x_fs_val, Head_quat, Head_quat_val = train_test_split(
        x_acc, x_gyro, x_mag, x_fs, quat, test_size=0.2, random_state=42, shuffle=True)
    model_checkpoint = ModelCheckpoint(
        'model_checkpoint.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    # tensorboard
    tensorboard = TensorBoard(log_dir="./logs/yaw/%s_%s" % (timestr,
                                                            pred_model.__name__),
                              histogram_freq=0, write_graph=True, write_images=True)
    # EarlyStopping
    earlystopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=1)
    # ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.75, patience=1, min_lr=0.0, verbose=1)
    callbacklist = [model_checkpoint, tensorboard,
                    earlystopping, reduce_lr, ]
    # shuffle data for training TqdmCallback(verbose=2)
    print("Learning rate: ", lr)
    model = pred_model(window_size)
    model.compile(optimizer=keras.optimizers.Adam(
        learning_rate=lr,), loss=QQuat_mult, metrics=[Quat_error_angle])
    history = model.fit(
        [x_acc, x_gyro, x_mag, x_fs],
        Head_quat,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[callbacklist],
        validation_data=([x_acc_val, x_gyro_val, x_mag_val,
                          x_fs_val], Head_quat_val),
        verbose=1,
        shuffle=True,
        max_queue_size=10,
        workers=8,
        use_multiprocessing=True)

    model.save('%s_B%s_E%s_V%s.hdf5' %
               (pred_model.__name__, batch_size, epochs, version))

    # plot training history
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'loss_%s_B%s_E%s_V%s.png' %
                (pred_model.__name__, batch_size, epochs, version))
    # plt.show()


def main():
    # for i in range(2, 11):
    pred_model = CNN_LSTM_yaw_mag_acc_gyro_fs

    # learningRate(pred_model)
    train(pred_model)


if __name__ == '__main__':

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    main()

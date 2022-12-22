from util import *
from dataset_loader import *
from model_yaw import *
from train_yaw import *
from keras.models import load_model
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import tensorflow_addons as tfa
from riann.riann import RIANN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
riann = RIANN()


q = 7
YAW = 1
stride = 4

fs = 286

window_size = 100
if window_size == 1:
    stride = 1


broad_test = 37
oxiod_test = []
oxiod_test.append(
    'Oxford Inertial Odometry Dataset/handheld/data1/syn/imu2.csv')
oxiod_test.append(
    'Oxford Inertial Odometry Dataset/handheld/data1/syn/imu5.csv')
oxiod_test.append(
    'Oxford Inertial Odometry Dataset/handheld/data1/syn/imu6.csv')
oxiod_test.append(
    'Oxford Inertial Odometry Dataset/handheld/data3/syn/imu1.csv')
oxiod_test.append(
    'Oxford Inertial Odometry Dataset/handheld/data4/syn/imu1.csv')
oxiod_test.append(
    'Oxford Inertial Odometry Dataset/handheld/data4/syn/imu3.csv')
oxiod_test.append(
    'Oxford Inertial Odometry Dataset/handheld/data5/syn/imu1.csv')


def data_broad(window_size, stride):
    # broad data
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    acc_temp, gyro_temp, mag_temp, quat_temp, fs = BROAD_data(
        BROAD_path(broad_test)[0], BROAD_path(broad_test)[1])
    acc = np.concatenate((acc, acc_temp), axis=0)
    gyro = np.concatenate((gyro, gyro_temp), axis=0)
    mag = np.concatenate((mag, mag_temp), axis=0)
    quat = np.concatenate((quat, quat_temp), axis=0)

    # acc = np.asanyarray(acc)
    # gyro = np.asanyarray(gyro)
    # mag = np.asanyarray(mag)
    # quat = np.asanyarray(quat)
    globals()['acc_main'] = acc
    globals()['gyro_main'] = gyro
    globals()['quat_main'] = quat
    globals()['fs_main'] = fs
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    return [gyro, acc, mag, fs], [quat]


def data_oxiod(window_size, stride):
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    for i in range(len(oxiod_test)):
        acc_temp, gyro_temp, mag_temp, quat_temp, fs = OxIOD_data(
            dataset_path+oxiod_test[i])
        acc = np.concatenate((acc, acc_temp), axis=0)
        gyro = np.concatenate((gyro, gyro_temp), axis=0)
        mag = np.concatenate((mag, mag_temp), axis=0)
        quat = np.concatenate((quat, quat_temp), axis=0)
    roll_ref, pitch_ref, yaw_ref = quat2eul(quat)
    quat = eul2quat(yaw_ref, pitch_ref, roll_ref)
    globals()['acc_main'] = acc
    globals()['gyro_main'] = gyro
    globals()['quat_main'] = quat
    globals()['fs_main'] = fs
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
    for i in range(len(df_TStick.values)-5, len(df_TStick.values)-3):
        acc_temp, gyro_temp, mag_temp, quat_temp, fs = repoIMU_data(
            df_TStick.values[i][0])
        acc = np.concatenate((acc, acc_temp), axis=0)
        gyro = np.concatenate((gyro, gyro_temp), axis=0)
        mag = np.concatenate((mag, mag_temp), axis=0)
        quat = np.concatenate((quat, quat_temp), axis=0)
    globals()['acc_main'] = acc
    globals()['gyro_main'] = gyro
    globals()['quat_main'] = quat
    globals()['fs_main'] = fs
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
    for i in range(len(df_Pendulum.values)-5, len(df_Pendulum.values)-3):
        acc_temp, gyro_temp, mag_temp, quat_temp, fs = repoIMU_data(
            df_Pendulum.values[i][0])
        acc = np.concatenate((acc, acc_temp), axis=0)
        gyro = np.concatenate((gyro, gyro_temp), axis=0)
        mag = np.concatenate((mag, mag_temp), axis=0)
        quat = np.concatenate((quat, quat_temp), axis=0)
    globals()['acc_main'] = acc
    globals()['gyro_main'] = gyro
    globals()['quat_main'] = quat
    globals()['fs_main'] = fs
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
        for j in range(len(MIMU)//2, len(MIMU)):
            acc_temp, gyro_temp, mag_temp, quat_temp, fs = sassari_data(
                file_list[i], MIMU[j])
            acc = np.concatenate((acc, acc_temp), axis=0)
            gyro = np.concatenate((gyro, gyro_temp), axis=0)
            mag = np.concatenate((mag, mag_temp), axis=0)
            quat = np.concatenate((quat, quat_temp), axis=0)
    globals()['acc_main'] = acc
    globals()['gyro_main'] = gyro
    globals()['quat_main'] = quat
    globals()['fs_main'] = fs
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    return [gyro, acc, mag, fs], [quat]


def data_ronin(window_size, stride, i):
    file_list = RoNIN_path()[0].values

    acc, gyro, mag, quat, fs = RoNIN_data(file_list[-i, 0])
    globals()['acc_main'] = acc
    globals()['gyro_main'] = gyro
    globals()['quat_main'] = quat
    globals()['fs_main'] = fs
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    return [gyro, acc, mag, fs], [quat]


def data_ridi(window_size, stride, i):

    file_list = RIDI_path()

    acc, gyro, mag, quat, fs = RIDI_data(file_list[-i])
    globals()['acc_main'] = acc
    globals()['gyro_main'] = gyro
    globals()['quat_main'] = quat
    globals()['fs_main'] = fs
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    return [gyro, acc, mag, fs], [quat]


def read_file_names(path):
    file_names = []
    for file in os.listdir(path):
        if file.endswith(".hdf5") or file.endswith(".h5"):
            file_names.append(file)
    return file_names


def model_load():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    model = load_model("./model_checkpoint.hdf5",
                       custom_objects={
                           'lossDCM2Quat': lossDCM2Quat,
                           "metric_dcm2quat_angle": metric_dcm2quat_angle,
                           'AdamW': tfa.optimizers.AdamW,
                           'Quat_mult': QQuat_mult, 'QQuat_mult': QQuat_mult,
                           "Quat_error_angle": Quat_error_angle, "Quat_error": Quat_error,
                           'TCN': TCN})
    return model


def main():

    for k in range(1):
        if i == 1:
            [gyro, acc, mag, fs], [quat_ref] = data_broad(window_size, stride)
        elif i == 2:
            [gyro, acc, mag, fs], [quat_ref] = data_oxiod(window_size, stride)
        elif i == 3:
            [gyro, acc, mag, fs], [quat_ref] = data_repoIMU_TStick(
                window_size, stride)
        elif i == 4:
            [gyro, acc, mag, fs], [quat_ref] = data_ronin(
                window_size, stride, k)
        elif i == 5:
            [gyro, acc, mag, fs], [quat_ref] = data_ridi(
                window_size, stride, k)
        elif i == 6:
            [gyro, acc, mag, fs], [quat_ref] = data_sassari(
                window_size, stride)
        elif i == 7:
            [gyro, acc, mag, fs], [quat_ref] = data_repoIMU_Pendulum(
                window_size, stride)
        model = model_load()
        headq = Head(quat_ref)
        quat_pred = model.predict(
            [acc, gyro, mag, fs], batch_size=500, verbose=1)
        _, _, yaw = quat2eul(quat_ref)
        roll_pred, pitch_pred, yaw_pred = quat2eul(quat_pred)
        plt.figure()
        plt.plot(yaw_pred, label='yaw_pred')
        plt.plot(roll_pred, label='roll_pred')
        plt.plot(pitch_pred, label='pitch_pred')
        plt.show()
        #roll_pred, pitch_pred, yaw_pred = quat2eul(quat)
        #acc_riann, gyro_riann = acc_main, gyro_main
        #attitude_riann = riann.predict(acc_riann, gyro_riann, fs_main)
        #roll_riann, pitch_riann, yaw_riann = quat2eul(attitude_riann)
        yaw_err = np.abs(yaw - yaw_pred)
        fs = fs[0] * stride
        t = np.arange(0, len(yaw)/fs, 1/fs)

        RMSE_yaw = np.sqrt(np.mean(yaw_err**2)) * 180/np.pi
        print("RMSE_yaw", RMSE_yaw)

        plt.figure()
        plt.plot(t, yaw_pred, label='Pred')
        plt.plot(t, yaw, label='ref')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    for i in range(1, q):
        main()

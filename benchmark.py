import sys
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from riann.riann import RIANN
from dataset_loader import *
from ahrs.filters import EKF, Madgwick, Mahony
import tensorflow_addons as tfa
from util import *
from keras.models import load_model
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
riann = RIANN()
n = 40
window_size, stride = 200, 4


# Load Data from:
# BROAD, OxIOD, Sassari, RoNIN, RIDI, euroc
# Load Model, in each fucntion, predict and calculate the error, then store the error in dataframe and save it
# error would be consist of:
# Simultanously, use EKF, Madgwick, Mahony, and RIANN for comparison study
# The result would be saved in a dataframe, and then save it in a csv file
# The saved dataframe would be consists of RMSE_roll, RMSE_Pitch, and Total_Rotation_Error for each trial and caluclted by the DL model, EKF, Madgwick, Mahony, and RIANN
#                           Total_Rotation_Error             |                  RMSE_roll                   |                       RMSE_pitch              |
# Trial No,     DL Model | EKF | Madgwick | Mahony | RIANN  ||| DL Model | EKF | Madgwick | Mahony | RIANN ||| DL Model | EKF | Madgwick | Mahony | RIANN  |||
#
model_name = "CuDNNLSTM_AAA_GGG_fs_B500_E250_V1"


def model_load():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    model = load_model(("./%s.hdf5" % model_name),
                       custom_objects={
                           'AdamW': tfa.optimizers.AdamW, 'Quat_mult': Quat_mult, "Quat_error_angle": Quat_error_angle, "Quat_error": Quat_error})
    return model


model = model_load()


def df_broad(window_size, stride):
    name = (sys._getframe().f_code.co_name).replace("df_", "")

    # tot
    broad_tot_RMSE_dl = np.zeros((0, 1))
    broad_tot_RMSE_ekf = np.zeros((0, 1))
    broad_tot_RMSE_madgwick = np.zeros((0, 1))
    broad_tot_RMSE_mahony = np.zeros((0, 1))
    broad_tot_RMSE_riann = np.zeros((0, 1))
    # roll and pitch
    broad_roll_RMSE_dl = np.zeros((0, 1))
    broad_roll_RMSE_ekf = np.zeros((0, 1))
    broad_roll_RMSE_madgwick = np.zeros((0, 1))
    broad_roll_RMSE_mahony = np.zeros((0, 1))
    broad_roll_RMSE_riann = np.zeros((0, 1))
    broad_pitch_RMSE_dl = np.zeros((0, 1))
    broad_pitch_RMSE_ekf = np.zeros((0, 1))
    broad_pitch_RMSE_madgwick = np.zeros((0, 1))
    broad_pitch_RMSE_mahony = np.zeros((0, 1))
    broad_pitch_RMSE_riann = np.zeros((0, 1))

    cols = np.empty((n-1,))
    for i in range(n-1):
        cols[i] = str(i+1)
    df_tot_dl = pd.DataFrame()
    df_roll_dl = pd.DataFrame()
    df_pitch_dl = pd.DataFrame()
    df_tot_ekf = pd.DataFrame()
    df_roll_ekf = pd.DataFrame()
    df_pitch_ekf = pd.DataFrame()
    df_tot_madgwick = pd.DataFrame()
    df_roll_madgwick = pd.DataFrame()
    df_pitch_madgwick = pd.DataFrame()
    df_tot_mahony = pd.DataFrame()
    df_roll_mahony = pd.DataFrame()
    df_pitch_mahony = pd.DataFrame()
    df_tot_riann = pd.DataFrame()
    df_roll_riann = pd.DataFrame()
    df_pitch_riann = pd.DataFrame()

    for i in range(1, n):
        # if i not in broad_set:
        # main data for filters and riann
        acc_main, gyro_main, mag_main, quat_main, fs_main = BROAD_data(
            BROAD_path(i)[0], BROAD_path(i)[1])
        quat_main = Att_q(quat_main)
        Quat_main = tf.Variable(quat_main, dtype=tf.float64)
        roll_main, pitch_main, yaw_main = quat2eul(quat_main)
        # ref data for dl model
        [gyro_ref, acc_ref, fs_ref], [quat_ref] = load_dataset_A_G_Fs(
            gyro_main, acc_main, quat_main, window_size, stride, fs_main)
        quat_ref = Att_q(quat_ref)
        Quat_ref = tf.Variable(quat_ref, dtype=tf.float64)
        roll_ref, pitch_ref, yaw_ref = quat2eul(quat_ref)

        # DL Model
        quat_dl = model.predict(
            [acc_ref, gyro_ref, fs_ref], batch_size=500, verbose=1)
        quat_dl = Att_q(quat_dl)
        roll_pred, pitch_pred, yaw_pred = quat2eul(quat_dl)
        Quat_dl = tf.Variable(quat_dl, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_ref, Quat_dl)
        tot = np.mean(np.sqrt(tot_err**2))
        broad_tot_RMSE_dl = np.vstack(
            [broad_tot_RMSE_dl, tot])
        # concatinate tot_error for each trial to a dataframe
        df_tot_dl = pd.concat(
            [df_tot_dl, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)

        # roll error
        err_roll = roll_ref - roll_pred
        # pitch error
        err_pitch = pitch_ref - pitch_pred
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        broad_roll_RMSE_dl = np.vstack(
            [broad_roll_RMSE_dl, RMSE_roll])
        broad_pitch_RMSE_dl = np.vstack(
            [broad_pitch_RMSE_dl, RMSE_pitch])
        df_roll_dl = pd.concat(
            [df_roll_dl, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
        df_pitch_dl = pd.concat(
            [df_pitch_dl, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)
        # RIANN
        attitude_riann = riann.predict(acc_main, gyro_main, fs_main)
        attitude_riann = Att_q(attitude_riann)
        Quat_riann = tf.Variable(attitude_riann, dtype=tf.float64)
        roll_riann, pitch_riann, yaw_riann = quat2eul(attitude_riann)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_riann)
        tot = np.mean(np.sqrt(tot_err**2))
        broad_tot_RMSE_riann = np.vstack(
            [broad_tot_RMSE_riann, tot])
        df_tot_riann = pd.concat(
            [df_tot_riann, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)
        # roll error
        err_roll = roll_main - roll_riann
        # pitch error
        err_pitch = pitch_main - pitch_riann
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        broad_roll_RMSE_riann = np.vstack(
            [broad_roll_RMSE_riann, RMSE_roll])
        broad_pitch_RMSE_riann = np.vstack(
            [broad_pitch_RMSE_riann, RMSE_pitch])

        df_roll_riann = pd.concat(
            [df_roll_riann, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
        df_pitch_riann = pd.concat(
            [df_pitch_riann, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)

        # EKF
        attitude_ekf = EKF(acc=acc_main, gyr=gyro_main,
                           frequency=fs_main).Q
        attitude_ekf = Att_q(attitude_ekf)
        roll_ekf, pitch_ekf, yaw_ekf = quat2eul(attitude_ekf)
        # limit roll and pitch to 2pi
        roll_ekf = (roll_ekf + np.pi)
        pitch_ekf = -pitch_ekf
        for k in range(len(roll_ekf)):
            if roll_ekf[k] > np.pi:
                roll_ekf[k] = roll_ekf[k]-2*np.pi
            elif roll_ekf[k] < -np.pi:
                roll_ekf[k] = roll_ekf[k]+2*np.pi
            if pitch_ekf[k] > np.pi:
                pitch_ekf[k] = pitch_ekf[k]-2*np.pi
            elif pitch_ekf[k] < -np.pi:
                pitch_ekf[k] = pitch_ekf[k]+2*np.pi

        attitude_ekf = eul2quat(yaw_ekf, pitch_ekf, roll_ekf)
        Quat_ekf = tf.Variable(attitude_ekf, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_ekf)
        tot = np.mean(np.sqrt(tot_err**2))
        broad_tot_RMSE_ekf = np.vstack(
            [broad_tot_RMSE_ekf, tot])

        df_tot_ekf = pd.concat(
            [df_tot_ekf, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)

        # roll error
        err_roll = roll_main - roll_ekf
        # pitch error
        err_pitch = pitch_main - pitch_ekf
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        broad_roll_RMSE_ekf = np.vstack(
            [broad_roll_RMSE_ekf, RMSE_roll])
        broad_pitch_RMSE_ekf = np.vstack(
            [broad_pitch_RMSE_ekf, RMSE_pitch])

        df_roll_ekf = pd.concat(
            [df_roll_ekf, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
        df_pitch_ekf = pd.concat(
            [df_pitch_ekf, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)

        # Mahony
        attitude_mahony = Mahony(
            acc=acc_main, gyr=gyro_main, frequency=fs_main).Q
        attitude_mahony = Att_q(attitude_mahony)
        roll_mahony, pitch_mahony, yaw_mahony = quat2eul(attitude_mahony)
        Quat_mahony = tf.Variable(attitude_mahony, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_mahony)
        tot = np.mean(np.sqrt(tot_err**2))
        broad_tot_RMSE_mahony = np.vstack(
            [broad_tot_RMSE_mahony, tot])

        df_tot_mahony = pd.concat(
            [df_tot_mahony, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)

        # roll error
        err_roll = roll_main - roll_mahony
        # pitch error
        err_pitch = pitch_main - pitch_mahony
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        broad_roll_RMSE_mahony = np.vstack(
            [broad_roll_RMSE_mahony, RMSE_roll])
        broad_pitch_RMSE_mahony = np.vstack(
            [broad_pitch_RMSE_mahony, RMSE_pitch])

        df_roll_mahony = pd.concat(
            [df_roll_mahony, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
        df_pitch_mahony = pd.concat(
            [df_pitch_mahony, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)

        # Madgwick
        attitude_madgwick = Madgwick(
            acc=acc_main, gyr=gyro_main, frequency=fs_main).Q
        attitude_madgwick = Att_q(attitude_madgwick)
        roll_madgwick, pitch_madgwick, yaw_madgwick = quat2eul(
            attitude_madgwick)
        Quat_madgwick = tf.Variable(attitude_madgwick, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_madgwick)
        tot = np.mean(np.sqrt(tot_err**2))
        broad_tot_RMSE_madgwick = np.vstack(
            [broad_tot_RMSE_madgwick, tot])
        df_tot_madgwick = pd.concat(
            [df_tot_madgwick, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)

        # roll error
        err_roll = roll_main - roll_madgwick
        # pitch error
        err_pitch = pitch_main - pitch_madgwick
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        broad_roll_RMSE_madgwick = np.vstack(
            [broad_roll_RMSE_madgwick, RMSE_roll])
        broad_pitch_RMSE_madgwick = np.vstack(
            [broad_pitch_RMSE_madgwick, RMSE_pitch])

        df_roll_madgwick = pd.concat(
            [df_roll_madgwick, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
        df_pitch_madgwick = pd.concat(
            [df_pitch_madgwick, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)

    broad_set = np.arange(1, n)

    # save dfs error
    df_tot_dl.to_csv('./Error/total/%s_%s_tot_dl.csv' % (model_name, name))
    df_tot_ekf.to_csv('./Error/total/%s_tot_ekf.csv' % (model_name, name))
    df_tot_mahony.to_csv('./Error/total/%s_tot_mahony.csv' %
                         (model_name, name))
    df_tot_madgwick.to_csv(
        './Error/total/%s_tot_madgwick.csv' % (model_name, name))
    df_tot_riann.to_csv('./Error/total/%s_tot_riann.csv' % (model_name, name))
    df_roll_dl.to_csv('./Error/roll/%s_roll_dl.csv' % (model_name, name))
    df_roll_ekf.to_csv('./Error/roll/%s_roll_ekf.csv' % (model_name, name))
    df_roll_mahony.to_csv('./Error/roll/%s_roll_mahony.csv' %
                          (model_name, name))
    df_roll_madgwick.to_csv(
        './Error/roll/%s_roll_madgwick.csv' % (model_name, name))
    df_roll_riann.to_csv('./Error/roll/%s_roll_riann.csv' % (model_name, name))
    df_pitch_dl.to_csv('./Error/pitch/%s_pitch_dl.csv' % (model_name, name))
    df_pitch_ekf.to_csv('./Error/pitch/%s_pitch_ekf.csv' % (model_name, name))
    df_pitch_mahony.to_csv(
        './Error/pitch/%s_pitch_mahony.csv' % (model_name, name))
    df_pitch_madgwick.to_csv(
        './Error/pitch/%s_pitch_madgwick.csv' % (model_name, name))
    df_pitch_riann.to_csv('./Error/pitch/%s_pitch_riann.csv' %
                          (model_name, name))

    # The saved dataframe would be consists of RMSE_roll, RMSE_Pitch, and Total_Rotation_Error for each trial and caluclted by the DL model, EKF, Madgwick, Mahony, and RIANN
    #                           Total_Rotation_Error             |                  RMSE_roll                   |                       RMSE_pitch              |
    # Trial No,     DL Model | EKF | Madgwick | Mahony | RIANN  ||| DL Model | EKF | Madgwick | Mahony | RIANN ||| DL Model | EKF | Madgwick | Mahony | RIANN  |||
    header = pd.MultiIndex.from_product([['Total_Rotation_Error', 'RMSE_roll', 'RMSE_pitch'], [
                                        'DL Model', 'EKF', 'Madgwick', 'Mahony', 'RIANN']], names=['', 'Trial No'])
    df = pd.DataFrame(np.hstack([broad_tot_RMSE_dl,
                                broad_tot_RMSE_ekf,
                                broad_tot_RMSE_madgwick,
                                broad_tot_RMSE_mahony,
                                broad_tot_RMSE_riann,
                                broad_roll_RMSE_dl,
                                broad_roll_RMSE_ekf,
                                broad_roll_RMSE_madgwick,
                                broad_roll_RMSE_mahony,
                                broad_roll_RMSE_riann,
                                broad_pitch_RMSE_dl,
                                broad_pitch_RMSE_ekf,
                                broad_pitch_RMSE_madgwick,
                                broad_pitch_RMSE_mahony,
                                broad_pitch_RMSE_riann]),
                      columns=header,
                      index=broad_set)

    # show df
    print(df)
    df.to_csv('%s_broad_set.csv' % (model_name))
    return


def df_oxiod(window_size, stride):
    name = (sys._getframe().f_code.co_name).replace("df_", "")
    # tot
    oxiod_tot_RMSE_dl = np.zeros((0, 1))
    oxiod_tot_RMSE_ekf = np.zeros((0, 1))
    oxiod_tot_RMSE_madgwick = np.zeros((0, 1))
    oxiod_tot_RMSE_mahony = np.zeros((0, 1))
    oxiod_tot_RMSE_riann = np.zeros((0, 1))
    # roll and pitch
    oxiod_roll_RMSE_dl = np.zeros((0, 1))
    oxiod_roll_RMSE_ekf = np.zeros((0, 1))
    oxiod_roll_RMSE_madgwick = np.zeros((0, 1))
    oxiod_roll_RMSE_mahony = np.zeros((0, 1))
    oxiod_roll_RMSE_riann = np.zeros((0, 1))
    oxiod_pitch_RMSE_dl = np.zeros((0, 1))
    oxiod_pitch_RMSE_ekf = np.zeros((0, 1))
    oxiod_pitch_RMSE_madgwick = np.zeros((0, 1))
    oxiod_pitch_RMSE_mahony = np.zeros((0, 1))
    oxiod_pitch_RMSE_riann = np.zeros((0, 1))

    files = OxIOD_path()
    cols = np.empty((len(OxIOD_path().values)-1,))
    for i in range(n-1):
        cols[i] = str(i+1)
    df_tot_dl = pd.DataFrame()
    df_roll_dl = pd.DataFrame()
    df_pitch_dl = pd.DataFrame()
    df_tot_ekf = pd.DataFrame()
    df_roll_ekf = pd.DataFrame()
    df_pitch_ekf = pd.DataFrame()
    df_tot_madgwick = pd.DataFrame()
    df_roll_madgwick = pd.DataFrame()
    df_pitch_madgwick = pd.DataFrame()
    df_tot_mahony = pd.DataFrame()
    df_roll_mahony = pd.DataFrame()
    df_pitch_mahony = pd.DataFrame()
    df_tot_riann = pd.DataFrame()
    df_roll_riann = pd.DataFrame()
    df_pitch_riann = pd.DataFrame()
    for i in range(len(OxIOD_path().values)):
        # main data for filters and riann
        acc_main, gyro_main, mag_main, quat_main, fs_main = OxIOD_data(
            OxIOD_path().values[i][0])
        quat_main = Att_q(quat_main)
        Quat_main = tf.Variable(quat_main, dtype=tf.float64)
        roll_main, pitch_main, yaw_main = quat2eul(quat_main)
        # ref data for DL model
        [gyro_ref, acc_ref, fs_ref], [quat_ref] = load_dataset_A_G_Fs(
            gyro_main, acc_main, quat_main, window_size, stride, fs_main)
        quat_ref = Att_q(quat_ref)
        Quat_ref = tf.Variable(quat_ref, dtype=tf.float64)
        roll_ref, pitch_ref, yaw_ref = quat2eul(quat_ref)

        # DL Model
        quat_dl = model.predict(
            [acc_ref, gyro_ref, fs_ref], batch_size=500, verbose=1)
        quat_dl = Att_q(quat_dl)
        roll_pred, pitch_pred, yaw_pred = quat2eul(quat_dl)
        Quat_dl = tf.Variable(quat_dl, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_ref, Quat_dl)
        tot = np.mean(np.sqrt(tot_err**2))
        oxiod_tot_RMSE_dl = np.vstack([oxiod_tot_RMSE_dl, tot])
        df_tot_dl = pd.concat(
            [df_tot_dl, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)
        # roll error
        err_roll = roll_ref - roll_pred
        # pitch error
        err_pitch = pitch_ref - pitch_pred
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        oxiod_roll_RMSE_dl = np.vstack(
            [oxiod_roll_RMSE_dl, RMSE_roll])
        oxiod_pitch_RMSE_dl = np.vstack(
            [oxiod_pitch_RMSE_dl, RMSE_pitch])
        df_roll_dl = pd.concat(
            [df_roll_dl, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
        df_pitch_dl = pd.concat(
            [df_pitch_dl, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)

        # RIANN
        attitude_riann = riann.predict(acc_main, gyro_main, fs_main)
        attitude_riann = Att_q(attitude_riann)
        roll_riann, pitch_riann, yaw_riann = quat2eul(attitude_riann)
        #roll_riann = -(roll_riann + np.pi)
        #pitch_riann = pitch_riann+np.pi/6
        for k in range(len(roll_riann)):
            while roll_riann[k] > np.pi:
                roll_riann[k] = roll_riann[k]-2*np.pi
            while roll_riann[k] < -np.pi:
                roll_riann[k] = roll_riann[k]+2*np.pi
            while pitch_riann[k] > np.pi:
                pitch_riann[k] = pitch_riann[k]-2*np.pi
            while pitch_riann[k] < -np.pi:
                pitch_riann[k] = pitch_riann[k]+2*np.pi
        attitude_riann = eul2quat(yaw_riann, pitch_riann, roll_riann)
        Quat_riann = tf.Variable(attitude_riann, dtype=tf.float64)
        roll_riann, pitch_riann, yaw_riann = quat2eul(attitude_riann)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_riann)
        tot = np.mean(np.sqrt(tot_err**2))
        oxiod_tot_RMSE_riann = np.vstack(
            [oxiod_tot_RMSE_riann, tot])

        df_tot_riann = pd.concat(
            [df_tot_riann, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)
        # roll error
        err_roll = roll_main - roll_riann
        # pitch error
        err_pitch = pitch_main - pitch_riann
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        oxiod_roll_RMSE_riann = np.vstack(
            [oxiod_roll_RMSE_riann, RMSE_roll])
        oxiod_pitch_RMSE_riann = np.vstack(
            [oxiod_pitch_RMSE_riann, RMSE_pitch])

        df_roll_riann = pd.concat(
            [df_roll_riann, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
        df_pitch_riann = pd.concat(
            [df_pitch_riann, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)

        # EKF
        attitude_ekf = EKF(acc=acc_main, gyr=gyro_main,
                           frequency=fs_main).Q
        attitude_ekf = Att_q(attitude_ekf)
        roll_ekf, pitch_ekf, yaw_ekf = quat2eul(attitude_ekf)
        # limit roll and pitch to 2pi
        for k in range(len(roll_ekf)):
            if roll_ekf[k] > np.pi:
                roll_ekf[k] = roll_ekf[k]-2*np.pi
            elif roll_ekf[k] < -np.pi:
                roll_ekf[k] = roll_ekf[k]+2*np.pi
            if pitch_ekf[k] > np.pi:
                pitch_ekf[k] = pitch_ekf[k]-2*np.pi
            elif pitch_ekf[k] < -np.pi:
                pitch_ekf[k] = pitch_ekf[k]+2*np.pi
        roll_ekf = -(roll_ekf + np.pi)
        attitude_ekf = eul2quat(yaw_ekf, pitch_ekf, roll_ekf)
        Quat_ekf = tf.Variable(attitude_ekf, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_ekf)
        tot = np.mean(np.sqrt(tot_err**2))
        oxiod_tot_RMSE_ekf = np.vstack(
            [oxiod_pitch_RMSE_ekf, tot])
        df_tot_ekf = pd.concat(
            [df_tot_ekf, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)
        # roll error
        err_roll = roll_main - roll_ekf
        # pitch error
        err_pitch = pitch_main - pitch_ekf
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        oxiod_roll_RMSE_ekf = np.vstack(
            [oxiod_roll_RMSE_ekf, RMSE_roll])
        oxiod_pitch_RMSE_ekf = np.vstack(
            [oxiod_pitch_RMSE_ekf, RMSE_pitch])

        df_roll_ekf = pd.concat(
            [df_roll_ekf, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
        df_pitch_ekf = pd.concat(
            [df_pitch_ekf, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)

        # Mahony
        attitude_mahony = Mahony(
            acc=acc_main, gyr=gyro_main, frequency=fs_main).Q
        attitude_mahony = Att_q(attitude_mahony)
        roll_mahony, pitch_mahony, yaw_mahony = quat2eul(attitude_mahony)
        Quat_mahony = tf.Variable(attitude_mahony, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_mahony)
        tot = np.mean(np.sqrt(tot_err**2))
        oxiod_tot_RMSE_mahony = np.vstack(
            [oxiod_tot_RMSE_mahony, tot])
        df_tot_mahony = pd.concat(
            [df_tot_mahony, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)
        # roll error
        err_roll = roll_main - roll_mahony
        # pitch error
        err_pitch = pitch_main - pitch_mahony
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        oxiod_roll_RMSE_mahony = np.vstack(
            [oxiod_roll_RMSE_mahony, RMSE_roll])
        oxiod_pitch_RMSE_mahony = np.vstack(
            [oxiod_pitch_RMSE_mahony, RMSE_pitch])
        df_roll_mahony = pd.concat(
            [df_roll_mahony, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
        df_pitch_mahony = pd.concat(
            [df_pitch_mahony, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)

        # Madgwick
        attitude_madgwick = Madgwick(
            acc=acc_main, gyr=gyro_main, frequency=fs_main).Q
        attitude_madgwick = Att_q(attitude_madgwick)
        roll_madgwick, pitch_madgwick, yaw_madgwick = quat2eul(
            attitude_madgwick)
        Quat_madgwick = tf.Variable(attitude_madgwick, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_madgwick)
        tot = np.mean(np.sqrt(tot_err**2))
        oxiod_tot_RMSE_madgwick = np.vstack(
            [oxiod_tot_RMSE_madgwick, tot])
        df_tot_madgwick = pd.concat(
            [df_tot_madgwick, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)
        # roll error
        err_roll = roll_main - roll_madgwick
        # pitch error
        err_pitch = pitch_main - pitch_madgwick
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
            # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        oxiod_roll_RMSE_madgwick = np.vstack(
            [oxiod_roll_RMSE_madgwick, RMSE_roll])
        oxiod_pitch_RMSE_madgwick = np.vstack(
            [oxiod_pitch_RMSE_madgwick, RMSE_pitch])
        df_roll_madgwick = pd.concat(
            [df_roll_madgwick, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
        df_pitch_madgwick = pd.concat(
            [df_pitch_madgwick, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)

    df = OxIOD_path()
    df['OxIOD File Path'] = df['OxIOD File Path'].str.replace(
        r'/home/arman/Neural Network in Python/RNN/Dataset/Oxford Inertial Odometry Dataset/', '')
    df['OxIOD File Path'] = df['OxIOD File Path'].str.replace(r'.csv', '')
    df['OxIOD File Path'] = df['OxIOD File Path'].str.replace(r'/syn', '')
    oxiod_set = df.values
    # The saved dataframe would be consists of RMSE_roll, RMSE_Pitch, and Total_Rotation_Error for each trial and caluclted by the DL model, EKF, Madgwick, Mahony, and RIANN
    #                           Total_Rotation_Error             |                  RMSE_roll                   |                       RMSE_pitch              |
    # Trial No,     DL Model | EKF | Madgwick | Mahony | RIANN  ||| DL Model | EKF | Madgwick | Mahony | RIANN ||| DL Model | EKF | Madgwick | Mahony | RIANN  |||
    header = pd.MultiIndex.from_product([['Total_Rotation_Error', 'RMSE_roll', 'RMSE_pitch'], [
                                        'DL Model', 'EKF', 'Madgwick', 'Mahony', 'RIANN']], names=['', 'Trial No'])
    df = pd.DataFrame(np.hstack([oxiod_tot_RMSE_dl,
                                oxiod_tot_RMSE_ekf,
                                oxiod_tot_RMSE_madgwick,
                                oxiod_tot_RMSE_mahony,
                                oxiod_tot_RMSE_riann,
                                oxiod_roll_RMSE_dl,
                                oxiod_roll_RMSE_ekf,
                                oxiod_roll_RMSE_madgwick,
                                oxiod_roll_RMSE_mahony,
                                oxiod_roll_RMSE_riann,
                                oxiod_pitch_RMSE_dl,
                                oxiod_pitch_RMSE_ekf,
                                oxiod_pitch_RMSE_madgwick,
                                oxiod_pitch_RMSE_mahony,
                                oxiod_pitch_RMSE_riann]),
                      columns=header,
                      index=df['OxIOD File Path'][:][:].values[:][:])
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv('%s_oxiod_set.csv' % (model_name))
    # save dfs error
    df_tot_dl.to_csv('./Error/total/%s_tot_dl.csv' % (model_name, name))
    df_tot_ekf.to_csv('./Error/total/%s_tot_ekf.csv' % (model_name, name))
    df_tot_mahony.to_csv('./Error/total/%s_tot_mahony.csv' %
                         (model_name, name))
    df_tot_madgwick.to_csv(
        './Error/total/%s_tot_madgwick.csv' % (model_name, name))
    df_tot_riann.to_csv('./Error/total/%s_tot_riann.csv' % (model_name, name))
    df_roll_dl.to_csv('./Error/roll/%s_roll_dl.csv' % (model_name, name))
    df_roll_ekf.to_csv('./Error/roll/%s_roll_ekf.csv' % (model_name, name))
    df_roll_mahony.to_csv('./Error/roll/%s_roll_mahony.csv' %
                          (model_name, name))
    df_roll_madgwick.to_csv(
        './Error/roll/%s_roll_madgwick.csv' % (model_name, name))
    df_roll_riann.to_csv('./Error/roll/%s_roll_riann.csv' % (model_name, name))
    df_pitch_dl.to_csv('./Error/pitch/%s_pitch_dl.csv' % (model_name, name))
    df_pitch_ekf.to_csv('./Error/pitch/%s_pitch_ekf.csv' % (model_name, name))
    df_pitch_mahony.to_csv(
        './Error/pitch/%s_pitch_mahony.csv' % (model_name, name))
    df_pitch_madgwick.to_csv(
        './Error/pitch/%s_pitch_madgwick.csv' % (model_name, name))
    df_pitch_riann.to_csv('./Error/pitch/%s_pitch_riann.csv' %
                          (model_name, name))

    return


def df_repoIMU_TStick(window_size, stride):
    name = (sys._getframe().f_code.co_name).replace("df_", "")
    # tot
    TStick_tot_RMSE_dl = np.zeros((0, 1))
    TStick_tot_RMSE_ekf = np.zeros((0, 1))
    TStick_tot_RMSE_madgwick = np.zeros((0, 1))
    TStick_tot_RMSE_mahony = np.zeros((0, 1))
    TStick_tot_RMSE_riann = np.zeros((0, 1))
    # roll and pitch
    TStick_roll_RMSE_dl = np.zeros((0, 1))
    TStick_roll_RMSE_ekf = np.zeros((0, 1))
    TStick_roll_RMSE_madgwick = np.zeros((0, 1))
    TStick_roll_RMSE_mahony = np.zeros((0, 1))
    TStick_roll_RMSE_riann = np.zeros((0, 1))
    TStick_pitch_RMSE_dl = np.zeros((0, 1))
    TStick_pitch_RMSE_ekf = np.zeros((0, 1))
    TStick_pitch_RMSE_madgwick = np.zeros((0, 1))
    TStick_pitch_RMSE_mahony = np.zeros((0, 1))
    TStick_pitch_RMSE_riann = np.zeros((0, 1))

    df_TStick, df_Pendulum = repoIMU_path()
    cols = np.empty((n-1,))
    for i in range(len(df_TStick.values)-1):
        cols[i] = str(i+1)
    df_tot_dl = pd.DataFrame()
    df_roll_dl = pd.DataFrame()
    df_pitch_dl = pd.DataFrame()
    df_tot_ekf = pd.DataFrame()
    df_roll_ekf = pd.DataFrame()
    df_pitch_ekf = pd.DataFrame()
    df_tot_madgwick = pd.DataFrame()
    df_roll_madgwick = pd.DataFrame()
    df_pitch_madgwick = pd.DataFrame()
    df_tot_mahony = pd.DataFrame()
    df_roll_mahony = pd.DataFrame()
    df_pitch_mahony = pd.DataFrame()
    df_tot_riann = pd.DataFrame()
    df_roll_riann = pd.DataFrame()
    df_pitch_riann = pd.DataFrame()
    for i in range(len(df_TStick.values)):
        acc_main, gyro_main, mag_main, quat_main, fs_main = repoIMU_data(
            df_TStick.values[i][0])
        quat_main = Att_q(quat_main)
        Quat_main = tf.Variable(quat_main, dtype=tf.float64)
        roll_main, pitch_main, yaw_main = quat2eul(quat_main)
        # ref data for dl model
        [gyro_ref, acc_ref, fs_ref], [quat_ref] = load_dataset_A_G_Fs(
            gyro_main, acc_main, quat_main, window_size, stride, fs_main)
        quat_ref = Att_q(quat_ref)
        Quat_ref = tf.Variable(quat_ref, dtype=tf.float64)
        roll_ref, pitch_ref, yaw_ref = quat2eul(quat_ref)

        # DL Model
        quat_dl = model.predict(
            [acc_ref, gyro_ref, fs_ref], batch_size=500, verbose=1)
        quat_dl = Att_q(quat_dl)
        roll_pred, pitch_pred, yaw_pred = quat2eul(quat_dl)
        Quat_dl = tf.Variable(quat_dl, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_ref, Quat_dl)
        tot = np.mean(np.sqrt(tot_err**2))
        TStick_tot_RMSE_dl = np.vstack([TStick_tot_RMSE_dl, tot])
        df_tot_dl = pd.concat(
            [df_tot_dl, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)

        # roll error
        err_roll = roll_ref - roll_pred
        # pitch error
        err_pitch = pitch_ref - pitch_pred
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        TStick_roll_RMSE_dl = np.vstack([TStick_roll_RMSE_dl, RMSE_roll])
        TStick_pitch_RMSE_dl = np.vstack([TStick_pitch_RMSE_dl, RMSE_pitch])
        df_roll_dl = pd.concat(
            [df_roll_dl, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
        df_pitch_dl = pd.concat(
            [df_pitch_dl, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)

        # RIANN
        attitude_riann = riann.predict(acc_main, gyro_main, fs_main)
        attitude_riann = Att_q(attitude_riann)
        Quat_riann = tf.Variable(attitude_riann, dtype=tf.float64)
        roll_riann, pitch_riann, yaw_riann = quat2eul(attitude_riann)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_riann)
        tot = np.mean(np.sqrt(tot_err**2))
        TStick_tot_RMSE_riann = np.vstack([TStick_tot_RMSE_riann, tot])
        df_tot_riann = pd.concat(
            [df_tot_riann, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)
        # roll error
        err_roll = roll_main - roll_riann
        # pitch error
        err_pitch = pitch_main - pitch_riann
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        TStick_roll_RMSE_riann = np.vstack([TStick_roll_RMSE_riann, RMSE_roll])
        TStick_pitch_RMSE_riann = np.vstack(
            [TStick_pitch_RMSE_riann, RMSE_pitch])

        df_roll_riann = pd.concat(
            [df_roll_riann, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
        df_pitch_riann = pd.concat(
            [df_pitch_riann, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)

        # EKF
        attitude_ekf = EKF(acc=acc_main, gyr=gyro_main,
                           frequency=fs_main, frame='NED').Q
        attitude_ekf = Att_q(attitude_ekf)
        roll_ekf, pitch_ekf, yaw_ekf = quat2eul(attitude_ekf)
        roll_ekf = roll_ekf - np.pi
        attitude_ekf = eul2quat(yaw_ekf, pitch_ekf, roll_ekf)
        Quat_ekf = tf.Variable(attitude_ekf, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_ekf)
        tot = np.mean(np.sqrt(tot_err**2))
        TStick_tot_RMSE_ekf = np.vstack([TStick_tot_RMSE_ekf, tot])
        df_tot_ekf = pd.concat(
            [df_tot_ekf, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)

        # roll error
        err_roll = roll_main - roll_ekf
        # pitch error
        err_pitch = pitch_main - pitch_ekf
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        TStick_roll_RMSE_ekf = np.vstack([TStick_roll_RMSE_ekf, RMSE_roll])
        TStick_pitch_RMSE_ekf = np.vstack([TStick_pitch_RMSE_ekf, RMSE_pitch])
        df_roll_ekf = pd.concat(
            [df_roll_ekf, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
        df_pitch_ekf = pd.concat(
            [df_pitch_ekf, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)

        # Mahony
        attitude_mahony = Mahony(
            acc=acc_main, gyr=gyro_main, frequency=fs_main).Q
        attitude_mahony = Att_q(attitude_mahony)
        roll_mahony, pitch_mahony, yaw_mahony = quat2eul(attitude_mahony)
        Quat_mahony = tf.Variable(attitude_mahony, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_mahony)
        tot = np.mean(np.sqrt(tot_err**2))
        TStick_tot_RMSE_mahony = np.vstack([TStick_tot_RMSE_mahony, tot])
        df_tot_mahony = pd.concat(
            [df_tot_mahony, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)

        # roll error
        err_roll = roll_main - roll_mahony
        # pitch error
        err_pitch = pitch_main - pitch_mahony
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        TStick_roll_RMSE_mahony = np.vstack(
            [TStick_roll_RMSE_mahony, RMSE_roll])
        TStick_pitch_RMSE_mahony = np.vstack(
            [TStick_pitch_RMSE_mahony, RMSE_pitch])
        df_roll_mahony = pd.concat(
            [df_roll_mahony, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
        df_pitch_mahony = pd.concat(
            [df_pitch_mahony, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)

        # Madgwick
        attitude_madgwick = Madgwick(
            acc=acc_main, gyr=gyro_main, frequency=fs_main).Q
        attitude_madgwick = Att_q(attitude_madgwick)
        roll_madgwick, pitch_madgwick, yaw_madgwick = quat2eul(
            attitude_madgwick)
        Quat_madgwick = tf.Variable(attitude_madgwick, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_madgwick)
        tot = np.mean(np.sqrt(tot_err**2))
        TStick_tot_RMSE_madgwick = np.vstack([TStick_tot_RMSE_madgwick, tot])
        df_tot_madgwick = pd.concat(
            [df_tot_madgwick, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)

        # roll error
        err_roll = roll_main - roll_madgwick
        # pitch error
        err_pitch = pitch_main - pitch_madgwick
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        TStick_roll_RMSE_madgwick = np.vstack(
            [TStick_roll_RMSE_madgwick, RMSE_roll])
        TStick_pitch_RMSE_madgwick = np.vstack(
            [TStick_pitch_RMSE_madgwick, RMSE_pitch])
        df_roll_madgwick = pd.concat(
            [df_roll_madgwick, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
        df_pitch_madgwick = pd.concat(
            [df_pitch_madgwick, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)

    df_TStick['TStick'] = df_TStick['TStick'].str.replace('TStick_', '')
    df_TStick['TStick'] = df_TStick['TStick'].str.replace('.csv', '')
    # The saved dataframe would be consists of RMSE_roll, RMSE_Pitch, and Total_Rotation_Error for each trial and caluclted by the DL model, EKF, Madgwick, Mahony, and RIANN
    #                           Total_Rotation_Error             |                  RMSE_roll                   |                       RMSE_pitch              |
    # Trial No,     DL Model | EKF | Madgwick | Mahony | RIANN  ||| DL Model | EKF | Madgwick | Mahony | RIANN ||| DL Model | EKF | Madgwick | Mahony | RIANN  |||
    header = pd.MultiIndex.from_product([['Total_Rotation_Error', 'RMSE_roll', 'RMSE_pitch'], [
                                        'DL Model', 'EKF', 'Madgwick', 'Mahony', 'RIANN']], names=['', 'Trial No'])
    df = pd.DataFrame(np.hstack([TStick_tot_RMSE_dl,
                                 TStick_tot_RMSE_ekf,
                                 TStick_tot_RMSE_madgwick,
                                 TStick_tot_RMSE_mahony,
                                 TStick_tot_RMSE_riann,
                                 TStick_roll_RMSE_dl,
                                 TStick_roll_RMSE_ekf,
                                 TStick_roll_RMSE_madgwick,
                                 TStick_roll_RMSE_mahony,
                                 TStick_roll_RMSE_riann,
                                 TStick_pitch_RMSE_dl,
                                 TStick_pitch_RMSE_ekf,
                                 TStick_pitch_RMSE_madgwick,
                                 TStick_pitch_RMSE_mahony,
                                 TStick_pitch_RMSE_riann]),
                      columns=header,
                      index=df_TStick['TStick'][:][:].values[:][:])
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(('%sTStick_RMSE.csv' % (model_name)), index=True)
    # save dfs error
    df_tot_dl.to_csv('./Error/total/%s_tot_dl.csv' % (model_name, name))
    df_tot_ekf.to_csv('./Error/total/%s_tot_ekf.csv' % (model_name, name))
    df_tot_mahony.to_csv('./Error/total/%s_tot_mahony.csv' %
                         (model_name, name))
    df_tot_madgwick.to_csv(
        './Error/total/%s_tot_madgwick.csv' % (model_name, name))
    df_tot_riann.to_csv('./Error/total/%s_tot_riann.csv' % (model_name, name))
    df_roll_dl.to_csv('./Error/roll/%s_roll_dl.csv' % (model_name, name))
    df_roll_ekf.to_csv('./Error/roll/%s_roll_ekf.csv' % (model_name, name))
    df_roll_mahony.to_csv('./Error/roll/%s_roll_mahony.csv' %
                          (model_name, name))
    df_roll_madgwick.to_csv(
        './Error/roll/%s_roll_madgwick.csv' % (model_name, name))
    df_roll_riann.to_csv('./Error/roll/%s_roll_riann.csv' % (model_name, name))
    df_pitch_dl.to_csv('./Error/pitch/%s_pitch_dl.csv' % (model_name, name))
    df_pitch_ekf.to_csv('./Error/pitch/%s_pitch_ekf.csv' % (model_name, name))
    df_pitch_mahony.to_csv(
        './Error/pitch/%s_pitch_mahony.csv' % (model_name, name))
    df_pitch_madgwick.to_csv(
        './Error/pitch/%s_pitch_madgwick.csv' % (model_name, name))
    df_pitch_riann.to_csv('./Error/pitch/%s_pitch_riann.csv' %
                          (model_name, name))
    return


def df_sassari(window_size, stride):
    name = (sys._getframe().f_code.co_name).replace("df_", "")
    # tot
    sassari_tot_RMSE_dl = np.zeros((0, 1))
    sassari_tot_RMSE_ekf = np.zeros((0, 1))
    sassari_tot_RMSE_madgwick = np.zeros((0, 1))
    sassari_tot_RMSE_mahony = np.zeros((0, 1))
    sassari_tot_RMSE_riann = np.zeros((0, 1))
    sassari_roll_RMSE_dl = np.zeros((0, 1))
    sassari_roll_RMSE_ekf = np.zeros((0, 1))
    sassari_roll_RMSE_madgwick = np.zeros((0, 1))
    sassari_roll_RMSE_mahony = np.zeros((0, 1))
    sassari_roll_RMSE_riann = np.zeros((0, 1))
    sassari_pitch_RMSE_dl = np.zeros((0, 1))
    sassari_pitch_RMSE_ekf = np.zeros((0, 1))
    sassari_pitch_RMSE_madgwick = np.zeros((0, 1))
    sassari_pitch_RMSE_mahony = np.zeros((0, 1))
    sassari_pitch_RMSE_riann = np.zeros((0, 1))
    MIMU = ["XS1", "AP2", "SH1", "XS2", "AP1", "SH2"]
    file_list = sassari_path()
    file_name = ["fast_v4", "slow_v4", "medium_v4"]
    files = []
    # add MIMU elements to file_name elements

    for j in range(len(file_name)):
        for i in range(len(MIMU)):
            temp = (file_name[j]+'/'+MIMU[i])
            files.append(temp)
    cols = files
    df_tot_dl = pd.DataFrame()
    df_roll_dl = pd.DataFrame()
    df_pitch_dl = pd.DataFrame()
    df_tot_ekf = pd.DataFrame()
    df_roll_ekf = pd.DataFrame()
    df_pitch_ekf = pd.DataFrame()
    df_tot_madgwick = pd.DataFrame()
    df_roll_madgwick = pd.DataFrame()
    df_pitch_madgwick = pd.DataFrame()
    df_tot_mahony = pd.DataFrame()
    df_roll_mahony = pd.DataFrame()
    df_pitch_mahony = pd.DataFrame()
    df_tot_riann = pd.DataFrame()
    df_roll_riann = pd.DataFrame()
    df_pitch_riann = pd.DataFrame()
    for i in range(len(file_list)):
        for j in range(len(MIMU)):
            acc_main, gyro_main, mag_main, quat_main, fs_main = sassari_data(
                file_list[i], MIMU[j])
            quat_main = Att_q(quat_main)
            Quat_main = tf.Variable(quat_main, dtype=tf.float64)
            roll_main, pitch_main, yaw_main = quat2eul(quat_main)
            # ref data for dl model
            [gyro_ref, acc_ref, fs_ref], [quat_ref] = load_dataset_A_G_Fs(
                gyro_main, acc_main, quat_main, window_size, stride, fs_main)
            quat_ref = Att_q(quat_ref)
            Quat_ref = tf.Variable(quat_ref, dtype=tf.float64)
            roll_ref, pitch_ref, yaw_ref = quat2eul(quat_ref)

            # DL Model
            quat_dl = model.predict(
                [acc_ref, gyro_ref, fs_ref], batch_size=500, verbose=1)
            quat_dl = Att_q(quat_dl)
            roll_pred, pitch_pred, yaw_pred = quat2eul(quat_dl)
            Quat_dl = tf.Variable(quat_dl, dtype=tf.float64)
            # tot error
            tot_err = quat_error_angle(Quat_ref, Quat_dl)
            tot = np.mean(np.sqrt(tot_err**2))
            sassari_tot_RMSE_dl = np.vstack([sassari_tot_RMSE_dl, tot])
            df_tot_dl = pd.concat(
                [df_tot_dl, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)
            # roll error
            err_roll = roll_ref - roll_pred
            # pitch error
            err_pitch = pitch_ref - pitch_pred
            # limit the max error to 180
            for k in range(len(err_roll)):
                if err_roll[k] > np.pi:
                    err_roll[k] = err_roll[k]-2*np.pi
                elif err_roll[k] < -np.pi:
                    err_roll[k] = err_roll[k]+2*np.pi
                if err_pitch[k] > np.pi:
                    err_pitch[k] = err_pitch[k]-2*np.pi
                elif err_pitch[k] < -np.pi:
                    err_pitch[k] = err_pitch[k]+2*np.pi
            # RMSE roll and pitch
            RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
            RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

            sassari_roll_RMSE_dl = np.vstack([sassari_roll_RMSE_dl, RMSE_roll])
            sassari_pitch_RMSE_dl = np.vstack(
                [sassari_pitch_RMSE_dl, RMSE_pitch])
            df_roll_dl = pd.concat(
                [df_roll_dl, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
            df_pitch_dl = pd.concat(
                [df_pitch_dl, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)
            # RIANN
            attitude_riann = riann.predict(acc_main, gyro_main, fs_main)
            attitude_riann = Att_q(attitude_riann)
            Quat_riann = tf.Variable(attitude_riann, dtype=tf.float64)
            roll_riann, pitch_riann, yaw_riann = quat2eul(attitude_riann)
            # tot error
            tot_err = quat_error_angle(Quat_main, Quat_riann)
            tot = np.mean(np.sqrt(tot_err**2))
            sassari_tot_RMSE_riann = np.vstack([sassari_tot_RMSE_riann, tot])
            df_tot_riann = pd.concat(
                [df_tot_riann, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)
            # roll error
            err_roll = roll_main - roll_riann
            # pitch error
            err_pitch = pitch_main - pitch_riann
            # limit the max error to 180
            for k in range(len(err_roll)):
                if err_roll[k] > np.pi:
                    err_roll[k] = err_roll[k]-2*np.pi
                elif err_roll[k] < -np.pi:
                    err_roll[k] = err_roll[k]+2*np.pi
                if err_pitch[k] > np.pi:
                    err_pitch[k] = err_pitch[k]-2*np.pi
                elif err_pitch[k] < -np.pi:
                    err_pitch[k] = err_pitch[k]+2*np.pi
            # RMSE roll and pitch
            RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
            RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

            sassari_roll_RMSE_riann = np.vstack(
                [sassari_roll_RMSE_riann, RMSE_roll])
            sassari_pitch_RMSE_riann = np.vstack(
                [sassari_pitch_RMSE_riann, RMSE_pitch])
            df_roll_riann = pd.concat(
                [df_roll_riann, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
            df_pitch_riann = pd.concat(
                [df_pitch_riann, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)

            # EKF
            attitude_ekf = EKF(acc=acc_main, gyr=gyro_main,
                               frequency=fs_main, frame='NED').Q
            attitude_ekf = Att_q(attitude_ekf)
            roll_ekf, pitch_ekf, yaw_ekf = quat2eul(attitude_ekf)
            roll_ekf = roll_ekf - np.pi
            attitude_ekf = eul2quat(yaw_ekf, pitch_ekf, roll_ekf)
            Quat_ekf = tf.Variable(attitude_ekf, dtype=tf.float64)
            # tot error
            tot_err = quat_error_angle(Quat_main, Quat_ekf)
            tot = np.mean(np.sqrt(tot_err**2))
            sassari_tot_RMSE_ekf = np.vstack([sassari_tot_RMSE_ekf, tot])
            df_tot_ekf = pd.concat(
                [df_tot_ekf, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)
            # roll error
            err_roll = roll_main - roll_ekf
            # pitch error
            err_pitch = pitch_main - pitch_ekf
            # limit the max error to 180
            for k in range(len(err_roll)):
                if err_roll[k] > np.pi:
                    err_roll[k] = err_roll[k]-2*np.pi
                elif err_roll[k] < -np.pi:
                    err_roll[k] = err_roll[k]+2*np.pi
                if err_pitch[k] > np.pi:
                    err_pitch[k] = err_pitch[k]-2*np.pi
                elif err_pitch[k] < -np.pi:
                    err_pitch[k] = err_pitch[k]+2*np.pi
            # RMSE roll and pitch
            RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
            RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

            sassari_roll_RMSE_ekf = np.vstack(
                [sassari_roll_RMSE_ekf, RMSE_roll])
            sassari_pitch_RMSE_ekf = np.vstack(
                [sassari_pitch_RMSE_ekf, RMSE_pitch])
            df_roll_ekf = pd.concat(
                [df_roll_ekf, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
            df_pitch_ekf = pd.concat(
                [df_pitch_ekf, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)
            # Mahony
            attitude_mahony = Mahony(
                acc=acc_main, gyr=gyro_main, frequency=fs_main).Q
            attitude_mahony = Att_q(attitude_mahony)
            roll_mahony, pitch_mahony, yaw_mahony = quat2eul(attitude_mahony)
            Quat_mahony = tf.Variable(attitude_mahony, dtype=tf.float64)
            # tot error
            tot_err = quat_error_angle(Quat_main, Quat_mahony)
            tot = np.mean(np.sqrt(tot_err**2))
            sassari_tot_RMSE_mahony = np.vstack([sassari_tot_RMSE_mahony, tot])
            df_tot_mahony = pd.concat(
                [df_tot_mahony, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)
            # roll error
            err_roll = roll_main - roll_mahony
            # pitch error
            err_pitch = pitch_main - pitch_mahony
            # limit the max error to 180
            for k in range(len(err_roll)):
                if err_roll[k] > np.pi:
                    err_roll[k] = err_roll[k]-2*np.pi
                elif err_roll[k] < -np.pi:
                    err_roll[k] = err_roll[k]+2*np.pi
                if err_pitch[k] > np.pi:
                    err_pitch[k] = err_pitch[k]-2*np.pi
                elif err_pitch[k] < -np.pi:
                    err_pitch[k] = err_pitch[k]+2*np.pi
            # RMSE roll and pitch
            RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
            RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

            sassari_roll_RMSE_mahony = np.vstack(
                [sassari_roll_RMSE_mahony, RMSE_roll])
            sassari_pitch_RMSE_mahony = np.vstack(
                [sassari_pitch_RMSE_mahony, RMSE_pitch])
            df_roll_mahony = pd.concat(
                [df_roll_mahony, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
            df_pitch_mahony = pd.concat(
                [df_pitch_mahony, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)
            # Madgwick
            attitude_madgwick = Madgwick(
                acc=acc_main, gyr=gyro_main, frequency=fs_main).Q
            attitude_madgwick = Att_q(attitude_madgwick)
            roll_madgwick, pitch_madgwick, yaw_madgwick = quat2eul(
                attitude_madgwick)
            Quat_madgwick = tf.Variable(attitude_madgwick, dtype=tf.float64)
            # tot error
            tot_err = quat_error_angle(Quat_main, Quat_madgwick)
            tot = np.mean(np.sqrt(tot_err**2))
            sassari_tot_RMSE_madgwick = np.vstack(
                [sassari_tot_RMSE_madgwick, tot])
            df_tot_madgwick = pd.concat(
                [df_tot_madgwick, pd.DataFrame(tot_err, columns=[cols[i-1]])], axis=1)
            # roll error
            err_roll = roll_main - roll_madgwick
            # pitch error
            err_pitch = pitch_main - pitch_madgwick
            # limit the max error to 180
            for k in range(len(err_roll)):
                if err_roll[k] > np.pi:
                    err_roll[k] = err_roll[k]-2*np.pi
                elif err_roll[k] < -np.pi:
                    err_roll[k] = err_roll[k]+2*np.pi
                if err_pitch[k] > np.pi:
                    err_pitch[k] = err_pitch[k]-2*np.pi
                elif err_pitch[k] < -np.pi:
                    err_pitch[k] = err_pitch[k]+2*np.pi
            # RMSE roll and pitch
            RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
            RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

            sassari_roll_RMSE_madgwick = np.vstack(
                [sassari_roll_RMSE_madgwick, RMSE_roll])
            sassari_pitch_RMSE_madgwick = np.vstack(
                [sassari_pitch_RMSE_madgwick, RMSE_pitch])
            df_roll_madgwick = pd.concat(
                [df_roll_madgwick, pd.DataFrame(err_roll, columns=[cols[i-1]])], axis=1)
            df_pitch_madgwick = pd.concat(
                [df_pitch_madgwick, pd.DataFrame(err_pitch, columns=[cols[i-1]])], axis=1)

    # The saved dataframe would be consists of RMSE_roll, RMSE_Pitch, and Total_Rotation_Error for each trial and caluclted by the DL model, EKF, Madgwick, Mahony, and RIANN
    #                           Total_Rotation_Error             |                  RMSE_roll                   |                       RMSE_pitch              |
    # Trial No,     DL Model | EKF | Madgwick | Mahony | RIANN  ||| DL Model | EKF | Madgwick | Mahony | RIANN ||| DL Model | EKF | Madgwick | Mahony | RIANN  |||
    header = pd.MultiIndex.from_product([['Total_Rotation_Error', 'RMSE_roll', 'RMSE_pitch'], [
                                        'DL Model', 'EKF', 'Madgwick', 'Mahony', 'RIANN']], names=['', 'Trial No'])
    df = pd.DataFrame(np.hstack([sassari_tot_RMSE_dl,
                                 sassari_tot_RMSE_ekf,
                                 sassari_tot_RMSE_madgwick,
                                 sassari_tot_RMSE_mahony,
                                 sassari_tot_RMSE_riann,
                                sassari_roll_RMSE_dl,
                                sassari_roll_RMSE_ekf,
                                sassari_roll_RMSE_madgwick,
                                sassari_roll_RMSE_mahony,
                                sassari_roll_RMSE_riann,
                                sassari_pitch_RMSE_dl,
                                sassari_pitch_RMSE_ekf,
                                sassari_pitch_RMSE_madgwick,
                                sassari_pitch_RMSE_mahony,
                                sassari_pitch_RMSE_riann]),
                      columns=header,
                      index=files)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv('Sassari_RMSE.csv' % (model_name))
    # save dfs error
    df_tot_dl.to_csv('./Error/total/%s_tot_dl.csv' % (model_name, name))
    df_tot_ekf.to_csv('./Error/total/%s_tot_ekf.csv' % (model_name, name))
    df_tot_mahony.to_csv('./Error/total/%s_tot_mahony.csv' %
                         (model_name, name))
    df_tot_madgwick.to_csv(
        './Error/total/%s_tot_madgwick.csv' % (model_name, name))
    df_tot_riann.to_csv('./Error/total/%s_tot_riann.csv' % (model_name, name))
    df_roll_dl.to_csv('./Error/roll/%s_roll_dl.csv' % (model_name, name))
    df_roll_ekf.to_csv('./Error/roll/%s_roll_ekf.csv' % (model_name, name))
    df_roll_mahony.to_csv('./Error/roll/%s_roll_mahony.csv' %
                          (model_name, name))
    df_roll_madgwick.to_csv(
        './Error/roll/%s_roll_madgwick.csv' % (model_name, name))
    df_roll_riann.to_csv('./Error/roll/%s_roll_riann.csv' % (model_name, name))
    df_pitch_dl.to_csv('./Error/pitch/%s_pitch_dl.csv' % (model_name, name))
    df_pitch_ekf.to_csv('./Error/pitch/%s_pitch_ekf.csv' % (model_name, name))
    df_pitch_mahony.to_csv(
        './Error/pitch/%s_pitch_mahony.csv' % (model_name, name))
    df_pitch_madgwick.to_csv(
        './Error/pitch/%s_pitch_madgwick.csv' % (model_name, name))
    df_pitch_riann.to_csv('./Error/pitch/%s_pitch_riann.csv' %
                          (model_name, name))
    return


def df_ridi(window_size, stride):
    name = (sys._getframe().f_code.co_name).replace("df_", "")
    # tot
    ridi_tot_RMSE_dl = np.empty((0, 1))
    ridi_tot_RMSE_ekf = np.empty((0, 1))
    ridi_tot_RMSE_madgwick = np.empty((0, 1))
    ridi_tot_RMSE_mahony = np.empty((0, 1))
    ridi_tot_RMSE_riann = np.empty((0, 1))
    ridi_roll_RMSE_dl = np.empty((0, 1))
    ridi_roll_RMSE_ekf = np.empty((0, 1))
    ridi_roll_RMSE_madgwick = np.empty((0, 1))
    ridi_roll_RMSE_mahony = np.empty((0, 1))
    ridi_roll_RMSE_riann = np.empty((0, 1))
    ridi_pitch_RMSE_dl = np.empty((0, 1))
    ridi_pitch_RMSE_ekf = np.empty((0, 1))
    ridi_pitch_RMSE_madgwick = np.empty((0, 1))
    ridi_pitch_RMSE_mahony = np.empty((0, 1))
    ridi_pitch_RMSE_riann = np.empty((0, 1))
    dir_list = RIDI_path()
    cols = dir_list
    df_tot_dl = pd.DataFrame()
    df_roll_dl = pd.DataFrame()
    df_pitch_dl = pd.DataFrame()
    df_tot_ekf = pd.DataFrame()
    df_roll_ekf = pd.DataFrame()
    df_pitch_ekf = pd.DataFrame()
    df_tot_madgwick = pd.DataFrame()
    df_roll_madgwick = pd.DataFrame()
    df_pitch_madgwick = pd.DataFrame()
    df_tot_mahony = pd.DataFrame()
    df_roll_mahony = pd.DataFrame()
    df_pitch_mahony = pd.DataFrame()
    df_tot_riann = pd.DataFrame()
    df_roll_riann = pd.DataFrame()
    df_pitch_riann = pd.DataFrame()
    for i in range(len(dir_list)):
        # load the data
        acc_main, gyro_main, mag_main, quat_main, fs_main = RIDI_data(
            dir_list[i])
        quat_main = Att_q(quat_main)
        Quat_main = tf.Variable(quat_main, dtype=tf.float64)
        roll_main, pitch_main, yaw_main = quat2eul(quat_main)
        # ref data for dl model
        [gyro_ref, acc_ref, fs_ref], [quat_ref] = load_dataset_A_G_Fs(
            gyro_main, acc_main, quat_main, window_size, stride, fs_main)
        quat_ref = Att_q(quat_ref)
        Quat_ref = tf.Variable(quat_ref, dtype=tf.float64)
        roll_ref, pitch_ref, yaw_ref = quat2eul(quat_ref)

        # DL Model
        quat_dl = model.predict(
            [acc_ref, gyro_ref, fs_ref], batch_size=500, verbose=1)
        quat_dl = Att_q(quat_dl)
        roll_pred, pitch_pred, yaw_pred = quat2eul(quat_dl)
        Quat_dl = tf.Variable(quat_dl, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_ref, Quat_dl)
        tot = np.mean(np.sqrt(tot_err**2))
        ridi_tot_RMSE_dl = np.vstack([ridi_tot_RMSE_dl, tot])
        # roll error
        err_roll = roll_ref - roll_pred
        # pitch error
        err_pitch = pitch_ref - pitch_pred
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        ridi_roll_RMSE_dl = np.vstack([ridi_roll_RMSE_dl, RMSE_roll])
        ridi_pitch_RMSE_dl = np.vstack([ridi_pitch_RMSE_dl, RMSE_pitch])

        # RIANN
        attitude_riann = riann.predict(acc_main, gyro_main, fs_main)
        attitude_riann = Att_q(attitude_riann)
        Quat_riann = tf.Variable(attitude_riann, dtype=tf.float64)
        roll_riann, pitch_riann, yaw_riann = quat2eul(attitude_riann)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_riann)
        tot = np.mean(np.sqrt(tot_err**2))
        ridi_tot_RMSE_riann = np.vstack([ridi_tot_RMSE_riann, tot])
        # roll error
        err_roll = roll_main - roll_riann
        # pitch error
        err_pitch = pitch_main - pitch_riann
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        ridi_roll_RMSE_riann = np.vstack([ridi_roll_RMSE_riann, RMSE_roll])
        ridi_pitch_RMSE_riann = np.vstack([ridi_pitch_RMSE_riann, RMSE_pitch])

        # EKF
        attitude_ekf = EKF(acc=acc_main, gyr=gyro_main,
                           frequency=fs_main, frame='NED').Q
        attitude_ekf = Att_q(attitude_ekf)
        roll_ekf, pitch_ekf, yaw_ekf = quat2eul(attitude_ekf)
        roll_ekf = roll_ekf - np.pi
        attitude_ekf = eul2quat(yaw_ekf, pitch_ekf, roll_ekf)
        Quat_ekf = tf.Variable(attitude_ekf, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_ekf)
        tot = np.mean(np.sqrt(tot_err**2))
        ridi_tot_RMSE_ekf = np.vstack([ridi_tot_RMSE_ekf, tot])
        # roll error
        err_roll = roll_main - roll_ekf
        # pitch error
        err_pitch = pitch_main - pitch_ekf
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        ridi_roll_RMSE_ekf = np.vstack([ridi_roll_RMSE_ekf, RMSE_roll])
        ridi_pitch_RMSE_ekf = np.vstack([ridi_pitch_RMSE_ekf, RMSE_pitch])

        # Mahony
        attitude_mahony = Mahony(
            acc=acc_main, gyr=gyro_main, frequency=fs_main).Q
        attitude_mahony = Att_q(attitude_mahony)
        roll_mahony, pitch_mahony, yaw_mahony = quat2eul(attitude_mahony)
        Quat_mahony = tf.Variable(attitude_mahony, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_mahony)
        tot = np.mean(np.sqrt(tot_err**2))
        ridi_tot_RMSE_mahony = np.vstack([ridi_tot_RMSE_mahony, tot])
        # roll error
        err_roll = roll_main - roll_mahony
        # pitch error
        err_pitch = pitch_main - pitch_mahony
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        ridi_roll_RMSE_mahony = np.vstack([ridi_roll_RMSE_mahony, RMSE_roll])
        ridi_pitch_RMSE_mahony = np.vstack(
            [ridi_pitch_RMSE_mahony, RMSE_pitch])
        # Madgwick
        attitude_madgwick = Madgwick(
            acc=acc_main, gyr=gyro_main, frequency=fs_main).Q
        attitude_madgwick = Att_q(attitude_madgwick)
        roll_madgwick, pitch_madgwick, yaw_madgwick = quat2eul(
            attitude_madgwick)
        Quat_madgwick = tf.Variable(attitude_madgwick, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_madgwick)
        tot = np.mean(np.sqrt(tot_err**2))
        ridi_tot_RMSE_madgwick = np.vstack([ridi_tot_RMSE_madgwick, tot])
        # roll error
        err_roll = roll_main - roll_madgwick
        # pitch error
        err_pitch = pitch_main - pitch_madgwick
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        ridi_roll_RMSE_madgwick = np.vstack(
            [ridi_roll_RMSE_madgwick, RMSE_roll])
        ridi_pitch_RMSE_madgwick = np.vstack(
            [ridi_pitch_RMSE_madgwick, RMSE_pitch])

    # The saved dataframe would be consists of RMSE_roll, RMSE_Pitch, and Total_Rotation_Error for each trial and caluclted by the DL model, EKF, Madgwick, Mahony, and RIANN
    #                           Total_Rotation_Error             |                  RMSE_roll                   |                       RMSE_pitch              |
    # Trial No,     DL Model | EKF | Madgwick | Mahony | RIANN  ||| DL Model | EKF | Madgwick | Mahony | RIANN ||| DL Model | EKF | Madgwick | Mahony | RIANN  |||
    header = pd.MultiIndex.from_product([['Total_Rotation_Error', 'RMSE_roll', 'RMSE_pitch'], [
                                        'DL Model', 'EKF', 'Madgwick', 'Mahony', 'RIANN']], names=['', 'Trial No'])
    df = pd.DataFrame(np.hstack([ridi_tot_RMSE_dl,
                                 ridi_tot_RMSE_ekf,
                                 ridi_tot_RMSE_madgwick,
                                 ridi_tot_RMSE_mahony,
                                 ridi_tot_RMSE_riann,
                                 ridi_roll_RMSE_dl,
                                 ridi_roll_RMSE_ekf,
                                 ridi_roll_RMSE_madgwick,
                                 ridi_roll_RMSE_mahony,
                                 ridi_roll_RMSE_riann,
                                 ridi_pitch_RMSE_dl,
                                 ridi_pitch_RMSE_ekf,
                                 ridi_pitch_RMSE_madgwick,
                                 ridi_pitch_RMSE_mahony,
                                 ridi_pitch_RMSE_riann]),
                      columns=header,
                      index=dir_list)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv('RIDI_RMSE.csv' % (model_name))
    return


def df_RoNIN(window_size, stride):
    # create empty array to store the RMSE values
    ronin_tot_RMSE_dl = np.empty((0, 1))
    ronin_tot_RMSE_ekf = np.empty((0, 1))
    ronin_tot_RMSE_madgwick = np.empty((0, 1))
    ronin_tot_RMSE_mahony = np.empty((0, 1))
    ronin_tot_RMSE_riann = np.empty((0, 1))
    ronin_roll_RMSE_dl = np.empty((0, 1))
    ronin_roll_RMSE_ekf = np.empty((0, 1))
    ronin_roll_RMSE_madgwick = np.empty((0, 1))
    ronin_roll_RMSE_mahony = np.empty((0, 1))
    ronin_roll_RMSE_riann = np.empty((0, 1))
    ronin_pitch_RMSE_dl = np.empty((0, 1))
    ronin_pitch_RMSE_ekf = np.empty((0, 1))
    ronin_pitch_RMSE_madgwick = np.empty((0, 1))
    ronin_pitch_RMSE_mahony = np.empty((0, 1))
    ronin_pitch_RMSE_riann = np.empty((0, 1))
    # load the data
    df1, df2, df3, df0 = RoNIN_path()
    file_path = np.empty((0, 1))
    file_name = np.empty((0, 1))
    for i in range(0, 4):
        df = locals()['df{}'.format(i)]
        file_path = np.vstack([file_path, df.values])
        df[df.columns.values[0]] = df[df.columns.values[0]].str.replace(
            r'/home/arman/Neural Network in Python/RNN/Dataset/Ronin/', '')
        df[df.columns.values[0]] = df[df.columns.values[0]
                                      ].str.replace(r'.csv', '')
        df[df.columns.values[0]] = df[df.columns.values[0]
                                      ].str.replace(r'/syn', '')
        file_name = np.vstack([file_name, df.values])

    for i in range(len(file_name)):
        # load the data
        acc_main, gyro_main, mag_main, quat_main, fs_main = RoNIN_data(
            file_path[i][0])
        quat_main = Att_q(quat_main)
        Quat_main = tf.Variable(quat_main, dtype=tf.float64)
        roll_main, pitch_main, yaw_main = quat2eul(quat_main)
        # ref data for dl model
        [gyro_ref, acc_ref, fs_ref], [quat_ref] = load_dataset_A_G_Fs(
            gyro_main, acc_main, quat_main, window_size, stride, fs_main)
        quat_ref = Att_q(quat_ref)
        Quat_ref = tf.Variable(quat_ref, dtype=tf.float64)
        roll_ref, pitch_ref, yaw_ref = quat2eul(quat_ref)

        # DL Model
        quat_dl = model.predict(
            [acc_ref, gyro_ref, fs_ref], batch_size=500, verbose=1)
        quat_dl = Att_q(quat_dl)
        roll_pred, pitch_pred, yaw_pred = quat2eul(quat_dl)
        Quat_dl = tf.Variable(quat_dl, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_ref, Quat_dl)
        tot = np.mean(np.sqrt(tot_err**2))
        ronin_tot_RMSE_dl = np.vstack([ronin_tot_RMSE_dl, tot])
        # roll error
        err_roll = roll_ref - roll_pred
        # pitch error
        err_pitch = pitch_ref - pitch_pred
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        ronin_roll_RMSE_dl = np.vstack([ronin_roll_RMSE_dl, RMSE_roll])
        ronin_pitch_RMSE_dl = np.vstack([ronin_pitch_RMSE_dl, RMSE_pitch])
        # RIANN
        attitude_riann = riann.predict(acc_main, gyro_main, fs_main)
        attitude_riann = Att_q(attitude_riann)
        Quat_riann = tf.Variable(attitude_riann, dtype=tf.float64)
        roll_riann, pitch_riann, yaw_riann = quat2eul(attitude_riann)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_riann)
        tot = np.mean(np.sqrt(tot_err**2))
        ronin_tot_RMSE_riann = np.vstack([ronin_tot_RMSE_riann, tot])
        # roll error
        err_roll = roll_main - roll_riann
        # pitch error
        err_pitch = pitch_main - pitch_riann
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        ronin_roll_RMSE_riann = np.vstack([ronin_roll_RMSE_riann, RMSE_roll])
        ronin_pitch_RMSE_riann = np.vstack(
            [ronin_pitch_RMSE_riann, RMSE_pitch])

        # EKF
        attitude_ekf = EKF(acc=acc_main, gyr=gyro_main,
                           frequency=fs_main, frame='NED').Q
        attitude_ekf = Att_q(attitude_ekf)
        roll_ekf, pitch_ekf, yaw_ekf = quat2eul(attitude_ekf)
        roll_ekf = roll_ekf - np.pi
        attitude_ekf = eul2quat(yaw_ekf, pitch_ekf, roll_ekf)
        Quat_ekf = tf.Variable(attitude_ekf, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_ekf)
        tot = np.mean(np.sqrt(tot_err**2))
        ronin_tot_RMSE_ekf = np.vstack([ronin_tot_RMSE_ekf, tot])
        # roll error
        err_roll = roll_main - roll_ekf
        # pitch error
        err_pitch = pitch_main - pitch_ekf
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        ronin_roll_RMSE_ekf = np.vstack([ronin_roll_RMSE_ekf, RMSE_roll])
        ronin_pitch_RMSE_ekf = np.vstack([ronin_pitch_RMSE_ekf, RMSE_pitch])

        # Mahony
        attitude_mahony = Mahony(
            acc=acc_main, gyr=gyro_main, frequency=fs_main).Q
        attitude_mahony = Att_q(attitude_mahony)
        roll_mahony, pitch_mahony, yaw_mahony = quat2eul(attitude_mahony)
        Quat_mahony = tf.Variable(attitude_mahony, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_mahony)
        tot = np.mean(np.sqrt(tot_err**2))
        ronin_tot_RMSE_mahony = np.vstack([ronin_tot_RMSE_mahony, tot])
        # roll error
        err_roll = roll_main - roll_mahony
        # pitch error
        err_pitch = pitch_main - pitch_mahony
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        ronin_roll_RMSE_mahony = np.vstack([ronin_roll_RMSE_mahony, RMSE_roll])
        ronin_pitch_RMSE_mahony = np.vstack(
            [ronin_pitch_RMSE_mahony, RMSE_pitch])

        # Madgwick
        attitude_madgwick = Madgwick(
            acc=acc_main, gyr=gyro_main, frequency=fs_main).Q
        attitude_madgwick = Att_q(attitude_madgwick)
        roll_madgwick, pitch_madgwick, yaw_madgwick = quat2eul(
            attitude_madgwick)
        Quat_madgwick = tf.Variable(attitude_madgwick, dtype=tf.float64)
        # tot error
        tot_err = quat_error_angle(Quat_main, Quat_madgwick)
        tot = np.mean(np.sqrt(tot_err**2))
        ronin_tot_RMSE_madgwick = np.vstack([ronin_tot_RMSE_madgwick, tot])
        # roll error
        err_roll = roll_main - roll_madgwick
        # pitch error
        err_pitch = pitch_main - pitch_madgwick
        # limit the max error to 180
        for k in range(len(err_roll)):
            if err_roll[k] > np.pi:
                err_roll[k] = err_roll[k]-2*np.pi
            elif err_roll[k] < -np.pi:
                err_roll[k] = err_roll[k]+2*np.pi
            if err_pitch[k] > np.pi:
                err_pitch[k] = err_pitch[k]-2*np.pi
            elif err_pitch[k] < -np.pi:
                err_pitch[k] = err_pitch[k]+2*np.pi
        # RMSE roll and pitch
        RMSE_roll = np.sqrt(np.mean(err_roll**2)) * 180 / np.pi
        RMSE_pitch = np.sqrt(np.mean(err_pitch**2)) * 180 / np.pi

        ronin_roll_RMSE_madgwick = np.vstack(
            [ronin_roll_RMSE_madgwick, RMSE_roll])
        ronin_pitch_RMSE_madgwick = np.vstack(
            [ronin_pitch_RMSE_madgwick, RMSE_pitch])

    # The saved dataframe would be consists of RMSE_roll, RMSE_Pitch, and Total_Rotation_Error for each trial and caluclted by the DL model, EKF, Madgwick, Mahony, and RIANN
    #                           Total_Rotation_Error             |                  RMSE_roll                   |                       RMSE_pitch              |
    # Trial No,     DL Model | EKF | Madgwick | Mahony | RIANN  ||| DL Model | EKF | Madgwick | Mahony | RIANN ||| DL Model | EKF | Madgwick | Mahony | RIANN  |||
    header = pd.MultiIndex.from_product([['Total_Rotation_Error', 'RMSE_roll', 'RMSE_pitch'], [
                                        'DL Model', 'EKF', 'Madgwick', 'Mahony', 'RIANN']], names=['', 'Trial No'])
    df = pd.DataFrame(np.hstack([ronin_tot_RMSE_dl,
                                 ronin_tot_RMSE_ekf,
                                 ronin_tot_RMSE_madgwick,
                                 ronin_tot_RMSE_mahony,
                                 ronin_tot_RMSE_riann,
                                 ronin_roll_RMSE_dl,
                                 ronin_roll_RMSE_ekf,
                                 ronin_roll_RMSE_madgwick,
                                 ronin_roll_RMSE_mahony,
                                 ronin_roll_RMSE_riann,
                                 ronin_pitch_RMSE_dl,
                                 ronin_pitch_RMSE_ekf,
                                 ronin_pitch_RMSE_madgwick,
                                 ronin_pitch_RMSE_mahony,
                                 ronin_pitch_RMSE_riann]),
                      columns=header,
                      index=file_name[0:2])
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv('ronin_RMSE.csv' % (model_name))


def main():
    print('BROAD data')
    df_broad(window_size, stride)
    df_oxiod(window_size, stride)
    #df_repoIMU_TStick(window_size, stride)
    prtin('Ridi data')
    df_ridi(window_size, stride)
    print('Ronin data')
    df_RoNIN(window_size, stride)
    prtin('sassari data')
    df_sassari(window_size, stride)


if __name__ == '__main__':
    main()

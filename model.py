
from keras.layers import Bidirectional, GaussianNoise, Activation, LSTM, ConvLSTM1D, CuDNNLSTM, Dropout, Dense, Input, Layer, Conv1D, MaxPooling1D, concatenate, GRU, CuDNNGRU, GlobalAveragePooling1D, Flatten, ZeroPadding1D
from keras.callbacks import ModelCheckpoint, TensorBoard
# import BatchNormalization from keras
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.losses import mean_absolute_error
from keras.models import Sequential, Model
from tensorflow.keras import backend as k
from keras.initializers import Constant
from keras.models import load_model
from keras import backend as K
from tcn import TCN
from fileinput import filename
from numba import cuda
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import random as rn
import numpy as np
import argparse
import math
import time
import os
from symbol import import_from
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'


def Quat_error_angle(y_true, y_pred):
    """
    The function takes in two quaternions, normalizes the predicted quaternion, and then calculates the
    angle between the two quaternions

    :param y_true: the true quaternion
    :param y_pred: the predicted quaternion
    :return: The angle between the two quaternions.
    """
    y_pred = tf.linalg.normalize(y_pred, ord='euclidean', axis=1)[0]
    w0, x0, y0, z0 = tf.split(
        (tf.multiply(y_pred, [1., -1, -1, -1]),), num_or_size_splits=4, axis=-1)
    w1, x1, y1, z1 = tf.split(y_true, num_or_size_splits=4, axis=-1)
    w = w0*w1 - x0*x1 - y0*y1 - z0*z1
    angle = (tf.abs(2 * tf.math.acos(tf.math.sqrt(tf.math.square(w))))) * 180/np.pi
    return tf.clip_by_value(angle, -1e3, 1e3)


def Quat_mult(y_true, y_pred):
    """
    The function takes in two quaternions, normalizes the first one, and then multiplies the two
    quaternions together.

    The function returns the absolute value of the vector part of the resulting quaternion.

    The reason for this is that the vector part of the quaternion is the axis of rotation, and the
    absolute value of the vector part is the angle of rotation.

    The reason for normalizing the first quaternion is that the first quaternion is the predicted
    quaternion, and the predicted quaternion is not always normalized.

    The reason for returning the absolute value of the vector part of the resulting quaternion is that
    the angle of rotation is always positive.

    The reason for returning the vector part of the resulting quaternion is that the axis of rotation is
    always a vector.

    :param y_true: the ground truth quaternion
    :param y_pred: the predicted quaternion
    :return: The absolute value of the quaternion multiplication of the predicted and true quaternions.
    """
    y_pred = tf.linalg.normalize(y_pred, ord='euclidean', axis=1)[0]
    w0, x0, y0, z0 = tf.split(
        (tf.multiply(y_pred, [1., -1, -1, -1]),), num_or_size_splits=4, axis=-1)
    w1, x1, y1, z1 = tf.split(y_true, num_or_size_splits=4, axis=-1)
    # w = w0*w1 - x0*x1 - y0*y1 - z0*z1
    x = w0*x1 + x0*w1 + y0*z1 - z0*y1
    y = w0*y1 - x0*z1 + y0*w1 + z0*x1
    z = w0*z1 + x0*y1 - y0*x1 + z0*w1
    return (tf.abs(tf.multiply(2.0, tf.concat(values=[x, y, z], axis=-1))))


def quaternion_mean_multiplicative_error(y_true, y_pred):
    return tf.reduce_mean(Quat_mult(y_true, y_pred))

# Custom loss layer


class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        # def __init__(self, nb_outputs=3, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        return config

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(
            ys_pred) == self.nb_outputs
        loss = 0

        # for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
        #    precision = K.exp(-log_var[0])
        #    loss += K.sum(precision * (y_true - y_pred)**2., -1) + log_var[0]

        precision = K.exp(-self.log_vars[0][0])
        loss += precision * \
            mean_absolute_error(ys_true[0], ys_pred[0]) + self.log_vars[0][0]
        precision = K.exp(-self.log_vars[1][0])
        loss += precision * \
            quaternion_mean_multiplicative_error(
                ys_true[1], ys_pred[1]) + self.log_vars[1][0]
        # loss += precision * quaternion_phi_4_error(ys_true[1], ys_pred[1]) + self.log_vars[1][0]

        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)


def create_train_model(pred_model):
    # inp = Input(shape=(window_size, 6), name='inp')
    # y1_pred, y2_pred = pred_model(inp)
    x1 = Input((9, 1), name='x1')
    y1_pred, y2_pred = pred_model(x1)

    y1_true = Input(shape=(3,), name='y1_true')
    y2_true = Input(shape=(4,), name='y2_true')

    out = CustomMultiLossLayer(nb_outputs=2)(
        [y1_true, y2_true, y1_pred, y2_pred])
    # train_model = Model([inp, y1_true, y2_true], out)
    train_model = Model([x1, y1_true, y2_true], out)
    train_model.summary()
    return train_model


def Train_Model():
    # LSTM Model
    # input size is 7 (IMU data + frequency/sampling rate) and output size is 4 quaternion (roll and pitch)
    inp = keras.Input(shape=(9, 1))
    x1 = LSTM(128, return_sequences=True)(inp)
    d1 = Dropout(0.2)(x1)
    x2 = LSTM(128, return_sequences=True)(d1)
    d2 = Dropout(0.2)(x2)
    y1 = LSTM(128)(d2)
    y1d = Dropout(0.2)(y1)
    y2 = LSTM(128)(d2)
    y2d = Dropout(0.2)(y2)
    pose = Dense(3, activation='linear')(y1d)
    ori = Dense(4, activation='linear')(y2d)
    model = keras.Model(inputs=inp, outputs=[pose, ori])
    model.summary()
    train_model = create_train_model(model)
    train_model.summary()
    return train_model


# Custom loss layer
class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        # def __init__(self, nb_outputs=3, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(
            ys_pred) == self.nb_outputs
        loss = 0

        # for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
        #    precision = K.exp(-log_var[0])
        #    loss += K.sum(precision * (y_true - y_pred)**2., -1) + log_var[0]

        precision = K.exp(-self.log_vars[0][0])
        loss += precision * \
            mean_absolute_error(ys_true[0], ys_pred[0]) + self.log_vars[0][0]
        precision = K.exp(-self.log_vars[1][0])
        loss += precision * \
            quaternion_mean_multiplicative_error(
                ys_true[1], ys_pred[1]) + self.log_vars[1][0]
        # loss += precision * quaternion_phi_4_error(ys_true[1], ys_pred[1]) + self.log_vars[1][0]

        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)


def create_pred_model_6d_quat(window_size=200):
    # inp = Input((window_size, 6), name='inp')
    # lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    convA1 = Conv1D(128, 11)(x1)
    convA2 = Conv1D(128, 11)(convA1)
    poolA = MaxPooling1D(3)(convA2)
    convB1 = Conv1D(128, 11)(x2)
    convB2 = Conv1D(128, 11)(convB1)
    poolB = MaxPooling1D(3)(convB2)
    AB = concatenate([poolA, poolB])
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(AB)
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    drop2 = Dropout(0.25)(lstm2)
    y1_pred = Dense(3)(drop2)
    y2_pred = Dense(4)(drop2)

    # model = Model(inp, [y1_pred, y2_pred])
    model = Model([x1, x2], [y1_pred, y2_pred])

    model.summary()

    return model


def create_train_model_6d_quat(pred_model, window_size=200):
    # inp = Input(shape=(window_size, 6), name='inp')
    # y1_pred, y2_pred = pred_model(inp)
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    y1_pred, y2_pred = pred_model([x1, x2])
    y1_true = Input(shape=(3,), name='y1_true')
    y2_true = Input(shape=(4,), name='y2_true')
    out = CustomMultiLossLayer(nb_outputs=2)(
        [y1_true, y2_true, y1_pred, y2_pred])
    # train_model = Model([inp, y1_true, y2_true], out)
    train_model = Model([x1, x2, y1_true, y2_true], out)
    train_model.summary()
    return train_model


def create_pred_model_3d(window_size=200):
    # inp = Input((window_size, 6), name='inp')
    # lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    convA1 = Conv1D(128, 11)(x1)
    convA2 = Conv1D(128, 11)(convA1)
    poolA = MaxPooling1D(3)(convA2)
    convB1 = Conv1D(128, 11)(x2)
    convB2 = Conv1D(128, 11)(convB1)
    poolB = MaxPooling1D(3)(convB2)
    AB = concatenate([poolA, poolB])
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(AB)
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    drop2 = Dropout(0.25)(lstm2)
    y1_pred = Dense(1)(drop2)
    y2_pred = Dense(1)(drop2)
    y3_pred = Dense(1)(drop2)

    # model = Model(inp, [y1_pred, y2_pred, y3_pred])
    model = Model([x1, x2], [y1_pred, y2_pred, y3_pred])

    model.summary()

    return model


def create_train_model_3d(pred_model, window_size=200):
    # inp = Input(shape=(window_size, 6), name='inp')
    # y1_pred, y2_pred, y3_pred = pred_model(inp)
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    y1_pred, y2_pred, y3_pred = pred_model([x1, x2])
    y1_true = Input(shape=(1,), name='y1_true')
    y2_true = Input(shape=(1,), name='y2_true')
    y3_true = Input(shape=(1,), name='y3_true')
    out = CustomMultiLossLayer(nb_outputs=3)(
        [y1_true, y2_true, y3_true, y1_pred, y2_pred, y3_pred])
    # train_model = Model([inp, y1_true, y2_true, y3_true], out)
    train_model = Model([x1, x2, y1_true, y2_true, y3_true], out)
    train_model.summary()
    return train_model


def create_model_6d_rvec(window_size=200):
    input_gyro_acc = Input((window_size, 6))
    lstm1 = Bidirectional(
        CuDNNLSTM(128, return_sequences=True))(input_gyro_acc)
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    drop2 = Dropout(0.25)(lstm2)
    output_delta_rvec = Dense(3)(drop2)
    output_delta_tvec = Dense(3)(drop2)

    model = Model(inputs=input_gyro_acc, outputs=[
                  output_delta_rvec, output_delta_tvec])
    model.summary()
    model.compile(optimizer=Adam(0.0001), loss='mean_squared_error')

    return model


def create_model_6d_quat(window_size=200):
    input_gyro_acc = Input((window_size, 6))
    lstm1 = Bidirectional(
        CuDNNLSTM(128, return_sequences=True))(input_gyro_acc)
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    drop2 = Dropout(0.25)(lstm2)
    output_delta_p = Dense(3)(drop2)
    output_delta_q = Dense(4)(drop2)

    model = Model(inputs=input_gyro_acc, outputs=[
                  output_delta_p, output_delta_q])
    model.summary()
    # model.compile(optimizer = Adam(0.0001), loss = 'mean_squared_error')
    model.compile(optimizer=Adam(0.0001), loss=[
                  'mean_absolute_error', quaternion_mean_multiplicative_error])
    # model.compile(optimizer = Adam(0.0001), loss = ['mean_absolute_error', quaternion_phi_4_error])

    return model


def create_model_3d(window_size=200):
    input_gyro_acc = Input((window_size, 6))
    lstm1 = Bidirectional(
        CuDNNLSTM(128, return_sequences=True))(input_gyro_acc)
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    drop2 = Dropout(0.25)(lstm2)
    output_delta_l = Dense(1)(drop2)
    output_delta_theta = Dense(1)(drop2)
    output_delta_psi = Dense(1)(drop2)

    model = Model(inputs=input_gyro_acc, outputs=[
                  output_delta_l, output_delta_theta, output_delta_psi])
    model.summary()
    model.compile(optimizer=Adam(0.0001), loss='mean_squared_error')

    return model


def create_model_2d(window_size=200):
    input_gyro_acc = Input((window_size, 6))
    lstm1 = Bidirectional(
        CuDNNLSTM(128, return_sequences=True))(input_gyro_acc)
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    drop2 = Dropout(0.25)(lstm2)
    output_delta_l = Dense(1)(drop2)
    output_delta_psi = Dense(1)(drop2)
    model = Model(inputs=input_gyro_acc, outputs=[
                  output_delta_l, output_delta_psi])
    model.summary()
    model.compile(optimizer=Adam(0.0001), loss='mean_squared_error')

    return model


def attitude_model_LSTM_2d(window_size=200):
    input_gyro = Input((window_size, 3))
    input_acc = Input((window_size, 3))

    CNN_gyro1 = Conv1D(128, 11)(input_gyro)
    CNN_gyro2 = Conv1D(128, 11)(CNN_gyro1)
    pool_gyro = MaxPooling1D(2)(CNN_gyro2)

    CNN_acc1 = Conv1D(128, 11)(input_acc)
    CNN_acc2 = Conv1D(128, 11)(CNN_acc1)
    pool_acc = MaxPooling1D(2)(CNN_acc2)

    con_lstm = concatenate([pool_acc, pool_gyro])
    lstm1 = Bidirectional(CuDNNLSTM(128))(con_lstm)
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(con_lstm)
    drop2 = Dropout(0.25)(lstm2)
    output_delta_r = Dense(1, activation="linear")(drop1)
    output_delta_p = Dense(1, activation="linear")(drop2)
    model = Model(inputs=[input_gyro, input_acc], outputs=[
                  output_delta_r, output_delta_p])
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(
    ), loss='mean_squared_error')

    return model


def attitude_model_LSTM_2d_fs(window_size=200):
    input_gyro = Input((window_size, 4), name='gyro')
    input_acc = Input((window_size, 4), name='acc')

    CNN_gyro1 = Conv1D(128, 15, name='CNN_gyro1')(input_gyro)
    CNN_gyro2 = Conv1D(128, 15, name='CNN_gyro2')(CNN_gyro1)
    pool_gyro = MaxPooling1D(4, name='pool_gyro')(CNN_gyro2)

    CNN_acc1 = Conv1D(128, 15, name='CNN_acc1')(input_acc)
    CNN_acc2 = Conv1D(128, 15, name='CNN_acc2')(CNN_acc1)
    pool_acc = MaxPooling1D(4, name='pool_acc')(CNN_acc2)

    con_lstm = concatenate([pool_acc, pool_gyro], name='Concatenate_CNNs')
    lstm = Bidirectional(
        CuDNNLSTM(128, return_sequences=True, name='LSTM'))(con_lstm)
    lstm = Dropout(0.2, name='Dropout_LSTM')(lstm)

    lstm1r = Bidirectional(
        CuDNNLSTM(128, return_sequences=True, name='LSTM_Roll1'))(lstm)
    drop1r = Dropout(0.2, name='Dropout_Roll1')(lstm1r)
    lstm2r = Bidirectional(CuDNNLSTM(128, name='LSTM_Roll2'))(drop1r)
    drop2r = Dropout(0.2, name='Dropout_Roll2')(lstm2r)
    output_delta_r = Dense(1, activation="linear", name='Roll')(drop2r)

    lstm1p = Bidirectional(
        CuDNNLSTM(128, return_sequences=True, name='LSTM_Pitch1'))(lstm)
    drop1p = Dropout(0.2, name='Dropout_Pitch1')(lstm1p)
    lstm2p = Bidirectional(CuDNNLSTM(128, name='LSTM_Pitch2'))(drop1p)
    drop2p = Dropout(0.2, name='Dropout_Pitch2')(lstm2p)
    output_delta_p = Dense(1, activation="linear", name='Pitch')(drop2p)

    model = Model(inputs=[input_gyro, input_acc], outputs=[
                  output_delta_r, output_delta_p])
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(
    ), loss='mean_squared_error')

    return model


def attitude_model_LSTM_2d_fs2(window_size=200):
    input_gyro = Input((window_size, 4), name='gyro')
    input_acc = Input((window_size, 4), name='acc')

    CNN_gyro1 = Conv1D(128, 15, name='CNN_gyro1')(input_gyro)
    CNN_gyro2 = Conv1D(128, 15, name='CNN_gyro2')(CNN_gyro1)
    pool_gyro = MaxPooling1D(4, name='pool_gyro')(CNN_gyro2)

    CNN_acc1 = Conv1D(128, 15, name='CNN_acc1')(input_acc)
    CNN_acc2 = Conv1D(128, 15, name='CNN_acc2')(CNN_acc1)
    pool_acc = MaxPooling1D(4, name='pool_acc')(CNN_acc2)

    con_lstm = concatenate([pool_acc, pool_gyro], name='Concatenate_CNNs')

    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(con_lstm)
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    drop2 = Dropout(0.25)(lstm2)
    quat = Dense(4, activation="linear")(drop2)
    model = Model(inputs=[input_gyro, input_acc], outputs=[
                  quat])
    model.summary()

    return model


def attitude_model_LSTM_2d_fs3(window_size=200):
    input_gyro = Input((window_size, 4), name='gyro')
    input_acc = Input((window_size, 4), name='acc')

    CNN_gyro1 = Conv1D(128, 11, name='CNN_gyro1')(input_gyro)
    CNN_gyro2 = Conv1D(128, 11, name='CNN_gyro2')(CNN_gyro1)
    pool_gyro = MaxPooling1D(3, name='pool_gyro')(CNN_gyro2)

    CNN_acc1 = Conv1D(128, 11, name='CNN_acc1')(input_acc)
    CNN_acc2 = Conv1D(128, 11, name='CNN_acc2')(CNN_acc1)
    pool_acc = MaxPooling1D(3, name='pool_acc')(CNN_acc2)

    con_lstm = concatenate([pool_acc, pool_gyro], name='Concatenate_CNNs')

    lstm1 = Bidirectional(
        CuDNNLSTM(128, return_sequences=True, name='LSTM1'))(con_lstm)
    drop1 = Dropout(0.2, name='Dropout1')(lstm1)

    lstmw = Bidirectional(CuDNNLSTM(128, name='LSTMw'))(drop1)
    dropw = Dropout(0.2, name='Dropoutw')(lstmw)
    w = Dense(1, activation="linear", name='Qw')(dropw)

    lstmx = Bidirectional(CuDNNLSTM(128, name='LSTMx'))(drop1)
    dropx = Dropout(0.2, name='Dropoutx')(lstmx)
    x = Dense(1, activation="linear", name='Qx')(dropx)

    lstmy = Bidirectional(CuDNNLSTM(128, name='LSTMy'))(drop1)
    dropy = Dropout(0.2, name='Dropouty')(lstmy)
    y = Dense(1, activation="linear", name='Qy')(dropy)

    lstmz = Bidirectional(CuDNNLSTM(128, name='LSTMz'))(drop1)
    dropz = Dropout(0.2, name='Dropoutz')(lstmz)
    z = Dense(1, activation="linear", name='Qz')(dropz)

    model = Model(inputs=[input_gyro, input_acc], outputs=[
                  w, x, y, z])
    model.summary()

    return model


def attitude_model_TCN_10_Fs(window_size):
    input = Input((window_size, 7), name='Input')
    tcn1 = TCN(nb_filters=200, kernel_size=3, return_sequences=True, dilations=[
               1024], dropout_rate=0.2, name='TCN1')(input)
    tcn10 = TCN(nb_filters=200, kernel_size=3, dilations=[
                1024], dropout_rate=0.2, name='TCN10')(tcn1)
    quat = Dense(4, activation="linear", name='Quat')(tcn10)
    model = Model(inputs=[input], outputs=[quat])
    model.compile(
        optimizer=keras.optimizers.Adam(), loss="mse")
    model.summary()
    return model


def attitude_model_LSTM_2d_fs4(window_size=200):
    input = Input((window_size, 7), name='Input')

    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(input)
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(drop1)
    drop2 = Dropout(0.25)(lstm2)
    lstm3 = Bidirectional(CuDNNLSTM(128))(drop2)
    drop3 = Dropout(0.25)(lstm3)
    quat = Dense(4, activation="linear")(drop3)

    model = Model(inputs=[input], outputs=[quat])
    model.summary()
    return model
####################################################################################################################################################
################################################################### Window of data #################################################################
####################################################################################################################################################

################################################################ Grouped Input Models ##############################################################
###########################
############### Acc + Gyro
###########################
# CuDNNLSTM only

# Quaternion Based


def CuDNNLSTM_AG1(window_size=200):
    x = Input((window_size, 6), name='Input')
    LSTM1 = Bidirectional(
        CuDNNLSTM(128, return_sequences=True), name='BiLSTM1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(LSTM1)
    LSTM2 = Bidirectional(CuDNNLSTM(128), name='BiLSTM2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(LSTM2)
    quat = Dense(4, activation="linear", name='Quat')(drop2)
    model = Model(inputs=[x], outputs=[quat])
    model.summary()
    return model


def CuDNNLSTM_AG2(window_size=200):
    x = Input((window_size, 6), name='Input')
    LSTM1 = Bidirectional(
        CuDNNLSTM(128, return_sequences=True), name='BiLSTM1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(LSTM1)
    LSTM2 = Bidirectional(CuDNNLSTM(128), name='BiLSTM2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(LSTM2)
    qw = Dense(1, activation="linear", name='Qw')(drop2)
    qx = Dense(1, activation="linear", name='Qx')(drop2)
    qy = Dense(1, activation="linear", name='Qy')(drop2)
    qz = Dense(1, activation="linear", name='Qz')(drop2)
    model = Model(inputs=[x], outputs=[qw, qx, qy, qz])
    model.summary()
    return model


def CuDNNLSTM_AG3(window_size=200):
    x = Input((window_size, 6), name='Input')
    LSTM1 = Bidirectional(
        CuDNNLSTM(128, return_sequences=True), name='BiLSTM1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(LSTM1)
    LSTM2 = Bidirectional(CuDNNLSTM(128), name='BiLSTM2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(LSTM2)
    qs = Dense(1, activation="linear", name='Qs')(drop2)
    qv = Dense(3, activation="linear", name='Qv')(drop2)
    model = Model(inputs=[x], outputs=[qs, qv])
    model.summary()
    return model

# Euler Based - Roll - Pitch


def CuDNNLSTM_AG4(window_size=200):
    x = Input((window_size, 6), name='Input')
    LSTM1 = Bidirectional(
        CuDNNLSTM(128, return_sequences=True), name='BiLSTM1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(LSTM1)
    LSTM2 = Bidirectional(CuDNNLSTM(128), name='BiLSTM2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(LSTM2)
    roll = Dense(1, activation="linear", name='Roll')(drop2)
    pitch = Dense(1, activation="linear", name='Pitch')(drop2)
    model = Model(inputs=[x], outputs=[roll, pitch])
    model.summary()
    return model


def CuDNNLSTM_AG5(window_size=200):
    x = Input((window_size, 6), name='Input')
    LSTM1 = Bidirectional(
        CuDNNLSTM(128, return_sequences=True), name='BiLSTM1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(LSTM1)
    LSTM2 = Bidirectional(CuDNNLSTM(128), name='BiLSTM2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(LSTM2)
    Eul = Dense(2, activation="linear", name='RollPitch')(drop2)
    model = Model(inputs=[x], outputs=[Eul])
    model.summary()
    return model


# CNN only

# Quaternion Based

def CNN_AG1(window_size=200):
    x = Input((window_size, 6), name='Input')
    conv1 = Conv1D(128, 3, activation="relu", name='Conv1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(conv1)
    conv2 = Conv1D(128, 3, activation="relu", name='Conv2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(conv2)
    conv3 = Conv1D(128, 3, activation="relu", name='Conv3')(drop2)
    drop3 = Dropout(0.25, name='Dropout3')(conv3)
    conv4 = Conv1D(128, 3, activation="relu", name='Conv4')(drop3)
    drop4 = Dropout(0.25, name='Dropout4')(conv4)
    conv5 = Conv1D(128, 3, activation="relu", name='Conv5')(drop4)
    drop5 = Dropout(0.25, name='Dropout5')(conv5)
    conv6 = Conv1D(128, 3, activation="relu", name='Conv6')(drop5)
    drop6 = Dropout(0.25, name='Dropout6')(conv6)
    conv7 = Conv1D(128, 3, activation="relu", name='Conv7')(drop6)
    drop7 = Dropout(0.25, name='Dropout7')(conv7)
    conv8 = Conv1D(128, 3, activation="relu", name='Conv8')(drop7)
    drop8 = Dropout(0.25, name='Dropout8')(conv8)
    conv9 = Conv1D(128, 3, activation="relu", name='Conv9')(drop8)
    drop9 = Dropout(0.25, name='Dropout9')(conv9)
    conv10 = Conv1D(128, 3, activation="relu", name='Conv10')(drop9)
    drop10 = Dropout(0.25, name='Dropout10')(conv10)
    pool = MaxPooling1D(7, name='MaxPooling1D')(drop10)
    flat = Flatten(name='Flatten')(pool)
    quat = Dense(4, activation="linear", name='Quat')(flat)
    model = Model(inputs=[x], outputs=[quat])
    model.summary()
    return model


def CNN_AG2(window_size=200):
    x = Input((window_size, 6), name='Input')
    conv1 = Conv1D(128, 3, activation="relu", name='Conv1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(conv1)
    conv2 = Conv1D(128, 3, activation="relu", name='Conv2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(conv2)
    conv3 = Conv1D(128, 3, activation="relu", name='Conv3')(drop2)
    drop3 = Dropout(0.25, name='Dropout3')(conv3)
    conv4 = Conv1D(128, 3, activation="relu", name='Conv4')(drop3)
    drop4 = Dropout(0.25, name='Dropout4')(conv4)
    conv5 = Conv1D(128, 3, activation="relu", name='Conv5')(drop4)
    drop5 = Dropout(0.25, name='Dropout5')(conv5)
    conv6 = Conv1D(128, 3, activation="relu", name='Conv6')(drop5)
    drop6 = Dropout(0.25, name='Dropout6')(conv6)
    conv7 = Conv1D(128, 3, activation="relu", name='Conv7')(drop6)
    drop7 = Dropout(0.25, name='Dropout7')(conv7)
    conv8 = Conv1D(128, 3, activation="relu", name='Conv8')(drop7)
    drop8 = Dropout(0.25, name='Dropout8')(conv8)
    conv9 = Conv1D(128, 3, activation="relu", name='Conv9')(drop8)
    drop9 = Dropout(0.25, name='Dropout9')(conv9)
    conv10 = Conv1D(128, 3, activation="relu", name='Conv10')(drop9)
    drop10 = Dropout(0.25, name='Dropout10')(conv10)
    pool = MaxPooling1D(7, name='MaxPooling1D')(drop10)
    flat = Flatten(name='Flatten')(pool)
    qw = Dense(1, activation="linear", name='qw')(flat)
    qx = Dense(1, activation="linear", name='qx')(flat)
    qy = Dense(1, activation="linear", name='qy')(flat)
    qz = Dense(1, activation="linear", name='qz')(flat)
    model = Model(inputs=[x], outputs=[qw, qx, qy, qz])
    model.summary()
    return model


def CNN_AG3(window_size=200):
    x = Input((window_size, 6), name='Input')
    conv1 = Conv1D(128, 3, activation="relu", name='Conv1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(conv1)
    conv2 = Conv1D(128, 3, activation="relu", name='Conv2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(conv2)
    conv3 = Conv1D(128, 3, activation="relu", name='Conv3')(drop2)
    drop3 = Dropout(0.25, name='Dropout3')(conv3)
    conv4 = Conv1D(128, 3, activation="relu", name='Conv4')(drop3)
    drop4 = Dropout(0.25, name='Dropout4')(conv4)
    conv5 = Conv1D(128, 3, activation="relu", name='Conv5')(drop4)
    drop5 = Dropout(0.25, name='Dropout5')(conv5)
    conv6 = Conv1D(128, 3, activation="relu", name='Conv6')(drop5)
    drop6 = Dropout(0.25, name='Dropout6')(conv6)
    conv7 = Conv1D(128, 3, activation="relu", name='Conv7')(drop6)
    drop7 = Dropout(0.25, name='Dropout7')(conv7)
    conv8 = Conv1D(128, 3, activation="relu", name='Conv8')(drop7)
    drop8 = Dropout(0.25, name='Dropout8')(conv8)
    conv9 = Conv1D(128, 3, activation="relu", name='Conv9')(drop8)
    drop9 = Dropout(0.25, name='Dropout9')(conv9)
    conv10 = Conv1D(128, 3, activation="relu", name='Conv10')(drop9)
    drop10 = Dropout(0.25, name='Dropout10')(conv10)
    pool = MaxPooling1D(7, name='MaxPooling1D')(drop10)
    flat = Flatten(name='Flatten')(pool)
    qs = Dense(1, activation="linear", name='Qs')(flat)
    qv = Dense(3, activation="linear", name='Qv')(flat)
    model = Model(inputs=[x], outputs=[qs, qv])
    model.summary()
    return model

# Euler Based


def CNN_AG4(window_size=200):
    x = Input((window_size, 6), name='Input')
    conv1 = Conv1D(128, 3, activation="relu", name='Conv1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(conv1)
    conv2 = Conv1D(128, 3, activation="relu", name='Conv2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(conv2)
    conv3 = Conv1D(128, 3, activation="relu", name='Conv3')(drop2)
    drop3 = Dropout(0.25, name='Dropout3')(conv3)
    conv4 = Conv1D(128, 3, activation="relu", name='Conv4')(drop3)
    drop4 = Dropout(0.25, name='Dropout4')(conv4)
    conv5 = Conv1D(128, 3, activation="relu", name='Conv5')(drop4)
    drop5 = Dropout(0.25, name='Dropout5')(conv5)
    conv6 = Conv1D(128, 3, activation="relu", name='Conv6')(drop5)
    drop6 = Dropout(0.25, name='Dropout6')(conv6)
    conv7 = Conv1D(128, 3, activation="relu", name='Conv7')(drop6)
    drop7 = Dropout(0.25, name='Dropout7')(conv7)
    conv8 = Conv1D(128, 3, activation="relu", name='Conv8')(drop7)
    drop8 = Dropout(0.25, name='Dropout8')(conv8)
    conv9 = Conv1D(128, 3, activation="relu", name='Conv9')(drop8)
    drop9 = Dropout(0.25, name='Dropout9')(conv9)
    conv10 = Conv1D(128, 3, activation="relu", name='Conv10')(drop9)
    drop10 = Dropout(0.25, name='Dropout10')(conv10)
    pool = MaxPooling1D(7, name='MaxPooling1D')(drop10)
    flat = Flatten(name='Flatten')(pool)
    roll = Dense(1, activation="linear", name='Roll')(flat)
    pitch = Dense(1, activation="linear", name='Pitch')(flat)
    model = Model(inputs=[x], outputs=[roll, pitch])
    model.summary()
    return model


def CNN_AG5(window_size=200):
    x = Input((window_size, 6), name='Input')
    conv1 = Conv1D(128, 3, activation="relu", name='Conv1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(conv1)
    conv2 = Conv1D(128, 3, activation="relu", name='Conv2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(conv2)
    conv3 = Conv1D(128, 3, activation="relu", name='Conv3')(drop2)
    drop3 = Dropout(0.25, name='Dropout3')(conv3)
    conv4 = Conv1D(128, 3, activation="relu", name='Conv4')(drop3)
    drop4 = Dropout(0.25, name='Dropout4')(conv4)
    conv5 = Conv1D(128, 3, activation="relu", name='Conv5')(drop4)
    drop5 = Dropout(0.25, name='Dropout5')(conv5)
    conv6 = Conv1D(128, 3, activation="relu", name='Conv6')(drop5)
    drop6 = Dropout(0.25, name='Dropout6')(conv6)
    conv7 = Conv1D(128, 3, activation="relu", name='Conv7')(drop6)
    drop7 = Dropout(0.25, name='Dropout7')(conv7)
    conv8 = Conv1D(128, 3, activation="relu", name='Conv8')(drop7)
    drop8 = Dropout(0.25, name='Dropout8')(conv8)
    conv9 = Conv1D(128, 3, activation="relu", name='Conv9')(drop8)
    drop9 = Dropout(0.25, name='Dropout9')(conv9)
    conv10 = Conv1D(128, 3, activation="relu", name='Conv10')(drop9)
    drop10 = Dropout(0.25, name='Dropout10')(conv10)
    pool = MaxPooling1D(7, name='MaxPooling1D')(drop10)
    flat = Flatten(name='Flatten')(pool)
    Eul = Dense(2, activation="linear", name='RollPitch')(flat)
    model = Model(inputs=[x], outputs=[Eul])
    model.summary()
    return model


################################################################ Splited Input Models ##############################################################
###########################
############### Acc + Gyro
###########################

# CuDNN only

# Quaternion Based

# Euler Based


# CNN only

# Quaternion Based

# Euler Based


###########################
########### Acc + Gyro + Fs
###########################
# CuDNNLSTM only

# Quaternion Based


def CuDNNLSTM_AG1(window_size=200):
    x = Input((window_size, 6), name='Input')
    LSTM1 = Bidirectional(
        CuDNNLSTM(128, return_sequences=True), name='BiLSTM1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(LSTM1)
    LSTM2 = Bidirectional(CuDNNLSTM(128), name='BiLSTM2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(LSTM2)
    quat = Dense(4, activation="linear", name='Quat')(drop2)
    model = Model(inputs=[x], outputs=[quat])
    model.summary()
    return model


def CuDNNLSTM_AG2(window_size=200):
    x = Input((window_size, 6), name='Input')
    LSTM1 = Bidirectional(
        CuDNNLSTM(128, return_sequences=True), name='BiLSTM1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(LSTM1)
    LSTM2 = Bidirectional(CuDNNLSTM(128), name='BiLSTM2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(LSTM2)
    qw = Dense(1, activation="linear", name='Qw')(drop2)
    qx = Dense(1, activation="linear", name='Qx')(drop2)
    qy = Dense(1, activation="linear", name='Qy')(drop2)
    qz = Dense(1, activation="linear", name='Qz')(drop2)
    model = Model(inputs=[x], outputs=[qw, qx, qy, qz])
    model.summary()
    return model


def CuDNNLSTM_AG3(window_size=200):
    x = Input((window_size, 6), name='Input')
    LSTM1 = Bidirectional(
        CuDNNLSTM(128, return_sequences=True), name='BiLSTM1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(LSTM1)
    LSTM2 = Bidirectional(CuDNNLSTM(128), name='BiLSTM2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(LSTM2)
    qs = Dense(1, activation="linear", name='Qs')(drop2)
    qv = Dense(3, activation="linear", name='Qv')(drop2)
    model = Model(inputs=[x], outputs=[qs, qv])
    model.summary()
    return model

# Euler Based - Roll - Pitch


def CuDNNLSTM_AG4(window_size=200):
    x = Input((window_size, 6), name='Input')
    LSTM1 = Bidirectional(
        CuDNNLSTM(128, return_sequences=True), name='BiLSTM1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(LSTM1)
    LSTM2 = Bidirectional(CuDNNLSTM(128), name='BiLSTM2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(LSTM2)
    roll = Dense(1, activation="linear", name='Roll')(drop2)
    pitch = Dense(1, activation="linear", name='Pitch')(drop2)
    model = Model(inputs=[x], outputs=[roll, pitch])
    model.summary()
    return model


def CuDNNLSTM_AG5(window_size=200):
    x = Input((window_size, 6), name='Input')
    LSTM1 = Bidirectional(
        CuDNNLSTM(128, return_sequences=True), name='BiLSTM1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(LSTM1)
    LSTM2 = Bidirectional(CuDNNLSTM(128), name='BiLSTM2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(LSTM2)
    Eul = Dense(2, activation="linear", name='RollPitch')(drop2)
    model = Model(inputs=[x], outputs=[Eul])
    model.summary()
    return model


# CNN only

# Quaternion Based

def CNN_AG1(window_size=200):
    x = Input((window_size, 6), name='Input')
    conv1 = Conv1D(128, 3, activation="relu", name='Conv1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(conv1)
    conv2 = Conv1D(128, 3, activation="relu", name='Conv2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(conv2)
    conv3 = Conv1D(128, 3, activation="relu", name='Conv3')(drop2)
    drop3 = Dropout(0.25, name='Dropout3')(conv3)
    conv4 = Conv1D(128, 3, activation="relu", name='Conv4')(drop3)
    drop4 = Dropout(0.25, name='Dropout4')(conv4)
    conv5 = Conv1D(128, 3, activation="relu", name='Conv5')(drop4)
    drop5 = Dropout(0.25, name='Dropout5')(conv5)
    conv6 = Conv1D(128, 3, activation="relu", name='Conv6')(drop5)
    drop6 = Dropout(0.25, name='Dropout6')(conv6)
    conv7 = Conv1D(128, 3, activation="relu", name='Conv7')(drop6)
    drop7 = Dropout(0.25, name='Dropout7')(conv7)
    conv8 = Conv1D(128, 3, activation="relu", name='Conv8')(drop7)
    drop8 = Dropout(0.25, name='Dropout8')(conv8)
    conv9 = Conv1D(128, 3, activation="relu", name='Conv9')(drop8)
    drop9 = Dropout(0.25, name='Dropout9')(conv9)
    conv10 = Conv1D(128, 3, activation="relu", name='Conv10')(drop9)
    drop10 = Dropout(0.25, name='Dropout10')(conv10)
    pool = MaxPooling1D(7, name='MaxPooling1D')(drop10)
    flat = Flatten(name='Flatten')(pool)
    quat = Dense(4, activation="linear", name='Quat')(flat)
    model = Model(inputs=[x], outputs=[quat])
    model.summary()
    return model


def CNN_AG2(window_size=200):
    x = Input((window_size, 6), name='Input')
    conv1 = Conv1D(128, 3, activation="relu", name='Conv1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(conv1)
    conv2 = Conv1D(128, 3, activation="relu", name='Conv2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(conv2)
    conv3 = Conv1D(128, 3, activation="relu", name='Conv3')(drop2)
    drop3 = Dropout(0.25, name='Dropout3')(conv3)
    conv4 = Conv1D(128, 3, activation="relu", name='Conv4')(drop3)
    drop4 = Dropout(0.25, name='Dropout4')(conv4)
    conv5 = Conv1D(128, 3, activation="relu", name='Conv5')(drop4)
    drop5 = Dropout(0.25, name='Dropout5')(conv5)
    conv6 = Conv1D(128, 3, activation="relu", name='Conv6')(drop5)
    drop6 = Dropout(0.25, name='Dropout6')(conv6)
    conv7 = Conv1D(128, 3, activation="relu", name='Conv7')(drop6)
    drop7 = Dropout(0.25, name='Dropout7')(conv7)
    conv8 = Conv1D(128, 3, activation="relu", name='Conv8')(drop7)
    drop8 = Dropout(0.25, name='Dropout8')(conv8)
    conv9 = Conv1D(128, 3, activation="relu", name='Conv9')(drop8)
    drop9 = Dropout(0.25, name='Dropout9')(conv9)
    conv10 = Conv1D(128, 3, activation="relu", name='Conv10')(drop9)
    drop10 = Dropout(0.25, name='Dropout10')(conv10)
    pool = MaxPooling1D(7, name='MaxPooling1D')(drop10)
    flat = Flatten(name='Flatten')(pool)
    qw = Dense(1, activation="linear", name='qw')(flat)
    qx = Dense(1, activation="linear", name='qx')(flat)
    qy = Dense(1, activation="linear", name='qy')(flat)
    qz = Dense(1, activation="linear", name='qz')(flat)
    model = Model(inputs=[x], outputs=[qw, qx, qy, qz])
    model.summary()
    return model


def CNN_AG3(window_size=200):
    x = Input((window_size, 6), name='Input')
    conv1 = Conv1D(128, 3, activation="relu", name='Conv1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(conv1)
    conv2 = Conv1D(128, 3, activation="relu", name='Conv2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(conv2)
    conv3 = Conv1D(128, 3, activation="relu", name='Conv3')(drop2)
    drop3 = Dropout(0.25, name='Dropout3')(conv3)
    conv4 = Conv1D(128, 3, activation="relu", name='Conv4')(drop3)
    drop4 = Dropout(0.25, name='Dropout4')(conv4)
    conv5 = Conv1D(128, 3, activation="relu", name='Conv5')(drop4)
    drop5 = Dropout(0.25, name='Dropout5')(conv5)
    conv6 = Conv1D(128, 3, activation="relu", name='Conv6')(drop5)
    drop6 = Dropout(0.25, name='Dropout6')(conv6)
    conv7 = Conv1D(128, 3, activation="relu", name='Conv7')(drop6)
    drop7 = Dropout(0.25, name='Dropout7')(conv7)
    conv8 = Conv1D(128, 3, activation="relu", name='Conv8')(drop7)
    drop8 = Dropout(0.25, name='Dropout8')(conv8)
    conv9 = Conv1D(128, 3, activation="relu", name='Conv9')(drop8)
    drop9 = Dropout(0.25, name='Dropout9')(conv9)
    conv10 = Conv1D(128, 3, activation="relu", name='Conv10')(drop9)
    drop10 = Dropout(0.25, name='Dropout10')(conv10)
    pool = MaxPooling1D(7, name='MaxPooling1D')(drop10)
    flat = Flatten(name='Flatten')(pool)
    qs = Dense(1, activation="linear", name='Qs')(flat)
    qv = Dense(3, activation="linear", name='Qv')(flat)
    model = Model(inputs=[x], outputs=[qs, qv])
    model.summary()
    return model

# Euler Based


def CNN_AG4(window_size=200):
    x = Input((window_size, 6), name='Input')
    conv1 = Conv1D(128, 3, activation="relu", name='Conv1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(conv1)
    conv2 = Conv1D(128, 3, activation="relu", name='Conv2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(conv2)
    conv3 = Conv1D(128, 3, activation="relu", name='Conv3')(drop2)
    drop3 = Dropout(0.25, name='Dropout3')(conv3)
    conv4 = Conv1D(128, 3, activation="relu", name='Conv4')(drop3)
    drop4 = Dropout(0.25, name='Dropout4')(conv4)
    conv5 = Conv1D(128, 3, activation="relu", name='Conv5')(drop4)
    drop5 = Dropout(0.25, name='Dropout5')(conv5)
    conv6 = Conv1D(128, 3, activation="relu", name='Conv6')(drop5)
    drop6 = Dropout(0.25, name='Dropout6')(conv6)
    conv7 = Conv1D(128, 3, activation="relu", name='Conv7')(drop6)
    drop7 = Dropout(0.25, name='Dropout7')(conv7)
    conv8 = Conv1D(128, 3, activation="relu", name='Conv8')(drop7)
    drop8 = Dropout(0.25, name='Dropout8')(conv8)
    conv9 = Conv1D(128, 3, activation="relu", name='Conv9')(drop8)
    drop9 = Dropout(0.25, name='Dropout9')(conv9)
    conv10 = Conv1D(128, 3, activation="relu", name='Conv10')(drop9)
    drop10 = Dropout(0.25, name='Dropout10')(conv10)
    pool = MaxPooling1D(7, name='MaxPooling1D')(drop10)
    flat = Flatten(name='Flatten')(pool)
    roll = Dense(1, activation="linear", name='Roll')(flat)
    pitch = Dense(1, activation="linear", name='Pitch')(flat)
    model = Model(inputs=[x], outputs=[roll, pitch])
    model.summary()
    return model


def CNN_AG5(window_size=200):
    x = Input((window_size, 7), name='Input')
    conv1 = Conv1D(128, 3, activation="relu", name='Conv1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(conv1)
    conv2 = Conv1D(128, 3, activation="relu", name='Conv2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(conv2)
    conv3 = Conv1D(128, 3, activation="relu", name='Conv3')(drop2)
    drop3 = Dropout(0.25, name='Dropout3')(conv3)
    conv4 = Conv1D(128, 3, activation="relu", name='Conv4')(drop3)
    drop4 = Dropout(0.25, name='Dropout4')(conv4)
    conv5 = Conv1D(128, 3, activation="relu", name='Conv5')(drop4)
    drop5 = Dropout(0.25, name='Dropout5')(conv5)
    conv6 = Conv1D(128, 3, activation="relu", name='Conv6')(drop5)
    drop6 = Dropout(0.25, name='Dropout6')(conv6)
    conv7 = Conv1D(128, 3, activation="relu", name='Conv7')(drop6)
    drop7 = Dropout(0.25, name='Dropout7')(conv7)
    conv8 = Conv1D(128, 3, activation="relu", name='Conv8')(drop7)
    drop8 = Dropout(0.25, name='Dropout8')(conv8)
    conv9 = Conv1D(128, 3, activation="relu", name='Conv9')(drop8)
    drop9 = Dropout(0.25, name='Dropout9')(conv9)
    conv10 = Conv1D(128, 3, activation="relu", name='Conv10')(drop9)
    drop10 = Dropout(0.25, name='Dropout10')(conv10)
    pool = MaxPooling1D(7, name='MaxPooling1D')(drop10)
    flat = Flatten(name='Flatten')(pool)
    Eul = Dense(2, activation="linear", name='RollPitch')(flat)
    model = Model(inputs=[x], outputs=[Eul])
    model.summary()
    return model

# CuDNN + BiLSTM


def CNN_CuDNN_AGFs1(window_size=200):
    x = Input((window_size, 7), name='Input')
    conv1 = Conv1D(128, 3, activation="relu", name='Conv1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(conv1)
    conv2 = Conv1D(128, 3, activation="relu", name='Conv2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(conv2)
    conv3 = Conv1D(128, 3, activation="relu", name='Conv3')(drop2)
    drop3 = Dropout(0.25, name='Dropout3')(conv3)
    pool = Dense(256, activation='relu')(drop3)
    biLSTM1 = Bidirectional(
        CuDNNLSTM(128, return_sequences=True), name='BiLSTM1')(pool)
    drop1 = Dropout(0.25, name='DropoutLSTM1')(biLSTM1)
    biLSTM2 = Bidirectional(CuDNNLSTM(128), name='BiLSTM2')(drop1)
    drop2 = Dropout(0.25, name='DropoutLSTM2')(biLSTM2)
    quat = Dense(4, activation="linear", name='Quat')(drop2)
    model = Model(inputs=[x], outputs=[quat])
    model.summary()
    return model


def CNN_CuDNN_AG2(window_size=200):
    x = Input((window_size, 6), name='Input')
    conv1 = Conv1D(128, 3, activation="relu", name='Conv1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(conv1)
    conv2 = Conv1D(128, 3, activation="relu", name='Conv2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(conv2)
    conv3 = Conv1D(128, 3, activation="relu", name='Conv3')(drop2)
    drop3 = Dropout(0.25, name='Dropout3')(conv3)
    conv4 = Conv1D(128, 3, activation="relu", name='Conv4')(drop3)
    drop4 = Dropout(0.25, name='Dropout4')(conv4)
    conv5 = Conv1D(128, 3, activation="relu", name='Conv5')(drop4)
    drop5 = Dropout(0.25, name='Dropout5')(conv5)
    conv6 = Conv1D(128, 3, activation="relu", name='Conv6')(drop5)
    drop6 = Dropout(0.25, name='Dropout6')(conv6)
    conv7 = Conv1D(128, 3, activation="relu", name='Conv7')(drop6)
    drop7 = Dropout(0.25, name='Dropout7')(conv7)
    conv8 = Conv1D(128, 3, activation="relu", name='Conv8')(drop7)
    drop8 = Dropout(0.25, name='Dropout8')(conv8)
    conv9 = Conv1D(128, 3, activation="relu", name='Conv9')(drop8)
    drop9 = Dropout(0.25, name='Dropout9')(conv9)
    conv10 = Conv1D(128, 3, activation="relu", name='Conv10')(drop9)
    drop10 = Dropout(0.25, name='Dropout10')(conv10)
    pool = MaxPooling1D(7, name='MaxPooling1D')(drop10)
    flat = Flatten(name='Flatten')(pool)
    qw = Dense(1, activation="linear", name='qw')(flat)
    qx = Dense(1, activation="linear", name='qx')(flat)
    qy = Dense(1, activation="linear", name='qy')(flat)
    qz = Dense(1, activation="linear", name='qz')(flat)
    model = Model(inputs=[x], outputs=[qw, qx, qy, qz])
    model.summary()
    return model


def CNN_CuDNN_AG3(window_size=200):
    x = Input((window_size, 6), name='Input')
    conv1 = Conv1D(128, 3, activation="relu", name='Conv1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(conv1)
    conv2 = Conv1D(128, 3, activation="relu", name='Conv2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(conv2)
    conv3 = Conv1D(128, 3, activation="relu", name='Conv3')(drop2)
    drop3 = Dropout(0.25, name='Dropout3')(conv3)
    conv4 = Conv1D(128, 3, activation="relu", name='Conv4')(drop3)
    drop4 = Dropout(0.25, name='Dropout4')(conv4)
    conv5 = Conv1D(128, 3, activation="relu", name='Conv5')(drop4)
    drop5 = Dropout(0.25, name='Dropout5')(conv5)
    conv6 = Conv1D(128, 3, activation="relu", name='Conv6')(drop5)
    drop6 = Dropout(0.25, name='Dropout6')(conv6)
    conv7 = Conv1D(128, 3, activation="relu", name='Conv7')(drop6)
    drop7 = Dropout(0.25, name='Dropout7')(conv7)
    conv8 = Conv1D(128, 3, activation="relu", name='Conv8')(drop7)
    drop8 = Dropout(0.25, name='Dropout8')(conv8)
    conv9 = Conv1D(128, 3, activation="relu", name='Conv9')(drop8)
    drop9 = Dropout(0.25, name='Dropout9')(conv9)
    conv10 = Conv1D(128, 3, activation="relu", name='Conv10')(drop9)
    drop10 = Dropout(0.25, name='Dropout10')(conv10)
    pool = MaxPooling1D(7, name='MaxPooling1D')(drop10)
    flat = Flatten(name='Flatten')(pool)
    qs = Dense(1, activation="linear", name='Qs')(flat)
    qv = Dense(3, activation="linear", name='Qv')(flat)
    model = Model(inputs=[x], outputs=[qs, qv])
    model.summary()
    return model

# Euler Based


def CNN_CuDNN_AGFs4(window_size=200):
    x = Input((window_size, 7), name='Input')
    conv1 = Conv1D(128, 3, activation="relu", name='Conv1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(conv1)
    conv2 = Conv1D(128, 3, activation="relu", name='Conv2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(conv2)
    conv3 = Conv1D(128, 3, activation="relu", name='Conv3')(drop2)
    drop3 = Dropout(0.25, name='Dropout3')(conv3)

    pool = Dense(256, name='Dense')(drop3)
    biLSTM1 = Bidirectional(
        CuDNNLSTM(128, return_sequences=True), name='BiLSTM1')(pool)
    drop1 = Dropout(0.25, name='DropoutLSTM1')(biLSTM1)
    biLSTM2 = Bidirectional(
        CuDNNLSTM(128, return_sequences=True), name='BiLSTM2')(drop1)
    drop2 = Dropout(0.25, name='DropoutLSTM2')(biLSTM2)

    biLSTMr = Bidirectional(LSTM(128), name='BiLSTMr')(drop2)
    dropr = Dropout(0.25, name='DropoutLSTMr')(biLSTMr)
    roll = Dense(1, activation="linear", name='Roll')(dropr)

    #biLSTMp = Bidirectional(LSTM(128), name='BiLSTMp')(drop2)
    #dropp = Dropout(0.25, name='DropoutLSTMp')(biLSTMp)
    #pitch = Dense(1, activation="linear", name='Pitch')(dropp)

    model = Model(inputs=[x], outputs=[roll])
    model.summary()
    return model


def CNN_CuDNN_AG5(window_size=200):
    x = Input((window_size, 7), name='Input')
    conv1 = Conv1D(128, 3, activation="relu", name='Conv1')(x)
    drop1 = Dropout(0.25, name='Dropout1')(conv1)
    conv2 = Conv1D(128, 3, activation="relu", name='Conv2')(drop1)
    drop2 = Dropout(0.25, name='Dropout2')(conv2)
    conv3 = Conv1D(128, 3, activation="relu", name='Conv3')(drop2)
    drop3 = Dropout(0.25, name='Dropout3')(conv3)
    conv4 = Conv1D(128, 3, activation="relu", name='Conv4')(drop3)
    drop4 = Dropout(0.25, name='Dropout4')(conv4)
    conv5 = Conv1D(128, 3, activation="relu", name='Conv5')(drop4)
    drop5 = Dropout(0.25, name='Dropout5')(conv5)
    conv6 = Conv1D(128, 3, activation="relu", name='Conv6')(drop5)
    drop6 = Dropout(0.25, name='Dropout6')(conv6)
    conv7 = Conv1D(128, 3, activation="relu", name='Conv7')(drop6)
    drop7 = Dropout(0.25, name='Dropout7')(conv7)
    conv8 = Conv1D(128, 3, activation="relu", name='Conv8')(drop7)
    drop8 = Dropout(0.25, name='Dropout8')(conv8)
    conv9 = Conv1D(128, 3, activation="relu", name='Conv9')(drop8)
    drop9 = Dropout(0.25, name='Dropout9')(conv9)
    conv10 = Conv1D(128, 3, activation="relu", name='Conv10')(drop9)
    drop10 = Dropout(0.25, name='Dropout10')(conv10)
    pool = MaxPooling1D(7, name='MaxPooling1D')(drop10)
    flat = Flatten(name='Flatten')(pool)
    Eul = Dense(2, activation="linear", name='RollPitch')(flat)
    model = Model(inputs=[x], outputs=[Eul])
    model.summary()
    return model


def alexnet_model(window_size=200):

    # Initialize model
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Conv1D(96,  11, input_shape=(window_size, 7),
                padding='same', kernel_regularizer=l2(0)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling1D(pool_size=2))

    # Layer 2
    alexnet.add(Conv1D(256, 5, padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling1D(pool_size=2))

    # Layer 3
    alexnet.add(ZeroPadding1D(1))
    alexnet.add(Conv1D(512, 3, padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling1D(pool_size=2))

    # Layer 4
    alexnet.add(ZeroPadding1D(1))
    alexnet.add(Conv1D(1024, 3, padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # Layer 5
    alexnet.add(ZeroPadding1D(1))
    alexnet.add(Conv1D(1024, 3, padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling1D(pool_size=2))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(3072))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 7
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 8
    alexnet.add(Dense(4))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('linear'))
    alexnet.summary()
    return alexnet


###########################
########### Acc + Fs
########### Gyro + Fs
###########################

# CuDNN only

# Quaternion Based

# Euler Based


# CNN only

# Quaternion Based

# Euler Based

###########################
# Acc
# Gyro
###########################

# CuDNN only

# Quaternion Based

# Euler Based


# CNN only

# Quaternion Based

# Euler Based

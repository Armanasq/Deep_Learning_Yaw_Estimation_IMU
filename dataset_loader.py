import os
import numpy as np
import pandas as pd
import quaternion
import scipy.interpolate
import scipy as sp
import h5py
from keras.utils import Sequence

dataset_path = "/home/arman/Neural Network in Python/RNN/Dataset/"


def interpolate_3dvector_linear(input, input_timestamp, output_timestamp):
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
    interpolated = func(output_timestamp)
    return interpolated


def BROAD_path(i):
    ''' i = [1,40]
    '''
    gt_path = dataset_path + 'BROAD/trial_gt{}.csv'.format(i)
    imu_path = dataset_path + 'BROAD/trial_imu{}.csv'.format(i)
    return imu_path, gt_path


def BROAD_data(imu_filename, gt_filename):
    fs = 286
    df = pd.read_csv(imu_filename, header=0).values
    acc = df[:, 0:3]
    gyro = df[:, 3:6]
    mag = df[:, 6:9]
    df = pd.read_csv(gt_filename, header=0).values
    quat = df[:, 3:7]
    return acc, gyro, mag, quat, fs


def EuRoC_MAV_path():
    imu_path = []
    gt_path = []
    os.chdir(dataset_path+"EuRoC_MAV_Dataset/")
    # list folders in directory
    folder_list = [folder for folder in os.listdir(
        '.') if os.path.isdir(folder)]
    for i in range(len(folder_list)):
        imu_path.append(os.path.join(folder_list[i], 'mav0/imu0/data.csv'))
        gt_path.append(os.path.join(
            folder_list[i], 'mav0/state_groundtruth_estimate0/data.csv'))
    df = pd.DataFrame({'imu_path': imu_path, 'gt_path': gt_path})
    return df


def EuRoC_MAV_data(imu_filename, gt_filename):
    fs = 200
    gt_data = pd.read_csv(gt_filename).values
    imu_data = pd.read_csv(imu_filename).values

    gyro = interpolate_3dvector_linear(
        imu_data[:, 1:4], imu_data[:, 0], gt_data[:, 0])
    acc = interpolate_3dvector_linear(
        imu_data[:, 4:7], imu_data[:, 0], gt_data[:, 0])
    # pos = gt_data[:, 1:4]
    quat = gt_data[:, 4:8]
    return acc, gyro, quat


def OxIOD_path():
    path = []
    file_path = []
    # change dirctroy to file path
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir(dataset_path+"Oxford Inertial Odometry Dataset/")
    # dir_list = [f for f in os.listdir(oxiod) if os.path.isdir(f)]
    dir_list = [name for name in os.listdir(
        ".") if os.path.isdir(name) if not os.path.isfile(name)]
    dir_list.remove('.ipynb_checkpoints')
    for i in range(len(dir_list)):
        sub_folders = [name for name in os.listdir(
            dir_list[i]) if os.path.isdir(os.path.join(dir_list[i], name))]
        for k in range(int(len(sub_folders))):
            path = (dataset_path+"Oxford Inertial Odometry Dataset/" +
                    dir_list[i]+'/'+sub_folders[k]+'/syn/')
            files = os.listdir(path)
            if 'Readme.txt' in files:
                files.remove('Readme.txt')
            for j in range(1, len(files)):
                if 'imu' in str(files[j]):
                    file_path.append(path+files[j])
    df = pd.DataFrame({"OxIOD File Path": file_path})
    #df.to_csv(dataset_path+"OxIOD_file_path.csv", index=False)
    return df


def OxIOD_data(filename):
    fs = 100
    oxiod_imu = pd.read_csv(filename).values
    oxiod_gt = pd.read_csv(filename.replace('imu', 'vi')).values
    acc = oxiod_imu[:, 10:13]
    gyro = oxiod_imu[:, 4:7]
    mag = oxiod_imu[:, 7:10]
    ori = np.concatenate((oxiod_gt[:, 8:9], oxiod_gt[:, 5:8]), axis=1)
    #pose = oxiod_gt[:, 2:5]
    return acc, gyro, mag, ori, fs


def repoIMU_path():
    path = []
    file_path = []
    os.chdir(dataset_path+"Repo IMU/Pendulum/")
    # list files in Pendulum folder
    dir_list_Pendulum = [name for name in os.listdir('.')]
    os.chdir(dataset_path+"Repo IMU/TStick/")
    dir_list_TStick = [name for name in os.listdir('.')]
    df_Pendulum = pd.DataFrame({"Pendulum": dir_list_Pendulum})
    df_TStick = pd.DataFrame({"TStick": dir_list_TStick})
    return df_TStick, df_Pendulum


def repoIMU_data(filename):
    if 'Pendulum' in filename:
        path = 'Pendulum/'
    elif 'TStick' in filename:
        path = 'TStick/'
    filename = dataset_path + "Repo IMU/" + path + filename
    fs = 100
    df = pd.read_csv(filename, delimiter=';', header=1)
    quat = df.values[1:, 1:5]

    acc = df.values[1:, 5:8]
    gyro = df.values[1:, 8:11]
    mag = df.values[1:, 11:14]
    return acc, gyro, mag, quat, fs


def RIDI_path():
    os.chdir(dataset_path+"RIDI/data_publish_v2/")
    dir_list = [name for name in os.listdir('.') if os.path.isdir(name)]
    return dir_list


def RIDI_data(filename):
    fs = 201
    df = pd.read_csv(filename+'/processed/data.csv', index_col=0)
    acc = df[['acce_x', 'acce_y', 'acce_z']].values
    gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values
    mag = df[['magnet_x', 'magnet_y', 'magnet_z']].values

    quat = df[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    # pose = df[['pos_x', 'pos_y', 'pos_z']]
    return acc, gyro, mag, quat, fs


def RoNIN_path():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    folder = ['train_dataset_2', 'unseen_subjects_test_set',
              'seen_subjects_test_set', 'train_dataset_1']
    for i in range(len(folder)):
        os.chdir(dataset_path+"Ronin/"+folder[i])
        folder_name = prefixed = [filename for filename in os.listdir(
            '.') if filename.startswith("a")]
        globals()['df{}'.format(i)] = pd.DataFrame(
            {folder[i]: folder_name})
        globals()['df{}'.format(i)] = dataset_path+"Ronin/" + \
            folder[i]+"/" + globals()['df{}'.format(i)]
    return df1, df2, df3, df0


def RoNIN_data(filename):
    '''folder_name = prefixed = [filename for filename in os.listdir('.') if filename.startswith("a")]
    RoNIN_data(RoNIN_path()[0].values[0][0]
    '''

    fs = 200
    load_file = filename + '/data.hdf5'
    df = h5py.File(load_file, 'r')
    header = np.array(df.get('synced'))
    for i in range(len(np.array(df.get('synced')))):
        if header[i] == 'acce':
            acc = np.array(df.get('synced')[header[i]])
        if header[i] == 'gyro':
            gyro = np.array(df.get('synced')[header[i]])
        if header[i] == 'magnet':
            mag = np.array(df.get('synced')[header[i]])
    header = np.array(df.get('pose'))
    for i in range(len(np.array(df.get('pose')))):
        if header[i] == 'ekf_ori':
            quat = np.array(df.get('pose')[header[i]])
        if header[i] == 'tango_pos':
            pose = np.array(df.get('pose')[header[i]])
    return acc, gyro, mag, quat, fs


def sassari_path():
    os.chdir(dataset_path+"Sassari/")
    file_list = [name for name in os.listdir('.') if os.path.isfile(name)]
    file_list = [dataset_path+"Sassari/" + s for s in file_list]
    return file_list


def sassari_data(filename, mimu):
    # MIMU = XS1, XS2, AP1, AP2, SH1, SH2
    fs = 100
    mat = sp.io.loadmat(filename)
    data = mat[mimu]
    imu = data[:, 1: 10]
    quat = data[:, 10: 14]
    acc = imu[:, 0:3]
    gyro = imu[:, 3:6]
    mag = imu[:, 6:9]
    return acc, gyro, mag, quat, fs

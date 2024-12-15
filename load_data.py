import numpy as np
import torch
import torch.nn.functional as F
from utils import NormalizeFeaTorch, get_Similarity
# from keras_preprocessing import image
from numpy import hstack
from scipy import misc
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import h5py
import random
import warnings
warnings.filterwarnings("ignore")

# 需要h5py读取
ALL_data = dict(
hos_160_5view_3c_25_simple={1: 'hos_160_5view_3c_25_simple', 'N': 160, 'K': 2, 'V': 5, 'n_input': [1380, 1826, 3319, 17939,25],
                      'n_hid': [256, 256, 512, 2056,32], 'n_output': 256},
    hos_160_4view_3c_simple={1: 'hos_160_4view_3c_simple', 'N': 160, 'K': 3, 'V': 4, 'n_input': [1380, 1826, 3319, 17939],
                      'n_hid': [256, 256, 512, 2056], 'n_output': 256},
)
# {1: , 'N': , 'K': , 'V': , 'n_input': [], 'n_hid': [], 'n_output': }

path = 'C:/Users/Admin/Desktop/20231214/1205/datasets/'

def load_data(dataset):
    data = h5py.File(path + dataset[1] + ".mat")
    X = []
    Y = []
    Label = np.array(data['Y']).T
    Label = Label.reshape(Label.shape[0])
    # print('Label.shape',Label.shape)
    mm = MinMaxScaler()
    for i in range(data['X'].shape[1]):
        diff_view = data[data['X'][0, i]]
        diff_view = np.array(diff_view, dtype=np.float32).T
        ###### 对features归一化 ######
        std_view = mm.fit_transform(diff_view)
        # std_view = diff_view

        ######
        X.append(std_view)
        Y.append(Label)

    np.random.seed(1)  # 改1
    size = len(Y[0])  # size=n, X与Y均为nx(d1+d2+···+d_V)
    view_num = len(X)
    index = [i for i in range(size)]
    np.random.shuffle(index)
    for v in range(view_num):
        X[v] = X[v][index]
        Y[v] = Y[v][index]
    for v in range(view_num):
        X[v] = torch.from_numpy(X[v])


    return X, Y






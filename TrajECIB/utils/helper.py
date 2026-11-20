import numpy as np
import os
import torch
import pickle
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pandas as pd
import copy
# 划分训练集
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
import matplotlib.pyplot as plt
import random
class MyData(Dataset):#X, X_norm, Y, M
    def __init__(self, X, X_norm, Y,M):
        self.raw_traj = X
        self.norm_traj = X_norm
        self.label = Y
        self.M = M

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.raw_traj[idx], self.norm_traj[idx], self.label[idx], self.M[idx]
def truncated_latlon_noise(data, meters=50):

    factor_lat = 1 / 111000 * meters 
    cos = np.cos(data[:,:,0].mean().item())
    bound_lo = - 5 * factor_lat
    bound_hi = 5 * factor_lat

    noise_lat = torch.randn_like(data[:,:,[0]]) * factor_lat
    noise_lng = torch.randn_like(data[:,:,[1]]) * factor_lat * cos


    noise_lat = torch.clamp(noise_lat, bound_lo, bound_hi)
    noise_lng = torch.clamp(noise_lng, bound_lo* cos, bound_hi* cos)

    noise = torch.cat([noise_lat, noise_lng], dim=-1)

    # 20%的点noise
    mask = torch.rand(data.shape[0], data.shape[1], 1) < 0.2
    noise = noise * mask.float()

    return noise 

def collate_fn(data, basic_model = 'gru'):
    # X, X_raw, Y
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_length = [len(sq[0]) for sq in data]
    x = [i[0] for i in data]
    x_norm = [i[1] for i in data]
    y = [i[2] for i in data]
    data_x = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
    x_norm = rnn_utils.pad_sequence(x_norm, batch_first=True, padding_value=0)
    # data_length = [data_x.size(1)] * len(data_length) # 为了可比
    if basic_model != 'cl':
        m = [i[3][:data_length[0]//2,:data_length[0]] for i in data]
        m = torch.stack(m)
    else:
        '''
        # data_x: batch_size, seq_len, d_dim
        data_x_new = data_x.clone() # batch_size, seq_len, d_dim
        data_x_new[:,:,[0,1]] = truncated_latlon_noise(data_x_new[:,:,[0,1]])


        data_x1 = data_x_new[:, ::2, :]  # batch_size, seq_len//2, d_dim
        data_x2 = data_x_new[:, 1::2, :]  # batch_size, seq_len//2, d_dim

        temp1 = data_x_new[:,1:-1:2,2] + data_x_new[:,2::2,2]
        temp1 = torch.cat([torch.zeros(data_x_new.size(0),1).to(data_x_new.device), temp1], dim=1)
        data_x1[:,:,2] =  temp1

        temp2 = data_x_new[:,2:-1:2,2] + data_x_new[:,3::2,2] # 
        temp2 = torch.cat([torch.zeros(data_x_new.size(0),1).to(data_x_new.device), temp2], dim=1)
        data_x2[:,:,2] =  temp2

        m = torch.stack([data_x1, data_x2], dim=1) # batch_size, 2, seq_len//2, d_dim
        '''
        m = data_x[:,:,[0,1]].clone() # batch_size, seq_len, d_dim
        m = truncated_latlon_noise(m)


    return data_x, data_length,  x_norm, torch.stack(y), m

def get_dataloader_dict(augs, datapath, batch_size=64,seed=5104):
    with open(datapath, "rb") as fp:  # Pickling
        dataset = pickle.load(fp)
    if ('wh'  in datapath) or ('yt' in datapath):
        if 'wh' in datapath:
            driver_list = [137649498, 207849566, 320033196, 330845915, 342998578, 372125240, 437705204, 445133775, 512642865, 546241304, 613996135, 777127476, 830601713, 861165921, 991933850, 1147360783, 1183000336, 1224803237, 1244087498, 1388789427, 1400604994, 1562245881, 1675438963, 1778090874, 1922821183, 1988645666]
        elif 'yt' in datapath:
            driver_list = [434376109, 318314496, 1152363661, 1240499650, 1037846550, 1311978570, 1424057282, 689475580, 172996644, 548247522, 1173863258, 659547234, 1641819463, 973659019, 1121658432, 1384905, 226580370, 1612154935, 1047179212, 982650525, 191287159, 1031821050, 5387325, 1524327524, 543550181, 1928887949, 2033905236, 480834070, 1376399892, 674156422, 1716543144, 405535186, 940632930, 1592957037, 933125043, 1490892827, 1673460953, 113469987, 1148485818, 1619148328, 392996919, 596129895, 619262519, 1636352545, 1759669349, 249626333, 141237039, 2038268758, 1162296944, 841034708]        # 随机种子

        # 随机种子
        np.random.seed(42)
        if augs.num_driver < len(driver_list):
            k = np.random.choice(driver_list, size=min(augs.num_driver,len(driver_list)),replace=False)
        else:
            k = np.random.choice(np.unique(dataset['Y']), size=augs.num_driver, replace=False)
        id = [i for i, uid in enumerate(dataset['Y']) if uid in k]
        for key, value in dataset.items():
            dataset[key] = [value[i] for i in id]
        # dataset['Y'] 重新编码
        uid_map = {old_uid: new_uid for new_uid, old_uid in enumerate(sorted(set(dataset['Y'])))}
        print('Selected drivers:', uid_map)
        dataset['Y'] = [uid_map[old_uid] for old_uid in dataset['Y']]









    X = dataset['X'] #特征
    Y = dataset['Y'] #标签
    M = dataset['M'] #
   
    
    Trip_id = dataset['Trip_id']
    User_id = dataset['User_id']
    groups_id = Trip_id#np.arange(len(X)) 
    # 划分训练集，验证集，测试集，我们希望相同的trip_id在同一个集合中

    random_state = 42
    # 1. 首先划分训练集和测试集
    test_size = 0.2
    if 'shl'  in datapath:
        test_size = 0.25
    group_splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(group_splitter.split(X, Y, groups=groups_id))
    X_train, Y_train, User_id_train, Trip_id_train, M_train = [X[i] for i in train_indices], [Y[i] for i in train_indices], [User_id[i] for i in train_indices], [Trip_id[i] for i in train_indices], [M[i] for i in train_indices]
    X_test, Y_test, User_id_test, Trip_id_test, M_test = [X[i] for i in test_indices], [Y[i] for i in test_indices], [User_id[i] for i in test_indices], [Trip_id[i] for i in test_indices], [M[i] for i in test_indices]
    # 2. 划分训练集和验证集
    groups_id = np.arange(len(X_test)) # 保证相同的trip_id在同一个集合中
    group_splitter_valid = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    test_indices, valid_indices = next(group_splitter_valid.split(X_test, Y_test, groups=groups_id))
    X_valid, Y_valid, User_id_valid, Trip_id_valid, M_valid= [X_test[i] for i in valid_indices], [Y_test[i] for i in valid_indices], [User_id_test[i] for i in valid_indices], [Trip_id_test[i] for i in valid_indices], [M_test[i] for i in valid_indices]
    X_test, Y_test, User_id_test, Trip_id_test, M_test = [X_test[i] for i in test_indices], [Y_test[i] for i in test_indices], [User_id_test[i] for i in test_indices], [Trip_id_test[i] for i in test_indices], [M_test[i] for i in test_indices]
    
    if augs.roubust == 'few':
        # 训练集中的一部分数据
        n = int(len(X_train) * augs.roubust_size)
        np.random.seed(42)
        choice_id = np.random.choice(len(X_train), n, replace=False)
        X_train, Y_train, M_train = [X_train[i] for i in choice_id], [Y_train[i] for i in choice_id], [M_train[i] for i in choice_id]

    elif augs.roubust == 'imbalance':
        
        half = augs.num_driver // 2

        X_train_0 = [X_train[i] for i in range(len(X_train)) if Y_train[i] < half]
        Y_train_0 = [Y_train[i] for i in range(len(Y_train)) if Y_train[i] < half]
        M_train_0 = [M_train[i] for i in range(len(M_train)) if Y_train[i] < half]
        
        X_train_1 = [X_train[i] for i in range(len(X_train)) if Y_train[i] >= half]
        Y_train_1 = [Y_train[i] for i in range(len(Y_train)) if Y_train[i] >= half]
        M_train_1 = [M_train[i] for i in range(len(M_train)) if Y_train[i] >= half]

        
        size_min = min(len(X_train_0), len(X_train_1))
        
        
        shuffle_idx_0 = np.random.permutation(len(X_train_0))
        X_train_0 = [X_train_0[i] for i in shuffle_idx_0]
        Y_train_0 = [Y_train_0[i] for i in shuffle_idx_0]
        M_train_0 = [M_train_0[i] for i in shuffle_idx_0]
        
        shuffle_idx_1 = np.random.permutation(len(X_train_1))
        X_train_1 = [X_train_1[i] for i in shuffle_idx_1]
        Y_train_1 = [Y_train_1[i] for i in shuffle_idx_1]
        M_train_1 = [M_train_1[i] for i in shuffle_idx_1]
        
        X_train_0 = [X_train_0[i] for i in range(size_min)]
        Y_train_0 = [Y_train_0[i] for i in range(size_min)]
        M_train_0 = [M_train_0[i] for i in range(size_min)]
        
        X_train_1 = [X_train_1[i] for i in range(size_min)]
        Y_train_1 = [Y_train_1[i] for i in range(size_min)]
        M_train_1 = [M_train_1[i] for i in range(size_min)]
        
        
        roubust_size = str(augs.roubust_size).split('.')
        roubust_size_0 = int(roubust_size[0])
        roubust_size_1 = int(roubust_size[1])
        
        # 可比的
        size=size_min // roubust_size_1
        X_train_choice_0 = [i for i in range(size)]
        size=size_min // roubust_size_0
        X_train_choice_1 = [i for i in range(size)]

        
        X_train_0 = [X_train_0[i] for i in X_train_choice_0]
        Y_train_0 = [Y_train_0[i] for i in X_train_choice_0]
        M_train_0 = [M_train_0[i] for i in X_train_choice_0]
        X_train_1 = [X_train_1[i] for i in X_train_choice_1]
        Y_train_1 = [Y_train_1[i] for i in X_train_choice_1]
        M_train_1 = [M_train_1[i] for i in X_train_choice_1]

        
        
        X_train = X_train_0 + X_train_1
        Y_train = Y_train_0 + Y_train_1
        M_train = M_train_0 + M_train_1
        

        np.random.seed(42)
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = [X_train[i] for i in shuffle_idx]
        Y_train = [Y_train[i] for i in shuffle_idx]
        M_train = [M_train[i] for i in shuffle_idx]
        
        # X_train, Y_train, User_id_train, Trip_id_train, M_train
    
    
    #elif augs.roubust == 'envshift':
    #    # 随机种子
    #    np.random.seed(42)
    #    def envshift(data):
    #        # data: seq_len, d_dim
    #        ra = np.random.uniform(0,0.5)
    #        rb = np.random.uniform(0,1.5)
    #        len_p = int(24 * ra) + 1
    #        choice_id = np.random.choice(24, len_p, replace=False) 
    #        data[:,choice_id] = data[:,choice_id] * rb
    #        return data

    #    X_test_list = [] #[:, -24:]
    #
    #    for data_i in X_test:
    #        # data_i: seq_len,d_dim
    #        data_i[:,-24:] = envshift(data_i[:,-24:])
    #        X_test_list.append(data_i)
    #    X_test = X_test_list
       

    elif augs.roubust == 'noise_history':
        # X_train seq_len,d_dim
        def liner_inter(data,mask):
            seq_len, feat = data.shape
            # 初始化一个同样形状的数组用于存放插补后的结果
            data = copy.copy(data)
            data[mask.astype(bool)] = np.nan
            # 遍历每个批次和每个特征

            data_sequence = data
            data_sequence = pd.DataFrame(data_sequence)
            # data_sequence_ = data_sequence.interpolate(method='linear',limit_direction='both',limit = 26).fillna(0)
            data_sequence_ = data_sequence.interpolate(method='linear',limit_direction='both',limit = 26).fillna(0)
            return data_sequence_.values  
        # 随机把X_test中部分值赋值为所在列的均值
        X_test_list = []
        for data_i in X_test:
            # data_i: seq_len,d_dim
            len_p = int(data_i.shape[0] * augs.roubust_size) + 1 
            choice_id = np.random.choice(data_i.shape[0], len_p, replace=False) 
            mask = np.zeros(data_i.shape)
            mask[choice_id,:] = 1
            data_i = liner_inter(data_i,mask)
            X_test_list.append(data_i)
        X_test = X_test_list
    # 如果X_train除以batch_size余数是1，那么就删除最后一个
    
    if len(X_train) % batch_size == 1:
        X_train = X_train[:-1]
        Y_train = Y_train[:-1]
        M_train = M_train[:-1]

    # 划分训练集，验证集，测试集
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42,stratify=Y) # 固定的划分
    X_test, X_valid, Y_test, Y_valid = train_test_split(X_test, Y_test, test_size=0.5, random_state=42,stratify=Y)
    '''
    data = {'train': [X_train, Y_train, M_train], 'valid': [X_valid, Y_valid, M_valid], 'test': [X_test, Y_test, M_test]}
    modes = ['train', 'valid', 'test']
    shuffles = [True, False, False]
    dataloader_dict = {}
    for mode, shuffle in zip(modes, shuffles):
        dataloader_dict[mode] = _get_dataloader(data[mode], shuffle, batch_size, basic_model = augs.base_encoder)
        print('# of intances in {}: {}'.format(mode, len(dataloader_dict[mode].dataset)))
    return dataloader_dict

from functools import partial

def _get_dataloader(data, shuffle=True, batch_size=64, basic_model='gru'):

    T = [torch.Tensor(t) for t in data[0]]
    Y = torch.LongTensor([y for y in data[1]])
    M = [torch.Tensor(m) for m in data[2]]
    #data [T,Y(标签)]

    E = [torch.Tensor(t[:, -24:]) for t in T] #后24个是环境信息
    X_raw = [torch.Tensor(t[:, 6:12]) for t in T] #中间6个是原始的轨迹信息
    X_norm = [torch.Tensor(t[:, :6]) for t in T] #前6个是归一化的轨迹信息
    X = [torch.cat((x, e), dim=1) for x, e in zip(X_raw, E)] # 拼接轨迹信息，使用原始的数据
    #normed_traj, X_raw, label
    data = MyData(X, X_norm, Y, M)
    dataloader = DataLoader(data, batch_size=batch_size,
                            shuffle=shuffle,collate_fn= partial(collate_fn, basic_model=basic_model))
    return dataloader


def get_data_dir():
    return  os.getcwd()


def seed_torch(seed=1):
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) #
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

# 如何初始化
import torch.nn as nn
def weight_init(m):
    if isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None and m.bias.requires_grad:
            nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式 
    elif isinstance(m, nn.Conv1d):
        if m.weight.requires_grad:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None and m.bias.requires_grad:
            nn.init.constant_(m.bias, 0)
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm1d):
        if m.weight.requires_grad:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None and m.bias.requires_grad:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        # Initialize LSTM weights
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # 为什么是2
                
    elif isinstance(m, nn.LSTMCell):
        # Initialize LSTM weights
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    elif isinstance(m, nn.GRU):
        # Initialize GRU weights
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    elif isinstance(m, nn.GRUCell):
        # Initialize GRU weights
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

def average_dict_values(dict_list):
    # 初始化一个字典来存储累加的值和计数
    counts = 0
    sums_counts = {}
    for d in dict_list:  # 遍历字典列表
        counts += 1
        for outer_key, middle_dict in d.items():
            if outer_key not in sums_counts:
                sums_counts[outer_key] = {}
            for middle_key, inner_dict in middle_dict.items():
                if middle_key not in sums_counts[outer_key]:
                    sums_counts[outer_key][middle_key] = {}
                for inner_key, inner_value in inner_dict.items():
                    if inner_key not in sums_counts[outer_key][middle_key]:
                        sums_counts[outer_key][middle_key][inner_key] = np.zeros_like(inner_value)
                    sums_counts[outer_key][middle_key][inner_key] += inner_value
    # 计算平均值
    averages = {}
    for outer_key, middle_dict in sums_counts.items():
        averages[outer_key] = {}
        for middle_key, inner_dict in middle_dict.items():
            averages[outer_key][middle_key] = {}
            for inner_key, value_count in inner_dict.items():
                averages[outer_key][middle_key][inner_key] = value_count / counts
    return averages



def ave_png(reserve_dict,save_dir):
    names = list(reserve_dict.keys())# [ 'acc', 'confidence_d', 'confidence_i']
    method_styles = {
        'IB_top': {'color': 'darkred', 'linestyle': '-', 'linewidth': 2,'marker':'o','markersize':5},
    }
    
    # 创建子图
    fig, ax = plt.subplots(1, len(names), figsize=(12, 4))
    # 绘制每个指标的图
    for i, name in enumerate(names):
        for method, value in reserve_dict[name].items():
            style = method_styles.get(method, {'color': None, 'linestyle': '--', 'linewidth': 1,'marker':'o','markersize':3})
            ax[i].plot(value, label=method, color=style['color'], marker=style['marker'], linestyle=style['linestyle'], markersize=style['markersize'], linewidth=style['linewidth'])
        ax[i].set_title(name)
    ax[2].legend(loc='upper center', bbox_to_anchor=(1.25, 1.1),ncol=1)
    plt.savefig(save_dir, dpi=300, bbox_inches='tight')
    plt.close()
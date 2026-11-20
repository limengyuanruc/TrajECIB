#%%
import os
# os.chdir('/home/DATA2/limengyuan/MYCL')
import torch
import pickle
import math
import numpy as np
import torch.nn as nn
from base.model import BaseModel
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import random

from model.base_block import *
from model.enc_block import *
from model.my_base_block import *


class TrajCausal(BaseModel):
    def __init__(self, augs, name = 'gru', c_in=6, c_env=0, c_out=4, embed_dim=64, base_encoder='gru',
                 qk_dim=64, envbook_num=50, envbook_dim=64,
                 cat_or_add='cat', with_random=True, without_point_attention=False):
        super(TrajCausal, self).__init__()
        self.augs = augs
        self.name = name
        self.c_in = c_in # 输入的维度  3
        self.c_env = c_env # 环境的维度
        self.c_out = c_out # 输出的维度
        self.cat_or_add = cat_or_add # CL模型中的参数
        self.with_random = with_random # CL模型中的参数
        self.without_point_attention = without_point_attention # CL模型中的参数

        self.point_att = None
        self.perplexity = None
        self.env_selects = None

        ###############################################Traj Encoder###############################################
        # 
        self.conv1d_traj = nn.Sequential(nn.Conv1d(c_in, embed_dim, 3, 1, 1)) 
                                        
        self.conv1d_env = nn.Sequential(nn.Conv1d(c_env, embed_dim, 3, 1, 1))
        self.concat = nn.Linear(embed_dim * 2, embed_dim)            
        self.conv1d_all = nn.Sequential(nn.ReLU(inplace=True),
                                        nn.Conv1d(embed_dim, embed_dim, 3, 1, 1))
        # 步幅和填充都是1，输入和输出一致
        self.env_encoder = EnvCodebook(num_code = envbook_num, input_dim= embed_dim, dim_code = embed_dim,output_dim = embed_dim, prior = augs.env_prior)

        self.object_encoder = UniEncBlock(hidden_size=embed_dim, cell=base_encoder)
        # 使用base_encoder GRU、BiLSTM、GRU-D、STGN、TrajFormer
        
        self.o_meanblock = TimePoolBlock() # 平均池化
        
        self.object_readout_layer = ReadoutLayer(embed_dim, c_out) # 分类器


    def forward(self, _x,_norm,_m, _len): # 输入的数据和长度
        inputs = _x

        envs = inputs[..., 6: 6+self.c_env] # 环境的维度
        # batch_size,n_time,24
        trajs = inputs[..., :self.c_in] # 输入的维度 lat,lon,timedelta
        # batch_size,n_time,3

        dt = inputs[..., [2]] # 时间
        dd = inputs[..., [3]] # 距离

        if self.c_env==0:
            x = trajs
        elif self.c_env!=0:
            x = torch.cat((trajs, envs), dim=2)

        # batch_size, n_time, 6+24
        x = self.conv1d_traj(trajs.permute(0, 2, 1)).permute(0, 2, 1)     # b, n_time, embed_dim
        e = self.conv1d_env(envs.permute(0, 2, 1)).permute(0, 2, 1)       # b, n_time, embed_dim
        
        
        if self.augs.env_encoder>0:
            e_hat = self.env_encoder(e,_len)   
            codebook_loss = torch.zeros(1).to(e.device)   
        else:
            e_hat = e
            codebook_loss = torch.zeros(1).to(e.device) 
            
        x = self.concat(torch.cat((x, e_hat), dim=-1))                         # b, n_time, embed_dim
        # x = x + e_hat
        x = self.conv1d_all(x.permute(0, 2, 1)).permute(0, 2, 1)         # b, n_time, embed_dim


        # batch_size, n_time, embed_dim
        xo = self.object_encoder(x, dt, dd, _len,_m)
        # batch_size, n_time, embed_dim
        mean_o = self.o_meanblock(xo, _len)

        out_o_logis = self.object_readout_layer(mean_o)
        
        model_outs = {}
        #out_c_logis, out_o_logis, out_co
        model_outs['out_c_logis'] = torch.zeros_like(out_o_logis)
        model_outs['out_o_logis'] = out_o_logis
        model_outs['out_co'] = torch.zeros_like(out_o_logis)
        model_outs['codebook_loss'] = codebook_loss

        return model_outs
#%%
if __name__ == "__main__":
    

    model0 = TrajCausal(name = '1',
                        c_in = 3,
                        c_env = 24,
                        c_out = 4,
                        embed_dim =64,
                        base_encoder = 'gru',
                        qk_dim = 64,
                        envbook_num = 25,
                        envbook_dim = 64,
                        cat_or_add = 'add',
                        with_random = True,
                        without_point_attention = True,
                        )
    # 加载final_model_1998.pt
    model0.load_state_dict(torch.load('/home/DATA2/limengyuan/MYCL/logs/20250211/bj/2-bj-base-gru/1998/final_model_1998.pt'))
    x = torch.tensor([ 3.9974e+01,  1.1633e+02,  5.0000e+00,  3.3911e-01,  6.7821e-02,
        -4.0193e-02,  0.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,
         0.0000e+00,  2.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  1.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         4.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  4.0000e+00,  0.0000e+00])
    trajs = x[:3].reshape(1,3,1)
    envs = x[6:30].reshape(1,24,1)
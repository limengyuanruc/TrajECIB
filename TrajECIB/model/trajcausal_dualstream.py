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
from torch.distributions import  kl_divergence,Bernoulli
from model.base_block import *
from model.enc_block import *
from model.my_base_block import *

def KL_ber_fun(p_i):
    p = Bernoulli(p_i)#batch_size,seq_len,1
    q = Bernoulli(torch.ones_like(p_i)*0.001)  # .001
    KL_no_noisy = kl_divergence(p,q)##batch_size,seq_len 
    return KL_no_noisy.mean() #batch_size,seq_len,1 

class TrajCausal(BaseModel):
    def __init__(self, augs, name = 'gru', c_in=6, c_env=0, c_out=4, embed_dim=64, base_encoder='gru',
                 qk_dim=64, envbook_num=25,promot = 25, envbook_dim=64,
                  cat_or_add='cat', with_random=True, without_point_attention=False):
        super(TrajCausal, self).__init__()
        self.augs = augs
        self.name = name
        self.c_in = c_in
        self.c_env = c_env
        self.c_out = c_out

        self.cat_or_add = cat_or_add
        self.with_random = with_random
        self.without_point_attention = without_point_attention


        self.point_att = None
        self.perplexity = None

        self.env_selects = None

        ###############################################Traj Encoder###############################################
        # 
        # self.conv1d_traj = nn.Sequential(nn.Conv1d(c_in+c_env, embed_dim, 3, 1, 1), nn.ReLU(inplace=True))

        self.conv1d_traj = nn.Sequential(nn.Conv1d(c_in + augs.motion_dim, embed_dim, 3, 1, 1)) 
                                        
        self.conv1d_env = nn.Sequential(nn.Conv1d(c_env, embed_dim, 3, 1, 1))
        self.concat = nn.Linear(embed_dim * 2, embed_dim)            
        self.conv1d_all = nn.Sequential(nn.ReLU(inplace=True),
                                        nn.Conv1d(embed_dim, embed_dim, 3, 1, 1))

        self.env_bn = nn.BatchNorm1d(c_env)
        self.x_bn = nn.BatchNorm1d(embed_dim)
        
        self.env_encoder = EnvCodebook(num_code = envbook_num, input_dim= embed_dim, dim_code = embed_dim,output_dim = embed_dim, prior = augs.env_prior)
        self.promtdecoder = PromtDecoder(num_code = promot, input_dim= embed_dim, dim_code = embed_dim,output_dim = embed_dim)
        ###############################################Environment Extracter###############################################
        # 
        self.PointAttention = PointAttention(augs, embed_dim, qk_dim)
            
        ###############################################Context ENC and Object ENC###############################################
        # 
        self.context_encoder = UniEncBlock(hidden_size=embed_dim, cell=base_encoder)
        self.object_encoder = UniEncBlock(hidden_size=embed_dim, cell=base_encoder)

        self.c_meanblock = TimePoolBlock()
        self.o_meanblock = TimePoolBlock()

        self.context_readout_layer = ReadoutLayer(embed_dim, c_out)
        self.object_readout_layer = ReadoutLayer(embed_dim, c_out)
        if self.augs.co>0:
            self.random_readout_layer = RandomReadoutLayer(embed_dim, c_out, cat_or_add, with_random)


        self.ODL_loss = ODL_loss(augs = augs, hidden_dim = embed_dim, class_size = c_out)

    def IB_loss(self, p_i):
        p_i = torch.clamp(p_i, 0.001, 0.999)
        IB_loss = KL_ber_fun(p_i) 
        entropy_loss = -p_i * torch.log(p_i + 1e-5)- (1 - p_i) * torch.log(1 - p_i+ 1e-5)
        return IB_loss + entropy_loss.mean()
    def Cat_loss(self, p_i):
        p_i = torch.clamp(p_i, 0.001, 0.999)
        cat_loss = p_i * torch.log(p_i + 1e-5) + p_i.shape[1]
        return cat_loss.mean()
    
    def forward(self, _x,_norm,_m, _len,Y):
        inputs = _x
        # print(inputs.shape)

        envs = inputs[..., 6: 6+self.c_env] # 环境的维度
        # batch_size,n_time,24
        trajs = inputs[..., :self.c_in] # 输入的维度 lat,lon,timedelta
        # batch_size,n_time,3

        dt = inputs[..., [2]] # 时间
        dd = inputs[..., [3]] # 距离

        b, n_time, _ = trajs.shape

        envs = self.env_bn(envs.permute(0,2,1)).permute(0,2,1)

        ############################################### env input ###############################################
        # only basicfeatures
        if self.c_env==0:
            x = trajs
            e = trajs
        elif self.c_env!=0:
            x = trajs
            e = envs
        
        if self.augs.motion_dim==2:
            motion = _norm[..., [4,5]]
            x = torch.cat((x, motion), dim=2)
        if self.augs.motion_dim==3:
            motion = _norm[..., [3,4,5]]
            x = torch.cat((x, motion), dim=2)   
        ############################################### input proj ###############################################
        # batch_size, n_time, 6+24
        x = self.conv1d_traj(x.permute(0, 2, 1)).permute(0, 2, 1)     # b, n_time, embed_dim
        e = self.conv1d_env(envs.permute(0, 2, 1)).permute(0, 2, 1)       # b, n_time, embed_dim
        
        if self.augs.env_encoder>0:
            e_hat = self.env_encoder(e,_len)   
        else:
            e_hat = e
        
        
        
        x_e = self.concat(torch.cat((x, e_hat), dim=-1))                         # b, n_time, embed_dim
        # x_e = x + e_hat
        x_e = self.conv1d_all(x_e.permute(0, 2, 1)).permute(0, 2, 1)         # b, n_time, embed_dim
        x_e = self.x_bn(x_e.permute(0, 2, 1)).permute(0, 2, 1)               # b, n_time, embed_dim
        
        
        ############################################### 不变学习 ###############################################

            
        if self.augs.train_ib:
            # query = self.quary_encoder(x_e, dt, dd, _len,_m) # query
            point_att, env_selects = self.PointAttention(x_e, x_e, _m,_len, train_ib = self.augs.train_ib, att_mlp = self.augs.att_mlp,  local_att = self.augs.local_att) # b, n_time, 2
            self.env_selects = env_selects # attention的分数
            IB_loss = self.IB_loss(env_selects) #KL_ber_fun(env_selects)
            # Cat_loss = self.Cat_loss(env_selects)
            
        else: 
            point_att = torch.ones(b, n_time, 1).to(inputs[:,:,[0]].device)   # context

            point_att_inv = 1 - point_att
            point_att = torch.cat((point_att, point_att_inv), dim=-1) # b, n_time, 2
            IB_loss = torch.tensor(0).to(inputs.device)
            # Cat_loss = torch.tensor(0).to(inputs.device)
        # for tensorboard record
        self.point_att = point_att

        ############################################### intervention ###############################################
        # 标准正太分布
        # std, mean = torch.std_mean(x_e, keepdim=True) # 
        if self.training:
            x_eo = point_att[:, :, [0]] * x_e + point_att[:, :, [1]] * torch.randn_like(x_e) 
            x_ec = point_att[:, :, [1]] * x_e + point_att[:, :, [0]] * torch.randn_like(x_e) # b, n_time, embed_dim
        else:
            x_eo = point_att[:, :, [0]] * x_e
            x_ec = point_att[:, :, [1]] * x_e



        if self.augs.promtd:
            xo = self.promtdecoder(x_eo,_len) # 增加了强化操作
        else:
            xo = x_eo
        xo, else_loss = self.object_encoder(xo, dt, dd, _len,_m)
        xc,_ = self.context_encoder(x_ec, dt, dd, _len,_m)


        epsilon = 1e-5
        mean_o = self.o_meanblock(xo, _len)  # b, embed_dim
        mean_c = self.c_meanblock(xc, _len)  # b, embed_dim
        
        loss_odlo = self.ODL_loss(mean_o, Y)
        loss_odlc = self.ODL_loss(mean_c, None)
        
        out_o_logis = self.object_readout_layer(mean_o) # ReadoutLayer
        out_c_logis = self.context_readout_layer(mean_c) # ReadoutLayer
        if self.augs.co>0:
            out_co = self.random_readout_layer(mean_c, mean_o) # RandomReadoutLayer
        else:
            out_co = out_o_logis.detach().clone() # 只输出object的logits
        
        model_outs = {}
        model_outs['out_c_logis'] = out_c_logis
        model_outs['out_o_logis'] = out_o_logis
        model_outs['out_co'] = out_co  # 只输出object的logits
        model_outs['IB_loss'] = IB_loss
        model_outs['loss_odlo'] = loss_odlo
        model_outs['loss_odlc'] = loss_odlc
        model_outs['emb'] = xo
        model_outs['mean_o'] = mean_o # causal embedding
        model_outs['mean_s'] = mean_c # spurious embedding
        model_outs['p_mean'] = point_att[:, :, [0]].mean()
        model_outs['else_loss'] = else_loss
        return model_outs
    







class TrajCausal_Decoder(BaseModel):
    def __init__(self, augs, model,name = 'gru', c_in=6, c_env=0, c_out=4, embed_dim=64, base_encoder='gru',
                 qk_dim=64, envbook_num=50, envbook_dim=64,
                  cat_or_add='cat', with_random=True, without_point_attention=False):
        super(TrajCausal_Decoder, self).__init__()
        self.model = model
        self.augs = augs
        self.name = name
        self.c_in = c_in
        self.c_env = c_env
        self.c_out = c_out

        self.cat_or_add = cat_or_add
        self.with_random = with_random
        self.without_point_attention = without_point_attention


        self.point_att = None
        self.perplexity = None

        self.env_selects = None

        ###############################################Traj Encoder###############################################
        # 
        # self.conv1d_traj = nn.Sequential(nn.Conv1d(c_in+c_env, embed_dim, 3, 1, 1), nn.ReLU(inplace=True))

        self.conv1d_traj = nn.Sequential(nn.Conv1d( embed_dim,embed_dim, 3, 1, 1)) 
                                        
        self.conv1d_env = nn.Sequential(nn.Conv1d( embed_dim,embed_dim, 3, 1, 1))
        self.concat = nn.Linear(embed_dim, embed_dim * 2)            
        self.conv1d_all = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, 3, 1, 1),
                                        nn.ReLU(inplace=True))

        self.env_bn = nn.BatchNorm1d(embed_dim)
        self.x_bn = nn.BatchNorm1d(embed_dim)
        
        self.env_encoder = EnvCodebook(num_code = envbook_num, input_dim= embed_dim, dim_code = embed_dim,output_dim = embed_dim, prior = augs.env_prior)
        if self.augs.promtd_lendy:
            self.promtdecoder = PromtDecoder_Dynamic(num_code = envbook_num, input_dim= embed_dim, dim_code = embed_dim,output_dim = embed_dim)
        else:
            self.promtdecoder = PromtDecoder(num_code = envbook_num, input_dim= embed_dim, dim_code = embed_dim,output_dim = embed_dim)

        self.object_encoder = UniEncBlock(hidden_size=embed_dim, cell=base_encoder)
        self.criterion = nn.MSELoss()

        self.pred1 =  nn.Linear(embed_dim,c_in)
        self.pred2 =  nn.Linear(embed_dim,c_env)
    
    def forward(self, _x,_m, _len,Y):
        emb = self.model(_x,_m, _len,Y)['emb']
        ############原始的数据########################
        inputs = _x

        envs = inputs[..., 6: 6+self.c_env] # 环境的维度
        trajs = inputs[..., :self.c_in] # 输入的维度 lat,lon,timedelta
        dt = inputs[..., [2]] # 时间
        dd = inputs[..., [3]] # 距离

        b, n_time, _ = trajs.shape
        
        ###########新的数据##########################
        x_e = self.object_encoder(emb, dt, dd, _len,_m)
        
        x_e = self.x_bn(x_e.permute(0, 2, 1)).permute(0, 2, 1) 
        x_e = self.conv1d_all(x_e.permute(0, 2, 1)).permute(0, 2, 1) 
        
        x_e = self.concat(x_e)   # batch_size, seq_len, d_model * 2

        x = x_e[:,:,:(x_e.shape[-1] // 2)]
        e = x_e[:,:,(x_e.shape[-1] // 2):]


        trajs_hat = self.conv1d_traj(x.permute(0, 2, 1)).permute(0, 2, 1)     # b, n_time, embed_dim
        envs_hat = self.conv1d_env(e.permute(0, 2, 1)).permute(0, 2, 1)       # b, n_time, embed_dim
        
        envs_hat = self.env_bn(envs_hat.permute(0,2,1)).permute(0,2,1)

        trajs_hat = self.pred1(trajs_hat)
        envs_hat = self.pred2(envs_hat)

        loss1 = self.criterion(trajs_hat,trajs)
        loss2 = self.criterion(envs_hat,envs)
              

        return loss1 + 0.1 * loss2
    
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
                        without_point_attention = False,
                        )
    # 加载final_model_1998.pt
    # model0.load_state_dict(torch.load('/home/DATA2/limengyuan/MYCL/logs/20250211/bj/2-bj-base-gru/1998/final_model_1998.pt'))
    x = torch.tensor([ 3.9974e+01,  1.1633e+02,  5.0000e+00,  3.3911e-01,  6.7821e-02,
        -4.0193e-02,  0.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,
         0.0000e+00,  2.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  1.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         4.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  4.0000e+00,  0.0000e+00])
    trajs = x[:3].reshape(1,3,1)
    envs = x[6:30].reshape(1,24,1)
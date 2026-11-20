import torch
import pickle
import math
import numpy as np
import torch.nn as nn
from base.model import BaseModel
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import random
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# from timm.models.layers import trunc_normal_


class VQCrossAttenSeg(nn.Module):
    def __init__(self, in_channels, qk_dim, envbook_num, envbook_dim):
        super(VQCrossAttenSeg, self).__init__()
        # c_env = 24, qk_dim = 64, envbook_num = 25, envbook_dim =64

        self.in_channels = in_channels
        self.qk_dim = qk_dim
        self.envbook_num = envbook_num
        self.envbook_dim = envbook_dim

        self.scaling = qk_dim ** -0.5
        
        # Learnable embedding (n*d)
        self.env_codebook = nn.Parameter(torch.randn(envbook_num, envbook_dim)) # 环境编码本
        # trunc_normal_(self.env_codebook, std=.02)
        
        # Query, key, value projections for attention
        self.query_proj = nn.Linear(in_channels, qk_dim, bias=False)
        self.key_proj = nn.Linear(envbook_dim, qk_dim, bias=False)
        self.value_proj = nn.Linear(envbook_dim, 1, bias=False) # 每个环境编码本的权重

    def forward(self, x):
        # x is the input tensor of shape [batch_size, sequence_length, in_channels]
        
        # Project trajectories to query, and embeddings to key and value

        query = self.query_proj(x) # Q是环境的线性变换  [batch_size, sequence_length, qk_dim]
        key = self.key_proj(self.env_codebook) # K是环境编码本的线性变换 # [envbook_num, qk_dim]
        value = self.value_proj(self.env_codebook) # V是环境编码本的线性变换  # [envbook_num, qk_dim]
        
        # Compute attention scores
        # Shape: [batch_size, sequence_length, envbook_num]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling  
        # # Shape: [batch_size, sequence_length, envbook_num]
        
        # Sample from the Gumbel-Softmax distribution to get hard assignments
        env_selects = F.gumbel_softmax(attention_scores, hard=True, dim=-1) # [batch_size, sequence_length, envbook_num]
        
        # Optionally multiply the probabilities with the value embeddings
        # This step depends on how you want to use the selected embeddings
        env_att = torch.matmul(env_selects, value)  # [batch_size, sequence_length, 1]

        env_att = torch.sigmoid(env_att)
        
        return env_att, env_selects
    


class ReadoutLayer(nn.Module):
    def __init__(self, embed_dim, c_out):
        super(ReadoutLayer, self).__init__()
        self.fc1_bn = nn.BatchNorm1d(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2_bn = nn.BatchNorm1d(embed_dim)
        self.fc2 = nn.Linear(embed_dim, c_out)
        
    def forward(self, x):
        x = self.fc1_bn(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2_bn(x)
        x = self.fc2(x)
        # x_logis = F.log_softmax(x, dim=-1) # log_softmax
        return x


class RandomReadoutLayer(nn.Module):
    def __init__(self, embed_dim, c_out, cat_or_add, with_random):
        super(RandomReadoutLayer, self).__init__()
        self.with_random = with_random
        self.cat_or_add = cat_or_add

        # Random mlp - do(x) Causal intervention
        if self.cat_or_add == "cat":
            self.fc1_bn_co = nn.BatchNorm1d(embed_dim * 2)
            self.fc1_co = nn.Linear(embed_dim * 2, embed_dim)
        elif self.cat_or_add == "add": ####!!!###
            self.fc1_bn_co = nn.BatchNorm1d(embed_dim)
            self.fc1_co = nn.Linear(embed_dim, embed_dim)
        else:
            assert False
        self.fc2_bn_co = nn.BatchNorm1d(embed_dim)
        self.fc2_co = nn.Linear(embed_dim, c_out)
        
    def forward(self, xc, xo):

        num = xc.shape[0]                   # bs
        l = [i for i in range(num)]
        if self.with_random:
            random.shuffle(l)
        random_idx = torch.tensor(l)        
        if self.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)   # bs, xc_feat+xo_feat (2*embed_dim)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        # x_logis = F.log_softmax(x, dim=-1)
        # return x_logis
        return x

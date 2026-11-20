import torch
import pickle
import math
import numpy as np
import torch.nn as nn
from base.model import BaseModel
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import random
import itertools

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops import rearrange, repeat
def log(t, eps = 1e-5):
    return t.clamp(min = eps).log()
def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)

class EnvCodebook(nn.Module):
    # inspired by https://github.com/YunshiWen/VQShape/blob/main/vqshape/model.py
    def __init__(
            self,
            input_dim: int = 64,
            output_dim: int = 64,
            num_code: int = 512,
            dim_code: int = 256,
            prior: str = 'uniform',
        ):
        super().__init__()
        self.num_codebook_vectors = num_code
        self.latent_dim = dim_code
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior = prior
        self.env_codebook = nn.Parameter(torch.randn(self.num_codebook_vectors, self.latent_dim)) # S = {s1,...,sK}
        if self.prior == 'learnable':
            # 可学习环境先验：每个环境代码对应一个先验logit，softmax归一化为分布 pi(k)
            # 零初始化 => pi(k) = 1/K，初始与均匀先验严格等价，先验偏置由训练逐渐学出
            self.env_prior_logits = nn.Parameter(torch.zeros(self.num_codebook_vectors))
            print("Using learnable prior for environment codebook")

        self.nheads = 2
        self.head_dim = self.latent_dim // self.nheads
        
        
        self.query_proj = nn.Linear(self.input_dim, self.latent_dim, bias=False) # W_q
        self.key_proj = nn.Linear(self.latent_dim, self.latent_dim, bias=False) # W_k
        self.value_proj = nn.Linear(self.latent_dim, self.latent_dim, bias=False) # W_v
        self.scaling = self.head_dim ** -0.5

        
        self.W_o = nn.Linear(self.latent_dim, self.output_dim) # W_o
        self.layernorm = nn.LayerNorm(self.latent_dim)

        # 环境代码本的初始化
    def forward(self, z, _len):

        batch_size, seq_len,_  = z.shape
        
        # mask = torch.arange(seq_len, device=z.device).expand(batch_size, seq_len) >= torch.tensor(_len, device=z.device).unsqueeze(1)
        # bath_size, seq_len
        # 有效位置是1，无效位置是0
        
        query = self.query_proj(z) # batch_size, seq_len, embed_dim
        key = self.key_proj(self.env_codebook) # num_codebook_vectors, embed_dim
        value = self.value_proj(self.env_codebook) # num_codebook_vectors, embed_dim
        
        # 拆分为多个头
        query = rearrange(query, 'b q (h d) -> b h q d', h=self.nheads) # batch_size, nheads, seq_len, head_dim
        key = rearrange(key, 'k (h d) -> h k d', h=self.nheads) # nheads, num_codebook_vectors, head_dim
        if self.prior == 'learnable':
            prior_w = F.softmax(self.env_prior_logits, dim=0) # num_codebook_vectors
            value = rearrange(value, 'k (h d) -> h k d', h=self.nheads) * prior_w.view(1, -1, 1) # 先验pi(k)在各头间共享
        else:
            value = rearrange(value, 'k (h d) -> h k d', h=self.nheads) / self.num_codebook_vectors # 均匀先验 1/K
        
        # 计算注意力分数
        attention_scores = torch.einsum('b h q d, h k d -> b h q k', query, key) * self.scaling 
        # attention_scores = attention_scores.masked_fill(mask.unsqueeze(1).unsqueeze(-1), -1e9)
        # batch_size, nheads, seq_len, num_codebook_vectors
        attention_scores = F.sigmoid(attention_scores) # 在seq_len维度上进行softmax
        
        # 计算注意力
        # value = value * 1 / self.num_codebook_vectors # 注释掉
        z_q = torch.einsum('b h q k, h k d -> b h q d', attention_scores, value) 
        z_q = rearrange(z_q, 'b h q d -> b q (h d)')
        z_q = self.W_o(z_q)
        z_q = self.layernorm(z_q)  + z
        return z_q
    







class PromtDecoder(nn.Module):
    def __init__(
            self, 
            input_dim: int = 64,
            output_dim: int = 64,
            num_code: int = 512, 
            dim_code: int = 256, 
        ):
        super().__init__()
        self.num_codebook_vectors = num_code
        self.latent_dim = dim_code
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.env_codebook = nn.Parameter(torch.randn(self.num_codebook_vectors, self.latent_dim)) # S = {s1,...,sK}
        self.nheads = 2
        self.muliti = nn.MultiheadAttention(input_dim, num_heads = self.nheads,batch_first=True)
        self.layernorm = nn.LayerNorm(self.latent_dim)

        # 环境代码本的初始化
    def forward(self, z,_len):
        # print(z.shape)
        dim_size = len(z.shape)
        query = z.unsqueeze(1) if dim_size == 2 else z
        key = self.env_codebook.unsqueeze(0).repeat(z.shape[0],1,1) # 1, num_codebook_vectors, embed_dim
        value = self.env_codebook.unsqueeze(0).repeat(z.shape[0],1,1) # 1, num_codebook_vectors, embed_dim
        z_q, _ = self.muliti(query, key, value) # batch_size, 1, embed_dim
        z_q = z_q.squeeze(1) if dim_size == 2 else z_q
        z_q = self.layernorm(z_q) + z
        return z_q


    
class PromtDecoder_Dynamic(nn.Module):
    def __init__(
            self, 
            input_dim: int = 64,
            output_dim: int = 64,
            num_code: int = 512, 
            dim_code: int = 256, 
        ):
        '''
        实现 class PromtDecoder_Dynamic部分的代码，满足审稿人要求：
        8. The front-door prompt decoder adopts fixed learnable prompt vectors 
        without adaptive update according to input trajectories. 
        Would dynamic prompt generation conditioned on trajectory length 
        further strengthen causal feature separation?
        
        '''
        super().__init__()
        self.num_codebook_vectors = num_code
        self.latent_dim = dim_code
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.env_codebook = nn.Parameter(torch.randn(self.num_codebook_vectors, self.latent_dim)) # S = {s1,...,sK}
        self.nheads = 2
        self.muliti = nn.MultiheadAttention(input_dim, num_heads = self.nheads,batch_first=True)
        self.layernorm = nn.LayerNorm(self.latent_dim)

        # 动态prompt生成：以轨迹长度 _len 为条件，通过超网络生成 FiLM 调制参数 (gamma, beta)，
        # 对代码本做逐轨迹仿射变换 s_k(l) = (1 + gamma_k(l)) * s_k + beta_k(l)
        self.len_dim = 64 # 轨迹长度正弦编码维度
        self.len_hyper = nn.Sequential(
            nn.Linear(self.len_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, 2 * self.latent_dim), # 同时输出 gamma 和 beta
        )
        # 超网络末层零初始化：训练初期 gamma=0, beta=0，动态prompt严格退化为静态代码本，
        # 即初始状态与 PromtDecoder 完全等价，长度条件的影响由训练逐渐学出
        nn.init.zeros_(self.len_hyper[-1].weight)
        nn.init.zeros_(self.len_hyper[-1].bias)

        # 环境代码本的初始化
    def _len_encoding(self, _len, device):
        # _len: 每条轨迹的有效长度 (batch_size,)
        # 对长度做多频正弦编码：无参数、不受最大长度上限约束、对不同数据集的长度范围鲁棒
        lengths = torch.as_tensor(_len, dtype=torch.float, device=device).view(-1, 1) # batch_size, 1
        lengths = torch.log1p(lengths) # log(1+l) 压缩长度动态范围
        half = self.len_dim // 2
        freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float) * (-math.log(10000.0) / max(half - 1, 1))) # half
        angles = lengths * freqs # batch_size, half
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) # batch_size, len_dim

    def forward(self, z,_len):
        dim_size = len(z.shape)
        query = z.unsqueeze(1) if dim_size == 2 else z

        # 长度条件 -> FiLM 调制参数
        len_feat = self._len_encoding(_len, z.device) # batch_size, len_dim
        gamma, beta = self.len_hyper(len_feat).chunk(2, dim=-1) # batch_size, latent_dim
        gamma = (1.0 + gamma).unsqueeze(1) # batch_size, 1, latent_dim
        beta = beta.unsqueeze(1) # batch_size, 1, latent_dim

        # 动态prompt：每条轨迹根据自身长度对代码本进行自适应调制，替代原始的固定可学习prompt
        prompts = gamma * self.env_codebook.unsqueeze(0) + beta # batch_size, num_codebook_vectors, latent_dim

        z_q, _ = self.muliti(query, prompts, prompts) # batch_size, 1 | seq_len, embed_dim
        z_q = z_q.squeeze(1) if dim_size == 2 else z_q
        z_q = self.layernorm(z_q) + z
        return z_q


def gumbel_sigmoid(logits, tau=1.0, eps=1e-10):
    """
    使用Gumbel-Sigmoid从logits中采样。logits是输入值，tau是温度，eps避免log(0)的错误。

    参数：
        logits: 输入logits，形状 [batch_size, num_features]
        tau: 温度参数，控制采样的平滑度
        eps: 防止数值错误的小常数

    返回：
        从Gumbel-Sigmoid分布采样后的值，形状与输入logits相同
    """
    # 1. 生成Gumbel噪声
    u = torch.rand_like(logits)  # 均匀分布[0, 1)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    # 2. 将Gumbel噪声与logits结合
    y = logits + gumbel_noise
    # 3. 应用Sigmoid函数并返回
    return torch.sigmoid(y / tau)

def concrete_sample(self, log_alpha, beta = 1.0, training = True):
    if training:
        eps=1e-10
        random_noise = torch.rand(log_alpha.shape)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
        gate_inputs = gate_inputs.sigmoid()

    else:
        gate_inputs = log_alpha

    return gate_inputs





class PointAttention(nn.Module):
    def __init__(self, augs, in_channels = 64, qk_dim = 64):
        super(PointAttention, self).__init__()
        self.nheads = 2
        self.augs = augs
        self.head_dim = qk_dim // self.nheads
        self.query_proj = nn.Linear(in_channels, qk_dim, bias=False)
        self.key_proj = nn.Linear(in_channels, qk_dim, bias=False)
        self.value_proj = nn.Linear(in_channels, qk_dim, bias=False) 
        
        
        self.query_proj_m = nn.Linear(in_channels, qk_dim, bias=False)
        self.key_proj_m = nn.Linear(in_channels, qk_dim, bias=False)
        self.value_proj_m = nn.Linear(in_channels, qk_dim, bias=False) 
        
        self.scaling = self.head_dim ** -0.5
        self.layernorm = nn.LayerNorm(qk_dim)
        self.output = nn.Linear(qk_dim, 2)



        self.mlp = nn.Sequential(
            nn.Linear(qk_dim, qk_dim),   # 输入为10维，输出为20维的全连接层
            nn.ReLU(),           # ReLU 激活函数
            nn.Linear(qk_dim, 2)     # 输入为20维，输出为1维的全连接层
        )
        if self.augs.att_mlp == 'cnn':
            self.cnn = nn.Conv1d(
            in_channels=qk_dim,    # 根据你的特征通道修改
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2)  # 保证输入输出长度相同
        if self.augs.att_mlp == 'gru':
            self.gru = nn.GRU(
            qk_dim,        # 每个时间步输入特征维度
            2,       # GRU隐层维度
            num_layers=1,      # GRU层数
            bias=True,
            batch_first=True, # 设True：输入[batch, seq_len, feat]，更常用             
            )
    def forward(self, x, q, _m, _len, train_ib = 1, att_mlp = 1, local_att = 1):
        # x: batch_size, seq_len, embed_dim
        # _m: batch_size, seq_len //2, seq_len 
        batch_size, seq_len,_  = x.shape  

        mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) >= torch.tensor(_len, device=x.device).unsqueeze(1)  

        if att_mlp=='0':
            res = self.layernorm(q)

            
            query = self.query_proj(q)  
            key = self.key_proj(x)
            value = self.value_proj(x)
            
            # 拆分为多个头
            query = rearrange(query, 'b q (h d) -> b h q d', h=self.nheads) # batch_size, nheads, seq_len, head_dim
            key = rearrange(key, 'b k (h d) -> b h k d', h=self.nheads) # nheads, num_codebook_vectors, head_dim
            value = rearrange(value, 'b k (h v) -> b h k v', h=self.nheads) # nheads, num_codebook_vectors, head_dim
            
            
            # 全局注意力
            attention_scores_gobal = torch.einsum('b h q d, b h k d -> b h q k', query, key) * self.scaling # batch_size, nheads, seq_len, seq_len
            # attention_scores_gobal = attention_scores_gobal.masked_fill(mask.unsqueeze(1).unsqueeze(1), -1e9)# batch_size, 1, 1, seq_len
            attention_scores_gobal = F.softmax(attention_scores_gobal, dim=-1)
            attention_gobal = torch.einsum('b h q k, b h k v -> b h q v', attention_scores_gobal, value)
            attention_gobal = rearrange(attention_gobal, 'b h q v -> b q (h v)')
            attention_gobal = self.layernorm(attention_gobal) 
            # attention_gobal = 0
            
            
            # 局部注意力
            if local_att == 1:
                query = self.query_proj_m(q)
                key = self.key_proj_m(x)
                value = self.value_proj_m(x)
                
                query = rearrange(query, 'b q (h d) -> b h q d', h=self.nheads) # batch_size, nheads, seq_len, head_dim
                key = rearrange(key, 'b k (h d) -> b h k d', h=self.nheads) # nheads, num_codebook_vectors, head_dim
                value = rearrange(value, 'b k (h v) -> b h k v', h=self.nheads) # nheads, num_codebook_vectors, head_dim

                key_new = torch.einsum('b m q, b h q d -> b h m d', _m, key) # batch_size, seq_len, seq_len
                value_new = torch.einsum('b m q, b h q v -> b h m v', _m, value) # batch_size, seq_len, 1
                
                attention_scores_local =  torch.einsum('b h q d, b h m d -> b h q m', query, key_new) * self.scaling
                attention_probs_local = F.softmax(attention_scores_local, dim=-1)
                attention_local = torch.einsum('b h q m, b h m v -> b h q v', attention_probs_local, value_new)
                attention_local = rearrange(attention_local, 'b h q v -> b q (h v)')
                attention_local = self.layernorm(attention_local)
            else:
                attention_local = 0 # 增加了局部注意力

            # 融合全局和局部
            attention = self.output(attention_gobal + attention_local + res) # batch_size, seq_len, 1
        
        elif att_mlp=='1':
            attention = self.mlp(x)
        elif att_mlp == 'cnn':
            x_per = x.permute(0,2,1)
            attention = self.cnn(x_per)
            attention = attention.permute(0,2,1)
        elif att_mlp == 'gru':
            attention, _ = self.gru(x)
        
            
        if self.training & train_ib:
            env_selects = F.softmax(attention, dim=-1)
            point_att = F.gumbel_softmax(attention, tau=self.augs.tau, hard=True, dim=-1)
        else:
            env_selects = F.softmax(attention, dim=-1)
            point_att = F.softmax(attention, dim=-1)
        # 将mask掉的部分置为0
        point_att = point_att.masked_fill(mask.unsqueeze(-1), 0) # batch_size, seq_len, 2
        env_selects = env_selects.masked_fill(mask.unsqueeze(-1), 0) # batch_size, seq_len, 2
        return point_att, env_selects[:,:,[0]]



class ODL_loss(nn.Module):
    def __init__(self,      
        augs,
        hidden_dim = 64,
        class_size = 4,
        lambda_inter = 0.1,
        lambda_ = 0.01,
        m = 0.1,
        ):
        super(ODL_loss, self).__init__()
        self.class_center = nn.Embedding(class_size, hidden_dim)
        self.class_size = class_size
        self.hidden_dim = hidden_dim
        self.lambda_inter = augs.lambda_inter
        self.lambda_ = augs.lambda_
        self.m = augs.m
        self.ausgs = augs

    def forward(self, z, y = None):
        # z: batch_size, embed_dim
        # y: batch_size
        if y is not None:
            # y = y.cpu()
            batch_size, _ = z.shape

            centers = self.class_center(y) # batch_size, embed_dim

            # 计算类内距离
            intra_distance = torch.sum((z - centers) ** 2, dim=-1) # batch_size
            L_Com = intra_distance.mean() # 希望和center的距离越小越好
            # print(L_Com, 'L_Com')
            
            y_onehot = F.one_hot(y, self.class_size).bool() # batch_size, class_size
            all_class = torch.arange(self.class_size).unsqueeze(0).repeat(batch_size,1).to(y.device)
            negtive_class = all_class[~y_onehot].view(batch_size, -1) # batch_size, class_size - 1

            v_i = self.class_center(y) - z # batch_size, embed_dim
            v_j = self.class_center(negtive_class) - z.unsqueeze(1) # batch_size, class_size - 1, embed_dim

            Dis = -torch.sum(v_j ** 2, dim=-1) # batch_size, class_size - 1
            Dis = F.softmax(Dis, dim=-1) # batch_size, class_size - 1

            cos_sim = F.cosine_similarity(v_i.unsqueeze(1), v_j, dim=-1) # batch_size, class_size - 1
            L_Dis = (1 - cos_sim) * Dis # batch_size, class_size - 1
            # print(Dis.shape)
            L_Dis = L_Dis.sum(dim=-1).mean() 
            # print(L_Dis, 'L_Dis')
            if self.ausgs.cos:
                L_Dis = 0
            L_intra = L_Com   + 0.01 * L_Dis # 注释了

            # 计算类间距离
            
            all_centers = self.class_center.weight  # 所有类别的中心 (num_classes, embed_dim)
            distance_matrix = torch.cdist(all_centers, all_centers) ** 2 # (num_classes, num_classes)
            # print(distance_matrix.shape)
            min_distances = distance_matrix + torch.eye(self.class_size).to(z.device) * 1000 # 防止对角线计算自己与自己之间的距离
            min_distances, min_class_j = min_distances.min(dim=1)  # (num_classes, )
            # print(min_class_j)
            # 计算类间分离损失 L_Sep
            separation_loss = torch.max(self.m - min_distances, torch.zeros_like(min_distances))  # 每个类别的最小距离与边际 m 的差值
            L_Sep = separation_loss.mean()  # 对所有类别的损失进行求平均
            if self.class_size > 2:
                c_i = self.class_center.weight # 
                c_j = self.class_center(min_class_j)
                v_s = c_j - c_i   # c_j - c_i

                one_hot_encoded = F.one_hot(min_class_j, num_classes=self.class_size) + torch.eye(self.class_size).to(min_class_j.device) * 1
                one_hot_encoded = one_hot_encoded.bool().reshape(c_i.shape[0],-1)
                all_class = torch.arange(self.class_size).unsqueeze(0).repeat(c_i.shape[0],1).to(one_hot_encoded.device)
                o_class = all_class[~one_hot_encoded].view(c_i.shape[0], -1) # batch_size, class_size - 2
                v_o = self.class_center(o_class) - c_i.unsqueeze(1) # c_o - c_i

                D_oj = -torch.sum((self.class_center(o_class) - c_j.unsqueeze(1)) ** 2, dim=-1) # batch_size, class_size - 2
                D_oj = F.softmax(D_oj, dim=-1) # batch_size, class_size - 2

                cos_theta = F.cosine_similarity(v_s.unsqueeze(1), v_o, dim=-1) # batch_size, class_size - 2
                orientation_loss = (1 + cos_theta) * D_oj # batch_size, class_size - 2
                orientation_loss = orientation_loss.sum(dim=-1).mean()


            else:
                orientation_loss = 0


            '''
            复合 OOD 损失（compound OOD loss）包含
            类内紧致性（intra-class compactness）、
            类间分离性（inter-class separation）以及
            伪特征排斥项（spurious feature repulsion）三个部分。
            
            您是否可以分别对每个子损失项进行独立的消融实验（ablation study），
            以明确哪一个单独的损失项对提升模型的抗噪声（anti-noise）
            抗类别不平衡（anti-imbalance）鲁棒性贡献最大？ 
            '''

            if self.ausgs.cos:
                orientation_loss = 0
            
            L_inter = L_Sep  + 0.01 *  orientation_loss # 注释了
            if self.ausgs.intra == 0:
                L_intra = 0
            if self.ausgs.inter == 0:
                L_inter = 0
            L_odl = L_intra + self.lambda_inter * L_inter
        else:
            L_intra = 0
            # z batch_size, embed_dim
            all_centers = self.class_center.weight  # 所有类别的中心 (num_classes, embed_dim)
            distance_matrix = torch.cdist(z, all_centers) ** 2 # batch_size, num_classes
            # print(distance_matrix.shape)
            min_distances, min_class_j = distance_matrix.min(dim=1)
            # print(min_distances.shape) # batch_size
            # print(min_class_j)
            # Encourage large distance between z and all class centers (separation loss)
            separation_loss = torch.max(self.m - min_distances, torch.zeros_like(min_distances))  # Apply margin m
            L_Sep = separation_loss.mean()  # Mean over all samples in the batch

            c_i = z # batch_size, embed_dim
            c_j = self.class_center(min_class_j) # batch_size, embed_dim
            v_s = c_j - c_i   # batch_size, embed_dim

            one_hot_encoded = F.one_hot(min_class_j, num_classes=self.class_size).bool().reshape(c_i.shape[0],-1) # batch_size, class_size
            all_class = torch.arange(self.class_size).unsqueeze(0).repeat(c_i.shape[0],1).to(one_hot_encoded.device)
            o_class = all_class[~one_hot_encoded].view(c_i.shape[0], -1) # batch_size, class_size - 1
            v_o = self.class_center(o_class) - c_i.unsqueeze(1) # batch_size, class_size - 1, embed_dim


            D_oj = -torch.sum((self.class_center(o_class) - c_j.unsqueeze(1)) ** 2, dim=-1) # batch_size, class_size - 1
            D_oj = F.softmax(D_oj, dim=-1) # batch_size, class_size - 1
            
            cos_theta = F.cosine_similarity(v_s.unsqueeze(1), v_o, dim=-1) # batch_size, class_size - 2
            orientation_loss = (1 + cos_theta) * D_oj # batch_size, class_size - 2
            orientation_loss = orientation_loss.sum(dim=-1).mean()
            if self.ausgs.cos:
                orientation_loss = 0
            L_inter = L_Sep   + 0.01 *  orientation_loss # 注释了

            L_odl = L_intra + self.lambda_inter * L_inter
            if self.ausgs.spurious == 0:
                L_odl = 0
        return self.lambda_ * L_odl
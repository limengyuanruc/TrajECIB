import torch
import pickle
import math
import numpy as np
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import random
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .grud import GRUDCell,CustomGRUD,STGNCell
from .TrajFormer import TrajFormer



class GRUHiddenBlock(nn.Module):
    def __init__(self,hidden_size=64, num_layers=2, dropout=0.2, bidirectional=False):
        super(GRUHiddenBlock, self).__init__()

        self.gru = nn.GRU(hidden_size, hidden_size // 2, num_layers, dropout=dropout, bidirectional=bidirectional) if bidirectional else nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout)
        
    def forward(self, x_unpacked, dt, dd, x_lens):
        # x_unpacked:  [bs, n_time, in_channels]  2048,100,64  -> [bs, hidden_size]  2048,64
        x_pack = rnn_utils.pack_padded_sequence(x_unpacked, x_lens, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(x_pack)  
        out, x_lens = rnn_utils.pad_packed_sequence(out, batch_first=True)    # out: bs, n_time, hidden_size*2
        # batch_size, seq_len,  num_directions * hidden_size
        return out


class LSTMHiddenBlock(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=True):
        super(LSTMHiddenBlock, self).__init__()

        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers, dropout=dropout, bidirectional=bidirectional) if bidirectional else nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout)
        
    def forward(self, x_unpacked, dt, dd, x_lens):
        # x_unpacked:  [bs, n_time, in_channels]  2048,100,64  -> [bs, hidden_size]  2048,64
        x_pack = rnn_utils.pack_padded_sequence(x_unpacked, x_lens, batch_first=True, enforce_sorted=False)
        out,_ = self.lstm(x_pack)  
        out, x_lens = rnn_utils.pad_packed_sequence(out, batch_first=True)    # out: bs, n_time, hidden_size*2

        return out
    


class GRUDHiddenBlock(nn.Module):
    def __init__(self,hidden_size=64, num_layers=2, dropout=0.2, bidirectional=False):
        super(GRUDHiddenBlock, self).__init__()

        self.gru = CustomGRUD(hidden_size, hidden_size // 2, num_layers, dropout=dropout, bidirectional=bidirectional,module=GRUDCell) if bidirectional \
            else CustomGRUD(hidden_size, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout,module=GRUDCell)
        
    def forward(self, x_unpacked, dt, dd, x_lens):
        out = self.gru(x_unpacked,dt=dt,dd=dd)  
        # batch_size, seq_len,  num_directions * hidden_size
        return out



class STGNHiddenBlock(nn.Module):
    def __init__(self,hidden_size=64, num_layers=2, dropout=0.2, bidirectional=False):
        super(STGNHiddenBlock, self).__init__()

        self.gru = CustomGRUD(hidden_size, hidden_size // 2, num_layers, dropout=dropout, bidirectional=bidirectional,module=STGNCell,mode='lstm') if bidirectional \
            else CustomGRUD(hidden_size, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout,module=STGNCell,mode='lstm')
        
    def forward(self, x_unpacked, dt, dd, x_lens):
        out = self.gru(x_unpacked,dt=dt,dd=dd)  
        # batch_size, seq_len,  num_directions * hidden_size
        return out



#  implement of AE based on LSTM
class AutoEncoder(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=False):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout)
        self.decoder = nn.LSTMCell(hidden_size, hidden_size)
        self.Dropout = nn.Dropout(dropout)
    def forward(self, x_unpacked, dt, dd, x_lens):
        # x_unpacked:  [bs, n_time, in_channels]  2048,100,64  -> [bs, hidden_size]  2048,64
        x_pack = rnn_utils.pack_padded_sequence(x_unpacked, x_lens, batch_first=True, enforce_sorted=False)
        out, (hidden, cell) = self.encoder(x_pack)   # 返回的hidden和cell是最后一个时间步的隐藏状态和细胞状态
        out, x_lens = rnn_utils.pad_packed_sequence(out, batch_first=True)    # out: bs, n_time, hidden_size*2
        # hidden.shape: (num_layers * num_directions, batch, hidden_size)
        emb = out
        
        output = []
        for t in range(out.size(1)): # out: bs, n_time, hidden_size*2
            hidden_t, cell_t = self.decoder(out[:, t, :], (hidden[-1], cell[-1]))  # out_t: bs, hidden_size
            out_t = self.Dropout(hidden_t)
            output.append(out_t.unsqueeze(1))
        output = torch.cat(output, dim=1)  # output: bs, n_time

        
        MSE_loss = F.mse_loss(output, out, reduction='mean') * 0.01 # 

        return emb, MSE_loss

class TrajFormerHiddenBlock(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=False):
        super(TrajFormerHiddenBlock, self).__init__()

        self.encoder_layer = TrajFormer(c_in = hidden_size, trans_layers = num_layers, token_dim = hidden_size, n_heads = 4, mlp_dim = hidden_size *4)
        
    def forward(self, x_unpacked, dt, dd, x_lens):
        # x_unpacked:  [bs, n_time, in_channels]  2048,100,64  -> [bs, hidden_size]  2048,64
        out = self.encoder_layer(x_unpacked,dt,dd,x_lens)
        return out

########################### UniEncBlock #############################

# implement of Contrastive based on LSTM
class ContrastiveEncoder(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=False):
        super(ContrastiveEncoder,self).__init__()

        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout)
        # 投影头 projection head
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.Dropout = nn.Dropout(dropout)

    def _encode(self, x_unpacked, x_lens):
        # x_unpacked:  [bs, n_time, in_channels] -> [bs, n_time, hidden_size]
        x_pack = rnn_utils.pack_padded_sequence(x_unpacked, x_lens, batch_first=True, enforce_sorted=False)
        out, _ = self.encoder(x_pack)
        out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True)
        return out

    def forward(self, x_unpacked, dt, dd, x_lens, _m):
        # m: batch_size, seq_len, 2 为noise

        emb_raw = self._encode(x_unpacked, x_lens) # 原本的emb
        
        data_x_new = x_unpacked.clone()
        data_x_new[:,:,[0,1]] = data_x_new[:,:,[0,1]] + _m
        data_x1 = data_x_new[:, ::2, :]  # batch_size, seq_len//2, d_dim
        data_x2 = data_x_new[:, 1::2, :]  # batch_size, seq_len//2, d_dim

        temp1 = data_x_new[:,1:-1:2,2] + data_x_new[:,2::2,2]
        temp1 = torch.cat([torch.zeros(data_x_new.size(0),1).to(data_x_new.device), temp1], dim=1)
        data_x1[:,:,2] =  temp1

        temp2 = data_x_new[:,2:-1:2,2] + data_x_new[:,3::2,2] # 
        temp2 = torch.cat([torch.zeros(data_x_new.size(0),1).to(data_x_new.device), temp2], dim=1)
        data_x2[:,:,2] =  temp2

        x_lens_new =  x_lens // 2
        # 1. 原始视图 
        emb = self._encode(data_x1, x_lens_new)                    # bs, n_time, hidden_size
        # 2. 遮盖视图: 按mask遮盖部分时间步后再编码
        emb_m = self._encode(data_x2, x_lens_new)                    # bs, n_time, hidden_size

        # 3. 序列级表示: 对有效时间步取平均后过投影头
        x_lens_new = x_lens_new.to(emb.device)
        mask_pad = torch.arange(emb.size(1), device=emb.device).unsqueeze(0) < x_lens_new.unsqueeze(1)  # bs, n_time
        z_ori = self.projector(self.Dropout(emb))
        z_msk = self.projector(self.Dropout(emb_m))
        z_ori = (z_ori * mask_pad.unsqueeze(-1)).sum(dim=1).div(x_lens_new.float().unsqueeze(dim=1))    # bs, hidden_size
        z_msk = (z_msk * mask_pad.unsqueeze(-1)).sum(dim=1).div(x_lens_new.float().unsqueeze(dim=1))    # bs, hidden_size

        # 4. InfoNCE对比损失: 同一轨迹的两个视图为正样本对, batch内其他轨迹为负样本
        z_ori = F.normalize(z_ori, dim=-1)
        z_msk = F.normalize(z_msk, dim=-1)
        logits = z_ori @ z_msk.t() / 0.1                          # bs, bs
        labels = torch.arange(z_ori.size(0), device=z_ori.device)
        cl_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) * 0.5 * 0.01

        return emb_raw, cl_loss


# UniEncBlock is the same as above GRU/LSTMHiddenBlock
# It can be replaced by any sequence encoder

class UniEncBlock(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=True, cell='gru'):
        super(UniEncBlock, self).__init__()
        self.cell = cell
        if cell == 'gru':
            self.hidden_block = GRUHiddenBlock(hidden_size, 1, dropout, False)
        elif cell == 'lstm':
            self.hidden_block = LSTMHiddenBlock(hidden_size, 1, dropout, bidirectional = True)
        elif cell == 'grud':
            self.hidden_block = GRUDHiddenBlock(hidden_size, 1, dropout, bidirectional = False)
        elif cell == 'stgn':
            self.hidden_block = STGNHiddenBlock(hidden_size, 1, dropout, bidirectional = False)
        elif cell == 'former':
            self.hidden_block = TrajFormerHiddenBlock(hidden_size, num_layers, dropout, bidirectional)
        elif cell == 'ae':
            self.hidden_block = AutoEncoder(hidden_size, num_layers, dropout, bidirectional)
        elif cell == 'cl':
            self.hidden_block = ContrastiveEncoder(hidden_size, num_layers, dropout, bidirectional = False)

    def forward(self,  x_unpacked, dt, dd, x_lens,_m):
        # _m：mask
        if self.cell == 'ae':
            emb, MSE_loss = self.hidden_block(x_unpacked, dt, dd, x_lens)
        elif self.cell == 'cl':
            emb, MSE_loss = self.hidden_block(x_unpacked, dt, dd, x_lens, _m)

        else:
            emb = self.hidden_block(x_unpacked, dt, dd, x_lens)
            MSE_loss = 0
        return emb, MSE_loss
        
########################### aggregate #############################

class TimeMeanBlock(nn.Module):
    def __init__(self):
        super(TimeMeanBlock, self).__init__()
        
    def forward(self, x_unpacked, x_lens):
        # x_unpacked:  [bs, n_time, hidden_size]  2048,100,64  -> [bs, hidden_size]  2048,64
        x_lens = x_lens.to(x_unpacked.device)
        mean_out = x_unpacked.sum(dim=1).div(x_lens.float().unsqueeze(dim=1))    # mean_out: bs, hidden_size
        # 平均池化
        return mean_out
    

class TimeLastBlock(nn.Module):
    def __init__(self):
        super(TimeLastBlock, self).__init__()
        
    def forward(self, x_unpacked, x_lens):
        # x_unpacked:  [bs, n_time, hidden_size]  2048,100,64  -> [bs, hidden_size]  2048,64
        out = x_unpacked
        indices = torch.LongTensor(np.array(x_lens) - 1).view(-1, 1).expand(out.size(0), out.size(2)).unsqueeze(1).to(out.device)
        last_encoded_outs = out.gather(dim=1, index=indices).squeeze(dim=1)
        return last_encoded_outs


class TimePoolBlock(nn.Module):
    def __init__(self):
        super(TimePoolBlock, self).__init__()
        self.adapool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x_unpacked, x_lens):
        # x_unpacked:  [bs, n_time, hidden_size]  2048,100,64  -> [bs, hidden_size]  2048,64
        x = self.adapool(x_unpacked.permute(0, 2, 1)).squeeze(dim=2)
        return x
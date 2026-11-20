import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.utils.rnn as rnn_utils
# # implement of GRUD and STGN

class TemporalDecay(nn.Module):
    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std_dev = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-std_dev, std_dev)
        if self.b is not None:
            self.b.data.uniform_(-std_dev, std_dev)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:

        if self.diag:
            gamma = F.relu(F.linear(delta, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(delta, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma
    

class GRUDCell(nn.Module):
    # reference: https://docs.pypots.com/en/dev/_modules/pypots/nn/modules/grud/backbone.html#BackboneGRUD
    def __init__(self, input_size, hidden_size,):
        super().__init__()
        self.rnn_cell = nn.GRUCell(input_size+hidden_size,hidden_size)
        self.input_size = input_size
        # input: [x, h] output: h
        self.temp_decay_h = TemporalDecay(input_size=input_size, output_size=hidden_size, diag=False)
        # First, xt and ht−1 are respectively replaced by ˆ xt and − ˆ ht 1 from Equations (11) and (12). 
        # Second, the masking vector mt are fed into the model, and Vz, Vr, V are new parameters for it.
    def forward(self,x, hidden_state,dt=None,dd=None):
        # x: batch_size, input_size
        # hidden_state: batch_size, hidden_size
        dt = dt.repeat(1,self.input_size).clone() # batch_size, input_size, 1 -> batch_size, input_size, hidden_size
        gamma_h = self.temp_decay_h(dt)
        hidden_state = hidden_state * gamma_h
        data_input = torch.cat([x, hidden_state], dim=-1)
        hidden_state = self.rnn_cell(data_input, hidden_state)
        return hidden_state


class STGNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(STGNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_i = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.W_h = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))

        self.W_xt1 = nn.Parameter(torch.Tensor(input_size, hidden_size ))
        self.W_t1 = nn.Parameter(torch.Tensor(1, hidden_size ))
        self.bias_xt1 = nn.Parameter(torch.Tensor(hidden_size ))

        self.W_xt2 = nn.Parameter(torch.Tensor(input_size, hidden_size ))
        self.W_t2 = nn.Parameter(torch.Tensor(1, hidden_size ))
        self.bias_xt2 = nn.Parameter(torch.Tensor(hidden_size ))

        self.W_xd1 = nn.Parameter(torch.Tensor(input_size, hidden_size ))
        self.W_d1 = nn.Parameter(torch.Tensor(1, hidden_size ))
        self.bias_xd1 = nn.Parameter(torch.Tensor(hidden_size ))

        self.W_xd2 = nn.Parameter(torch.Tensor(input_size, hidden_size ))
        self.W_d2 = nn.Parameter(torch.Tensor(1, hidden_size ))
        self.bias_xd2 = nn.Parameter(torch.Tensor(hidden_size ))

        self.W_to = nn.Parameter(torch.Tensor(1, hidden_size ))
        self.W_do = nn.Parameter(torch.Tensor(1, hidden_size ))


        self.init_weights()


    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def forward(self, input, hx,dt,dd):
        h, c = hx # 二元组
        gates = input @ self.W_i + h @ self.W_h + self.bias
        i, f, c_var, o = gates.chunk(4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        c_var = torch.tanh(c_var)

        self.W_t1.data.clamp_(min=-float('inf'), max=0)
        self.W_d1.data.clamp_(min=-float('inf'), max=0)

        t1 = input @ self.W_xt1 + torch.sigmoid(dt @ self.W_t1) + self.bias_xt1
        t2 = input @ self.W_xt2 + torch.sigmoid(dt @ self.W_t2) + self.bias_xt2
        d1 = input @ self.W_xd1 + torch.sigmoid(dd @ self.W_d1) + self.bias_xd1
        d2 = input @ self.W_xd2 + torch.sigmoid(dd @ self.W_d2) + self.bias_xd2
        t1 = torch.sigmoid(t1)
        t2 = torch.sigmoid(t2)
        d1 = torch.sigmoid(d1)
        d2 = torch.sigmoid(d2)
    
        c_hat  = f * c + i * t1 * d1 * c_var
        c_next = f * c + i * t2 * d2 * c_var


        o = torch.sigmoid(o + dt @ self.W_to + dd @ self.W_do)
        # c_next = f * c + i * c_var
        h_next = o * torch.tanh(c_hat)
        return h_next, c_next

class CustomGRUD(nn.Module):
    """
    自定义单向多层 GRU / LSTM 模型。
    支持 GRU 和 LSTM 模式，但不支持双向模式。
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0,module = nn.GRUCell,mode = 'gru',bidirectional=False):
        super(CustomGRUD, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mode = mode.lower()
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # 定义单层的 GRUCell，分为正向和反向
        self.cells_f = nn.ModuleList([module(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])


    def forward(self, unpacked_x,dt=None,dd=None, h_0=None, c_0=None):
        """
        前向传播函数。
        Args:
            unpacked_x: 输入张量 [batch_size, seq_len, input_size]
            dt: 时间间隔张量 (可选) [batch_size, seq_len, 1]
            dd: 时间差异张量 (可选) [batch_size, seq_len, 1]
            h_0: 初始隐藏状态 [batch_size, num_layers, hidden_size]
            c_0: 初始单元状态 (仅 LSTM 模式下) [batch_size, num_layers, hidden_size]
        Returns:
            output_f: 最终输出张量 [batch_size, seq_len, hidden_size]
        """
        x_data = unpacked_x  # 输入数据 [batch_size, seq_len, input_size]
        batch_size, seq_len, _ = x_data.shape

        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.num_layers,  self.hidden_size, device=x_data.device,dtype=x_data.dtype)
        if self.mode == 'lstm' and c_0 is None:
            c_0 = torch.zeros(batch_size, self.num_layers,  self.hidden_size, device=x_data.device,dtype=x_data.dtype)
        # h_0: [batch_size, num_layers,  hidden_size]
        # c_0: [batch_size, num_layers,  hidden_size]

        hidden_states_f = [h_0[:, layer_idx] for layer_idx in range(self.num_layers)]
        if self.mode == 'lstm':
            cell_states_f = [c_0[:, layer_idx] for layer_idx in range(self.num_layers)]
        
        output_f = []
        # 时间步遍历
        for t in range(seq_len): 
            current_input = x_data[:, t, :] # 当前时间步的输入
            current_dt = dt[:,t,:] # batch_size, 1
            current_dd = dd[:,t,:] # batch_size, 1
            
            for layer_idx in range(self.num_layers): # 每一层的输入是上一层的输出
                if self.mode == 'gru':
                    hx_f = hidden_states_f[layer_idx] # batch_size,seq_len                 
                    hx_f = self.cells_f[layer_idx](current_input, hx_f,current_dt,current_dd)
                    hidden_states_f[layer_idx] = hx_f # 更新状态
                elif self.mode == 'lstm':
                    hx_f, cx_f = hidden_states_f[layer_idx], cell_states_f[layer_idx]
                    hx_f, cx_f = self.cells_f[layer_idx](current_input, (hx_f, cx_f),current_dt,current_dd)
                    hidden_states_f[layer_idx] = hx_f
                    cell_states_f[layer_idx] = cx_f

                if (self.dropout_layer is not None) and (layer_idx != self.num_layers - 1):
                    hx_f = self.dropout_layer(hx_f)                
                current_input = hx_f   # 记录当前时间步的最后一层输出
            output_f.append(hx_f)
        # [batch_size, hidden_size] * num_layers
        # hidden_states_f_copy = torch.stack(hidden_states_f_copy, dim=1) # [batch_size, num_layers, hidden_size]
        output_f = torch.stack(output_f, dim=1)  # batch_size, hidden_size
        return output_f







    

  
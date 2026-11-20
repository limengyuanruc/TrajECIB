from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from fvcore.nn import FlopCountAnalysis

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        :return: Model output
        这是所有 PyTorch 模型中的核心方法，定义了模型的前向传播逻辑。每个继承 BaseModel 的具体模型都必须实现这个方法
        """
        raise NotImplementedError
    
    def param_num(self, str):
        # 该方法用于计算模型中所有参数的数量，包括可训练参数和不可训练参数。
        return sum([param.nelement() for param in self.parameters()])

    def flops(self, inputs):
        #这个方法使用 FlopCountAnalysis 工具来计算模型的 FLOPs，衡量模型在给定输入下的计算复杂度。
        flops = FlopCountAnalysis(self, inputs)
        return flops.total()

    def __str__(self):
        """
        Model prints with number of trainable parameters
        该方法重写了 nn.Module 的 __str__ 方法，增加了对模型的可训练参数（trainable parameters）数量的显示。
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
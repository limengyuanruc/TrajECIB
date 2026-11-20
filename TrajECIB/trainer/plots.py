import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from captum.attr import (
    Saliency,#0
    IntegratedGradients,#1
    NoiseTunnel,#2
    InputXGradient,#3
    GuidedBackprop,#4
    DeepLift,#5
    GradientShap,#6
    DeepLiftShap,#7

    Occlusion,#0
    FeatureAblation,#1
    FeaturePermutation,#2

    ShapleyValueSampling,#0
    KernelShap,#1
)
from torch.autograd import Variable

def _check_device(tensors,_device):
    if isinstance(tensors, list):
        return [tensor.to(_device) for tensor in tensors]
    else:
        return tensors.to(_device)
def get_batch_idx(len_data,batch_size):
    batch_idx = []
    data_idx = np.arange(len_data)
    for i in range(0,len_data,batch_size):
        batch_idx.append(data_idx[i:min(i+batch_size,len_data)])
    return batch_idx
def liner_inter(data,mask):
    batch_size, seq_len, feat = data.shape
    # 初始化一个同样形状的数组用于存放插补后的结果
    data = copy.copy(data)
    interpolated_data = np.empty_like(data)
    data[mask.astype(bool)] = np.nan
    # 遍历每个批次和每个特征
    for i in range(batch_size):

        data_sequence = data[i]
        data_sequence = pd.DataFrame(data_sequence)
        # data_sequence_ = data_sequence.interpolate(method='linear',limit_direction='both',limit = 26).fillna(0)
        data_sequence_ = data_sequence.interpolate(method='linear',limit_direction='both',limit = 26).fillna(0)
        interpolated_data[i] = data_sequence_.values

    return torch.tensor(interpolated_data) 

def givenAttGetRescaledSaliency(attributions,is_abs = False):
    if  torch.is_tensor(attributions):
        saliency =attributions.cpu().data
    else:
        saliency = torch.tensor(attributions)
    if is_abs:
        saliency = torch.abs(saliency)
    saliency= saliency.mean(dim=2) if saliency.dim() == 3 else saliency # 重塑为 batch_size, time_feat 2D

    row_min, _  = saliency.min(dim=1,keepdim=True)
    row_max, _  = saliency.max(dim=1,keepdim=True)
    
    # 进行归一化
    normalized_matrix = (saliency - row_min) / (row_max - row_min)
    return normalized_matrix
def secondary_sort_criteria(tensor):
    # 加入随机性，防止相同大小的点同时选中。
    s_tensor = torch.rand_like(tensor)
    if tensor.dim() > 2:
        s_tensor = s_tensor.view(1, -1, 1)
    return tensor + s_tensor * 1e-3

def mask_base_ib(pretrained_model,baselines_results_four_,baselines_outs,conditions_sum_,method = 'IB_top'):
    pretrained_model.eval()
    with torch.no_grad():
        baselines_results_four = {method:copy.deepcopy(baselines_results_four_),} # 临时的
        conditions_sum = {i: copy.deepcopy(conditions_sum_)  for i in baselines_results_four.keys()}
        # baselines_outs 要修改的

        baselines_outs[method]['base_attributions'] = []

        batch_idx = get_batch_idx(len(baselines_outs['true_label']),pretrained_model.augs.batch_size)
        for idx in tqdm(batch_idx):
            with torch.no_grad():
                # 取出数据
                X = torch.tensor(baselines_outs['X'][idx]).to(pretrained_model.augs.device)
                X_norm = torch.tensor(baselines_outs['X_norm'][idx]).to(pretrained_model.augs.device)
                M = torch.tensor(baselines_outs['M'][idx]).to(pretrained_model.augs.device)
                _len = torch.tensor(baselines_outs['_len'][idx])
                
                labels = torch.tensor(baselines_outs['true_label'][idx]).to(pretrained_model.augs.device)
                pred_idx = torch.tensor(baselines_outs['pred_label'][idx]).to(pretrained_model.augs.device)
                confidence_o = torch.tensor(baselines_outs['confidence_o'][idx])
                
                target_idx = labels
                input_data = X
                input_data_numpy = input_data.cpu().numpy()
                k_indices_log = target_idx.cpu()
            
                model_outs = pretrained_model(X,X_norm, M, _len,labels)
                
                attributions = givenAttGetRescaledSaliency(pretrained_model.env_selects)
                baselines_outs[method]['base_attributions'].append(attributions.cpu().numpy())

                seq_len = (torch.ones((input_data.shape[0])) * input_data.shape[1]).long() 
                row_indices = torch.arange(input_data.shape[0])
                for j,top_ratio in enumerate([0.1 * i for i in range(1,10)]):
                    k_indices = (seq_len * top_ratio).long() - 1

                    sort_out = attributions.sort(dim=1, descending=True,stable=True)[0]
                    mask_temp = (attributions >= sort_out[row_indices,k_indices].reshape(-1,1)).float()                    
                    mask_temp_input = mask_temp.unsqueeze(-1).repeat(1,1,input_data.shape[-1])
                    mask_temp_norm = mask_temp.unsqueeze(-1).repeat(1,1,X_norm.shape[-1])
                    input_data_mask = liner_inter(input_data_numpy,mask_temp_input.numpy()).to(pretrained_model.augs.device)
                    X_norm_mask = liner_inter(X_norm.cpu().numpy(),mask_temp_norm.numpy()).to(pretrained_model.augs.device)
                    
                    model_outs = pretrained_model(input_data_mask,X_norm_mask, M, _len,labels)
                    logits = model_outs['out_o_logis']
                    _,pred_mask = torch.max(logits.data,1)
                    logits = F.softmax(logits,dim=-1).cpu()
                    confidence_s = logits[row_indices,k_indices_log]

                    get_result_sum(baselines_results_four[method],conditions_sum[method],target_idx,pred_idx,pred_mask,confidence_o,confidence_s,j+1)


        for method,middle_dict in  baselines_results_four.items():# [ 'acc', 'confidence_d', 'confidence_i']
            for condition,inner_dict in middle_dict.items():
                for name in inner_dict.keys():    
                    baselines_results_four[method][condition][name] = baselines_results_four[method][condition][name].numpy() /( conditions_sum[method][condition].numpy() )
                    baselines_results_four[method][condition][name] = np.nan_to_num(baselines_results_four[method][condition][name], nan=0.0)
        baselines_outs[method]['base_attributions'] = np.concatenate(baselines_outs[method]['base_attributions'])
    return baselines_results_four,baselines_outs

def get_resluts_dict():
    names = [ 'accuracy', 'confidence_d', 'confidence_i'] # 
    top_ratio_list = [0.1 * i for i in range( 10)]
    baselines_results = {name: torch.zeros(len(top_ratio_list)) for name in names} # 
    # 'accuracy':[]
    baselines_results_four = {
        'all': copy.deepcopy(baselines_results),

    }
    # all:'accuracy':[]
    conditions_sum = {key: torch.zeros(len(top_ratio_list)) for key in baselines_results_four.keys()}
    return baselines_results_four,conditions_sum
def get_result_sum(baselines_results_four,conditions_sum,target_idx,pred_idx,pred_mask,confidence_o,confidence_s,j):
    '''
    baselines_results_four: 存储结果的字典，用于后续的处理
    target_idx: 真实的label
    pred_idx: 预测的label
    pred_mask；mask的label
    confidence_o：
    confidence_s：
    j
    '''
    conditions = {
        'all': target_idx == target_idx,
    }
    for condition in conditions.keys():
        temp_sum =  conditions[condition].cpu().sum().float().item()
        conditions_sum[condition][j] += temp_sum
        baselines_results_four[condition]['accuracy'][j] += (target_idx[conditions[condition]] == pred_mask[conditions[condition]]).sum().float().item() if temp_sum >0 else 0
        baselines_results_four[condition]['confidence_d'][j] += torch.relu(confidence_o[conditions[condition].cpu()] - confidence_s[conditions[condition].cpu()]).sum() if temp_sum >0 else 0
        baselines_results_four[condition]['confidence_i'][j] +=(confidence_o[conditions[condition].cpu()] <= confidence_s[conditions[condition].cpu()] + 0.).sum() if temp_sum >0 else 0


def mask_raw_data(pretrained_model,test_loader):
    # 对不加mask的结果进行处理
    baselines_outs = {}
    pretrained_model.eval() 
    # X, M, _len,Y
    baselines_outs['true_label'] = []
    baselines_outs['pred_label'] = []
    baselines_outs['X'] = []
    baselines_outs['X_norm'] = []
    baselines_outs['M'] = []
    baselines_outs['_len'] = []
    baselines_outs['confidence_o'] = []
    baselines_results_four,conditions_sum = get_resluts_dict()
    counts = 0 # 计数
    
    with torch.no_grad():
        for batch_data in test_loader: # 对每一个epoch 对每一个       
            #X, Y ,M = _check_device([batch_data[0], batch_data[3],batch_data[4]],pretrained_model.augs.device)
            X,X_norm, Y ,M = _check_device([batch_data[0], batch_data[2],batch_data[3],batch_data[4]],pretrained_model.augs.device)

            _len = torch.tensor(batch_data[1])

            model_outs = pretrained_model(X,X_norm, M, _len,Y)
            logits_raw = model_outs['out_o_logis'] # 原始的logit
            _,pred_idx = torch.max(logits_raw.data,1)
            ###########################################
            target_idx = Y# 
            row_indices_log = torch.arange(logits_raw.shape[0])  # 生成行号
            k_indices_log = target_idx.cpu()
            confidence_o = F.softmax(logits_raw.cpu(), dim=-1)[row_indices_log,k_indices_log]



            baselines_outs['true_label'].append(target_idx.cpu().numpy())
            baselines_outs['pred_label'].append(pred_idx.cpu().numpy())
            baselines_outs['X'].append(X.data.cpu().numpy())
            baselines_outs['X_norm'].append(X_norm.data.cpu().numpy())
            baselines_outs['M'].append(M.data.cpu().numpy())
            baselines_outs['_len'].append(_len.data.cpu().numpy())
            baselines_outs['confidence_o'].append(confidence_o.cpu().numpy())

            
            confidence_s = confidence_o
            pred_mask = pred_idx
            get_result_sum(baselines_results_four,conditions_sum,target_idx,pred_idx,pred_mask,confidence_o,confidence_s,0)

    for k,_ in baselines_outs.items():
        baselines_outs[k] = np.concatenate(baselines_outs[k])
    
    baselines_outs['IB_top'] = {}
    for flag in ['GradFlag', 'IGFlag', 'SGFlag', 'TNGFlag', 'GBFlag', 'DLFlag', 'GSFlag', 'DLSFlag', 'FOFlag', 'FAFlag', 'FPFlag', 'SVSFlag', 'KSFlag', 'RNDFlag', 'FIXFlag', 'FREFlag', 'STDFlag']:
        if getattr(pretrained_model.augs, flag):
            baselines_outs[flag] = {}

    return baselines_results_four,baselines_outs,conditions_sum

class RND:
    def __init__(self, forward_func) -> None:
        self.forward_func = forward_func
    def attribute(
        self,
        inputs,
        target,
        abs = True,
        additional_forward_args = None,
    ):
        #att = torch.sin(torch.arange(inputs.shape[1])* (1/(inputs.shape[1]/2.5)) * 2 * torch.pi ).reshape(1,-1,1).repeat(inputs.shape[0],1,1)
        att = torch.randn_like(inputs)
        return att


def mask_base_baseline(pretrained_model,baselines_results_four_,conditions_sum_,baselines_outs,method = 'IGFlag'):
    pretrained_model = copy.deepcopy(pretrained_model)
    batch_size = pretrained_model.augs.batch_size
    attribution_methods = { # baseline的方法
        'GradFlag': Saliency,
        'IGFlag': IntegratedGradients,
        'SGFlag': lambda model: NoiseTunnel(Saliency(model)),
        'TNGFlag': InputXGradient,
        'GBFlag': GuidedBackprop,
        'DLFlag': DeepLift,
        'GSFlag': GradientShap,
        'DLSFlag': DeepLiftShap,
        'FOFlag': Occlusion,
        'FAFlag': FeatureAblation,
        'FPFlag': FeaturePermutation,
        'SVSFlag': ShapleyValueSampling,
        'KSFlag': KernelShap,
        'RNDFlag':RND,
    }

    attribution_batch_size = { # baseline的方法
        'GradFlag': batch_size,
        'IGFlag': batch_size,
        'SGFlag': batch_size,
        'TNGFlag': batch_size,
        'GBFlag': batch_size,
        'DLFlag': batch_size // 2,
        'GSFlag': batch_size // 5,
        'DLSFlag': int(np.sqrt(400 // 10)),
        'FOFlag': batch_size,
        'FAFlag': batch_size,
        'FPFlag': batch_size,
        'SVSFlag': batch_size,
        'KSFlag': batch_size,
        'RNDFlag':batch_size,
        'FIXFlag':batch_size,
        'FREFlag':batch_size,
        'STDFlag':batch_size,
    }

    class BaseModel(torch.nn.Module): # 对base_model的归因分析，时间序列
        def __init__(self, model):
            super(BaseModel, self).__init__()
            self.model = model
        def forward(self, _x,_x_norm,_m, _len, Y):
            self.model.augs.train_ib = 0
            model_outs = self.model(_x,_x_norm,_m, _len,Y)  
            logits = model_outs['out_o_logis'] # 原始的logit  
            return F.softmax(logits,dim=-1) # 返回的logits
    
    base_net = BaseModel(pretrained_model)
    attribution_method = attribution_methods[method](base_net)

    
    with torch.no_grad():
        baselines_results_four = {method:copy.deepcopy(baselines_results_four_),} # 临时的
        conditions_sum = {i: copy.deepcopy(conditions_sum_)  for i in baselines_results_four.keys()}  # 临时的
        # baselines_outs 要修改的

        baselines_outs[method]['base_attributions'] = []

        batch_idx = get_batch_idx(len(baselines_outs['true_label']),attribution_batch_size[method])
        for idx in tqdm(batch_idx):
            
            with torch.no_grad():
                X = torch.tensor(baselines_outs['X'][idx]).to(pretrained_model.augs.device)
                X_norm = torch.tensor(baselines_outs['X_norm'][idx]).to(pretrained_model.augs.device)
                M = torch.tensor(baselines_outs['M'][idx]).to(pretrained_model.augs.device)
                _len = torch.tensor(baselines_outs['_len'][idx])
                
                labels = torch.tensor(baselines_outs['true_label'][idx]).to(pretrained_model.augs.device)
                pred_idx = torch.tensor(baselines_outs['pred_label'][idx]).to(pretrained_model.augs.device)
                confidence_o = torch.tensor(baselines_outs['confidence_o'][idx])

                target_idx = labels
                input_data = X
                input_data_numpy = input_data.cpu().numpy()
                k_indices_log = target_idx.cpu()


                '''
                归因分析
                '''

                input_data_grad =  Variable(input_data,  volatile=False, requires_grad=True) # 设置为含有梯度的
                baseline_single = torch.zeros_like(input_data_grad) 
                baseline_multiple = baseline_single.repeat(5,1,1)
                mask_single=torch.zeros_like((input_data[[0],:,:])).long()
                for i in  range(input_data.shape[1]):
                    mask_single[:,i,:]=i
                attribution_params = {
                    'GradFlag': {},
                    'IGFlag': {'baselines':baseline_single,'internal_batch_size':input_data.shape[0]},
                    'SGFlag': {'nt_samples_batch_size':1},
                    'TNGFlag': {},
                    'GBFlag': {},
                    'DLFlag': {'baselines':baseline_single},
                    'GSFlag': {'baselines':baseline_multiple, 'stdevs':0.09},
                    'DLSFlag': {'baselines':baseline_multiple},
                    'FOFlag': {'sliding_window_shapes':(1,X.shape[-1]),'baselines':baseline_single,'perturbations_per_eval': 1},
                    'FAFlag': {'feature_mask':mask_single,'perturbations_per_eval': 1},
                    'FPFlag': {'perturbations_per_eval': 1},
                    'SVSFlag': {'baselines':baseline_single,'feature_mask':mask_single,'perturbations_per_eval': 1},
                    'KSFlag': {'baselines':baseline_single,'feature_mask':mask_single,'perturbations_per_eval': input_data.shape[0]},
                    'RNDFlag':{},
                    'FIXFlag':{},
                    'FREFlag':{},
                    'STDFlag':{},
                }
                pretrained_model.train()

                attributions = attribution_method.attribute(input_data_grad, target=pred_idx,**attribution_params[method],additional_forward_args=(X_norm, M,_len,labels))
                
                attributions = secondary_sort_criteria(givenAttGetRescaledSaliency(attributions))
                baselines_outs[method]['base_attributions'].append(attributions.cpu().numpy())

                seq_len = (torch.ones((input_data.shape[0])) * input_data.shape[1]).long() 
                row_indices = torch.arange(input_data.shape[0])
                pretrained_model.eval()
                for j,top_ratio in enumerate([0.1 * i for i in range(1,10)]):
                    k_indices = (seq_len * top_ratio).long() - 1

                    sort_out = attributions.sort(dim=1, descending=True,stable=True)[0]
                    mask_temp = (attributions >= sort_out[row_indices,k_indices].reshape(-1,1)).float()
     
                    mask_temp_input = mask_temp.unsqueeze(-1).repeat(1,1,input_data.shape[-1])
                    mask_temp_norm = mask_temp.unsqueeze(-1).repeat(1,1,X_norm.shape[-1])
                    input_data_mask = liner_inter(input_data_numpy,mask_temp_input.numpy()).to(pretrained_model.augs.device)
                    X_norm_mask = liner_inter(X_norm.cpu().numpy(),mask_temp_norm.numpy()).to(pretrained_model.augs.device)
                     
                    model_outs = pretrained_model(input_data_mask,X_norm_mask, M, _len,labels)
                    logits = model_outs['out_o_logis']
                    _,pred_mask = torch.max(logits.data,1)
                    logits = F.softmax(logits,dim=-1).cpu()
                    confidence_s = logits[row_indices,k_indices_log]

                    get_result_sum(baselines_results_four[method],conditions_sum[method],target_idx,pred_idx,pred_mask,confidence_o,confidence_s,j+1)

        for method,middle_dict in  baselines_results_four.items():# [ 'acc', 'confidence_d', 'confidence_i']
            for condition,inner_dict in middle_dict.items():
                for name in inner_dict.keys():    
                    baselines_results_four[method][condition][name] = baselines_results_four[method][condition][name].numpy() /( conditions_sum[method][condition].numpy() )
                    baselines_results_four[method][condition][name] = np.nan_to_num(baselines_results_four[method][condition][name], nan=0.0)
        baselines_outs[method]['base_attributions'] = np.concatenate(baselines_outs[method]['base_attributions'])
    return baselines_results_four,baselines_outs








def get_reserve_dict(baselines_results):
    reserve_dict = {inner_key: {} for outer_value in baselines_results.values() for inner_key in outer_value}
    # 然后，填充这些字典
    for outer_key, out_value in baselines_results.items():
        for inner_key, inner_value in out_value.items():
            reserve_dict[inner_key][outer_key] = inner_value
    return reserve_dict
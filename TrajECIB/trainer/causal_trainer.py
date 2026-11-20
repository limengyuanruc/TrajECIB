import logging
import os
import time
from typing import Optional, List, Union
import random
import numpy as np
from scipy import optimize
import torch
import pickle
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from utils.helper import ave_png, weight_init

from utils.logger import get_logger
from base.trainer import BaseTrainer
from model.base_block import ReadoutLayer
import pandas as pd
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
def accuracy(output, target, topk=(1,1,1)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
        # Computes Macro@F
        return res
from .plots import *
class RND:
    def __init__(self, forward_func) -> None:
        self.forward_func = forward_func

    def attribute(
        self,
        inputs,
        target,
        abs: bool=True,
    ):
        att = torch.randn_like(inputs)
        return att


class CausalTrainer(BaseTrainer):
    def __init__(self, augs,model_deocoder,o, c, co, only_object,  **args):
        super(CausalTrainer, self).__init__(**args)
        self.augs = augs
        self.o = o # 权重
        self.c = c # 权重
        self.co = co # 权重
        self.only_object = only_object # 只有目标 原始的模型
        

        # self.writer = writer
        self.tb_graph = 0
        self.model_deocoder = model_deocoder
    

    def train_batch_aug(self, batch_data, iter):
        pass
        '''
        # data_x, data_length,  x_raw, torch.stack(y)
        X, Y ,M = self._check_device([batch_data[0], batch_data[3],batch_data[4]])
        _len = torch.tensor(batch_data[1])
        with torch.no_grad():
            model_outs = self.model(X, M, _len,Y)
        self.optimizer.zero_grad()
        attributions = self.model.env_selects #属性归因
        num_top_k = int(X.shape[1] * 0.85)  # 最大的85%个点
        _, indices = torch.topk(attributions, num_top_k, dim=1)
        mask_temp = torch.zeros_like(attributions)
        mask_temp.scatter_(1, indices, 1.0)
        X_new = X
        X_new[:,:,6:] = X_new[:,:,6:] * mask_temp  # 防止分布偏移
        X_new[:,:,:2] = X_new[:,:,:2] * mask_temp + self.truncated_latlon_noise(X_new[:,:,:2]) * (1 - mask_temp)
        model_outs = self.model(X_new, M, _len,Y)
        out_c_logis = model_outs['out_c_logis']
        out_o_logis = model_outs['out_o_logis']
        out_co = model_outs['out_co']
        
        uniform_target = torch.ones_like(out_c_logis, dtype=torch.float).to(self._device) / self.model.c_out

        c_loss = F.kl_div(F.log_softmax(out_c_logis, dim=-1), uniform_target, reduction='batchmean')  # 模型输出分布 和 均匀分布 之间的 KL 散度
        o_loss = self.loss_fn(out_o_logis, Y)  # 交叉熵损失
        co_loss = self.loss_fn(out_co, Y)

        if self.only_object:
            loss = o_loss 
        else:
            loss = self.c * c_loss + self.o * o_loss + self.co * co_loss + \
            self.augs.kl_loss * model_outs['IB_loss'] + self.augs.cat_loss * model_outs['Cat_loss'] +\
            + model_outs['loss_odlo'] + model_outs['loss_odlc']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                    max_norm=self._clip_grad_value)
        self.optimizer.step()
        '''
        
    def train_batch(self, batch_data, iter):
        #data_x, data_length,  x_raw, torch.stack(y)
        X,X_norm, Y ,M = self._check_device([batch_data[0], batch_data[2],batch_data[3],batch_data[4]])
        _len = torch.tensor(batch_data[1])

        if self.augs.aug:

            with torch.no_grad():
                model_outs = self.model(X,X_norm, M, _len,Y)
            attributions = self.model.point_att[:,:,[0]] #属性归因
            X_new = X.clone()
            X_new = X_new * attributions # +  noise * (1 - mask_temp)  # 添加噪声
            # X_new[:,:,6:] = X_new[:,:,6:] * mask_temp  # 防止分布偏移
            # X_new[:,:,:2] = X_new[:,:,:2] * mask_temp  + self.truncated_latlon_noise(X_new[:,:,:2]) * (1 - mask_temp)
            X = torch.concat([X,X_new],dim=0)
            M = torch.concat([M,M],dim=0)
            _len = torch.concat([_len,_len],dim=0)
            Y = torch.concat([Y,Y],dim=0)





        self.optimizer.zero_grad()
        model_outs = self.model(X,X_norm, M, _len,Y)
        out_c_logis = model_outs['out_c_logis']
        out_o_logis = model_outs['out_o_logis']
        out_co = model_outs['out_co']

        # 混淆，非混淆，混淆+非混淆
        if self.tb_graph:
            # self.writer.add_graph(self.model, (X,_len))
            self.tb_graph=0


        uniform_target = torch.ones_like(out_c_logis, dtype=torch.float).to(self._device) / self.model.c_out

        c_loss = F.kl_div(F.log_softmax(out_c_logis, dim=-1), uniform_target, reduction='batchmean')  # 模型输出分布 和 均匀分布 之间的 KL 散度
        o_loss = self.loss_fn(out_o_logis, Y)  # 交叉熵损失
        co_loss = self.loss_fn(out_co, Y)

        if self.only_object:
            loss = o_loss + model_outs['else_loss']
        else:
            loss = self.augs.c * c_loss + self.augs.o * o_loss + self.augs.co * co_loss + \
            self.augs.kl_loss * model_outs['IB_loss']  +\
            + self.augs.odl_loss * (model_outs['loss_odlo'] + model_outs['loss_odlc']) + model_outs['else_loss']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       max_norm=self._clip_grad_value)
        self.optimizer.step()

        losses = {}
        losses['train_loss'] = loss.item()
        losses['c_loss'] = c_loss.item()
        losses['o_loss'] = o_loss.item()
        losses['co_loss'] = co_loss.item()
        losses['IB_loss'] = model_outs['IB_loss'].item()
        losses['p_mean'] = model_outs['p_mean'].item()




        return losses
       


    def truncated_latlon_noise(self, data, meters=50):

        factor_lat = 1 / 111000 * meters 
        cos = np.cos(data[:,:,0].mean().item())
        bound_lo = - 5 * factor_lat
        bound_hi = 5 * factor_lat

        noise_lat = torch.randn_like(data[:,:,[0]]) * factor_lat
        noise_lng = torch.randn_like(data[:,:,[1]]) * factor_lat * cos


        noise_lat = torch.clamp(noise_lat, bound_lo, bound_hi)
        noise_lng = torch.clamp(noise_lng, bound_lo* cos, bound_hi* cos)

        noise = torch.cat([noise_lat, noise_lng], dim=-1)
        return noise + data

    def drop_block(self, x, drop_prob=0.25, block_size=5):        
        gamma = drop_prob / block_size
        gamma = gamma * (x.shape[1] / (x.shape[1] - block_size + 1)) *1.1

        x_per = x.permute(0,2,1)
        mask = torch.bernoulli(torch.ones((x_per[:,[0],:].shape)) * gamma).repeat(1, x_per.shape[1], 1)

        # place mask on input device
        mask = mask.to(x.device)

        # compute block mask
        block_mask = F.max_pool1d(
            # input=mask[:, None, :],
            input=mask,
            kernel_size=block_size,
            stride=1,
            padding=block_size // 2,
        )

        if block_size % 2 == 0:
            block_mask = block_mask[:, :,:-1]

        block_mask = 1 - block_mask
        block_mask = block_mask.permute(0,2,1)

        # apply block mask
        out = x
        out[:,:,6:] = x[:,:,6:] * block_mask[:,:,6:]
        # 
        out[:,:,:2] = out[:,:,:2] * block_mask[:,:,:2] + self.truncated_latlon_noise(out[:,:,:2]) * (1 - block_mask[:,:,:2])
        return out
    def neg_samp_loss(self, center_emb, pos_emb, neg_emb):#负采样损失
        # batch_size,dim
        temperature = 0.1
        pos_loss = (pos_emb * center_emb).sum(dim=1) 
        # batch_size,dim, batch_size,dim -> batch_size,dim
        #计算正样本的损失
        pos_loss = F.logsigmoid(pos_loss)
        neg_loss = (neg_emb * center_emb).sum(dim=1) 
        #计算负样本的损失
        neg_loss = F.logsigmoid(neg_loss)
        total_loss = torch.mean(pos_loss + neg_loss)
        return -total_loss 
    def pretrain(self):
        self.logger.info("start pre training !!!!!")
        self.model_deocoder.to(self.device)
        # training phase
        self.model_deocoder.train()
        iter = 0
        val_accs = [-1]
        saved_epoch = -1
        optimizer = Adam(self.model_deocoder.parameters())
        for epoch in range(20):
            start_time = time.time()
            for i, batch_data in enumerate(self.dataloader_dict['train']):
                
                #data_x, data_length,  x_raw, torch.stack(y)
                X,X_norm, Y ,M = self._check_device([batch_data[0], batch_data[2],batch_data[3],batch_data[4]])
                _len = torch.tensor(batch_data[1])
                
                model_outs_1 = self.model(X,X_norm, M, _len,Y)
                
                attributions = self.model.point_att[:,:,[0]] #属性归因
                env_selects = self.model.env_selects.detach() #属性归因
                # attributions = self.model.env_selects #属性归因
                num_bottom_k = int(X.shape[1] * 0.25)  # 最小的10%个点
                _, indices_bottom_k = torch.topk(env_selects, num_bottom_k, dim=1, largest=False) 
                # mask = torch.ones_like(attributions) # 1.0表示保留，0.0表示删除
                attributions.scatter_(1, indices_bottom_k, 0.0)
                
                X_new = X.clone()
                std,mean = torch.std_mean(X_new, dim=1, keepdim=True)
                noise = torch.randn_like(X_new) * std 
                noise = torch.clamp(noise, -2 * std, 2 * std)  # 限制噪声范围
                X_new = X_new  +  noise * (1 - attributions)
                
                X_norm_new = X_norm.clone()
                std,mean = torch.std_mean(X_norm_new, dim=1, keepdim=True)
                noise = torch.randn_like(X_norm_new) * std
                noise = torch.clamp(noise, -2 * std, 2 * std)  # 限制噪声范围
                X_norm_new = X_norm_new  +  noise * (1 - attributions)
                
                model_outs_2 = self.model(X_new,X_norm_new, M, _len,Y)
                loss = self.NT_Xent(model_outs_1['mean_o'], model_outs_2['mean_o'])
                
                # X_new = X.clone()
                # X_new = X_new * (1 - attributions)
                # model_outs_3 = self.model(X_new, M, _len,Y)
                
                
                # loss = self.neg_samp_loss(model_outs_1['mean_o'], model_outs_2['mean_o'], model_outs_3['mean_o'])
                
                
                
                # X1 = self.drop_block(X)
                # loss = self.model_deocoder(X1, M, _len,Y)
               
                optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model_deocoder.parameters(),
                                            max_norm=self._clip_grad_value)
                optimizer.step()
                iter += 1
            end_time = time.time()
            self.logger.info("epoch complete, evaluating now!")



            if self.lr_scheduler is not None:
                self.lr_scheduler.step() # 根据当前的训练进展更新学习率
                  
            loss_message = 'Epoch [{}/{}] ({})'.format(epoch, self._max_epochs, iter)

            loss_message += ', loss: {:.6f}, {:.1f}s'.format(loss, (end_time - start_time))

            self._logger.info(loss_message)

            ########################################################################

        model_file_name = self.save_model(epoch, self._save_path, self._n_exp)
        self.logger.info('Save model at epoch {}, file at {}'.format(saved_epoch, self._save_path+model_file_name))
        # 关闭日志记录器


    def NT_Xent(self,q, k):
        # 计算query和key的相似度得分
        similarity_scores = torch.matmul(q, k.t())  # 矩阵乘法计算相似度得分
        # 计算相似度得分的温度参数
        temperature = 0.1
        # 计算logits
        logits = similarity_scores / temperature
        # 构建labels（假设有N个样本）
        N = q.size(0)
        labels = torch.arange(N).to(logits.device)
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)
        return loss




    def train(self):
        self.logger.info("start training !!!!!")

        # training phase
        iter = 0
        val_accs = [-1]
        saved_epoch = -1
        for epoch in range(self._max_epochs):
            self.model.train()
            train_losses_dict = {
                'train_loss': [],
                'c_loss': [],
                'o_loss': [],
                'co_loss': [],
                'IB_loss': [],
                'p_mean': []
            }
            


            if epoch - saved_epoch > self._patience: # 早停 35
                self.early_stop(epoch, max(val_accs))
                break

            start_time = time.time()
            for i, batch_data in enumerate(self.dataloader_dict['train']):
                losses = self.train_batch(batch_data, iter)
                for key in train_losses_dict.keys():
                    train_losses_dict[key].append(losses[key])

                iter += 1
            end_time = time.time()
            self.logger.info("epoch complete, evaluating now!")

            for key in train_losses_dict.keys():
                train_losses_dict[key] = np.mean(train_losses_dict[key])


            if self.lr_scheduler is not None:
                self.lr_scheduler.step() # 根据当前的训练进展更新学习率
            
            acc_c, acc_o, acc_co = self.evaluate(mode='valid') # 这里调用 evaluate() 方法来评估模型在验证集或测试集上的性能。
            val_acc = acc_o

            if self.lr_scheduler is None:
                new_lr = self._base_lr
            else:
                new_lr = self.lr_scheduler.get_last_lr()[0] #则调用 self.lr_scheduler.get_last_lr() 获取调度器的最新学习率
            

            
            loss_message = 'Epoch [{}/{}] ({})'.format(epoch, self._max_epochs, iter)
            for key, value in train_losses_dict.items():
                loss_message += ' {}: {:.4f}'.format(key, value)
            loss_message += ', lr: {:.6f}, {:.1f}s'.format(new_lr, (end_time - start_time))


            val_message = 'Epoch [{}/{}] ({}) val_acc: c:{:.4f} / o:{:.4f} / co:{:.4f}'.format(
                epoch,
                self._max_epochs,
                iter,
                acc_c,
                acc_o,
                acc_co)

            self._logger.info(loss_message)
            self._logger.info(val_message)
            ########################################################################

            if val_acc > np.max(val_accs):
                model_file_name = self.save_model(epoch, self._save_path, self._n_exp)
                self._logger.info(
                    'Val accuracy increases from {:.4f} to {:.4f}, '
                    'saving to {}'.format(np.max(val_accs), val_acc, model_file_name))
                val_accs.append(val_acc)
                saved_epoch = epoch
        # saved_epoch = epoch
        # model_file_name = self.save_model(epoch, self._save_path, self._n_exp)

        self.logger.info('Save model at epoch {}, file at {}'.format(saved_epoch, self._save_path+model_file_name))
        # 关闭日志记录器
        self.logger.handlers.clear()

    def test_batch(self, batch_data):
        X,X_norm, Y ,M = self._check_device([batch_data[0], batch_data[2],batch_data[3],batch_data[4]])
        _len = torch.tensor(batch_data[1])
        model_outs = self.model(X,X_norm, M, _len,Y)
        out_c_logis = model_outs['out_c_logis']
        out_o_logis = model_outs['out_o_logis']
        out_co = model_outs['out_co']
        return out_c_logis, out_o_logis, out_co


    def evaluate(self, mode='valid'):
        labels = []
        preds_c = []
        preds_o = []
        preds_co = []
        with torch.no_grad():
            self.model.eval()
            for _, batch_data in enumerate(self.dataloader_dict[mode]):
                out_c_logis, out_o_logis, out_co = self.test_batch(batch_data)
                labels.append(batch_data[3])
                preds_c.append(out_c_logis.cpu())
                preds_o.append(out_o_logis.cpu())
                preds_co.append(out_co.cpu())

        labels = torch.cat(labels, dim=0)
        preds_c = torch.cat(preds_c, dim=0)
        preds_o = torch.cat(preds_o, dim=0)
        preds_co = torch.cat(preds_co, dim=0)

        _, outputs_c = torch.max(preds_c, 1)
        _, outputs_o = torch.max(preds_o, 1)
        _, outputs_co = torch.max(preds_co, 1)

        acc_c = (outputs_c == labels).sum() * 1.0 / len(labels)
        acc_o = (outputs_o == labels).sum() * 1.0 / len(labels)
        acc_co = (outputs_co == labels).sum() * 1.0 / len(labels)

        return acc_c, acc_o, acc_co    
    

    def test(self, epoch):
        self.load_model(epoch, self.save_path, self._n_exp) # 加载模型

        val_acc_c, val_acc_o, val_acc_co = self.evaluate('valid')
        test_acc_c, test_acc_o, test_acc_co = self.evaluate('test')
        if self.model.augs.mode == 'test_crossdata':
            test_acc_c, test_acc_o, test_acc_co = self.evaluate('train')
        print('val_acc: c:{:.6f} / o:{:.6f} / co:{:.6f}'.format(val_acc_c, val_acc_o, val_acc_co))
        print('test_acc: c:{:.6f} / o:{:.6f} / co:{:.6f}'.format(test_acc_c, test_acc_o, test_acc_co))

        filename = os.path.join(self.save_path, 'test_results_{}.csv'.format(self._n_exp))

        # save to csv
        results = np.array([[val_acc_c, val_acc_o, val_acc_co], [test_acc_c, test_acc_o, test_acc_co]])
        np.savetxt(filename, results, fmt='%.6f', delimiter=',')

        return val_acc_c, val_acc_o, val_acc_co, test_acc_c, test_acc_o, test_acc_co
    
    


    def train_crossdata_classifier(self):
        # 跨数据集迁移: 加载bj训练好的模型, 冻结编码器, 用bj的mean_o作为shl的特征,
        # 只在shl上重新训练轻量级分类器(object_readout_layer)
        self.load_model(-1, self.save_path, self._n_exp) # 加载从log_dir复制的bj权重

        # 1. 冻结全部参数; 先解冻head再weight_init(weight_init会跳过requires_grad=False的参数)
        for p in self.model.parameters():
            p.requires_grad_(False)
        head = self.model.object_readout_layer
        for p in head.parameters():
            p.requires_grad_(True)
        head.apply(weight_init)

        # 2. 冻结模型上缓存shl的mean_o特征(train/valid), model前向直接返回mean_o
        def extract_features(mode):
            f_list, y_list = [], []
            with torch.no_grad():
                self.model.eval()
                for _, batch_data in enumerate(self.dataloader_dict[mode]):
                    X, X_norm, Y, M = self._check_device([batch_data[0], batch_data[2], batch_data[3], batch_data[4]])
                    _len = torch.tensor(batch_data[1])
                    model_outs = self.model(X, X_norm, M, _len, Y)
                    f_list.append(model_outs['mean_o'].detach())
                    y_list.append(Y)
            return torch.cat(f_list, dim=0), torch.cat(y_list, dim=0)

        train_feats, train_labels = extract_features('train')
        valid_feats, valid_labels = extract_features('valid')
        self._logger.info('crossdata classifier: cached {} train / {} valid mean_o features'.format(
            len(train_labels), len(valid_labels)))

        # 3. 只训练轻量分类器(在缓存的mean_o特征上)
        head_optimizer = Adam(head.parameters(), lr=self._base_lr)
        head_loader = DataLoader(
            TensorDataset(train_feats, train_labels),
            batch_size=self.dataloader_dict['train'].batch_size,
            shuffle=True,
        )

        iter = 0
        val_accs = [-1]
        saved_epoch = -1
        for epoch in range(200):
            head.train()
            train_losses = []
            if epoch - saved_epoch > self._patience: # 早停
                self.early_stop(epoch, max(val_accs))
                break
            start_time = time.time()
            for i, (x_batch, y_batch) in enumerate(head_loader):
                if x_batch.shape[0] == 1: # BN在batch_size=1时报错
                    continue
                head_optimizer.zero_grad()
                out_o_logis = head(x_batch)
                o_loss = self.loss_fn(out_o_logis, y_batch)
                o_loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),
                                               max_norm=self._clip_grad_value)
                head_optimizer.step()
                train_losses.append(o_loss.item())
                iter += 1
            end_time = time.time()

            # 在缓存的valid特征上评估
            head.eval()
            with torch.no_grad():
                preds_o = head(valid_feats)
            val_acc = ((torch.max(preds_o, 1)[1] == valid_labels).sum() * 1.0 / len(valid_labels)).item()

            loss_message = 'Epoch [{}/{}] ({}) crossdata_head_loss: {:.4f}, {:.1f}s'.format(
                epoch, self._max_epochs, iter, np.mean(train_losses), (end_time - start_time))
            val_message = 'Epoch [{}/{}] ({}) crossdata_val_acc: o:{:.4f}'.format(
                epoch, self._max_epochs, iter, val_acc)

            self._logger.info(loss_message)
            self._logger.info(val_message)

            if val_acc > np.max(val_accs):
                # 保存完整模型: 冻结的bj编码器 + shl上重训的分类器(覆盖log_dir中复制的checkpoint)
                model_file_name = self.save_model(epoch, self._save_path, self._n_exp)
                self._logger.info(
                    'Val accuracy increases from {:.4f} to {:.4f}, '
                    'saving to {}'.format(np.max(val_accs), val_acc, model_file_name))
                val_accs.append(val_acc)
                saved_epoch = epoch

        self.logger.info('Save model at epoch {}, file at {}'.format(saved_epoch, self._save_path+model_file_name))
        self.logger.handlers.clear()

    def explain(self):
        baselines_results_base = {}
        baselines_results_four_,baselines_outs,conditions_sum_ = mask_raw_data(self.model, self.dataloader_dict['test'])
        baselines_results_four,baselines_outs = mask_base_ib(self.model,baselines_results_four_,baselines_outs,conditions_sum_,method = 'IB_top')
        baselines_results_base.update(baselines_results_four)


        for flag in ['GradFlag', 'IGFlag', 'SGFlag', 'TNGFlag', 'GBFlag', 'DLFlag', 'GSFlag', 'DLSFlag', 'FOFlag', 'FAFlag', 'FPFlag', 'SVSFlag', 'KSFlag', 'RNDFlag', 'FIXFlag', 'FREFlag', 'STDFlag']:
            if getattr(self.augs, flag):
                print(flag)
                baselines_results_four,baselines_outs = mask_base_baseline(self.model,baselines_results_four_,conditions_sum_,baselines_outs,method = flag) 
                baselines_results_base.update(baselines_results_four)



        baselines_results_base =  {k:get_reserve_dict(v)    for k,v in get_reserve_dict(baselines_results_base).items()}
        pickle.dump(baselines_results_base, open('{}/base_result_dict_{}.pkl'.format(self.save_path,self.augs.seed), 'wb'))
        # ave_png(baselines_results_base['all'],args.inter,'{}/base_mask_{}.png'.format(args.save_dir,args.seed))  # 百分位数的图
        ave_png(baselines_results_base['all'],'{}/base_mask_{}.png'.format(self.save_path,self.augs.seed))  # 百分位数的图

        return baselines_results_base


    def test_repinter(self, mode='test', testcausal=True, num_points=13):
        '''
        表示干预测试：point_att[:, :, 0] 是每个点属于 causal 的概率 (batch_size, seq_len)
        testcausal = True    去掉 causal 概率最高的 num_points 个点
        testcausal = False   去掉 causal 概率最低（spurious 最强）的 num_points 个点
        去掉的点缺失位置线性插值补全（X 和 X_norm 同步处理，只在有效长度内插值），重新前向，
        与干预前（干净输入）的 accuracy / 预测一致率 / macro P R F1 / 分类边际对比
        '''
        inter_tag = 'causal' if testcausal else 'spurious'
        labels = []
        preds_o = []
        preds_o_inter = []
        removed_att = []
        with torch.no_grad():
            self.model.eval()
            for _, batch_data in enumerate(self.dataloader_dict[mode]):
                X, X_norm, Y, M = self._check_device([batch_data[0], batch_data[2], batch_data[3], batch_data[4]])
                _len = torch.tensor(batch_data[1])

                # 1. 干净输入前向，取 point_att 的 causal 通道 (batch_size, seq_len)
                model_outs = self.model(X, X_norm, M, _len, Y)
                preds_o.append(model_outs['out_o_logis'].cpu())
                labels.append(batch_data[3])
                att = self.model.point_att[:, :, 0].cpu()  # padding 位置两通道均为 0，选点必须限制在有效长度内

                # 2. 每个样本去掉 att 最高/最低的 num_points 个点，缺失位置线性插值补全
                X_inter = X.clone()
                X_norm_inter = X_norm.clone()
                for i in range(att.shape[0]):
                    li = att.shape[1]
                    k = min(num_points, li)
                    att_i = att[i, :li]
                    if testcausal:
                        vals, idx = torch.topk(att_i, k)  # causal 最强的 k 个点
                    else:
                        vals, idx = torch.topk(att_i, k, largest=False)  # causal 最弱的 k 个点
                    removed_att.append(vals)
                    remove = np.zeros((1, li, 1))
                    remove[0, idx.numpy(), 0] = 1  # 1=去掉该点（liner_inter 中被置 nan 后插值）
                    X_inter[i:i + 1, :li] = liner_inter(X[i:i + 1, :li].cpu().numpy(),
                                                        np.repeat(remove, X.shape[2], axis=2)).to(X.device)
                    X_norm_inter[i:i + 1, :li] = liner_inter(X_norm[i:i + 1, :li].cpu().numpy(),
                                                             np.repeat(remove, X_norm.shape[2], axis=2)).to(X_norm.device)

                # 3. 干预后的输入重新前向
                model_outs_inter = self.model(X_inter, X_norm_inter, M, _len, Y)
                preds_o_inter.append(model_outs_inter['out_o_logis'].cpu())

        labels = torch.cat(labels, dim=0)
        preds_o = torch.cat(preds_o, dim=0)
        preds_o_inter = torch.cat(preds_o_inter, dim=0)
        _, outputs_o = torch.max(preds_o, 1)
        _, outputs_o_inter = torch.max(preds_o_inter, 1)

        ####### 计算指标
        y_true = labels.numpy()
        y_pred = outputs_o_inter.numpy()
        test_repinter_dict = {}
        test_repinter_dict['accuracy'] = (outputs_o == labels).sum().item() / len(labels)  # 干预前
        test_repinter_dict['accuracy_inter'] = (outputs_o_inter == labels).sum().item() / len(labels)  # 干预后
        test_repinter_dict['accuracyshift_inter'] = (outputs_o_inter == outputs_o).sum().item() / len(labels)  # 干预后与干预前预测一致的比例        
        test_repinter_dict['precision_inter'] = precision_score(y_true, y_pred, average='macro')
        test_repinter_dict['recall_inter'] = recall_score(y_true, y_pred, average='macro')
        test_repinter_dict['f1_inter'] = f1_score(y_true, y_pred, average='macro')
        test_repinter_dict['cm'] = self.classification_margin(preds_o, labels).mean().item()
        test_repinter_dict['cm_inter'] = self.classification_margin(preds_o_inter, labels).mean().item()
        test_repinter_dict['removed_att_mean'] = torch.cat(removed_att).mean().item()  # 被去掉点的平均 causal 概率

        # 保存结果
        filename = os.path.join(self.save_path, 'test_repinter_{}_results_{}.csv'.format(inter_tag, self._n_exp))
        results = pd.DataFrame(test_repinter_dict, index=[0])
        results.to_csv(filename, index=False)

        # 打印结果
        print('Test Repinter Results (remove {} {} points):'.format(num_points, inter_tag))
        for key, value in test_repinter_dict.items():
            print(f"{key}: {value:.4f}")

        return test_repinter_dict

    def test_intervene(self, mode='test'):
        '''
        环境干预测试（对环境特征做 do-干预）：
        固定 np.random.seed(42)，保证不同模型在完全相同的干预下对比
        （test 的 dataloader shuffle=False、collate 按长度降序排序，批次顺序跨模型一致）。
        测试集 X 维度是 batch_size x seq_len x feature_dim，环境特征为 X[..., 6:6+c_env]，
        GPS 等非环境特征（前 6 维及 X_norm）保持不变。
        对每个样本，在 c_env 个环境特征维度中随机选择 np.random.uniform(0, 1) 比例的维度
        （按比例向下取整，比例过小则该样本不干预；batch 只有 1 个样本时无法干预），
        干预方式是从同 batch 其他样本中随机选一个，用其同一环境特征序列替换
        （只在两者有效长度内替换，避免把 padding 引入有效区域）。
        将原始输入 e0 与干预输入 e1 分别送入同一个模型（out_o_logis），
        比较：1) 干预前后 accuracy 及差值；
              2) 每个类别的 softmax 概率变化 Delta_k = 1/N * sum_i[p_ik(e1) - p_ik(e0)]；
              3) S = 1/K * sum_k |Delta_k|，K 为类别数。
        模型对环境（伪相关）依赖越强，干预后输出变化越大，S 与 accuracy 下降越大。
        '''
        np.random.seed(42)  # 不同模型下干预完全一致
        c_env = self.model.c_env
        if c_env == 0:
            print('警告: env_dim=0，无环境特征可干预，干预结果将与原始输出相同')
        labels = []
        preds_o = []
        preds_o_inter = []
        ratio_list = []
        with torch.no_grad():
            self.model.eval()
            for _, batch_data in enumerate(self.dataloader_dict[mode]):
                X, X_norm, Y, M = self._check_device([batch_data[0], batch_data[2], batch_data[3], batch_data[4]])
                _len = torch.tensor(batch_data[1])
                b = X.shape[0]

                # 1. 原始输入 e0 前向
                model_outs = self.model(X, X_norm, M, _len, Y)
                preds_o.append(model_outs['out_o_logis'].cpu())
                labels.append(batch_data[3])

                # 2. 逐样本干预环境特征：X[..., 6:6+c_env] 中随机选 uniform(0,0.5) 比例的维度，
                #    用同 batch 内随机另一个样本的同一维度序列替换（限两者有效长度内）
                X_inter = X.clone()
                if b > 1 and c_env > 0:
                    len_arr = batch_data[1]
                    for i in range(b):
                        ratio = np.random.uniform(0, 0.5)
                        ratio_list.append(ratio)
                        
                        k_dims = int(c_env * ratio)  # 按比例向下取整
                        if k_dims == 0:
                            continue
                        
                        dims = np.random.choice(c_env, size=k_dims, replace=False)
                        for d in dims:
                            j = np.random.randint(b - 1)  # 均匀取 i 以外的样本
                            if j >= i:
                                j += 1
                            L_ij = min(len_arr[i], len_arr[j])
                            X_inter[i, :L_ij, 6 + d] = X[j, :L_ij, 6 + d]
                            # ratio2 = np.random.uniform(0.5, 1.5)
                            # X_inter[i, :L_ij, 6 + d] = X_inter[i, :L_ij, 6 + d] * ratio2
                        '''
                        k_dims = int(len_arr[i] * ratio)  # 改成在seq_len维度上随机选择比例的点进行干预
                        if k_dims == 0:
                            continue
                        dims = np.random.choice(len_arr[i], size=k_dims, replace=False)
                        for d in dims:
                            j = np.random.randint(b - 1)  # 均匀取 i 以外的样本
                            if j >= i:
                                j += 1
                            L_ij = min(len_arr[i], len_arr[j])
                            if d < L_ij:  # 确保索引在有效长度内
                                X_inter[i, d, 6:6+c_env] = X[j, d, 6:6+c_env]
                        '''
                # 3. 干预输入 e1 重新前向
                model_outs_inter = self.model(X_inter, X_norm, M, _len, Y)
                preds_o_inter.append(model_outs_inter['out_o_logis'].cpu())

        labels = torch.cat(labels, dim=0)
        preds_o = torch.cat(preds_o, dim=0)
        preds_o_inter = torch.cat(preds_o_inter, dim=0)
        _, outputs_o = torch.max(preds_o, 1)
        _, outputs_o_inter = torch.max(preds_o_inter, 1)

        ####### 计算指标
        test_intervene_dict = {}
        test_intervene_dict['accuracy'] = (outputs_o == labels).sum().item() / len(labels)  # 干预前 e0
        test_intervene_dict['accuracy_inter'] = (outputs_o_inter == labels).sum().item() / len(labels)  # 干预后 e1
        test_intervene_dict['accuracy_diff'] = test_intervene_dict['accuracy'] - test_intervene_dict['accuracy_inter']  # 准确率差值 e0-e1
        test_intervene_dict['pred_shift'] = (outputs_o_inter == outputs_o).sum().item() / len(labels)  # 干预后与干预前预测一致的比例

        # 每类 softmax 概率变化 Delta_k = 1/N sum_i[p_ik(e1)-p_ik(e0)]，及 S = 1/K sum_k|Delta_k|
        prob_e0 = F.softmax(preds_o, dim=-1)
        prob_e1 = F.softmax(preds_o_inter, dim=-1)
        delta_k = (prob_e1 - prob_e0).mean(dim=0).numpy()  # K
        for k in range(delta_k.shape[0]):
            test_intervene_dict['delta_class_{}'.format(k)] = float(delta_k[k])
        test_intervene_dict['S'] = float(np.abs(delta_k).mean())
        test_intervene_dict['intervene_ratio_mean'] = float(np.mean(ratio_list)) if len(ratio_list) > 0 else 0.0

        # 保存结果
        filename = os.path.join(self.save_path, 'test_intervene_results_{}.csv'.format(self._n_exp))
        results = pd.DataFrame(test_intervene_dict, index=[0])
        results.to_csv(filename, index=False)

        # 打印结果
        print('Test Intervene Results (env do-intervention, ratio~U(0,0.5)):')
        for key, value in test_intervene_dict.items():
            print(f"{key}: {value:.4f}")

        return test_intervene_dict

    def test_spurious_withoutretraing(self, mode='test'):
        '''
        仅用伪相关表示 Z_S（model_outs['mean_s']）评估分类性能，不重新训练任何参数：
        直接用原始模型里训练好的 object_readout_layer（平时吃 mean_o 的头）来分类 mean_s，
        若 Z_S 确实与标签无关（或与 Z_C 不在同一标签子空间），swap 后 accuracy 应接近随机水平 1/c_out。
        同时记录正常输入 mean_o 时同一头的 accuracy 作为对照。
        '''
        labels = []
        preds_o = []
        preds_o_swap = []
        with torch.no_grad():
            self.model.eval()
            for _, batch_data in enumerate(self.dataloader_dict[mode]):
                X, X_norm, Y, M = self._check_device([batch_data[0], batch_data[2], batch_data[3], batch_data[4]])
                _len = torch.tensor(batch_data[1])
                model_outs = self.model(X, X_norm, M, _len, Y)
                preds_o.append(model_outs['out_o_logis'].cpu())  # 正常: mean_o -> object_readout_layer
                # 表示交换: mean_s(Z_S) -> object_readout_layer（不训练，直接前向）
                preds_o_swap.append(self.model.object_readout_layer(model_outs['mean_s']).cpu())
                labels.append(batch_data[3])

        labels = torch.cat(labels, dim=0)
        preds_o = torch.cat(preds_o, dim=0)
        preds_o_swap = torch.cat(preds_o_swap, dim=0)
        _, outputs_o = torch.max(preds_o, 1)
        _, outputs_o_swap = torch.max(preds_o_swap, 1)

        ####### 计算指标
        y_true = labels.numpy()
        y_pred = outputs_o_swap.numpy()
        test_spurious_wotr_dict = {}
        test_spurious_wotr_dict['accuracy'] = (outputs_o == labels).sum().item() / len(labels)  # Z_C 正常分类
        test_spurious_wotr_dict['accuracy_swap'] = (outputs_o_swap == labels).sum().item() / len(labels)  # Z_S 交换后分类
        test_spurious_wotr_dict['chance'] = 1.0 / self.model.c_out  # 随机水平
        test_spurious_wotr_dict['accuracyshift_swap'] = (outputs_o_swap == outputs_o).sum().item() / len(labels)  # 与正常预测一致的比例
        test_spurious_wotr_dict['precision_swap'] = precision_score(y_true, y_pred, average='macro')
        test_spurious_wotr_dict['recall_swap'] = recall_score(y_true, y_pred, average='macro')
        test_spurious_wotr_dict['f1_swap'] = f1_score(y_true, y_pred, average='macro')
        test_spurious_wotr_dict['cm_swap'] = self.classification_margin(preds_o_swap, labels).mean().item()

        # 保存结果
        filename = os.path.join(self.save_path, 'test_spurious_withoutretraing_results_{}.csv'.format(self._n_exp))
        results = pd.DataFrame(test_spurious_wotr_dict, index=[0])
        results.to_csv(filename, index=False)

        # 打印结果
        print('Test Spurious Without Retraining Results (mean_s -> object_readout_layer):')
        for key, value in test_spurious_wotr_dict.items():
            print(f"{key}: {value:.4f}")

        return test_spurious_wotr_dict
        
    def test_envprob(self, mode='test', num_epochs=100, patience=50):
        '''
        环境探测（environment probing）实验：
        冻结原始模型，用训练集的 causal embedding（model_outs['mean_o']，b×embed_dim）作为输入，
        训练一个自动解码器（LSTMCell(embed_dim, embed_dim) 自回归展开 + Linear(embed_dim, c_env)），
        希望重建环境变量序列 envs = inputs[..., 6:6+c_env]（用模型自带的 env_bn 归一化后作目标，
        各维尺度可比，且与编码器实际输入一致），valid 上按 MAE 早停选 epoch，
        在 test 上评估 MAE、RMSE，并给出"训练集逐维均值"这一平凡 baseline 作参照——
        若 causal 表示确实不含环境信息，解码器不应显著优于 baseline。
        feat_key 可换成 'mean_s'（spurious 表示）作对照，预期重建误差更小。
        '''
        feat_key = self.augs.feat_key
        class EnvProbDecoder(nn.Module):
            def __init__(self, embed_dim, c_env):
                super(EnvProbDecoder, self).__init__()
                self.decoder = nn.LSTMCell(embed_dim, embed_dim)  # hidden_size, hidden_size
                self.head = nn.Linear(embed_dim, c_env)

            def forward(self, feat, n_time):
                # feat: b, embed_dim -> 输出 b, n_time, c_env（自回归展开，不喂任何真实环境信息）
                h, c = feat, torch.zeros_like(feat)
                x_in = torch.zeros_like(feat)
                preds = []
                for _ in range(n_time):
                    h, c = self.decoder(x_in, (h, c))
                    preds.append(self.head(h))
                    x_in = h
                return torch.stack(preds, dim=1)

        # 1. 冻结原始模型，提取各 split 的 mean_o 特征与 env_bn 归一化后的环境序列目标
        feats, envs, lens = {}, {}, {}
        with torch.no_grad():
            self.model.eval()
            for split in ['train', 'valid', mode]:
                f_list, e_list, l_list = [], [], []
                for _, batch_data in enumerate(self.dataloader_dict[split]):
                    X, X_norm, Y, M = self._check_device([batch_data[0], batch_data[2], batch_data[3], batch_data[4]])
                    _len = torch.tensor(batch_data[1])
                    model_outs = self.model(X, X_norm, M, _len, Y)
                    f_list.append(model_outs[feat_key].cpu())
                    env = X[..., 6:6 + self.model.c_env]
                    e_list.append(env.cpu())
                    l_list.append(_len)
                feats[split] = torch.cat(f_list, dim=0)
                envs[split] = torch.cat(e_list, dim=0)
                lens[split] = torch.cat(l_list, dim=0)

        n_time = envs['train'].shape[1]
        arange_t = torch.arange(n_time)

        def masked_metrics(pred, target, length):
            # 只在有效长度内计算 MAE / RMSE（padding 不计）
            w = (arange_t[None, :] < length[:, None]).unsqueeze(-1).float()  # b, n_time, 1
            diff = pred - target
            mae = (diff.abs() * w).sum().item() / (w.sum().item() * target.shape[2])
            rmse = torch.sqrt(((diff ** 2) * w).sum() / (w.sum() * target.shape[2])).item()
            return mae, rmse

        # 2. 新建解码器，只在训练集上训练（原始模型参数不参与）
        decoder = EnvProbDecoder(self.augs.embed_dim, self.model.c_env).to(self._device)
        optimizer = Adam(decoder.parameters())
        dataset = TensorDataset(feats['train'], envs['train'], lens['train'])
        loader = DataLoader(dataset, batch_size=self.augs.batch_size, shuffle=True)
        best_val_mae, best_state, saved_epoch = float('inf'), None, -1
        for epoch in range(num_epochs):
            decoder.train()
            for f_batch, e_batch, l_batch in loader:
                f_batch, e_batch = f_batch.to(self._device), e_batch.to(self._device)
                optimizer.zero_grad()
                pred = decoder(f_batch, n_time)
                w = (arange_t[None, :] < l_batch[:, None]).unsqueeze(-1).float().to(self._device)
                loss = ((pred - e_batch) ** 2 * w).sum() / (w.sum() * e_batch.shape[2])  # masked MSE
                loss.backward()
                optimizer.step()

            decoder.eval()
            with torch.no_grad():
                val_pred = decoder(feats['valid'].to(self._device), n_time).cpu()
            val_mae, _ = masked_metrics(val_pred, envs['valid'], lens['valid'])
            if val_mae < best_val_mae:
                best_val_mae, saved_epoch = val_mae, epoch
                best_state = copy.deepcopy(decoder.state_dict())
            if epoch - saved_epoch > patience:  # 早停
                break
        decoder.load_state_dict(best_state)
        decoder.eval()

        # 3. test 上评估，并计算"训练集逐维均值"平凡 baseline
        with torch.no_grad():
            pred = decoder(feats[mode].to(self._device), n_time).cpu()
        mae, rmse = masked_metrics(pred, envs[mode], lens[mode])

        w_train = (arange_t[None, :] < lens['train'][:, None]).unsqueeze(-1).float()
        train_mean = (envs['train'] * w_train).sum(dim=(0, 1)) / w_train.sum(dim=(0, 1))  # c_env
        baseline_pred = train_mean[None, None, :].expand_as(envs[mode])
        mae_baseline, rmse_baseline = masked_metrics(baseline_pred, envs[mode], lens[mode])

        test_envprob_dict = {}
        test_envprob_dict['val_mae'] = best_val_mae
        test_envprob_dict['mae'] = mae
        test_envprob_dict['rmse'] = rmse
        test_envprob_dict['mae_baseline'] = mae_baseline  # 训练集逐维均值
        test_envprob_dict['rmse_baseline'] = rmse_baseline
        test_envprob_dict['mae_ratio'] = mae / (mae_baseline + 1e-10)  # <1 说明解码器优于平凡baseline
        test_envprob_dict['epochs'] = saved_epoch + 1

        # 保存结果
        filename = os.path.join(self.save_path, 'test_envprob_{}_results_{}.csv'.format(feat_key, self._n_exp))
        results = pd.DataFrame(test_envprob_dict, index=[0])
        results.to_csv(filename, index=False)

        # 打印结果
        print('Test EnvProb Results ({} -> envs):'.format(feat_key))
        for key, value in test_envprob_dict.items():
            print(f"{key}: {value:.4f}")

        return test_envprob_dict


    def test_spurious(self, mode='test', feat_key='mean_s', num_epochs=10, patience=10):
        '''
        仅用伪相关表示 Z_S（model_outs['mean_s']）评估分类性能，验证其是否确实与标签无关：
        冻结原始模型（不更新），提取各 split 的 mean_s 特征 (num, embed_dim)，
        在训练集特征上训练一个新的 ReadoutLayer 分类器，valid 上选最优 epoch，test 上评估，
        accuracy 应接近随机水平 1/c_out。feat_key 可换成 'mean_o' 作为因果表示的对照实验。
        '''
        # 1. 冻结原始模型，提取 train/valid/test 的 mean_s 特征
        feats, labels = {}, {}
        with torch.no_grad():
            self.model.eval()
            for split in ['train', 'valid', mode]:
                f_list, y_list = [], []
                for _, batch_data in enumerate(self.dataloader_dict[split]):
                    X, X_norm, Y, M = self._check_device([batch_data[0], batch_data[2], batch_data[3], batch_data[4]])
                    _len = torch.tensor(batch_data[1])
                    model_outs = self.model(X, X_norm, M, _len, Y)
                    f_list.append(model_outs[feat_key].cpu())
                    y_list.append(batch_data[3])
                feats[split] = torch.cat(f_list, dim=0)
                labels[split] = torch.cat(y_list, dim=0)

        # 2. 新建 ReadoutLayer，只在训练集特征上训练（原始模型参数不参与）
        readout = ReadoutLayer(self.augs.embed_dim, self.model.c_out).to(self._device)
        optimizer = Adam(readout.parameters())
        dataset = TensorDataset(feats['train'], labels['train'])
        loader = DataLoader(dataset, batch_size=self.augs.batch_size, shuffle=True)
        best_val_acc, best_state, saved_epoch = -1, None, -1
        for epoch in range(num_epochs):
            readout.train()
            for f_batch, y_batch in loader:
                f_batch, y_batch = f_batch.to(self._device), y_batch.to(self._device)
                optimizer.zero_grad()
                loss = self.loss_fn(readout(f_batch), y_batch)
                loss.backward()
                optimizer.step()

            readout.eval()
            with torch.no_grad():
                val_logits = readout(feats['valid'].to(self._device)).cpu()
            _, val_pred = torch.max(val_logits, 1)
            val_acc = (val_pred == labels['valid']).sum().item() / len(labels['valid'])
            if val_acc > best_val_acc:
                best_val_acc, saved_epoch = val_acc, epoch
                best_state = copy.deepcopy(readout.state_dict())
            if epoch - saved_epoch > patience:  # 早停
                break
        readout.load_state_dict(best_state)
        readout.eval()

        # 3. test 上评估
        with torch.no_grad():
            logits = readout(feats[mode].to(self._device)).cpu()
        _, pred = torch.max(logits, 1)
        y_true = labels[mode].numpy()
        y_pred = pred.numpy()

        test_spurious_dict = {}
        test_spurious_dict['val_accuracy'] = best_val_acc
        test_spurious_dict['accuracy'] = (pred == labels[mode]).sum().item() / len(labels[mode])
        test_spurious_dict['chance'] = 1.0 / self.model.c_out  # 随机水平
        test_spurious_dict['precision'] = precision_score(y_true, y_pred, average='macro')
        test_spurious_dict['recall'] = recall_score(y_true, y_pred, average='macro')
        test_spurious_dict['f1'] = f1_score(y_true, y_pred, average='macro')
        test_spurious_dict['cm'] = self.classification_margin(logits, labels[mode]).mean().item()
        test_spurious_dict['epochs'] = saved_epoch + 1

        # 保存结果
        filename = os.path.join(self.save_path, 'test_spurious_{}_results_{}.csv'.format(feat_key, self._n_exp))
        results = pd.DataFrame(test_spurious_dict, index=[0])
        results.to_csv(filename, index=False)

        # 打印结果
        print('Test Spurious Results ({}):'.format(feat_key))
        for key, value in test_spurious_dict.items():
            print(f"{key}: {value:.4f}")

        return test_spurious_dict
         
    
    def test_roubust(self, mode='test'):
        test_roubust_dict= {}
        labels = []
        preds_o = []
        out_o_logis_list = []
        with torch.no_grad():
            self.model.eval()
            for _, batch_data in enumerate(self.dataloader_dict[mode]):
                out_c_logis, out_o_logis, out_co = self.test_batch(batch_data)
                labels.append(batch_data[3])
                preds_o.append(out_o_logis.cpu())

        labels = torch.cat(labels, dim=0)
        preds_o = torch.cat(preds_o, dim=0)

        _, outputs_o = torch.max(preds_o, 1)

        
        ####### 计算指标
        # 将标签和预测结果从GPU转移到CPU
        y_true = labels.cpu().numpy()
        y_pred = outputs_o.cpu().numpy()
        accuracy_list = accuracy(preds_o, labels, topk=(1,1,1))
        # accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')  # 'macro'表示宏平均
        f1 = f1_score(y_true, y_pred, average='macro')
        
        test_roubust_dict['accuracy@1'] = accuracy_list[0].item()
        test_roubust_dict['accuracy@2'] = accuracy_list[1].item()
        test_roubust_dict['accuracy@5'] = accuracy_list[2].item()
        test_roubust_dict['precision'] = precision
        test_roubust_dict['recall'] = recall
        test_roubust_dict['f1'] = f1
        
        
        # 分类边界（Classification Margin）：衡量模型在正确类别与最可能错误类别之间的输出差距。

        cm = self.classification_margin(preds_o, labels)
        test_roubust_dict['cm'] = cm.mean().item()  # 计算平均分类边际
        
        if self.model.augs.roubust == 'imbalance':
            # 审稿人意见 #20: 不平衡设置下除整体准确率外，还需报告每类 F1 和每类 G-mean
            n_classes = preds_o.shape[1]
            class_range = np.arange(n_classes)
            # 每类 F1（average=None 返回每个类别的 F1）
            per_class_f1 = f1_score(y_true, y_pred, labels=class_range, average=None, zero_division=0)
            # 每类召回率（即 one-vs-rest 的 TPR）
            per_class_recall = recall_score(y_true, y_pred, labels=class_range, average=None, zero_division=0)
            # 每类 G-mean（one-vs-rest）: sqrt(召回率TPR * 特异度TNR)
            # TNR = 其余类样本中被正确判为"非该类"的比例，TN/(TN+FP)
            cm = confusion_matrix(y_true, y_pred, labels=class_range)
            TP = np.diag(cm).astype(np.float64)
            FP = cm.sum(axis=0) - TP
            TN = cm.sum() - TP - FP - (cm.sum(axis=1) - TP)
            tpr = per_class_recall
            tnr = TN / np.maximum(TN + FP, 1)  # 分母为0（该类无负样本）时记为0
            per_class_gmean = np.sqrt(tpr * tnr)

            for cls_idx in range(n_classes):
                test_roubust_dict['f1_class_{}'.format(cls_idx)] = per_class_f1[cls_idx]
                test_roubust_dict['recall_class_{}'.format(cls_idx)] = per_class_recall[cls_idx]
                test_roubust_dict['gmean_class_{}'.format(cls_idx)] = per_class_gmean[cls_idx]
            test_roubust_dict['g_mean'] = per_class_gmean.mean()  # 宏平均 G-mean
        
        if self.model.augs.roubust == 'noise':  # 扰动破坏
            with torch.no_grad():

                roubu_ratio = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
                roubu_ratio = [1-i for i in roubu_ratio]
                for roubust_size in roubu_ratio:
                    test_roubust_dict['accuracy_{}'.format(roubust_size)] = 0
                    test_roubust_dict['accuracyshift_{}'.format(roubust_size)] = 0
                    test_roubust_dict['precision_{}'.format(roubust_size)] = 0
                    test_roubust_dict['recall_{}'.format(roubust_size)] = 0
                    test_roubust_dict['f1_{}'.format(roubust_size)] = 0
                    sum_len = 0
                    pred_i_list = []

                    if ('bj' in self.model.augs.model_name) or  ('shl' in self.model.augs.model_name) :
                        for i, batch_data in enumerate(self.dataloader_dict[mode]):
                            X,X_norm, Y ,M = self._check_device([batch_data[0], batch_data[2],batch_data[3],batch_data[4]])
                            _len = torch.tensor(batch_data[1])

                            len_p = int(X.shape[1] * roubust_size) + 1 
                            seed_torch(seed=i+42)
                            choice_id = np.random.choice(X.shape[1], len_p, replace=False) 
                            mask = np.zeros(X.shape)
                            mask[:,choice_id,:] = 1 # 被mask的点是1
                            
                            mask = torch.tensor(mask).long().to(self.model.augs.device)
                            X_noisy = liner_inter(X.cpu().numpy(),(1-mask).cpu().numpy()).to(self.model.augs.device)
                            # X_noisy = F.relu(X + mask * torch.randn_like(X))
                            # X_noisy = self.truncated_latlon_noise(X[:,:,:2]) * (1 - mask[:,:,:2]) # 添加噪声
                            std, mean = torch.std_mean(X_noisy, dim=1, keepdim=True) # batch_size,1, dim
                            noise = torch.randn_like(X_noisy) 
                            #noise = torch.clamp(noise,  std,  std)
                            X_noisy = X_noisy  # + noise * (1 - mask)
                            
                            mask =  mask[:,:,[0]].expand_as(X_norm)
                            # X_norm_noisy = X_norm
                            X_norm_noisy = liner_inter(X_norm.cpu().numpy(),(1-mask).cpu().numpy()).to(self.model.augs.device)
                            std, mean = torch.std_mean(X_norm_noisy, dim=1, keepdim=True) # batch_size,1, dim
                            noise = torch.randn_like(X_norm_noisy) 
                            #noise = torch.clamp(noise,  std,  std)
                            X_norm_noisy = X_norm_noisy # + noise * (1 - mask) 
                            model_outs = self.model(X_noisy, X_norm_noisy, M, _len,Y)
                            out_o_logis = model_outs['out_o_logis']
                            pred_i_list.append(out_o_logis)
                    elif 'wh' in self.model.augs.model_name: #  原始的wh
                        for i, batch_data in enumerate(self.dataloader_dict[mode]):
                            X,X_norm, Y ,M = self._check_device([batch_data[0], batch_data[2],batch_data[3],batch_data[4]])
                            _len = torch.tensor(batch_data[1])

                            len_p = int(X.shape[1] * roubust_size) + 1 
                            seed_torch(seed=i+42)
                            choice_id = np.random.choice(X.shape[1], len_p, replace=False) 
                            mask = np.zeros(X.shape)
                            mask[:,choice_id,:] = 1 # 被mask的点是1
                            
                            mask = torch.tensor(mask).long().to(self.model.augs.device)
                            # X_noisy = F.relu(X + mask * torch.randn_like(X)) # # liner_inter(X.cpu().numpy(),mask).to(self.model.augs.device)
                            # X_noisy = F.relu(X + mask * torch.randn_like(X))
                            X_noisy = X 
                            # X_noisy = self.truncated_latlon_noise(X[:,:,:2]) * (1 - mask[:,:,:2]) # 添加噪声
                            std, mean = torch.std_mean(X_noisy, dim=1, keepdim=True) # batch_size,1, dim
                            noise = torch.randn_like(X_noisy) *  std 
                            #noise = torch.clamp(noise,  std,  std)
                            X_noisy = X_noisy  + noise * (1 - mask)
                            
                            X_norm_noisy = X_norm
                            std, mean = torch.std_mean(X_norm_noisy, dim=1, keepdim=True) # batch_size,1, dim
                            noise = torch.randn_like(X_norm_noisy) * std 
                            #noise = torch.clamp(noise,  std,  std)
                            X_norm_noisy = X_norm_noisy  + noise * (1 - mask[:,:,[0]])  
                            
                            model_outs = self.model(X_noisy, X_norm_noisy, M, _len,Y)
                            out_o_logis = model_outs['out_o_logis']
                            pred_i_list.append(out_o_logis)


                    preds_o = torch.cat(pred_i_list, dim=0)
                    _, preds_o = torch.max(preds_o, 1)
                    test_roubust_dict['accuracy_{}'.format(roubust_size)] += (preds_o == labels.to(preds_o.device)).sum().item()  / len(labels)
                    test_roubust_dict['accuracyshift_{}'.format(roubust_size)] += (preds_o == outputs_o.to(preds_o.device)).sum().item() / len(labels)
                    test_roubust_dict['precision_{}'.format(roubust_size)] += precision_score(y_true, preds_o.cpu().numpy(), average='macro')
                    test_roubust_dict['recall_{}'.format(roubust_size)] += recall_score(y_true, preds_o.cpu().numpy(), average='macro')  # 'macro'表示宏平均
                    test_roubust_dict['f1_{}'.format(roubust_size)] += f1_score(y_true, preds_o.cpu().numpy(), average='macro')
                    
            for i in roubu_ratio: 
                test_roubust_dict['accratio_{}'.format(i)] = test_roubust_dict['accuracy_{}'.format(i)] / (test_roubust_dict['accuracy@1'] + 1e-10)  # 防止除以0
            
        # 保存结果
        test_roubust_dict = dict(sorted(test_roubust_dict.items(), reverse=True))
        filename = os.path.join(self.save_path, 'test_roubust_results_{}.csv'.format(self._n_exp))
        results = pd.DataFrame(test_roubust_dict, index=[0])
        results.to_csv(filename, index=False)
        
        # 打印结果
        print('Test Robust Results:')
        for key, value in test_roubust_dict.items():
            print(f"{key}: {value:.4f}")

        return test_roubust_dict
    
    
    def classification_margin(self, logits, label):
        eps = 1e-10  # 防止对数计算中的数值稳定性问题
        # 获取类别的概率分布（通过softmax函数）
        probs = F.softmax(logits, dim=-1)
        
        # 真实类别的概率（softmax后的概率）
        true_class_probs = probs[torch.arange(len(label)), label.cpu()]
        
        # 计算对数概率
        log_true_class_probs = torch.log(true_class_probs + eps)
        
        # 计算所有其他类别的对数概率
        other_class_probs = probs.clone()
        other_class_probs[torch.arange(len(label)), label.cpu()] = 0  # 将真实类别的概率设为0，避免计算
        log_other_class_probs = torch.log(other_class_probs + eps)
        
        # 分类边际计算
        cm = log_other_class_probs.max(dim=-1).values - log_true_class_probs
        
        return -cm # 越大越好
def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) #
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    
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
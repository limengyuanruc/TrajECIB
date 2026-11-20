#%%
import os
#os.chdir('/home/DATA2/limengyuan/MYCL')
import re
from datetime import datetime
from args import get_config
from utils.helper import get_dataloader_dict,get_data_dir,seed_torch,weight_init,average_dict_values,ave_png
# from torch.utils.tensorboard import SummaryWriter
from model.trajcausal_baseline import TrajCausal as TrajCausalBase
from model.trajcausal_dualstream import TrajCausal as TrajCausalFinal
from model.trajcausal_dualstream import TrajCausal_Decoder
from trainer.causal_trainer import CausalTrainer
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
import pickle
import shutil
#%%
def main(augs):
    augs.n_exp = augs.seed # 实现的次数
    seed_torch(seed=augs.seed)
    # 格式化日期为 "YYYYMMDD" 格式
    augs_str = '  \n'.join(f'{k}: {v}' for k, v in vars(augs).items())
    # writer = SummaryWriter(log_dir=os.path.join(augs.my_data_dir,'runs',augs.log_dir_i))
    # writer.add_text('Config Parameters', augs_str)
    print(augs)
    if 'base' in augs.model_name:
        close_causal = True
        model = TrajCausalBase(augs= augs,
                        name = augs.model_name,
                        c_in = augs.input_dim,
                        c_env = augs.env_dim,
                        c_out = output_dim,
                        embed_dim = augs.embed_dim,
                        base_encoder = augs.base_encoder,
                        qk_dim = augs.latent_dim,
                        envbook_num = augs.envbook_num,
                        promot = augs.promot_num,
                        envbook_dim = augs.envbook_dim,
                        cat_or_add = augs.cat_or_add,
                        with_random = augs.with_random,
                        without_point_attention = close_causal,
                        )
        
    else:
        close_causal = False
        model = TrajCausalFinal(
                        augs= augs,
                        name = augs.model_name,
                        c_in = augs.input_dim,
                        c_env = augs.env_dim,
                        c_out = output_dim,
                        embed_dim = augs.embed_dim,
                        base_encoder = augs.base_encoder,
                        qk_dim = augs.latent_dim,
                        envbook_num = augs.envbook_num,
                        envbook_dim = augs.envbook_dim,
                        promot = augs.promot_num,
                        cat_or_add = augs.cat_or_add,
                        with_random = augs.with_random,
                        without_point_attention = close_causal,
                        )
        model_deocoder = TrajCausal_Decoder(
                        augs= augs,
                        model = model,
                        name = augs.model_name,
                        c_in = augs.input_dim,
                        c_env = augs.env_dim,
                        c_out = output_dim,
                        embed_dim = augs.embed_dim,
                        base_encoder = augs.base_encoder,
                        qk_dim = augs.latent_dim,
                        envbook_num = augs.envbook_num,
                        envbook_dim = augs.envbook_dim,
                        cat_or_add = augs.cat_or_add,
                        with_random = augs.with_random,
                        without_point_attention = close_causal,
                        )

    print('=========param_num=========', model.param_num('none'))
    # writer.add_text('Model param_num', str(model.param_num('none')))
    # 初始化
    model.apply(weight_init)
    model_deocoder.apply(weight_init)
    # model = torch.nn.DataParallel(model)
    trainer = CausalTrainer(
                    augs=augs,
                    model=model,# model
                    model_deocoder = model_deocoder,
                    dataloader_dict=dataloader_dict,
                    base_lr=augs.base_lr,
                    steps=[35,70],
                    lr_decay_ratio=augs.lr_decay_ratio,
                    log_dir=augs.log_dir,
                    n_exp=augs.n_exp,
                    save_iter=augs.save_iter,
                    clip_grad_value=augs.max_grad_norm,
                    max_epochs=augs.max_epochs,
                    patience=augs.patience,
                    o = augs.o,
                    c = augs.c,
                    co = augs.co,
                    only_object = close_causal, # close_causal
                    # writer = writer,
                    device = 'cuda:{}'.format(augs.gpu) if (augs.gpu >= 0 & torch.cuda.is_available()) else 'cpu',
                    )

    print(model)
    if augs.mode == 'train':
        trainer.train()
        
        val_acc_c, val_acc_o, val_acc_co, test_acc_c, test_acc_o, test_acc_co = trainer.test(-1)
        text = f'val_acc_c: {val_acc_c}, val_acc_o: {val_acc_o}, val_acc_co: {val_acc_co}, ' + \
                f'test_acc_c: {test_acc_c}, test_acc_o: {test_acc_o}, test_acc_co: {test_acc_co}'
        # writer.add_text('TestAcc', text)

        print(f"val_acc_o: {val_acc_o}")
        print(f"test_acc_o: {test_acc_o}")

        print(augs.model_name)
        if augs.robusttest:
            print('robust test')
            test_roubust_list = trainer.test_roubust()
        else:
            test_roubust_list = {}
        if augs.explain:
            print('explain')
            baselines_results_base = trainer.explain()
        else:
            baselines_results_base = {}

        
    elif augs.mode == 'test':        
        # 将MYCL/trained中训练好的模型复制到log_dir中
        if not os.path.exists(augs.log_dir):
            os.makedirs(augs.log_dir)
        
        # 在 augs.my_data_dir/trained_new/{city}/{imb|few|all}/ 下查找训练好的模型文件夹
        if 'shl' in augs.model_name:
            city = 'shl'
        elif 'wh' in augs.model_name:
            city = 'wh'
        else:
            city = 'bj'
        if augs.roubust == 'imbalance':
            subdir = 'imb'
            roubust_size_temp = '_im' + ''.join(str(augs.roubust_size).split('.')) + '_'
        elif augs.roubust == 'few':
            subdir = 'few'
            roubust_size_temp = 'few' + ''.join(str(augs.roubust_size).split('.'))
        else:
            subdir = 'all'
            roubust_size_temp = ''
        # 文件夹名需以 _{name最后一段}-{base_encoder} 结尾（如 ..._cl-cl），
        # 用完整后缀匹配避免子串误匹配（如 base_encoder=cl 时误选 dann-cl/base-cl 等文件夹）
        name_last = augs.name.split('_')[-1]
        folder_suffix = '_' + name_last + '-' + augs.base_encoder
        model_dir = os.path.join(augs.my_data_dir, 'trained_new', city, subdir)
        model_path = None
        if os.path.isdir(model_dir):
            model_folders = sorted(f for f in os.listdir(model_dir)
                                   if os.path.isdir(os.path.join(model_dir, f))
                                   and f.endswith(folder_suffix)
                                   and roubust_size_temp in f)
            if len(model_folders) > 0:
                # 多个匹配时取排序后最后一个（日期最新的）
                model_path = os.path.join(model_dir, model_folders[-1], str(augs.seed), 'final_model_{}.pt'.format(augs.seed))
                print('load trained model from {}'.format(model_path))
            else:
                print('警告: 在{}下未找到以{}结尾的模型文件夹'.format(model_dir, folder_suffix))
        else:
            print('警告: 目录{}不存在'.format(model_dir))
        # 将model_path复制到augs.log_dir中
        if (model_path is not None) and os.path.exists(model_path):

            shutil.copy(model_path, os.path.join(augs.log_dir, 'final_model_{}.pt'.format(augs.seed)))
        
        val_acc_c, val_acc_o, val_acc_co, test_acc_c, test_acc_o, test_acc_co = trainer.test(-1)
        text = f'val_acc_c: {val_acc_c}, val_acc_o: {val_acc_o}, val_acc_co: {val_acc_co}, ' + \
                f'test_acc_c: {test_acc_c}, test_acc_o: {test_acc_o}, test_acc_co: {test_acc_co}'
        # writer.add_text('TestAcc', text)

        print(f"val_acc_o: {val_acc_o}")
        print(f"test_acc_o: {test_acc_o}")

        print(augs.model_name)
        if augs.robusttest:
            print('robust test')
            test_roubust_list = trainer.test_roubust()
        else:
            test_roubust_list = {}
        if augs.repinter == 'c':
            print('repinter test')
            test_roubust_list = trainer.test_repinter(testcausal=True)   # 去掉 causal 概率最高的10个点
        if augs.repinter == 's':
            test_roubust_list = trainer.test_repinter(testcausal=False)  # 去掉 causal 概率最低（spurious最强）的10个点
        if augs.spurioustest:
            print('spurious test')
            test_roubust_list = trainer.test_spurious()  # 冻结模型，仅用Z_S(mean_s)训练ReadoutLayer分类
        if augs.spuriouswotr:
            print('spurious without retraining test')
            test_roubust_list = trainer.test_spurious_withoutretraing()  # Z_S(mean_s)直接过已训练的object_readout_layer
        if augs.envprob:
            print('envprob test')
            test_roubust_list = trainer.test_envprob()  # 冻结模型，用mean_o重建envs环境序列（MAE/RMSE）
        if augs.explain:
            print('explain')
            baselines_results_base = trainer.explain()
        if augs.intervene:
            print('intervene test')
            # 环境干预测试：固定np.random.seed(42)保证不同模型下干预一致，对测试集每个样本随机选
            # uniform(0,1)比例的环境特征维度，用同batch其他样本的同一环境特征替换，
            # 原始/干预输入分别前向，比较accuracy差值、每类softmax概率变化Delta_k及S=mean|Delta_k|
            # 实现见MYCL/trainer/causal_trainer.py的test_intervene
            test_roubust_list = trainer.test_intervene()
        if augs.explain:
            print('explain')
            baselines_results_base = trainer.explain()
        else:
            baselines_results_base = {}
    elif augs.mode == 'test_crossdata_shl_withtr': # 用bj的模型测试shl数据集(带轻量分类器重训练)
        # 将trained中训练好的bj模型复制到log_dir中(与test_crossdata_shl相同的checkpoint来源)
        if not os.path.exists(augs.log_dir):
            os.makedirs(augs.log_dir)
        model_name_temp = augs.model_name.replace('shl','bj')
        model_path = os.path.join(augs.my_data_dir, 'trained',model_name_temp, augs.base_encoder, str(augs.seed),'final_model_{}.pt'.format(augs.seed))
        # 将model_path复制到augs.log_dir中
        if os.path.exists(model_path):
            shutil.copy(model_path, os.path.join(augs.log_dir, 'final_model_{}.pt'.format(augs.seed)))
        else:
            print('警告: 未找到bj模型 {}'.format(model_path))

        # 冻结bj编码器, 用bj的mean_o作为shl的mean_o, 只重训轻量级分类器(见trainer/causal_trainer.py)
        trainer.train_crossdata_classifier()

        val_acc_c, val_acc_o, val_acc_co, test_acc_c, test_acc_o, test_acc_co = trainer.test(-1)
        text = f'val_acc_c: {val_acc_c}, val_acc_o: {val_acc_o}, val_acc_co: {val_acc_co}, ' + \
                f'test_acc_c: {test_acc_c}, test_acc_o: {test_acc_o}, test_acc_co: {test_acc_co}'
        # writer.add_text('TestAcc', text)

        print(f"val_acc_o: {val_acc_o}")
        print(f"test_acc_o: {test_acc_o}")

        print(augs.model_name)
        if augs.robusttest:
            print('robust test')
            test_roubust_list = trainer.test_roubust()
        else:
            test_roubust_list = {}
        if augs.explain:
            print('explain')
            baselines_results_base = trainer.explain()
        else:
            baselines_results_base = {}
    elif augs.mode == 'test_crossdata_shl': # 用bj的模型测试shl数据集
        # 将MYCL/trained中训练好的模型复制到log_dir中
        if not os.path.exists(augs.log_dir):
            os.makedirs(augs.log_dir)
        model_name_temp = augs.model_name.replace('shl','bj')
        model_path = os.path.join(augs.my_data_dir, 'trained',model_name_temp, augs.base_encoder, str(augs.seed),'final_model_{}.pt'.format(augs.seed))
        # 将model_path复制到augs.log_dir中
        if os.path.exists(model_path):
            
            shutil.copy(model_path, os.path.join(augs.log_dir, 'final_model_{}.pt'.format(augs.seed)))
        
        val_acc_c, val_acc_o, val_acc_co, test_acc_c, test_acc_o, test_acc_co = trainer.test(-1)
        text = f'val_acc_c: {val_acc_c}, val_acc_o: {val_acc_o}, val_acc_co: {val_acc_co}, ' + \
                f'test_acc_c: {test_acc_c}, test_acc_o: {test_acc_o}, test_acc_co: {test_acc_co}'
        # writer.add_text('TestAcc', text)

        print(f"val_acc_o: {val_acc_o}")
        print(f"test_acc_o: {test_acc_o}")

        print(augs.model_name)
        if augs.robusttest:
            print('robust test')
            test_roubust_list = trainer.test_roubust()
        else:
            test_roubust_list = {}
        if augs.explain:
            print('explain')
            baselines_results_base = trainer.explain()
        else:
            baselines_results_base = {}

    return [val_acc_c.item(), val_acc_o.item(), val_acc_co.item(), test_acc_c.item(), test_acc_o.item(), test_acc_co.item()], baselines_results_base,test_roubust_list
#%%
if __name__ == "__main__":
    

    augs = get_config().parse_args()

    
    

    now = datetime.now()
    augs.formatted_date = augs.name[:6]
    augs.device = 'cuda:{}'.format(augs.gpu) if (augs.gpu >= 0 & torch.cuda.is_available()) else 'cpu'
    seed_list = [2017, 2024, 4751, 2021, 1998, 11111, 12345, 54321, 2022, 2023]
    results = []
    base_result_dict_list = []
    test_roubust_list = []
    for seed_i in seed_list:
        if 'bj' in augs.model_name:
            augs.log_dir_i = '{}/bj/{}-{}/{}/'.format(augs.formatted_date,augs.name,augs.base_encoder,seed_i)
            augs.seed = seed_i
            print(augs)
            dataloader_dict = get_dataloader_dict(augs,os.path.join(augs.my_data_dir,'trajdata', 'geolife', 'processed_data', 'dataset.pkl'), augs.batch_size,augs.seed)
            if augs.roubust == 'regionshift':
                dataloader_dict = get_dataloader_dict(augs,os.path.join(augs.my_data_dir,'trajdata', 'geolife', 'processed_data', 'dataset_region1.pkl'), augs.batch_size,augs.seed)
                dataloader_dict2 = get_dataloader_dict(augs,os.path.join(augs.my_data_dir,'trajdata', 'geolife', 'processed_data', 'dataset_region2.pkl'), augs.batch_size,augs.seed)
                dataloader_dict['test'] = dataloader_dict2['test']
            output_dim = 4
            augs.log_dir = os.path.join(augs.my_data_dir,'logs',augs.log_dir_i) 
            # 如果存在路径，删除
            if os.path.exists(augs.log_dir):
                import shutil
                shutil.rmtree(augs.log_dir)
        if 'ja' in augs.model_name:
            augs.log_dir_i = '{}/ja/{}-{}/{}/'.format(augs.formatted_date,augs.name,augs.base_encoder,seed_i)
            augs.seed = seed_i
            print(augs)
            dataloader_dict = get_dataloader_dict(augs,os.path.join(augs.my_data_dir,'trajdata', 'jakarta', 'processed_data', 'dataset.pkl'), augs.batch_size,augs.seed)
            output_dim = 2
            augs.log_dir = os.path.join(augs.my_data_dir,'logs',augs.log_dir_i) 
            # 如果存在路径，删除
            if os.path.exists(augs.log_dir):
                import shutil
                shutil.rmtree(augs.log_dir)
        if 'sa' in augs.model_name:
            augs.log_dir_i = '{}/sa/{}-{}/{}/'.format(augs.formatted_date,augs.name,augs.base_encoder,seed_i)
            augs.seed = seed_i
            print(augs)
            dataloader_dict = get_dataloader_dict(augs,os.path.join(augs.my_data_dir,'trajdata', 'singapore', 'processed_data', 'dataset.pkl'), augs.batch_size,augs.seed)
            output_dim = 2
            augs.log_dir = os.path.join(augs.my_data_dir,'logs',augs.log_dir_i) 
            # 如果存在路径，删除
            if os.path.exists(augs.log_dir):
                import shutil
                shutil.rmtree(augs.log_dir)
        if 'shl' in augs.model_name:
            augs.log_dir_i = '{}/shl/{}-{}/{}/'.format(augs.formatted_date,augs.name,augs.base_encoder,seed_i)
            augs.seed = seed_i
            print(augs)
            dataloader_dict = get_dataloader_dict(augs,os.path.join(augs.my_data_dir,'trajdata', 'shl', 'processed_data', 'shl_dataset.pkl'), augs.batch_size,augs.seed)
            output_dim = 4
            augs.log_dir = os.path.join(augs.my_data_dir,'logs',augs.log_dir_i) 
            # 如果存在路径，删除
            if os.path.exists(augs.log_dir):
                import shutil
                shutil.rmtree(augs.log_dir)
        
        if 'wh' in augs.model_name:
            augs.log_dir_i = '{}/wh/{}-{}/{}/'.format(augs.formatted_date,augs.name,augs.base_encoder,seed_i)
            augs.seed = seed_i
            print(augs)
            augs.city = 'wh'
            dataloader_dict = get_dataloader_dict(augs, os.path.join(augs.my_data_dir,'trajdata', 'weihai', 'processed_data', 'wh_dataset.pkl'), augs.batch_size,augs.seed)
            if augs.roubust == 'regionshift':
                dataloader_dict = get_dataloader_dict(augs,os.path.join(augs.my_data_dir,'trajdata', 'weihai', 'processed_data', 'wh_dataset_region1.pkl'), augs.batch_size,augs.seed)
                dataloader_dict2 = get_dataloader_dict(augs,os.path.join(augs.my_data_dir,'trajdata', 'weihai', 'processed_data', 'wh_dataset_region2.pkl'), augs.batch_size,augs.seed)
                dataloader_dict['test'] = dataloader_dict2['test']
            output_dim = augs.num_driver
            augs.log_dir = os.path.join(augs.my_data_dir,'logs',augs.log_dir_i) 
            # 如果存在路径，删除
            if os.path.exists(augs.log_dir):
                import shutil
                shutil.rmtree(augs.log_dir)


        if 'yt' in augs.model_name:
            augs.log_dir_i = '{}/yt/{}-{}/{}/'.format(augs.formatted_date,augs.name,augs.base_encoder,seed_i)
            augs.seed = seed_i
            print(augs)
            augs.city = 'yt'
            dataloader_dict = get_dataloader_dict(augs, os.path.join(augs.my_data_dir,'trajdata', 'yantai', 'processed_data', 'yt_dataset.pkl'), augs.batch_size,augs.seed)
            output_dim = augs.num_driver
            augs.log_dir = os.path.join(augs.my_data_dir,'logs',augs.log_dir_i) 
            # 如果存在路径，删除
            if os.path.exists(augs.log_dir):
                import shutil
                shutil.rmtree(augs.log_dir)

        results_i,baselines_results_base_i,test_roubust_list_i = main(augs)
        results.append(results_i)
        base_result_dict_list.append(baselines_results_base_i)
        test_roubust_list.append(test_roubust_list_i)   
    print("所有任务完成")
    results = pd.DataFrame(results) 
    results.columns = ['val_acc_c', 'val_acc_o', 'val_acc_co', 'test_acc_c', 'test_acc_o', 'test_acc_co']
    reaults_i_mean = results.mean()
    reaults_i_std = results.std()
    print('mean')
    print(reaults_i_mean)
    print('std')
    print(reaults_i_std)
    reaults_i = pd.concat([reaults_i_mean,reaults_i_std],axis=1)
    reaults_i.columns = ['mean', 'std']
    # filename = os.path.join(os.path.join(augs.my_data_dir,'logs/{}/bj/{}-{}/'.format(augs.formatted_date,augs.name,augs.base_encoder)), 'test_results.csv')
    filename = os.path.join(os.path.join(augs.my_data_dir,'logs',augs.log_dir_i.replace(str(seed_i),'')), 'test_results.csv')

    reaults_i.to_csv(filename)
    ##############################################
    #############可解释性的结果########################
    ####################################################
    if augs.explain:
        print('explain')
        base_result_dict_mean = average_dict_values(base_result_dict_list)
        save_dir = os.path.join(os.path.join(augs.my_data_dir,'logs',augs.log_dir_i.replace(str(seed_i),'')))
        pickle.dump(base_result_dict_mean, open('{}/base_result_dict_mean.pkl'.format(save_dir), 'wb'))
        result_dict_df_base = {}
        for condition in base_result_dict_mean.keys():
            ave_png(base_result_dict_mean[condition],'{}/base_mask_mean{}.png'.format(save_dir,condition))                
            result_dict_df_base[condition] = pd.DataFrame(base_result_dict_mean[condition]).applymap(np.mean)
        # print('base_model_con\n',result_dict_df_base['all'])


        result_dict_df_base['all']['accuracy'] = base_result_dict_mean['all']['accuracy']['IB_top'][0] - result_dict_df_base['all']['accuracy']
        print('base_model_con_{}\n'.format(augs.base_encoder),result_dict_df_base['all'])
        filename = os.path.join(os.path.join(augs.my_data_dir,'logs',augs.log_dir_i.replace(str(seed_i),'')), 'eplain_results.csv')

        result_dict_df_base['all'].to_csv(filename, index=True)
    if augs.robusttest:
        # 对于test_roubust_list中的字典按照key进行平均
        test_roubust_list_mean = {}
        
        for key in test_roubust_list[0].keys():
            test_roubust_list_mean[key] = [np.mean([d[key] for d in test_roubust_list]),
                                            np.std([d[key] for d in test_roubust_list])
                                           ]
        # 保存test_roubust_list_mean为csv
        for key, value in test_roubust_list_mean.items():
            print(f"{key}: {value[0]:.4f}")
        test_roubust_list_df = pd.DataFrame(test_roubust_list_mean)
        save_dir = os.path.join(os.path.join(augs.my_data_dir,'logs',augs.log_dir_i.replace(str(seed_i),'')))
        test_roubust_list_df.to_csv(os.path.join(save_dir, 'test_roubust_list_mean.csv'), index=False)
    
# %%

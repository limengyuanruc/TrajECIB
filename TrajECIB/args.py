import argparse

def get_config():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', type=str, default='1-bj-base')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to run')
    parser.add_argument('--base_encoder', type=str, default='lstm',help= 'gru,lstm,grud,stgn,former') #GRU、BiLSTM、GRU-D、STGN、TrajFormer
    parser.add_argument('--num_driver', type=int, default=10,help= 'gru,lstm,grud,stgn,former') #GRU、BiLSTM、GRU-D、STGN、TrajFormer
    
    # model
    parser.add_argument('--input_dim', type=int, default=3)   # lat lon timedelta dis v a
    parser.add_argument('--env_dim', type=int, default=24)    # 0=deactivate---last n dim: road_n=13   full=24
    parser.add_argument('--embed_dim', type=int, default=64)  # total network emb dim
    parser.add_argument('--motion_dim', type=int, default=64)  # total network emb dim
    parser.add_argument('--gru_layers', type=int, default=1)  # gru encoder layers
    parser.add_argument('--batch_size', type=int, default=256)  # gru encoder layers

    # envsegmenter
    parser.add_argument('--latent_dim', type=int, default=64)  # qk dim
    parser.add_argument('--envbook_num', type=int, default=25,help='k = 25 in GeoLife and k = 50 in Grab-Posisi') # book num
    parser.add_argument('--promot_num', type=int, default=25,help='k = 25 in GeoLife and k = 50 in Grab-Posisi') # book num
    parser.add_argument('--envbook_dim', type=int, default=64) # book dim

    # causal option
    parser.add_argument('--cat_or_add', type=str, default='add')
    parser.add_argument('--with_random', type=str2bool, default=True)

    parser.add_argument('--o', type=float, default=1.0)     # object:  origin strength
    parser.add_argument('--c', type=float, default=0)     # context: disentanglement strength
    parser.add_argument('--co', type=float, default=0)    # causal intervention strength
    parser.add_argument('--kl_loss', type=float, default=0.001) # causal intervention strength
    parser.add_argument('--cat_loss', type=float, default=0) # causal intervention strength
    parser.add_argument('--k_ratio', type=float, default=0) # causal intervention strength
    parser.add_argument('--train_ib', type=float, default=1) # causal intervention strength
    parser.add_argument('--pre', type=float, default=0) # causal intervention strength
    parser.add_argument('--aug', type=float, default=0) # causal intervention strength
    
    
    # training
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=100) # early stop patience
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--save_iter', type=int, default=100)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)
    
    parser.add_argument('--roubust', type=str, default='0')
    parser.add_argument('--roubust_size', type=float, default=0)
    parser.add_argument('--att_mlp', type=str, default='1')
    parser.add_argument('--local_att', type=float, default=0)
    parser.add_argument('--promtd', type=float, default=1)
    parser.add_argument('--promtd_lendy', type=float, default=0)
    parser.add_argument('--odl_loss', type=float, default=0.001)
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--env_prior', type=str, default='uniform') # uniform=均匀先验1/K, learnable=可学习环境先验pi(k)


    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--name', type=str, default='train')
    parser.add_argument('--log_dir_i', type=str, default='train')
    parser.add_argument('--my_data_dir', type=str, default='')
    parser.add_argument('--formatted_date', type=str, default='train')
    parser.add_argument('--n_exp', type=int, default=1,help='experiment index') # experiment index
    parser.add_argument('--seed', type=int, default=2017)
    
    parser.add_argument('--env_encoder', type=float, default=1)
    parser.add_argument('--cos', type=float, default=1)

    parser.add_argument('--intra', type=float, default=1)
    parser.add_argument('--inter', type=float, default=1)
    parser.add_argument('--spurious', type=float, default=1)

    parser.add_argument('--lambda_inter', type=float, default=0.1)
    parser.add_argument('--lambda_', type=float, default=0.01)
    parser.add_argument('--m', type=float, default=0.1)


    parser.add_argument('--explain', type=float, default=1) # causal intervention strength
    parser.add_argument('--robusttest', type=float, default=1) # causal intervention strength
    parser.add_argument('--repinter', type=str, default='0') # 表示干预测试（去掉point_att前/后10个点）
    parser.add_argument('--spurioustest', type=float, default=0) # 仅用spurious表示Z_S训练ReadoutLayer分类测试
    parser.add_argument('--spuriouswotr', type=float, default=0) # 仅用spurious表示Z_S直接过object_readout_layer分类（不训练）
    parser.add_argument('--envprob', type=float, default=0) # 环境探测：mean_o重建envs序列评估MAE/RMSE
    parser.add_argument('--intervene', type=float, default=0) # 环境干预：随机替换环境特征维度，对比干预前后accuracy差值/每类概率变化Delta_k/S
    parser.add_argument('--feat_key', type=str, default='mean_o') # 环境探测：mean_o/mean_c重建envs序列评估MAE/RMSE
    parser.add_argument('--GradFlag', type=bool, default=True, help='Saliency') #0 111111
    parser.add_argument('--IGFlag', type=bool, default=False, help='IntegratedGradients') #1 1111
    parser.add_argument('--SGFlag', type=bool, default=False, help='NoiseTunnel(Saliency)') #2
    parser.add_argument('--TNGFlag', type=bool, default=False, help='InputXGradient') #3
    parser.add_argument('--GBFlag', type=bool, default=False, help='GuidedBackprop') #4
    parser.add_argument('--DLFlag', type=bool, default=False, help='DeepLIFT') #5 11111
    parser.add_argument('--GSFlag', type=bool, default=False, help='GradientShap') #6
    parser.add_argument('--DLSFlag', type=bool, default=False, help='DeepLiftShap') #7
    # 基于扰动的
    parser.add_argument('--FOFlag', type=bool, default=False, help='Occlusion') #0  11111
    parser.add_argument('--FAFlag', type=bool, default=False, help='FeatureAblation') #1 
    parser.add_argument('--FPFlag', type=bool, default=False, help='FeaturePermutation') #2
    # shap
    parser.add_argument('--SVSFlag', type=bool, default=False, help='ShapleyValueSampling') # 0  111111111
    parser.add_argument('--KSFlag', type=bool, default=False, help='KernelShap') #1
    # other
    parser.add_argument('--RNDFlag', type=bool, default=True, help='random') # 0
    parser.add_argument('--FIXFlag', type=bool, default=False, help='fix') #1
    parser.add_argument('--FREFlag', type=bool, default=False, help='frequency') #2
    parser.add_argument('--STDFlag', type=bool, default=False, help='std') #3
    augs = parser


    # os.environ["CUDA_VISIBLE_DEVICES"] = str(augs.gpu)
    return augs
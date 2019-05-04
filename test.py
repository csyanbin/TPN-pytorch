#-------------------------------------
# Project: Transductive Propagation Network for Few-shot Learning
# Date: 2018.1.12
# Author: Yanbin Liu
# All Rights Reserved
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import argparse
import models
import math
from tqdm import tqdm
import scipy as sp
import scipy.stats

from dataset_mini import *
from dataset_tiered import *


parser = argparse.ArgumentParser(description='Train transudctive propagation networks')
# basic params
parser.add_argument('--gpu',    type=str,   default=0,      metavar='GPU',
                    help="gpus, default:0")
parser.add_argument('--repeat', type=int,   default=10,     metavar='REPEAT',
                    help="run count")
# model params
n_examples = 600
parser.add_argument('--x_dim',  type=str,   default="84,84,3",metavar='XDIM',
                    help='input image dims')
parser.add_argument('--h_dim',  type=int,   default=64,     metavar='HDIM',
                    help="dimensionality of hidden layers (default: 64)")
parser.add_argument('--z_dim',  type=int,   default=64,     metavar='ZDIM',
                    help="dimensionality of output channels (default: 64)")

# basic training hyper-parameters
n_episodes = 100 # test interval
parser.add_argument('--n_way',  type=int,   default=5,      metavar='NWAY',
                    help="nway")
parser.add_argument('--n_shot', type=int,   default=5,      metavar='NSHOT',
                    help="nshot")
parser.add_argument('--n_query',type=int,   default=15,     metavar='NQUERY',
                            help="nquery")
parser.add_argument('--n_epochs',type=int,  default=2100,   metavar='NEPOCHS',
                    help="nepochs")

# val and test hyper-parameters
parser.add_argument('--n_test_way',type=int,default=5,      metavar='NTESTWAY',
                    help="ntestway")
parser.add_argument('--n_test_shot', type=int, default=5,   metavar='NTESTSHOT',
                    help="ntestshot")
parser.add_argument('--n_test_query',type=int, default=15,  metavar='NTESTQUERY',
                    help="ntestquery")
parser.add_argument('--n_test_episodes',type=int, default=600, metavar='NTESTEPI',
                    help="ntestepisodes")

# optimization params
parser.add_argument('--lr',     type=float, default=0.001,  metavar='LR',
                    help="base learning rate")
parser.add_argument('--step_size', type=int, default=10000, metavar='STEPSIZE',
                    help="lr decay step size")
parser.add_argument('--gamma', type=float,  default=0.5,    metavar='GAMMA',
                    help="decay rate")

# dataset params
parser.add_argument('--dataset',type=str,   default='mini', metavar='DATASET',
                    help="mini or tiered")
parser.add_argument('--ratio',  type=float, default=1.0,    metavar='RATIO',
                    help="ratio of labeled data each class")
parser.add_argument('--pkl', type=bool, default=True, metavar='PKL',
                    help="load pkl preprocessed data")

# label propagation params
parser.add_argument('--alg',    type=str,   default='TPN',  metavar='ALG',
                    help="algorithm used, TPN")
parser.add_argument('--k',      type=int,   default=-1,     metavar='K',
                    help="top k in constructing the graph W")
parser.add_argument('--sigma',  type=float, default=0.25,    metavar='SIGMA',
                    help="Initial sigma in label propagation")
parser.add_argument('--alpha',  type=float, default=0.99,    metavar='ALPHA',
                    help="Initial alpha in label propagation")
parser.add_argument('--rn',     type=int,   default=1,      metavar='RN',
                    help="relation types" + 
                         "30:learned sigma and alpha,    300:learned sigma, fixed alpha")

# restore params
parser.add_argument('--iters',  type=int,   default=0,      metavar='ITERS',
                    help="iteration to restore params")
parser.add_argument('--exp_name',type=str,  default='exp',  metavar='EXPNAME',
                    help="experiment name")
parser.add_argument('--seed', type=int, default=1000, metavar='SEED',
                    help="random seed for code and data sample")


# init args
args = vars(parser.parse_args())
print(args)
for key,v in args.items(): exec(key+'=v')


# RANDOM SEED
#torch.manual_seed(seed)
#if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
#np.random.seed(seed)
#random.seed(seed)


# set environment variables: gpu, num_thread
os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu'])
torch.set_num_threads(2)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h


def main():
    # init dataloader
    print("init data loader")
    args_data = {}
    args_data['x_dim'] = '84,84,3'
    args_data['ratio'] = 1.0
    args_data['seed'] = seed
    print('seed:',seed)
    if dataset=='mini':
        loader_test = dataset_mini(n_examples, n_episodes, 'test', args_data)
    elif dataset=='tiered':
        loader_test = dataset_tiered(n_examples, n_episodes, 'test', args_data)
    
    if not pkl:
        loader_test.load_data()
    else:
        loader_test.load_data_pkl()
    

    # Step 2: init neural networks
    print("init neural networks")

    # construct the model
    model = models.LabelPropagation(args)
    model.cuda(0)

    # load the saved model
    if iters>0:
        model.load_state_dict(torch.load('checkpoints/%s/models/%s_%d_model.t7' %(args['exp_name'],alg,iters) ))
        print('Loading Parameters from %s: %d' %(args['exp_name'], iters))


    # Step 3: build graph
    print("Testing...")

    all_acc = []
    all_std = []
    all_ci95 = []

    ce_list = []

    for rep in range(repeat):
        list_acc = []

        for epi in tqdm(range(n_test_episodes), desc='test:{}'.format(rep)):

            model.eval()

            # sample data for next batch
            support, s_labels, query, q_labels, unlabel = loader_test.next_data(n_test_way, n_test_shot, n_test_query)
            support = np.reshape(support, (support.shape[0]*support.shape[1],)+support.shape[2:])
            support = torch.from_numpy(np.transpose(support, (0,3,1,2)))
            query   = np.reshape(query, (query.shape[0]*query.shape[1],)+query.shape[2:])
            query   = torch.from_numpy(np.transpose(query, (0,3,1,2)))
            s_labels = torch.from_numpy(np.reshape(s_labels,(-1,)))
            q_labels = torch.from_numpy(np.reshape(q_labels,(-1,)))
            s_labels = s_labels.type(torch.LongTensor)
            q_labels = q_labels.type(torch.LongTensor)
            s_onehot = torch.zeros(n_test_way*n_test_shot, n_test_way).scatter_(1, s_labels.view(-1,1), 1)
            q_onehot = torch.zeros(n_test_way*n_test_query, n_test_way).scatter_(1, q_labels.view(-1,1), 1)

            with torch.no_grad():
                inputs = [support.cuda(0), s_onehot.cuda(0), query.cuda(0), q_onehot.cuda(0)]
                loss, acc = model(inputs)
            
            list_acc.append(acc.item())

        mean_acc = np.mean(list_acc)
        std_acc  = np.std(list_acc)
        ci95 = 1.96*std_acc/np.sqrt(n_test_episodes)
        m,ci = mean_confidence_interval(list_acc)
        
        print('label, acc:{:.4f},std:{:.4f},ci95:{:.4f},ci:{:.4f}'.format(mean_acc, std_acc, ci95, ci))
        all_acc.append(mean_acc)
        all_std.append(std_acc)
        all_ci95.append(ci95)

    ind = np.argmax(all_acc)
    print('Max acc:{:.5f}, std:{:.5f}, ci95: {:.5f}'.format(all_acc[ind], all_std[ind], all_ci95[ind]))
    print('Avg over {} runs: mean:{:.5f}, std:{:.5f}, ci95: {:.5f}'.format(repeat,np.mean(all_acc),np.mean(all_std),np.mean(all_ci95)))
            


if __name__ == "__main__":
    main()




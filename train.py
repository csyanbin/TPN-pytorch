#-------------------------------------
# Project: Transductive Propagation Network for Few-shot Learning
# Date: 2019.5.4
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

from dataset_mini import *
from dataset_tiered import *


parser = argparse.ArgumentParser(description='Train transudctive propagation networks')
# parse gpu
parser.add_argument('--gpu',        type=str,   default=0,          metavar='GPU',
                    help="gpus, default:0")
# model params
n_examples = 600
parser.add_argument('--x_dim',      type=str,   default="84,84,3",  metavar='XDIM',
                    help='input image dims')
parser.add_argument('--h_dim',      type=int,   default=64,         metavar='HDIM',
                    help="dimensionality of hidden layers (default: 64)")
parser.add_argument('--z_dim',      type=int,   default=64,         metavar='ZDIM',
                    help="dimensionality of output channels (default: 64)")

# training hyper-parameters
n_episodes = 100 # test interval
parser.add_argument('--n_way',      type=int,   default=5,          metavar='NWAY',
                    help="nway")
parser.add_argument('--n_shot',     type=int,   default=5,          metavar='NSHOT',
                    help="nshot")
parser.add_argument('--n_query',    type=int,   default=15,         metavar='NQUERY',
                    help="nquery")
parser.add_argument('--n_epochs',   type=int,   default=2100,       metavar='NEPOCHS',
                    help="nepochs")
# test hyper-parameters
parser.add_argument('--n_test_way', type=int,   default=5,          metavar='NTESTWAY',
                    help="ntestway")
parser.add_argument('--n_test_shot',type=int,   default=5,          metavar='NTESTSHOT',
                    help="ntestshot")
parser.add_argument('--n_test_query',type=int,  default=15,         metavar='NTESTQUERY',
                    help="ntestquery")

# optimization params
parser.add_argument('--lr',         type=float, default=0.001,      metavar='LR',
                    help="base learning rate")
parser.add_argument('--step_size',  type=int,   default=10000,      metavar='STEPSIZE',
                    help="lr decay step size")
parser.add_argument('--gamma',      type=float, default=0.5,        metavar='GAMMA',
                    help="decay rate")
parser.add_argument('--patience',   type=int,   default=200,        metavar='PATIENCE',
                    help="train patience until stop")

# dataset params
parser.add_argument('--dataset',    type=str,   default='mini',     metavar='DATASET',
                    help="mini or tiered")
parser.add_argument('--ratio',      type=float, default=1.0,        metavar='RATIO',
                    help="ratio of labeled data each class")
parser.add_argument('--pkl',        type=bool,  default=True,       metavar='PKL',
                    help="load pkl preprocessed data")

# label propagation params
parser.add_argument('--alg',        type=str,   default='TPN',      metavar='ALG',
                    help="algorithm used, TPN")
parser.add_argument('--k',          type=int,   default=20,         metavar='K',
                    help="top k in constructing the graph W")
parser.add_argument('--sigma',      type=float, default=0.25,       metavar='SIGMA',
                    help="Initial sigma in label propagation")
parser.add_argument('--alpha',      type=float, default=0.99,       metavar='ALPHA',
                    help="Initial alpha in label propagation")
parser.add_argument('--rn',         type=int,   default=300,        metavar='RN',
                    help="graph construction types: "
                    "300: sigma is learned, alpha is fixed" +
                    "30:  both sigma and alpha learned")

# save and restore params
parser.add_argument('--seed',       type=int,   default=1000,       metavar='SEED',
                    help="random seed for code and data sample")
parser.add_argument('--exp_name',   type=str,   default='exp',      metavar='EXPNAME',
                    help="experiment name")
parser.add_argument('--iters',      type=int,   default=0,          metavar='ITERS',
                    help="iteration to restore params")


# deal with params
args = vars(parser.parse_args())
im_width, im_height, channels = list(map(int, args['x_dim'].split(',')))
print(args)
for key,v in args.items(): exec(key+'=v')

# RANDOM SEED
#torch.manual_seed(seed)
#if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
#np.random.seed(seed)
#random.seed(seed)


# set environment variables: gpu, num_thread
os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu'])
#os.environ["OMP_NUM_THREADS"] = "4"
#os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(2)

## if "THCudaCheck FAIL file=/pytorch/aten/src/THC/THCGeneral.cpp line=405 error=11 : invalid argument" error occurs on GTX 2080Ti, set the following to False
torch.backends.cudnn.benchmark = True


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args['exp_name']):
        os.makedirs('checkpoints/'+args['exp_name'])
    if not os.path.exists('checkpoints/'+args['exp_name']+'/'+'models'):
        os.makedirs('checkpoints/'+args['exp_name']+'/'+'models')
    os.system('cp train.py checkpoints'+'/'+args['exp_name']+'/'+'train.py.backup')
    os.system('cp models.py checkpoints' + '/' + args['exp_name'] + '/' + 'models.py.backup')
    f = open('checkpoints/'+args['exp_name']+'/log.txt', 'a')
    print(args, file=f)
    f.close()
_init_()



def main():
    # Step 1: init dataloader
    print("init data loader")
    args_data = {}
    args_data['x_dim'] = x_dim
    args_data['ratio'] = ratio
    args_data['seed'] = seed
    if dataset=='mini':
        loader_train = dataset_mini(n_examples, n_episodes, 'train', args_data)
        loader_val   = dataset_mini(n_examples, n_episodes, 'val', args_data)
    elif dataset=='tiered':
        loader_train = dataset_tiered(n_examples, n_episodes, 'train', args_data)
        loader_val   = dataset_tiered(n_examples, n_episodes, 'val', args_data)
    
    if not pkl:
        loader_train.load_data()
        loader_val.load_data()
    else:
        loader_train.load_data_pkl()
        loader_val.load_data_pkl()
    

    # Step 2: init neural networks
    print("init neural networks")

    # construct the model
    model = models.LabelPropagation(args)
    model.cuda(0)

    # optimizer
    model_optim = torch.optim.Adam(model.parameters(), lr=lr)
    model_scheduler = StepLR(model_optim, step_size=step_size, gamma=gamma)

    # load the saved model
    if iters>0:
        model.load_state_dict(torch.load('checkpoints/%s/models/%s_%d_model.t7' %(args['exp_name'],alg,iters) ))
        print('Loading Parameters from %s: %d' %(args['exp_name'], iters))

    # Step 3: Train and validation
    print("Training...")

    best_acc = 0.0
    best_loss = np.inf
    wait = 0
    for ep in range(iters, n_epochs):
        loss_tr = []
        ce_list = []

        acc_tr  = []
        loss_val= []
        acc_val = []

        for epi in tqdm(range(n_episodes), desc='train_epoc:{}'.format(ep)):

            model_scheduler.step(ep*n_episodes+epi)

            # set train mode
            model.train()

            # sample data for next batch
            support, s_labels, query, q_labels, unlabel = loader_train.next_data(n_way, n_shot, n_query)
            support = np.reshape(support, (support.shape[0]*support.shape[1],)+support.shape[2:])
            support = torch.from_numpy(np.transpose(support, (0,3,1,2)))
            query   = np.reshape(query, (query.shape[0]*query.shape[1],)+query.shape[2:])
            query   = torch.from_numpy(np.transpose(query, (0,3,1,2)))
            s_labels = torch.from_numpy(np.reshape(s_labels,(-1,)))
            q_labels = torch.from_numpy(np.reshape(q_labels,(-1,)))
            s_labels = s_labels.type(torch.LongTensor)
            q_labels = q_labels.type(torch.LongTensor)
            s_onehot = torch.zeros(n_way*n_shot, n_way).scatter_(1, s_labels.view(-1,1), 1)
            q_onehot = torch.zeros(n_way*n_query, n_way).scatter_(1, q_labels.view(-1,1), 1)

            inputs = [support.cuda(0), s_onehot.cuda(0), query.cuda(0), q_onehot.cuda(0)]
            
            loss, acc = model(inputs)
            loss_tr.append(loss.item())
            acc_tr.append(acc.item())

            model.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm(model.parameters(), 4.0)
            model_optim.step()

        
        for epi in tqdm(range(n_episodes), desc='val epoc:{}'.format(ep)):
            # set eval mode
            model.eval()

            # sample data for next batch
            support, s_labels, query, q_labels, unlabel = loader_val.next_data(n_test_way, n_test_shot, n_test_query)
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

            loss_val.append(loss.item() )
            acc_val.append(acc.item())

        print('epoch:{}, loss_tr:{:.5f}, acc_tr:{:.5f}, loss_val:{:.5f}, acc_val:{:.5f}'.format(ep, np.mean(loss_tr), np.mean(acc_tr), np.mean(loss_val), np.mean(acc_val)))
        

        # Model Save and Stop Criterion
        cond1 = (np.mean(acc_val)>best_acc)
        cond2 = (np.mean(loss_val)<best_loss)
        
        if cond1 or cond2:
            best_acc = np.mean(acc_val)
            best_loss = np.mean(loss_val)
            print('best val loss:{:.5f}, acc:{:.5f}'.format(best_loss, best_acc))

            # save model
            torch.save(model.state_dict(), 'checkpoints/%s/models/%s_%d_model.t7' %(args['exp_name'],alg,(ep+1)*n_episodes) )
            
            f = open('checkpoints/'+args['exp_name']+'/log.txt', 'a')
            print('{} {:.5f} {:.5f}'.format((ep+1)*n_episodes, best_loss, best_acc), file=f)
            f.close()

            wait = 0

        else:
            wait += 1
            if ep%100==0:
                torch.save(model.state_dict(), 'checkpoints/%s/models/%s_%d_model.t7' %(args['exp_name'],alg,(ep+1)*n_episodes) )

                f = open('checkpoints/'+args['exp_name']+'/log.txt', 'a')
                print('{} {:.5f} {:.5f}'.format((ep+1)*n_episodes, np.mean(loss_val), np.mean(acc_val)), file=f)
                f.close()

        if wait>patience and ep>n_epochs:
            break


if __name__ == "__main__":
    main()




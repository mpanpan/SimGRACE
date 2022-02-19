from __future__ import print_function

import os
import sys
import numpy as np
import torch
import networkx as nx
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args
from graph_embedding import S2VGraph
sys.path.append('%s/../data_generator' % os.path.dirname(os.path.realpath(__file__)))
from data_util import load_pkl
from copy import deepcopy

@torch.no_grad()
def gen_adv_output(data, model, z):
    z = Variable(z.detach().data, requires_grad=False)
    model_adv = deepcopy(model)
    adv_optim = optim.Adam(model_adv.parameters(), lr=cmd_args.lr_inner)
    def closure(z):
        adv_optim.zero_grad()
        z_tmp = model_adv.forward_cl(data)
        loss_tmp = model_adv.loss_cl(z, z_tmp)
        loss_tmp.backward()
        torch.nn.utils.clip_grad_norm_(model_adv.parameters(), cmd_args.clip_norm)
    closure = torch.enable_grad()(closure)
    closure(z)
    state = dict()
    for i in range(2): 
        for name, param in model_adv.named_parameters():          
            if name.split('.')[0] != 'mlp' and name.split('.')[0] != 'projection_head':
                if i == 0:
                    state[name] = torch.zeros_like(param.grad)               
                dev = state[name] + cmd_args.lr_inner * param.grad
                clip_coef = cmd_args.epison / (dev.norm() + 1e-12)
                dev = clip_coef * dev if clip_coef < 1 else dev
                param.sub_(state[name]).add_(dev)
                state[name] = dev           
        closure(z)
    z2 = model_adv.forward_cl(data)
    return z2

def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size, epoch=0):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]
        batch_graph = [g_list[idx] for idx in selected_idx]
        if epoch <= 150:
            x1 = classifier.forward_cl(batch_graph)
            x2 = gen_adv_output(batch_graph, classifier, x1)
            x2 = Variable(x2.detach().data, requires_grad=False)
            loss = classifier.loss_cl(x1, x2)
            acc = torch.zeros(1)
        else:
            _, loss, acc = classifier(batch_graph)
        acc = acc.sum().item() / float(acc.size()[0])
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        loss = loss.data.cpu().numpy()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))
        total_loss.append( np.array([loss, acc]) * len(selected_idx))
        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    return avg_loss

def load_er_data():
    frac_train = 0.9
    pattern = 'nrange-%d-%d-n_graph-%d-p-%.2f' % (cmd_args.min_n, cmd_args.max_n, cmd_args.n_graphs, cmd_args.er_p)
    num_train = int(frac_train * cmd_args.n_graphs)
    train_glist = []
    test_glist = []
    label_map = {}
    for i in range(cmd_args.min_c, cmd_args.max_c + 1):
        cur_list = load_pkl('%s/ncomp-%d-%s.pkl' % (cmd_args.data_folder, i, pattern), cmd_args.n_graphs)
        assert len(cur_list) == cmd_args.n_graphs
        train_glist += [S2VGraph(cur_list[j], i) for j in range(num_train)]
        test_glist += [S2VGraph(cur_list[j], i) for j in range(num_train, len(cur_list))]
        label_map[i] = i - cmd_args.min_c
    cmd_args.num_class = len(label_map)
    cmd_args.feat_dim = 1
    print('# train:', len(train_glist), ' # test:', len(test_glist))

    return label_map, train_glist, test_glist

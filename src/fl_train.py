from audioop import cross
from cProfile import label
from cgitb import text
from enum import Flag
from heapq import merge
import torch
from torch import device, nn
import sys
from src.models import MULTModel, Trans_Encoder, mmTransformer
from src import ctc
from src.utils import *
from src.lc_update import LocalUpdate
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
import copy
# from scipy import linalg
from torch import linalg
import torch.nn.functional as F

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *
import src.temperature_scalling as ts
MODA = {
    0:'language',
    1:'vision',
    2:'audio'
}

def initiate(hyp_params, train_data, valid_data, test_data, dict_users):

    # model initialization
    # model = MULTModel(hyp_params)
    model = Trans_Encoder(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    return train_model(settings, hyp_params, train_data, valid_data, test_data, dict_users)


def iid_sampling(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users



####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_data, valid_data, test_data, dict_users):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']   
    scheduler = settings['scheduler'] 
    

    def multfl_train(model, optimizer, criterion, dict_users, temperatures):
        epoch_loss = 0

        model.train()
        # copy global weight
        net_glob = model.state_dict()

        #sample users and build trainset
        idxs_users = np.random.choice(range(hyp_params.clients), hyp_params.train_users, replace=False)
        ep_loss = []
        w_locals = []
        wl_locals = []
        wa_locals = []
        wv_locals = []

        for idx in idxs_users:
            start_time = time.time()
            if hyp_params.rand_moda:
                flag = list(idxs_users).index(idx) % 3
                temperatures[idx].set_flag(flag)
                # temperatures[idx].reset_temperature(0)
            else:
                flag = temperatures[idx].get_flag()

            local_net = copy.deepcopy(model)

            local_train = LocalUpdate(args=hyp_params, dataset=train_data, idxs=dict_users[idx])
            if hyp_params.MCMCtrain:
                local_w, idxs_loss, local_delta = local_train.MCMC_train(net=local_net, optimizer=optimizer, criterion=criterion, flag=flag, id=idx, scale_model=temperatures[idx])
                if flag == 0:
                    wl_locals.append({'weight':copy.deepcopy(local_w),"delta":local_delta})
                elif flag == 1:
                    wv_locals.append({'weight':copy.deepcopy(local_w),"delta":local_delta})
                else:
                    wa_locals.append({'weight':copy.deepcopy(local_w),"delta":local_delta})
            else:
                temperatures[idx].update_model(local_net)
                local_w, idxs_loss = local_train.train(net=local_net, optimizer=optimizer, criterion=criterion, flag=flag, id=idx, scale_model=temperatures[idx])
                temperatures[idx].update_model(None)
                if flag == 0:
                    wl_locals.append({'weight':copy.deepcopy(local_w),"temperature":temperatures[idx]})
                    # print('moda 0, temperature = {}'.format(temperatures[idx].get_temperature()))
                elif flag == 1:
                    wv_locals.append({'weight':copy.deepcopy(local_w),"temperature":temperatures[idx]})
                else:
                    wa_locals.append({'weight':copy.deepcopy(local_w),"temperature":temperatures[idx]})

            elapsed_time = time.time() - start_time
            print('Epoch {:2d} | Client {:3d}({}) | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f} | temperature{:5.2f}'.
                    format(epoch, idx,MODA[flag], elapsed_time * 1000 / hyp_params.log_interval, idxs_loss, temperatures[idx].get_temperature().item()))
            start_time = time.time()
            ep_loss.append(copy.deepcopy(idxs_loss))



        # update global weight
        if hyp_params.MCMCtrain:
            wl = FedAvg(wl_locals)
            wa = FedAvg(wa_locals)
            wv = FedAvg(wv_locals)    
        else:
            wl = fed_merge_with_uncertainty(wl_locals,local_net,hyp_params.use_temperature)
            wa = fed_merge_with_uncertainty(wa_locals,local_net,hyp_params.use_temperature)
            wv = fed_merge_with_uncertainty(wv_locals,local_net,hyp_params.use_temperature)

        w_locals = [wl, wa, wv]

        nets = []

        for i in range(len(w_locals)):
            l_net = copy.deepcopy(model)
            net = reload_model(w_locals[i], l_net)
            nets.append(net)

        net_merge = weight_merge(w_locals, net_glob)

        # load global weight
        model.load_state_dict(net_merge)
                
        return sum(ep_loss) / len(idxs_users)

    def reload_model(w, net):
        net.load_state_dict(w)
        return net

    def fed_merge_with_uncertainty(w, local_net, use_temperature):
        if w == []:
            return local_net.state_dict()
        
        uncertaintys = []
        for i in range(len(w)):
            if use_temperature:
                uncertaintys.append(w[i]['temperature'].get_uncertainty())
            else: 
                uncertaintys.append(1)
        uncertaintys = torch.tensor(uncertaintys)
        uncertaintys = uncertaintys- torch.min(uncertaintys)
        # uncertaintys *= 100
        uncertaintys = torch.exp(uncertaintys) / sum(torch.exp(uncertaintys))
        # quant uncertainty

        w_avg = copy.deepcopy(w[0]['weight'])
        for k in w_avg.keys():
            w_avg[k] = torch.mul(w_avg[k],uncertaintys[i].item())
            for i in range(1, len(w)):
                w_avg[k] += torch.mul(w[i]['weight'][k],uncertaintys[i].item())

        return w_avg

    def fed_merge_MCMC(w,net_glob):
        w_avg = copy.deepcopy(w[0]['weight'])
        for k in w_avg.keys():
            w_avg[k] = torch.mul(w[0]['weight'][k],w[0]['delta'][k])
            for i in range(1, len(w)):
                w_avg[k] = w_avg[k] + torch.mul(w[0]['weight'][k],w[i]['delta'][k])
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg


    def FedAvg(w):
        w_avg = copy.deepcopy(w[0]['weight'])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i]['weight'][k]
            w_avg[k] = torch.div(w_avg[k], len(w))

        return w_avg

    def weight_merge(w, net_glob):
        w_avg = w[0]
        for k in w_avg.keys():
            if "trans_l_with_a" in k:
                w_avg[k] = w[2][k]
            elif "trans_l_with_v" in k:
                w_avg[k] = w[1][k]
            elif "trans_l_with_l" in k:
                w_avg[k] = w[0][k]
            elif "proj_" not in k:
                # Cross-modal attention merge
                if hyp_params.use_cross_model:
                    w_avg[k] = cross_modal_merge(w, k)
                else:
                    for i in range(1, len(w)):
                        w_avg[k] += w[i][k]
                    w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg



    def cross_modal_merge(w, k):
        w_mult = torch.zeros_like(w[0][k])
        w_m = []
        for i in range(3):
            w_glob = w.pop(0)
            w_mult += aggregate_att(w, w_glob, k, stepsize=1.2, metric=2, dp=0.001)
            w_m.append(w_mult)
            w.append(w_glob)
        
        return torch.div(w_mult, len(w))

    def aggregate_att(w_clients, w_server, k, stepsize, metric, dp):
        """
        Attentive aggregation
        :param w_clients: list of client model parameters
        :param w_server: server model parameters
        :param stepsize: step size for aggregation
        :param metric: similarity
        :param dp: magnitude of randomization
        :return: updated server model parameters
        """

        w_next = copy.deepcopy(w_server)
        att, att_mat = {}, {}

        w_next[k] = torch.zeros_like(w_server[k])
        att[k] = torch.zeros(len(w_clients))

        for i in range(0, len(w_clients)):
            w_diff = w_server[k]-w_clients[i][k]
            # if not w_diff.device == torch.device('cpu'): w_diff = w_diff.cpu()
            # att[k][i] = torch.from_numpy(np.array(linalg.norm(w_diff, ord=metric)))
            att[k][i] = linalg.norm(w_diff, ord=metric)

        att[k] = F.softmax(att[k], dim=0)

        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            att_weight += torch.mul(w_server[k]-w_clients[i][k], att[k][i])
        w_next[k] = w_server[k] - torch.mul(att_weight, stepsize) + torch.mul(torch.randn(w_server[k].shape).to(w_server[k].device), dp)

        return w_next[k]

    def cross_train(wl, wa, wv, w, hyp_params):
        device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
        mu = 1e-2

        cross_model = mmTransformer(hyp_params)
        cross_model.train()

        optimizer = getattr(optim, hyp_params.optim)(cross_model.parameters(), lr=hyp_params.lr)

        net_glob = copy.deepcopy(wl)
        net_glob.load_state_dict(w)
        net_glob_para = w

        wl_para = wl.state_dict()
        wa_para = wa.state_dict()
        wv_para = wv.state_dict()
        nets = []

        w_diff1 = torch.tensor(0., device=device)
        w_diff2 = torch.tensor(0., device=device)
        w_diff3 = torch.tensor(0., device=device)
        
        
        for k in wl_para.keys():
            if "trans_l_mem" in k or "proj1.weight" in k or "proj2.weight" in k or "out_layer.weight" in k:
                merged_model = cross_model(wl_para[k], wa_para[k], wv_para[k], w[k])

                crossed_model = torch.reshape(merged_model, net_glob_para[k].size())

                wl_para[k] = crossed_model
                wa_para[k] = crossed_model
                wv_para[k] = crossed_model

                Avg_model = torch.div((wl_para[k]+wa_para[k]+wv_para[k]), 3)
                net_glob_para[k] = Avg_model

            elif "proj1.bias" in k or "proj2.bias" in k or "out_layer.bias" in k:
                crossed_model = torch.div((wl_para[k]+wa_para[k]+wv_para[k]), 3)

                wl_para[k] = crossed_model
                wa_para[k] = crossed_model
                wv_para[k] = crossed_model

                net_glob_para[k] = crossed_model

        nets = [wl_para, wa_para, wv_para]

        wl.load_state_dict(wl_para)
        wa.load_state_dict(wa_para)
        wv.load_state_dict(wv_para)
        net_glob.load_state_dict(net_glob_para)

        # # model comparision calculation

        for w, w_t in zip(net_glob.parameters(), wl.parameters()):

            w_diff1 += torch.pow(torch.norm(w - w_t), 2)

        for w, w_t in zip(net_glob.parameters(), wa.parameters()):

            w_diff2 += torch.pow(torch.norm(w - w_t), 2)

        for w, w_t in zip(net_glob.parameters(), wv.parameters()):

            w_diff3 += torch.pow(torch.norm(w - w_t), 2)        

        w_diff = w_diff1 + w_diff2 + w_diff3
        loss = mu / 2. * w_diff
        print("server loss: " + str(loss))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(cross_model.parameters(), 0.8)
        optimizer.step()

        return nets


    # evaluating function
    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_data if test else valid_data
        total_loss = 0.0
    
        results = []
        truths = []
        id = 0
        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1) # if num of labels is 1
            
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        if hyp_params.dataset == 'iemocap':
                            eval_attr = eval_attr.long()
                        
                batch_size = text.size(0)
                gpu_id = [0,1,2]
                flag = 3
                net = nn.DataParallel(model, device_ids=[0]  ) if batch_size > 10 else model
                preds = net(text, audio, vision, flag, id)
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                total_loss += criterion(preds, eval_attr).item() * batch_size

                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)
                
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_valid = 1e8

    loss_list = []
    temperatures = []
    for i in range(hyp_params.clients): temperatures.append(ts.ModelWithTemperature(i%3))
    #start training process
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()

        multfl_train(model, optimizer, criterion, dict_users, temperatures)

        val_loss, _, _ = evaluate(model, criterion, test=False)
        test_loss, _, _ = evaluate(model, criterion, test=True)
        
        end = time.time()
        duration = end-start
        scheduler.step(val_loss)    # Decay learning rate by validation loss

        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
        print("-"*50)

        loss_list.append(test_loss)
        
        if val_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths = evaluate(model, criterion, test=True)

    print(loss_list)

    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi':
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        eval_iemocap(results, truths)

    sys.stdout.flush()
    input('[Press Any Key to start another run]')

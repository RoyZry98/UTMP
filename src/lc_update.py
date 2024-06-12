import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import torch.optim as optim
import copy
import src.temperature_scalling as ts


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        batch_X, batch_Y, batch_META = self.dataset[self.idxs[item]]
        return batch_X, batch_Y, batch_META


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.train_loader = DataLoader(DatasetSplit(
            dataset, idxs), batch_size=self.args.batch_size, shuffle=True)

    def train(self, net, optimizer, criterion, flag, id, scale_model=None, aband_data=0):
        opti = getattr(optim, self.args.optim)(
            net.parameters(), lr=self.args.lr)
        glob_net = copy.deepcopy(net)
        net.train()
        epoch_loss = []
        local_ep = 2
        raw_loss = combined_loss = 0

        for i in range(local_ep):
            batch_loss = []
            # mini-batch training
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(self.train_loader, start=aband_data):
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1

                if self.args.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(
                        ), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        if self.args.dataset == 'iemocap':
                            eval_attr = eval_attr.long()

                net.zero_grad()

                preds = net(text, audio, vision, flag, id)

                if self.args.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)

                raw_loss = criterion(preds, eval_attr)

                raw_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), self.args.clip)
                opti.step()

                batch_loss.append(raw_loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # if scale_model!=None:
        #     scale_model.set_temperature(self.train_loader)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
        # return net, sum(epoch_loss) / len(epoch_loss)
    def eval(self, net, optimizer, criterion, flag, id, scale_model=None, aband_data=0):
        net.eval()
        batch_loss = []
        # mini-batch training
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(self.train_loader):
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            if self.args.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(
                    ), audio.cuda(), vision.cuda(), eval_attr.cuda()
                    if self.args.dataset == 'iemocap':
                        eval_attr = eval_attr.long()
            net.zero_grad()
            preds = net(text, audio, vision, flag, id)
            if self.args.dataset == 'iemocap':
                preds = preds.view(-1, 2)
                eval_attr = eval_attr.view(-1)
            raw_loss = criterion(preds, eval_attr)
            batch_loss.append(raw_loss.item())
        return sum(batch_loss)/len(batch_loss)
        # print(f"eval loss: {sum(batch_loss)/len(batch_loss)}")

    def get_delta(self, samples, means, glob_weight):
        dim = len(samples[0].shape)
        cov_matrix = None
        if dim == 1:  # no inverse
            return glob_weight-means
        for s in range(len(samples)):
            input_vec = samples[s]
            if dim == 3:
                a, b, c = input_vec.size()
                input_vec = input_vec.view(a, b)
                means = means.view(a, b)
                glob_weight = glob_weight.view(a, b)
            x = input_vec
            if cov_matrix == None:
                cov_matrix = torch.matmul(x.mT, x)
            else:
                cov_matrix += torch.matmul(x.mT, x)
        cov_matrix = torch.div(cov_matrix, len(samples))  # E(theta theta^T)
        # E(theta theta^T) - E(theta)^2
        cov_matrix -= torch.matmul(means.mT, means)
        try:
            inverse_cov = torch.linalg.inv(cov_matrix)
        except: 
            inverse_cov = torch.linalg.pinv(cov_matrix)
            
        delta = torch.matmul(glob_weight-means, inverse_cov)
        return delta.reshape(samples[0].shape)

    def MCMC_train(self, net, optimizer, criterion, flag, id, sample_times=5, moda_list=['l', 'v', 'a'], scale_model=None):
        """
        return the different: delta weight
        """
        samples = []
        epoch_loss = []
        for s in range(sample_times):
            s_net, s_loss = self.train(
                net, optimizer, criterion, flag, id, None, 3)
            samples.append(s_net)
            epoch_loss.append(s_loss)
        # record the relate weight
        delta = copy.deepcopy(net).state_dict()
        client_q = copy.deepcopy(net).state_dict()
        for key in samples[0].keys():
            weight_sum = torch.zeros_like(samples[0][key]).cuda()
            layer_sample = []
            for i in range(sample_times):
                weight_sum += samples[i][key]
                layer_sample.append(samples[i][key])
            means = torch.div(weight_sum, sample_times)
            delta[key] = self.get_delta(layer_sample, means, delta[key])
            client_q[key] = means

        globnet = net.state_dict()
        for key in samples[0].keys():
            client_q[key] = torch.mul(globnet[key]-client_q[key],delta[key]) + globnet[key]
        net.load_state_dict(client_q)
        loss = self.eval(net, optimizer, criterion, flag, id, None, 3)
        return client_q, loss, delta

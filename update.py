#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np

class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs, args):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.args = args

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.args.dataset == 'dbpedia':
            label, image = self.dataset[self.idxs[item]]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, device, vocab=0):

        self.args = args
        self.trainloader, self.validloader = self.train_val_test(
            dataset, list(idxs), args)
        _, (self.x_spt, self.y_spt) = list(enumerate(self.trainloader))[0]
        _, (self.x_qry, self.y_qry) = list(enumerate(self.validloader))[0]
        self.x_spt, self.y_spt, self.x_qry, self.y_qry = self.x_spt.to(device), self.y_spt.to(device), \
                                                         self.x_qry.to(device), self.y_qry.to(device)

        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs, args):
        idxs_train = idxs[:int(self.args.percentage * len(idxs))]
        idxs_val = idxs[int(self.args.percentage * len(idxs)):]


        idxs_train = np.random.choice(idxs_train,  max( int(args.frac_train_support * len(idxs_train) ), 1) , replace=False)
        idxs_val = np.random.choice(idxs_val, max(int(args.frac_train_query * len(idxs_val) ), 1), replace=False)

        if self.args.dataset == 'dbpedia':
            pass

        else:
            BATCH_SIZE = 500
            trainloader = DataLoader(DatasetSplit(dataset, idxs_train, self.args),
                                     batch_size = BATCH_SIZE, shuffle = True)

            validloader = DataLoader(DatasetSplit(dataset, idxs_val, self.args),
                                     batch_size = BATCH_SIZE, shuffle = True)

        return trainloader, validloader

    def update_weights(self, model, global_round,local_epoch, train_period=1):

        model.train()
        local_epoch_loss = []
        local_epoch_acc = []

        if local_epoch != 0:
            for iter in range(local_epoch):
                batch_loss = []

                if  train_period==1:
                    _, loss, fast_weights, grad = model(self.x_spt, self.y_spt)
                elif train_period==2:
                    _, loss, fast_weights, grad = model(self.x_qry, self.y_qry)
                batch_loss.append(loss.item())
                local_epoch_loss.append(sum(batch_loss) / len(batch_loss))

            return model.state_dict(), local_epoch_loss[-1], fast_weights, grad

        elif local_epoch == 0:
            local_max_w = model.state_dict()
            return local_max_w, [0,0], [0,0], [0,0]

        else: raise

#-----------------------test_inference-------------------------------

def test_inference_fastweight(args, model, train_dataset,user_groups, device, vocab=0):


    if args.dataset == 'dbpedia':
        pass

    total, correct =  0.0, 0.0

    if args.dataset == 'dbpedia':
        pass

    else:
        trainloader = DataLoader(DatasetSplit(train_dataset, user_groups, args),
                                 batch_size=500, shuffle=True)

    _, (x_spt, y_spt) = list(enumerate(trainloader))[0]
    x_spt, y_spt  = x_spt.to(device), y_spt.to(device)

    _, _, fast_weights, _ = model(x_spt, y_spt)


    return fast_weights



def test_inference(args, model, model_new, train_dataset,user_groups, test_dataset, vcn_test_all_class, device, vocab=0):



    if args.dataset == 'dbpedia':
        pass

    model.eval()
    total, correct =  0.0, 0.0

    if args.dataset == 'dbpedia':
        pass
    else:
        testloader_corresponding = DataLoader(DatasetSplit(test_dataset, vcn_test_all_class, args),batch_size=len(vcn_test_all_class),
                                              shuffle=False)


    _, (x_qry_corr, y_qry_corr) = list(enumerate(testloader_corresponding))[0]
    x_qry_corr, y_qry_corr = x_qry_corr.to(device), y_qry_corr.to(device)
    with torch.no_grad():
        test_accs_corresponding_class = model.finetunning( x_qry_corr, y_qry_corr)
    del x_qry_corr, y_qry_corr
    torch.cuda.empty_cache()



    return test_accs_corresponding_class

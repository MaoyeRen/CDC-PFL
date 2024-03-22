#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from torchvision import datasets, transforms
from sampling import cifar_full_class_fill, cifar100_full_class_fill, miniimagenet_full_class_fill
from MiniImagenet import MiniImagenet



def get_dataset(args):

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)


        user_groups, user_groups_test = cifar_full_class_fill(train_dataset, test_dataset, args.num_vcnnization, args.num_each_vcn_user)

    elif args.dataset == 'cifar100':
        data_dir = '../data/cifar100/'
        apply_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=apply_transform)

        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=apply_transform)


        user_groups, user_groups_test = cifar100_full_class_fill(train_dataset, test_dataset, args.num_vcnnization, args.num_each_vcn_user)

    elif args.dataset == 'miniimagenet':

        train_dataset = MiniImagenet(root='../data/miniimagenet', train=True, resize= 84)
        test_dataset = MiniImagenet(root='../data/miniimagenet', train=False, resize= 84)

        user_groups, user_groups_test = miniimagenet_full_class_fill(train_dataset, test_dataset,
                                                                         args.num_vcnnization, args.num_each_vcn_user)

    return train_dataset, test_dataset, user_groups , user_groups_test

def add_weights(w_avg, w):

    for key in w_avg.keys():
        w_avg[key] += w[key]
    return w_avg


def average_weights(w_avg, client_num):

    for key in w_avg.keys():
        w_avg[key] = torch.div(w_avg[key], client_num)
    return w_avg


def add_per_model_weights(w_avg, w):

    for key in range(len(w_avg)):
        w_avg[key] += w[key]
    return w_avg

def average_per_model_weights(w_avg, client_num):

    for key in range(len(w_avg)):
        w_avg[key] = torch.div(w_avg[key], client_num)
    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.vcniid:
        print('    vcn IID')
    else:
        print('    vcn Non-IID')
    if args.useriid:
        print('    user IID')
    else:
        print('    user Non-IID')
    print(f'    Fraction of vcns  : {args.frac_vcns}')
    print(f'    Fraction of users  : {args.frac_users}')
    print(f'    Fraction of train_support  : {args.frac_train_support}')
    print(f'    Fraction of train_query  : {args.frac_train_query}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

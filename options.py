#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_vcn', type=int, default=5,
                        help="number of users: K")  #vcn indicate the virtual cluster node
    parser.add_argument('--num_each_vcn_user', type=str, default=[20]*5,
                        help="number of users: K")

    parser.add_argument('--frac_vcns', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--frac_users', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--frac_train_support', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--frac_train_query', type=float, default=0.1,
                        help='the fraction of clients: C')

    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--vcn_epoch', type=int, default=1,
                        help="the number of local epochs: E")

    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default='None', help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--useriid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--vcniid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')

    parser.add_argument('--two_classes', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--full_class_fill', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')


    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--local_plot_every', type=int, default=20, help='plot local 1th 2th acc loss ')
    parser.add_argument('--print_every', type=int, default=1, help='print_every')
    parser.add_argument('--plot_every', type=int, default=50, help='plot_every')
    parser.add_argument('--record_every', type=int, default=100, help='record pth pkl txt _every')
    parser.add_argument('--test_every', type=int, default=1, help='test every')


    parser.add_argument('--more1_ep', type=int, default=10, help='more1_epoch')

    parser.add_argument('--continue_training', type=int, default=0, help='continue_training')
    parser.add_argument('--continue_training_epoch', type=int, default=0, help='continue_training')

    parser.add_argument('--percentage', type=float, default=0.5, help='split_percentage')
    parser.add_argument('--update_lr', type=float, default=0.4, help='update_lr rate')
    parser.add_argument('--meta_lr', type=float, default=0.01, help='meta_lr')


    args = parser.parse_args()
    return args

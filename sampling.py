#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import random


def cifar_full_class_fill(dataset_train, dataset_test, num_vcnnization, num_each_vcn_user, flag=0):
    num_shards, num_imgs = 10, 5000

    dict_vcn = [np.array([]) for j in range(num_vcnnization)]
    dict_users = [{i: np.array([]) for i in range(num_each_vcn_user[j])} for j in range(num_vcnnization)]
    idxs = np.arange(num_shards * num_imgs)

    labels = np.array(dataset_train.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # test_dataset
    num_shards_test, num_imgs_test = 10, 1000

    dict_vcn_test = [np.array([]) for j in range(num_vcnnization)]
    dict_users_test = [{i: np.array([]) for i in range(num_each_vcn_user[j])} for j in range(num_vcnnization)]
    idxs_test = np.arange(num_shards_test * num_imgs_test)

    labels_test = np.array(dataset_test.targets)

    # sort labels
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]

    idx_shard = [i for i in range(num_shards)]
    for i in range(num_vcnnization):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_vcn[i] = np.concatenate(
                (dict_vcn[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_vcn_test[i] = np.concatenate(
                (dict_vcn_test[i], idxs_test[rand * num_imgs_test: (rand + 1) * num_imgs_test]), axis=0)

    for i in range(len(dict_vcn)):
        random.shuffle(dict_vcn[i])
        random.shuffle(dict_vcn_test[i])

    for i in range(num_vcnnization):
        idx_shard = np.arange(20)
        for j in range(num_each_vcn_user[i]):
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i][j] = np.concatenate(
                    (dict_users[i][j], dict_vcn[i][rand * 500: (rand + 1) * 500]), axis=0)
                dict_users_test[i][j] = np.concatenate(
                    (dict_users_test[i][j], dict_vcn_test[i][rand * 100: (rand + 1) * 100]), axis=0)

    return dict_users, dict_users_test


def cifar100_full_class_fill(dataset_train, dataset_test, num_vcnnization, num_each_vcn_user, flag=0):


    # train_dataset
    num_shards, num_imgs = 100, 500

    dict_vcn = [ np.array([]) for j in range(num_vcnnization)]
    dict_users = [{i: np.array([]) for i in range(num_each_vcn_user[j])} for j in range(num_vcnnization)]
    idxs = np.arange(num_shards * num_imgs)

    labels = np.array(dataset_train.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # test_dataset
    num_shards_test, num_imgs_test = 100, 100

    dict_vcn_test = [np.array([]) for j in range(num_vcnnization)]
    dict_users_test = [{i: np.array([]) for i in range(num_each_vcn_user[j])} for j in range(num_vcnnization)]
    idxs_test = np.arange(num_shards_test * num_imgs_test)

    labels_test = np.array(dataset_test.targets)

    # sort labels
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]

    idx_shard = [i for i in range(num_shards)]
    for i in range(num_vcnnization):
        rand_set = set(np.random.choice(idx_shard, 20, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_vcn[i] = np.concatenate(
                (dict_vcn[i], idxs[ rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_vcn_test[i] = np.concatenate(
                (dict_vcn_test[i], idxs_test[rand * num_imgs_test: (rand + 1) * num_imgs_test]), axis=0)

    for i in range(len(dict_vcn)):
            random.shuffle(dict_vcn[i])
            random.shuffle(dict_vcn_test[i])

    for i in range(num_vcnnization):
        idx_shard = np.arange(20)
        for j in range(num_each_vcn_user[i]):
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i][j] = np.concatenate(
                (dict_users[i][j], dict_vcn[i][rand * 500: (rand + 1) * 500]), axis=0)

                dict_users_test[i][j] = np.concatenate(
                    (dict_users_test[i][j], dict_vcn_test[i][rand * 100: (rand + 1) * 100]), axis=0)

    return dict_users, dict_users_test


def miniimagenet_full_class_fill(dataset_train, dataset_test, num_vcnnization, num_each_vcn_user, flag=0):
    # train_dataset
    num_shards, num_imgs = 100, 500

    dict_vcn = [np.array([]) for j in range(num_vcnnization)]
    dict_users = [{i: np.array([]) for i in range(num_each_vcn_user[j])} for j in range(num_vcnnization)]
    idxs = np.arange(num_shards * num_imgs)

    labels = np.array(dataset_train.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # test_dataset
    num_shards_test, num_imgs_test = 100, 100

    dict_vcn_test = [np.array([]) for j in range(num_vcnnization)]
    dict_users_test = [{i: np.array([]) for i in range(num_each_vcn_user[j])} for j in range(num_vcnnization)]
    idxs_test = np.arange(num_shards_test * num_imgs_test)

    labels_test = np.array(dataset_test.targets)

    # sort labels
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]

    idx_shard = [i for i in range(num_shards)]
    for i in range(num_vcnnization):
        rand_set = set(np.random.choice(idx_shard, 20, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_vcn[i] = np.concatenate(
                (dict_vcn[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_vcn_test[i] = np.concatenate(
                (dict_vcn_test[i], idxs_test[rand * num_imgs_test: (rand + 1) * num_imgs_test]), axis=0)

    for i in range(len(dict_vcn)):
        random.shuffle(dict_vcn[i])
        random.shuffle(dict_vcn_test[i])

    for i in range(num_vcnnization):
        idx_shard = np.arange(20)
        for j in range(num_each_vcn_user[i]):
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i][j] = np.concatenate(
                    (dict_users[i][j], dict_vcn[i][rand * 500: (rand + 1) * 500]), axis=0)

                dict_users_test[i][j] = np.concatenate(
                    (dict_users_test[i][j], dict_vcn_test[i][rand * 100: (rand + 1) * 100]), axis=0)

    return dict_users, dict_users_test

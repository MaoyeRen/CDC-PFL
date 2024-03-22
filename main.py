#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from options import args_parser
from update import LocalUpdate, test_inference, test_inference_fastweight
from meta import Meta
from meta_vgg11_cifar100 import Meta_vgg11_cifar100
from utils import get_dataset, average_weights, exp_details, add_weights,   add_per_model_weights, average_per_model_weights
import torchvision
from torch.nn.parameter import Parameter
import random

# PLOTTING (optional)
import matplotlib
matplotlib.use('Agg')

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    args.num_each_vcn_user = eval(str(args.num_each_vcn_user))
    exp_details(args)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    device = "cpu"  if args.gpu == 'None' else "cuda:%s" % args.gpu



    train_dataset, test_dataset, user_groups, user_groups_test  = get_dataset(args)


    if args.model == 'cnn':
        if args.dataset == 'cifar':
            global_model = Meta(args)
            per_model = Meta(args)
            per_model_new = Meta(args)

    elif args.model == 'vgg11':
        if args.dataset ==  'cifar100':

            conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
            global_model = Meta_vgg11_cifar100(args)
            per_model = Meta_vgg11_cifar100(args)
            per_model_new = Meta_vgg11_cifar100(args)

            global_model_1 = torchvision.models.vgg11(pretrained=True)
            global_model_1.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            global_model_1.classifier[0] = torch.nn.Linear(512 * 1 * 1, 1024)
            del global_model_1.classifier[2]
            global_model_1.classifier[2] = torch.nn.Linear(1024, 1024)
            del global_model_1.classifier[4]
            global_model_1.classifier[4] = torch.nn.Linear(1024, 100)

            for i in range(len(global_model.net.parameters())):
                global_model.net.parameters()[i] = Parameter(list(global_model_1.parameters())[i])

        elif args.dataset == 'miniimagenet':

            conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
            global_model = Meta_vgg11_cifar100(args)
            per_model = Meta_vgg11_cifar100(args)
            per_model_new = Meta_vgg11_cifar100(args)

            global_model_1 = torchvision.models.vgg11(pretrained=True)
            global_model_1.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            global_model_1.classifier[0] = torch.nn.Linear(512 * 1 * 1, 1024)
            del global_model_1.classifier[2]
            global_model_1.classifier[2] = torch.nn.Linear(1024, 1024)
            del global_model_1.classifier[4]
            global_model_1.classifier[4] = torch.nn.Linear(1024, 100)

            for i in range(len(global_model.net.parameters())):
                global_model.net.parameters()[i] = Parameter(list(global_model_1.parameters())[i])

    else:
        exit('Error: unrecognized model')



    global_model.to(device)
    per_model.to(device)
    per_model_new.to(device)


    global_model.train()
    print("-" * 50)
    print(global_model)

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    test_accuracy_corr,test_accuracy_all, test_accuracy_new_corr, test_accuracy_new_all=[], [], [], []
    args.num_users = sum(args.num_each_vcn_user)


    new_user_test_all_class = []
    for i in range( len(user_groups_test) ):
        for j in range(len(user_groups_test[0])):
            random.shuffle(user_groups_test[i][j])
            random.shuffle(user_groups[i][j])
            new_user_test_all_class.extend(user_groups_test[i][j][int(args.percentage * len(user_groups_test[i][j])) :])


    for epoch in tqdm(range(args.epochs), ascii=" .oO0"):
        global_weights, all_local_losses = [], []
        one_vcn_list_user_losses, all_vcn_losses=[], []
        if (epoch + 1) % args.print_every == 0:
            tqdm.write(f'\r\n | Global Training Round : {epoch + 1} |\r\n')


        m = max(int(args.frac_vcns * args.num_vcnnization), 1)
        idxs_vcns = np.random.choice(range(args.num_vcnnization), m, replace=False)

        global_model.train()
        for vcn in idxs_vcns:

            vcnnization_weights = []


            m = max( int(args.frac_users * args.num_each_vcn_user[vcn]), 1)
            idxs_users = np.random.choice( range( args.num_each_vcn_user[vcn] ), m, replace=False)

            for vcn_ep in range(args.vcn_epoch):
                avg_fast_weight = []

                for idx in idxs_users:
                    train_period = 1
                    if args.dataset == 'dbpedia':
                        pass
                    else:
                        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                                  idxs=copy.deepcopy(user_groups[vcn][idx]), logger=logger, device=device)

                    if vcn_ep == 0:
                        _,  _, fast_weights,_ = local_model.update_weights(
                            model=copy.deepcopy(global_model), global_round=epoch, local_epoch=args.local_ep,  train_period=train_period)
                    else:
                        _, _, fast_weights, _ = local_model.update_weights(
                            model=copy.deepcopy(per_model), global_round=epoch, local_epoch=args.local_ep,
                            train_period=train_period)

                    if avg_fast_weight == []:
                        avg_fast_weight = fast_weights
                    else:
                        avg_fast_weight = add_per_model_weights(avg_fast_weight, fast_weights)

                    del fast_weights
                    torch.cuda.empty_cache()

                with torch.no_grad():
                    avg_fast_weight = average_per_model_weights(avg_fast_weight, len(idxs_users))

                for i in range( len(per_model.net.parameters()) ):
                    per_model.net.parameters()[i] = Parameter( avg_fast_weight[i] )

                del avg_fast_weight
                torch.cuda.empty_cache()

            avg_per_grad = []

            #
            for idx in idxs_users:
                train_period = 2
                if args.dataset == 'dbpedia':
                    pass
                else:
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                              idxs=copy.deepcopy(user_groups[vcn][idx]), logger=logger, device=device)

                _, user_loss, _, grad = local_model.update_weights(
                    model=copy.deepcopy(per_model), global_round=epoch, local_epoch=args.local_ep, train_period=train_period)
                if avg_per_grad == []:
                    avg_per_grad = list(grad)
                else:
                    avg_per_grad = add_per_model_weights(avg_per_grad, list(grad) )
    
                del grad
                torch.cuda.empty_cache()

                one_vcn_list_user_losses.append(user_loss)

            avg_per_grad = average_per_model_weights(avg_per_grad, len(idxs_users))

            vcnnization_weights = copy.deepcopy(global_model.net.parameters())
            vcnnization_weights = list(map(lambda p: p[1] - args.meta_lr * p[0], zip(avg_per_grad, vcnnization_weights)))
            if global_weights == []:
                global_weights = vcnnization_weights
            else:
                global_weights = add_per_model_weights(global_weights, vcnnization_weights)

            del vcnnization_weights
            torch.cuda.empty_cache()

            all_vcn_losses.extend(one_vcn_list_user_losses)

    

        with torch.no_grad():
            global_weights = average_per_model_weights(global_weights, args.num_vcnnization)

        for i in range(len(global_model.net.parameters())):
            global_model.net.parameters()[i] = Parameter(global_weights[i])

        all_vcn_losses_avg = sum(all_vcn_losses) / len(all_vcn_losses)
        train_loss.append(all_vcn_losses_avg)

        del global_weights
        torch.cuda.empty_cache()



        overall_train_accs, overall_train_corrects, overall_train_totals = [], [] , []
        overall_test_accs, overall_test_corrects, overall_test_totals =  [], [], []



        if (epoch + 1) % args.test_every == 0 or (epoch + 1) == args.epochs:
            list_test_accs_corr, list_test_accs_all, list_test_accs_new_corr, list_test_accs_new_all = [], [], [], []

            each_vcn_test_accs_corresponding_class = [[] for _ in range(args.num_vcnnization)]
            each_vcn_test_accs_all_class = [[] for _ in range(args.num_vcnnization)]
            each_vcn_test_accs_new_corresponding_class = [[] for _ in range(args.num_vcnnization)]
            each_vcn_test_accs_new_all_class = [[] for _ in range(args.num_vcnnization)]

            for vcn in range(args.num_vcnnization) :

                vcn_test_all_class = []
                vcn_new_test_all_class = []
                for j in range(args.num_each_vcn_user[vcn]):
                    vcn_test_all_class.extend(user_groups_test[vcn][j])
                    vcn_new_test_all_class.extend(
                        user_groups_test[vcn][j][int(args.percentage * len(user_groups_test[vcn][j])):])


                for vcn_ep in range(args.vcn_epoch):
                    avg_fast_weight =[]
                    avg_fast_weights_new = []

                    for idx in range( args.num_each_vcn_user[vcn] ):

                        if args.dataset == 'dbpedia':
                            pass
                        else:
                            if vcn_ep == 0:
                                fast_weights = test_inference_fastweight(args, copy.deepcopy(global_model), train_dataset, user_groups[vcn][idx],
                                                                                                device)
                            else:
                                fast_weights = test_inference_fastweight(args, copy.deepcopy(per_model), train_dataset, user_groups[vcn][idx],
                                                                                           device)

                        if avg_fast_weight == []:
                            avg_fast_weight = fast_weights
                        else:
                            avg_fast_weight = add_per_model_weights(avg_fast_weight, fast_weights)
                        del fast_weights
                        torch.cuda.empty_cache()

                    with torch.no_grad():
                        avg_fast_weight = average_per_model_weights(avg_fast_weight, args.num_each_vcn_user[vcn])

                    for i in range( len(per_model.net.parameters())):
                        per_model.net.parameters()[i] = Parameter(avg_fast_weight[i])
                    del avg_fast_weight
                    torch.cuda.empty_cache()

                    if args.dataset == 'dbpedia':
                        pass
                    else:
                        test_accs_corresponding_class= test_inference(
                            args, copy.deepcopy(per_model), copy.deepcopy(per_model_new), train_dataset,
                            user_groups[vcn][idx], test_dataset, vcn_test_all_class,
                            device)

                    each_vcn_test_accs_corresponding_class[vcn].append(test_accs_corresponding_class[0])


            mean_each_vcn_epoch_test_accs_corresponding_class = np.mean(each_vcn_test_accs_corresponding_class,0)


            test_accuracy_corr.append( mean_each_vcn_epoch_test_accs_corresponding_class[-1] )


        if args.test_every != args.print_every:
            args.print_every = args.test_every
        if (epoch+1) % args.print_every == 0:

            tqdm.write(f'\nResult  after {epoch + 1} global rounds:')
            tqdm.write("Training Loss : {:.4f}".format(train_loss[-1]), end='\t\t')

            print("Test Accuracy corr:", 100 * mean_each_vcn_epoch_test_accs_corresponding_class, end='\t\t')

        if (epoch + 1) % args.record_every == 0 or (epoch + 1) == args.epochs:
            PATH = '../save/model/{}_{}_{}_C[{}_{}_{}_{}]_Oiid[{}]_Uiid[{}]_F[{}]_E[{}_{}]_B[{}]_epoch[{}]_S[{}].pth'. \
                format(args.dataset, args.model, args.epochs, args.frac_vcns, args.frac_users, args.frac_train_support, args.frac_train_query,
                       args.vcniid, args.useriid, args.full_class_fill, args.vcn_epoch, args.local_ep, args.local_bs, (epoch + 1), args.seed)

            torch.save(global_model.state_dict(), PATH)

            file_name = '../save/objects/{}_{}_{}_C[[{}_{}_{}_{}]]_Oiid[{}]_Uiid[{}]_F[{}]_E[{}_{}]_B[{}]_S[{}].pkl'. \
                format(args.dataset, args.model, args.epochs + args.continue_training_epoch, args.frac_vcns, args.frac_users, args.frac_train_support, args.frac_train_query,
                       args.vcniid, args.useriid, args.full_class_fill, args.vcn_epoch, args.local_ep, args.local_bs, args.seed)

            with open(file_name, 'wb') as f:
                pickle.dump([train_loss, test_accuracy_corr, test_accuracy_all, test_accuracy_new_corr, test_accuracy_new_all], f)


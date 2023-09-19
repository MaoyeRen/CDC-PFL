# CEA-CDCPFL
Official code for paper: Personalized federated learning: A Clustered Distributed Co-Meta-Learning approach.

## Requirments
* Python3
* Pytorch
* Torchvision

## Data
* Cifar-10.
* Mimiimagenet
* Cifar-100.

## Running the experiments on cpu or gpu
* To run the  experiment  using CPU:
```
python main.py --gpu=None 
```
* Or to run it on GPU 0:
```
python main.py --gpu=0
```

## Running the experiments

* To run the  experiment  with MiniImagenet using gpu:
```
python main.py --model=vgg11 --dataset=miniimagenet --gpu=0 --orgiid=0 --useriid=1 --full_class_fill=1 --epochs=20000 --frac_orgs=1 --frac_users=0.1 --frac_train_support=0.1 --frac_train_query=0.1 --optimizer=sgd   --test_every=1 --num_org=5 --num_each_org_user=[20]*5 --percentage=0.5 --update_lr=0.2 --meta_lr=0.01 --org_epoch=2 --num_classes=100
```
## Pre-steps of using other datasets or distributions
* Firstly using any of the federated cluster methods to get the clusters of clents
* Secondly writing the code to input the clusters of the clents to virtual cluster nodes, in sampling.py    

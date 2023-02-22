
import numpy as np
import argparse
import os
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Compose, Lambda


def get_data(args: argparse.Namespace): 
    """
    Load the proper dataset based on the parsed arguments
    :param args: The arguments in which is specified which dataset should be used
    :return: a 5-tuple consisting of:
                - The train data set
                - The project data set (usually train data set without augmentation)
                - The test data set
                - a tuple containing all possible class labels
                - a tuple containing the shape (depth, width, height) of the input images
    """
    if args.dataset =='CUB-200-2011':
        return get_birds(True, '/fastscratch/harishbabu/data/CUB_200_2011/dataset/train_corners',
                                '/fastscratch/harishbabu/data/CUB_200_2011/dataset/train_crop',
                                 '/fastscratch/harishbabu/data/CUB_200_2011/dataset/test_full')
    if args.dataset == 'CARS':
        return get_cars(True, './data/cars/dataset/train', './data/cars/dataset/train', './data/cars/dataset/test')
    if args.dataset == 'FISH-224':
        return get_fish(True, '/home/harishbabu/data/Fish/phylo-VQVAE/train_padded_256', '/home/harishbabu/data/Fish/phylo-VQVAE/test_padded_256', 
                                '/home/harishbabu/data/Fish/phylo-VQVAE/test_padded_256')
    if args.dataset == 'FISH-256':
        return get_fish(True, '/home/harishbabu/data/Fish/phylo-VQVAE/train_padded_256', '/home/harishbabu/data/Fish/phylo-VQVAE/test_padded_256', 
                                '/home/harishbabu/data/Fish/phylo-VQVAE/test_padded_256', 256)
    if args.dataset == 'FISH-256-aug':
        return get_fish(True, '/home/harishbabu/data/Fish/phylo-VQVAE/train_padded_256', '/home/harishbabu/data/Fish/phylo-VQVAE/test_padded_256', 
                                '/home/harishbabu/data/Fish/phylo-VQVAE/test_padded_256', 256)
    if args.dataset == 'FISH-512':
        return get_fish(True, '/home/harishbabu/data/Fish/phylo-VQVAE/train_padded_512', '/home/harishbabu/data/Fish/phylo-VQVAE/test_padded_512', 
                                '/home/harishbabu/data/Fish/phylo-VQVAE/test_padded_512', 512)
    if args.dataset == 'CUB-224-imgnetmean':
        return get_cub_imgnetmean(True, '/fastscratch/harishbabu/data/CUB_190_pt/dataset_imgnet_pt_bb_crop/train_segmented_imagenet_background_corners',
                                '/fastscratch/harishbabu/data/CUB_190_pt/dataset_imgnet_pt_bb_crop/train_segmented_imagenet_background_crop', 
                                '/fastscratch/harishbabu/data/CUB_190_pt/dataset_imgnet_pt_bb_crop/test_segmented_imagenet_background_full', 224)
    if args.dataset == 'CUB-subset1-224-imgnetmean':
        return get_cub_imgnetmean(True, '/fastscratch/harishbabu/data/CUB_bb_crop/cub_subset1_train',
                                '/fastscratch/harishbabu/data/CUB_bb_crop/cub_subset1_train', 
                                '/fastscratch/harishbabu/data/CUB_bb_crop/cub_subset1_test', 224)
    if args.dataset =='CUB-190-2011':
        return get_birds(True, '/fastscratch/mridul/cub_190_split_resized/official/CUB_200_2011/train',
                                '/fastscratch/mridul/cub_190_split_resized/official/CUB_200_2011/train',
                                '/fastscratch/mridul/cub_190_split_resized/official/CUB_200_2011/test', 224)
    raise Exception(f'Could not load data set "{args.dataset}"!')

def get_dataloaders(args: argparse.Namespace):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, projectset, testset, classes, shape  = get_data(args)
    c, w, h = shape
    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=cuda
                                              )
    projectloader = torch.utils.data.DataLoader(projectset,
                                            #    batch_size=args.batch_size,
                                              batch_size=int(args.batch_size/4), #make batch size smaller to prevent out of memory errors during projection
                                              shuffle=False,
                                              pin_memory=cuda
                                              )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=cuda
                                             )
    print("Num classes (k) = ", len(classes), flush=True)
    return trainloader, projectloader, testloader, classes, c


def get_birds(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size = 224): 
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406) # imagenet mean and std
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])
    if augment:
        transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.RandomOrder([
            transforms.RandomPerspective(distortion_scale=0.2, p = 0.5),
            transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.02,0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, shear=(-2,2),translate=[0.05,0.05]),
            ]),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transform_no_augment

    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    classes = trainset.classes
    for i in range(len(classes)):
        classes[i]=classes[i].split('.')[1]
    return trainset, projectset, testset, classes, shape


def get_cub_imgnetmean(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size = 224): 
    shape = (3, img_size, img_size)
    # mean = (0.4699, 0.4381, 0.3879) # calculated using all the train images with imgnet background and bb crop
    # std = (0.1360, 0.1292, 0.1311)
    mean = (0.485, 0.456, 0.406) # imagenet mean and std
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])
    if augment:
        transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.RandomOrder([
            transforms.RandomPerspective(distortion_scale=0.2, p = 0.5),
            transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.02,0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, shear=(-2,2),translate=[0.05,0.05]),
            ]),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transform_no_augment

    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    classes = trainset.classes
    for i in range(len(classes)):
        classes[i]=classes[i].split('.')[1]
    return trainset, projectset, testset, classes, shape


def get_cars(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size = 224): 
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406) # imagenet mean and std
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])

    if augment:
        transform = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)), #resize to 256x256
            transforms.RandomOrder([
            transforms.RandomPerspective(distortion_scale=0.5, p = 0.5),
            transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.4,0.4)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15,shear=(-2,2)),
            ]),
            transforms.RandomCrop(size=(img_size, img_size)), #crop to 224x224
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transform_no_augment

    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    classes = trainset.classes
    
    return trainset, projectset, testset, classes, shape


def get_fish(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size = 224): 
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406) # imagenet mean and std
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])
    if augment:
        transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.RandomOrder([
            transforms.RandomPerspective(distortion_scale=0.2, p = 0.5),
            transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.02,0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, shear=(-2,2),translate=[0.05,0.05]),
            ]),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transform_no_augment

    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    classes = trainset.classes
    print(classes)
    # for i in range(len(classes)):
    #     classes[i]=classes[i].split('.')[1]
    return trainset, projectset, testset, classes, shape


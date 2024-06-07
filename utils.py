#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import random
import torch
import csv
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from sampling import unbc_noniid
import os.path
from PIL import Image
import torch
import os
import torch.utils.data as data



class unbc(Dataset):
    def __init__(self, root_dir, csv_file, indx, transform=None):
        self.data = []
        self.root_dir = root_dir
        self.transform = transform

        # Read the CSV file with format: index, image_path, label
        with open(csv_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                index = int(row[0])
                if index in indx:
                    self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data[idx][1])
        image = Image.open(img_path)
        label = int(self.data[idx][2])
        
        if self.transform:
            image = self.transform(image)

        return image, label 


class unbc_lab(Dataset):
    def __init__(self, root_dir, csv_file, indx, transform=None):
        self.data = []
        self.root_dir = root_dir
        self.transform = transform

        # Read the CSV file with format: index, image_path, label
        with open(csv_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                index = int(row[0])
                if index in indx:
                    self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data[idx][1])
        image = Image.open(img_path)
        label = int(self.data[idx][2])
        index = int(self.data[idx][0])  # Extract index from the data
        
        if self.transform:
            image = self.transform(image)

        return image, label, index 


class biovid(Dataset):
    def __init__(self, root_dir, csv_file, indx, transform=None):
        self.data = []
        self.root_dir = root_dir
        self.transform = transform

        # Read the CSV file with format: index, image_path, label
        with open(csv_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';')
            for row in csvreader:
                index = int(row[0])
                if index in indx:
                    self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data[idx][1])
        image = Image.open(img_path)
        label = int(self.data[idx][2])
        
        if self.transform:
            image = self.transform(image)

        return image, label  


class biovid_lab(Dataset):
    def __init__(self, root_dir, csv_file, indx, transform=None):
        self.data = []
        self.root_dir = root_dir
        self.transform = transform

        # Read the CSV file with format: index, image_path, label
        with open(csv_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';')
            for row in csvreader:
                index = int(row[0])
                if index in indx:
                    self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data[idx][1])
        image = Image.open(img_path)
        label = int(self.data[idx][2])
        index = int(self.data[idx][0])  # Extract index from the data
        
        if self.transform:
            image = self.transform(image)

        return image, label, index 





def get_dataset(args, fold_index):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'biovid':
        
        root_dir = '/Users/jmejia/Thesis/video'

        train_data_dir = '/Users/jmejia/Thesis/video/Bio_path_def2.csv'    
        test_data_dir = '/Users/jmejia/Thesis/img-label.csv'    
        train_indices = list(range(1, 60 + 1))
        test_indices = list(range(1, 25 + 1))

        apply_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
        
        
        train_dataset = biovid(root_dir, train_data_dir, train_indices, transform=apply_transform)
        test_dataset = unbc(root_dir, test_data_dir,test_indices, transform=apply_transform)
        train_dataset_lab=biovid_lab(root_dir, train_data_dir, train_indices, transform=apply_transform)
        user_groups = unbc_noniid(train_dataset_lab, args.num_users, train_indices)

        
        print(f"Training Groups: {train_indices}")
        print(f"Testing Groups: {test_indices}")
        print(f"Number of Training Samples: {len(train_dataset)}")
        print(f"Number of Testing Samples: {len(test_dataset)}")
        print()

        return train_dataset, test_dataset, user_groups


    elif args.dataset == 'unbc':
        root_dir = '/Users/jmejia/Thesis/video'

        test_data_dir = '/Users/jmejia/Thesis/video/Bio_path_def2.csv'    
        train_data_dir = '/Users/jmejia/Thesis/img-label.csv'    
        test_indices = list(range(1, 60 + 1))
        train_indices = list(range(1, 25 + 1))

        apply_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
        
        


        train_dataset = unbc(root_dir, train_data_dir, train_indices, transform=apply_transform)
        test_dataset = biovid(root_dir, test_data_dir,test_indices, transform=apply_transform)
        train_dataset_lab=unbc_lab(root_dir, train_data_dir, train_indices, transform=apply_transform)
        user_groups = unbc_noniid(train_dataset_lab, args.num_users, train_indices)

        
        print(f"Training Groups: {train_indices}")
        print(f"Testing Groups: {test_indices}")
        print(f"Number of Training Samples: {len(train_dataset)}")
        print(f"Number of Testing Samples: {len(test_dataset)}")
        print()

        return train_dataset, test_dataset, user_groups
    
    elif args.dataset == 'both':
        root_dir = '/Users/jmejia/Thesis/'

        biovid_data_dir = '/Users/jmejia/Thesis/video/Bio_path_def2.csv'    
        unbc_data_dir = '/Users/jmejia/Thesis/img-label.csv'    
       

        # Define the total number of samples
        biovid_samples = 60
        unbc_samples = 25
        k_folds = 5
        # Generate a list of all indices
        biovid_all_indices = list(range(1, biovid_samples + 1))
        unbc_all_indices = list(range(1, unbc_samples + 1))

        
        # Select the indices for the biovid dataset
        fold_size = len(biovid_all_indices) // k_folds
        folds = [biovid_all_indices[i:i + fold_size] for i in range(0, len(biovid_all_indices), fold_size)]
        biovid_test_indices = folds[fold_index]
        biovid_train_indices = [idx for i, fold in enumerate(folds) if i != fold_index for idx in fold]

        # Select the indices for the unbc dataset
        fold_size = len(unbc_all_indices) // k_folds
        folds = [unbc_all_indices[i:i + fold_size] for i in range(0, len(unbc_all_indices), fold_size)]
        unbc_test_indices = folds[fold_index]
        unbc_train_indices = [idx for i, fold in enumerate(folds) if i != fold_index for idx in fold]

        apply_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),  # Add this line for grayscale conversion
                transforms.ToTensor(),
            ])


        biovid_train_dataset = biovid(root_dir, biovid_data_dir, biovid_train_indices, transform=apply_transform)
        biovid_test_dataset = biovid(root_dir, biovid_data_dir,biovid_test_indices, transform=apply_transform)
        biovid_train_dataset_lab=biovid_lab(root_dir, biovid_data_dir, biovid_train_indices, transform=apply_transform)
        biovid_user_groups = unbc_noniid(biovid_train_dataset_lab, args.num_users, biovid_train_indices)

        unbc_train_dataset = unbc(root_dir, unbc_data_dir, unbc_train_indices, transform=apply_transform)
        unbc_test_dataset = unbc(root_dir, unbc_data_dir,unbc_test_indices, transform=apply_transform)
        unbc_train_dataset_lab=unbc_lab(root_dir, unbc_data_dir, unbc_train_indices, transform=apply_transform)
        unbc_user_groups = unbc_noniid(unbc_train_dataset_lab, args.num_users, unbc_train_indices)

        
        print(f"Training Groups Biovid: {biovid_train_indices}")
        print(f"Testing Groups Biovid: {biovid_test_indices}")
        print(f"Number of Training Samples Biovid: {len(biovid_train_dataset)}")
        print(f"Number of Testing Samples Biovid: {len(biovid_test_dataset)}")
        print()
        print(f"Training Groups UNBC: {unbc_train_indices}")
        print(f"Testing Groups UNBC: {unbc_test_indices}")
        print(f"Number of Training Samples UNBC: {len(unbc_train_dataset)}")
        print(f"Number of Testing Samples UNBC: {len(unbc_test_dataset)}")
        print()

        return biovid_train_dataset, biovid_test_dataset, biovid_user_groups, unbc_train_dataset, unbc_test_dataset, unbc_user_groups

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return



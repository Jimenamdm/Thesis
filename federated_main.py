#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import os
import time
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNUnbc, CNNBiovid
from utils import get_dataset, average_weights, exp_details

def train_and_collect_updates(dataset, user_groups, global_model, args, logger, epoch):
    local_weights = []
    local_losses = []  # Initialize a list to store losses
    for idx in user_groups:
        local_model = LocalUpdate(args=args, dataset=dataset, idxs=user_groups[idx], logger=logger)
        w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
        local_weights.append(copy.deepcopy(w))
        local_losses.append(loss)  # Collect the loss
    average_loss = sum(local_losses) / len(local_losses) if local_losses else 0  # Calculate the average loss
    return local_weights, average_loss  # Return both weights and the average loss

if __name__ == '__main__':
    start_time = time.time()

    # define paths and logger
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)
    device = 'cpu'
    k_folds = 5

    # Separate storage for results of Biovid and UNBC
    fold_results_biovid = []
    fold_results_unbc = []

    for fold_index in range(k_folds):
        print(f"\nFold {fold_index+1}/{k_folds}")
        train_dataset_1, test_dataset_1, user_groups_1, train_dataset_2, test_dataset_2, user_groups_2 = get_dataset(args, fold_index)

        # Initialize model
        if args.model == 'cnn':
            global_model = CNNUnbc(args=args)
        else:
            exit('Error: unrecognized model')

        global_model.to(device)
        global_model.train()
        print(global_model)

        for epoch in tqdm(range(args.epochs)):
            global_model.train()
            
            # Collect local model updates from each dataset
            local_updates_1, avg_loss_1 = train_and_collect_updates(train_dataset_1, user_groups_1, global_model, args, logger, epoch)
            local_updates_2, avg_loss_2 = train_and_collect_updates(train_dataset_2, user_groups_2, global_model, args, logger, epoch)

            # Aggregate updates from both datasets
            combined_updates = local_updates_1 + local_updates_2
            global_weights = average_weights(combined_updates)

            # Update the global model with the aggregated weights
            global_model.load_state_dict(global_weights)
            

           # Determine the number of users to participate from each dataset
            m_1 = max(int(args.frac * len(user_groups_1)), 1)
            m_2 = max(int(args.frac * len(user_groups_2)), 1)

            # Select users for this round for each dataset
            idxs_users_1 = np.random.choice(list(user_groups_1.keys()), m_1, replace=False)
            idxs_users_2 = np.random.choice(list(user_groups_2.keys()), m_2, replace=False)


           # Dataset 1: Train and collect updates
            for idx in idxs_users_1:
                local_model = LocalUpdate(args=args, dataset=train_dataset_1,
                                  idxs=user_groups_1[idx], logger=logger)
                w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                local_updates_1.append(copy.deepcopy(w))

            # Dataset 2: Train and collect updates (similar to Dataset 1)
            for idx in idxs_users_2:
                local_model = LocalUpdate(args=args, dataset=train_dataset_2,
                                  idxs=user_groups_2[idx], logger=logger)
                w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                local_updates_2.append(copy.deepcopy(w))

            # Aggregate updates from both datasets
            combined_updates = local_updates_1 + local_updates_2    
            global_weights = average_weights(combined_updates)
            global_model.load_state_dict(global_weights)
            test_loss_1, test_acc_1, true_labels_1, pred_labels_1, f1_1 = test_inference(args, global_model, test_dataset_1) 
            test_loss_2, test_acc_2, true_labels_2, pred_labels_2, f1_2 = test_inference(args, global_model, test_dataset_2) 
            print(f"\nFold {fold_index+1} F1 Score: {f1_1:.2f}")
            print(f"Fold {fold_index+1} F1 Score: {f1_2:.2f}")

    
                
        # Final evaluation on test datasets
        test_loss_1, test_acc_1, true_labels_1, pred_labels_1, f1_1 = test_inference(args, global_model, test_dataset_1) 
        test_loss_2, test_acc_2, true_labels_2, pred_labels_2, f1_2 = test_inference(args, global_model, test_dataset_2) 
        conf_matrix_1 = confusion_matrix(true_labels_1, pred_labels_1)
        report_1 = classification_report(true_labels_1, pred_labels_1, output_dict=True, zero_division=0)
        conf_matrix_2 = confusion_matrix(true_labels_2, pred_labels_2)
        report_2 = classification_report(true_labels_2, pred_labels_2, output_dict=True, zero_division=0)

        fold_results_biovid.append({
            "accuracy": test_acc_1,
            "F1_score": f1_1,
            "confusion_matrix": conf_matrix_1,
            "classification_report": report_1
        })

        fold_results_unbc.append({
            "accuracy": test_acc_2,
            "F1_score": f1_2,
            "confusion_matrix": conf_matrix_2,
            "classification_report": report_2
        })
        
        print(f"\nFold {fold_index+1} Final Test Accuracy on Dataset 1: {test_acc_1*100:.2f}%, F1 Score: {f1_1:.2f}")
        print(f"Fold {fold_index+1} Final Test Accuracy on Dataset 2: {test_acc_2*100:.2f}%, F1 Score: {f1_2:.2f}")
        print("Confusion Matrix Biovid:")
        print(conf_matrix_1)
        print("Classification Report Biovid:")
        print(classification_report(true_labels_1, pred_labels_1, zero_division=0))
        print("Confusion Matrix UNBC:")
        print(conf_matrix_2)
        print("Classification Report UNBC:")
        print(classification_report(true_labels_2, pred_labels_2, zero_division=0))

    # Calculate and print average results for Biovid
    avg_accuracy_biovid = sum(fold['accuracy'] for fold in fold_results_biovid) / k_folds
    avg_f1_score_biovid = sum(fold['F1_score'] for fold in fold_results_biovid) / k_folds
    total_confusion_matrix_biovid = np.sum([fold['confusion_matrix'] for fold in fold_results_biovid], axis=0)

    print(f"\nBiovid - Average Results Across All Folds: Accuracy: {avg_accuracy_biovid*100:.2f}%, F1 Score: {avg_f1_score_biovid:.2f}")
    print("Biovid - Total Confusion Matrix Across All Folds:")
    print(total_confusion_matrix_biovid)

    # Calculate and print average results for UNBC
    avg_accuracy_unbc = sum(fold['accuracy'] for fold in fold_results_unbc) / k_folds
    avg_f1_score_unbc = sum(fold['F1_score'] for fold in fold_results_unbc) / k_folds
    total_confusion_matrix_unbc = np.sum([fold['confusion_matrix'] for fold in fold_results_unbc], axis=0)

    print(f"\nUNBC - Average Results Across All Folds: Accuracy: {avg_accuracy_unbc*100:.2f}%, F1 Score: {avg_f1_score_unbc:.2f}")
    print("UNBC - Total Confusion Matrix Across All Folds:")
    print(total_confusion_matrix_unbc)



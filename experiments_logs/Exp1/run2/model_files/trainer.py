import os
import sys
from tqdm import tqdm
import time
sys.path.append("..")
import numpy as np
from utils import _logger, set_requires_grad, _calc_metrics, copy_Files
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataloader.augmentations import initial_plot
from models.loss import NTXentLoss


def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode, home_path):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(temp_cont_optimizer, 'min')
    pbar = tqdm(range(1, config.num_epoch + 1))
    for epoch in pbar:
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode)
        valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode, config)
        if training_mode != 'self_supervised':  # use scheduler in all other modes.
            scheduler1.step(valid_loss)
            scheduler2.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')
        time.sleep(0.01)
    pbar.close()

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(),
                'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        total_loss, total_acc, pred_labels, true_labels = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode, config)
        logger.debug(f'Test loss      :{total_loss:0.4f}\t | Test Accuracy      : {total_acc:0.4f}')
        _calc_metrics(pred_labels, true_labels, experiment_log_dir, home_path)

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config,
                device, training_mode):
    total_loss = []
    total_acc = []
    outs = []
    trgs = []
    model.train()
    temporal_contr_model.train()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised":
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)

            temp_cont_loss1, temp_cont_lstm_feat1, ret = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_lstm_feat2, ret = temporal_contr_model(features2, features1)

            # normalize projection feature vectors
            zis = temp_cont_lstm_feat1
            zjs = temp_cont_lstm_feat2

        else:
            output, features = model(data)
            if config.output_type =="vit":
                _, _, output = temporal_contr_model(features, features)

        # compute loss
        if training_mode == "self_supervised":
            lambda1 = config.vit #원본은 1
            lambda2 = config.simclr #원본은 0.7
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + nt_xent_criterion(zis, zjs) * lambda2

        else:  # supervised training or fine tuining
            predictions = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
            pred = predictions.max(1, keepdim=True)[1]
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    print('train set')
    print(trgs[:25])
    print(outs[:25])

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode, config):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])
    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                output, feature = model(data)
                if config.output_type =="vit":
                    _,_,output = temporal_contr_model(feature, feature)

            # compute loss
            if training_mode != "self_supervised":
                predictions = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if training_mode == "self_supervised":
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
    print('valid set')
    print(trgs[:25])
    print(outs[:25])
    return total_loss, total_acc, outs, trgs

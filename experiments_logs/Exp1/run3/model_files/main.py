import torch, gc
import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad, _calc_metrics, copy_Files
from dataloader.dataloader import data_generator
from trainer.trainer import Trainer, model_evaluate
from models.TC import TC
from models.model import make_patch_logits, resnet_model, densenet_model

# Args selections

start_time = datetime.now()
gc.collect()
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run3', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default= 0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='self_supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--selected_dataset', default='pFD', type=str,
                    help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
args = parser.parse_args()

device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = '개발중...'
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()
print(configs.__dict__)
# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
import random
random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                  training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
data_path = f"./data/{data_type}"
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
logger.debug("Data loaded ...")

# Load Model


if configs.modelname == 'CNN':
    model = make_patch_logits(configs).to(device)
    print(model, configs.modelname, configs.lr, configs.batch_size, configs.weightdecay)
elif configs.modelname == 'densenet121':
    model = densenet_model(configs).to(device)
    print(configs.modelname, configs.pretrain, configs.output_type, configs.lr, configs.batch_size, configs.weightdecay)
else:
    model = resnet_model(configs).to(device)
    print(configs.modelname, configs.pretrain, configs.output_type, configs.lr, configs.batch_size, configs.weightdecay)
print(configs.simclr, configs.vit)
temporal_contr_model = TC(configs, device).to(device)

if training_mode == "fine_tune":
    # load saved model of this experiment
    load_from = os.path.join(
        os.path.join(logs_save_dir, experiment_description, run_description, "self_supervised_seed_0",
                     "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    pretrained_dict = chkpoint["temporal_contr_model_state_dict"]
    model_dict = temporal_contr_model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    temporal_contr_model.load_state_dict(model_dict)

if training_mode == "train_linear" or "tl" in training_mode:
    load_from = os.path.join(
        os.path.join(logs_save_dir, experiment_description, run_description, "self_supervised_seed_0",
                     "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # delete these parameters (Ex: the linear layer at the end)
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

    pretrained_dict = chkpoint["temporal_contr_model_state_dict"]
    model_dict = temporal_contr_model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # delete these parameters (Ex: the linear layer at the end)
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]

    model_dict.update(pretrained_dict)
    temporal_contr_model.load_state_dict(model_dict)
    set_requires_grad(temporal_contr_model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.


if training_mode == "random_init":
    model_dict = model.state_dict()
    # delete all the parameters except for logits
    del_list = ['logits']
    pretrained_dict_copy = model_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del model_dict[i]
    set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.

    model_dict = temporal_contr_model.state_dict()
    # delete all the parameters except for logits
    del_list = ['logits']
    pretrained_dict_copy = model_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del model_dict[i]
    set_requires_grad(temporal_contr_model, model_dict, requires_grad=False)  # Freeze everything except last layer.

model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, weight_decay=configs.weightdecay)
temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, weight_decay=configs.weightdecay)

if training_mode == "self_supervised":  # to do it only once
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

# Trainer
Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device,
        logger, configs, experiment_log_dir, training_mode, args.home_path)

logger.debug(f"Training time is : {datetime.now() - start_time}")

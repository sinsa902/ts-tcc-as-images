import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.utils.data import Dataset
import os
import numpy as np
from .augmentations import DataTransform, encoding_image, rgb_image


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, random_int, config, training_mode, dataset_type):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.datafolder_name = config.datafolder_name
        print(self.datafolder_name)
        self.len = X_train.shape[0]
        if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = DataTransform(self.x_data, random_int, config, dataset_type)
        else:
            encoding_image(self.x_data, random_int, self.datafolder_name, config, dataset_type)
            self.x_data = rgb_image(random_int, self.datafolder_name, config, dataset_type)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, configs, training_mode):
    if configs.dataset == "pFD":
        train_dataset1 = torch.load(os.path.join(data_path, "train_a.pt"))
        train_dataset2 = torch.load(os.path.join(data_path, "test_a.pt"))
        train_dataset3 = torch.load(os.path.join(data_path, "val_a.pt"))
        for subdata in ['b', 'c', 'd']:
            temp_trainset = torch.load(os.path.join(data_path, f"train_{subdata}.pt"))
            temp_testset = torch.load(os.path.join(data_path, f"test_{subdata}.pt"))
            temp_valset = torch.load(os.path.join(data_path, f"val_{subdata}.pt"))
            train_dataset1['samples'] = torch.concat((train_dataset1['samples'], temp_trainset['samples']))
            train_dataset2['samples'] = torch.concat((train_dataset2['samples'], temp_testset['samples']))
            train_dataset3['samples'] = torch.concat((train_dataset3['samples'], temp_valset['samples']))
            train_dataset1['labels'] = torch.concat((train_dataset1['labels'], temp_trainset['labels']))
            train_dataset2['labels'] = torch.concat((train_dataset2['labels'], temp_testset['labels']))
            train_dataset3['labels'] = torch.concat((train_dataset3['labels'], temp_valset['labels']))


    else:
        train_dataset1 = torch.load(os.path.join(data_path, "train.pt"))
        train_dataset2 = torch.load(os.path.join(data_path, "test.pt"))
        train_dataset3 = torch.load(os.path.join(data_path, "val.pt"))

    train_dataset = {}
    train_dataset['samples'] = torch.concat(
        (train_dataset1['samples'], train_dataset2['samples'], train_dataset3['samples']))
    train_dataset['labels'] = torch.concat(
        (train_dataset1['labels'], train_dataset2['labels'], train_dataset3['labels']))
    print(train_dataset['samples'].shape)

    sample_size = configs.datalen
    if sample_size == 0:
        sample_size = train_dataset['samples'].shape[0]

    total_range = np.arange(0, train_dataset['samples'].shape[0])
    random_int = np.random.choice(total_range, sample_size, replace=False)
    random_int = torch.from_numpy(random_int).long()
    train_dataset['samples'] = train_dataset['samples'][random_int]
    train_dataset['labels'] = train_dataset['labels'][random_int]

    dataset_size = len(train_dataset['samples'])
    if training_mode == "self_supervised":
        train_size = dataset_size
        validation_size = 0
        test_size = 0
    else:
        train_size = int(dataset_size * configs.train_ratio)
        validation_size = int(dataset_size * configs.valid_ratio)
        test_size = dataset_size - train_size - validation_size

    train_dataset = Load_Dataset(train_dataset, random_int, configs, training_mode, "train")
    if dataset_size == len(train_dataset):
        print("dataset length 제대로 들어옴")
        print("randomseed", configs.seed, train_size, validation_size, test_size)

    train_dataset, valid_dataset, test_dataset = random_split(train_dataset, [train_size, validation_size, test_size])
    if train_size < 100:
        batch_size = 8
    else:
        batch_size = configs.batch_size

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader
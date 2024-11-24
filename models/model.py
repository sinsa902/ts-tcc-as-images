from torch import nn

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import timm


class densenet_model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        model = timm.create_model(
            configs.modelname,
            pretrained=configs.pretrain,
            num_classes=configs.num_classes,
        )
        self.densenet_channel = model.classifier.in_features
        self.embed_dim = configs.embed_dim

        self.densenet_model = list(model.children())[:-2]
        self.densenet_model = nn.Sequential(*self.densenet_model)

        self.project = nn.Conv2d(
            self.densenet_channel,
            configs.embed_dim,
            kernel_size=configs.patch_size,
            stride=1,
            padding="same",
        )

        self.logits = nn.Sequential(*list(model.children())[-2:])

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.densenet_model(x)
        embedding = self.project(x)
        z = embedding.view(B, self.embed_dim, -1).permute(0, 2, 1)  # BCHW -> BNC
        logits = self.logits(x)
        return logits, z


class resnet_model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        model = timm.create_model(
            configs.modelname,
            pretrained=configs.pretrain,
            num_classes=configs.num_classes,
        )
        self.resnet_channel = model.fc.in_features
        self.embed_dim = configs.embed_dim

        self.res_model = list(model.children())[:-2]
        self.res_model = nn.Sequential(*self.res_model)

        self.project = nn.Conv2d(
            self.resnet_channel,
            configs.embed_dim,
            kernel_size=configs.patch_size,
            stride=1,
            padding="same",
        )

        self.logits = nn.Sequential(*list(model.children())[-2:])

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.res_model(x)
        embedding = self.project(x)
        z = embedding.view(B, self.embed_dim, -1).permute(0, 2, 1)  # BCHW -> BNC
        logits = self.logits(x)
        return logits, z


class make_patch_logits(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.embed_dim = configs.embed_dim
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                configs.input_channels,
                12,
                kernel_size=configs.kernel_size,
                padding=(configs.kernel_size // 2),
            ),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(configs.dropout)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                12,
                32,
                kernel_size=configs.kernel_size,
                padding=(configs.kernel_size // 2),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(configs.dropout)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=(3 // 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(configs.dropout)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=(3 // 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(configs.dropout)
        )
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(128, configs.final_out_channels, kernel_size=1, padding=(1 // 2)),
            nn.BatchNorm2d(configs.final_out_channels),
            nn.ReLU(),
            # nn.Dropout2d(configs.dropout)
        )
        self.project = nn.Conv2d(
            configs.final_out_channels,
            configs.embed_dim,
            kernel_size=configs.patch_size,
            stride=configs.patch_size,
        )
        self.logits = nn.Sequential(
            nn.Linear(
                configs.features_len * configs.final_out_channels, configs.num_classes
            ),
            # nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        embedding = self.project(x)
        z = embedding.view(B, self.embed_dim, -1).permute(0, 2, 1)  # BCHW -> BNC
        z_flat = x.reshape(B, -1)
        logits = self.logits(z_flat)
        return logits, z


class base_Model_rev1(nn.Module):
    def __init__(self, configs):
        super(base_Model_rev1, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                configs.input_channels,
                32,
                kernel_size=configs.kernel_size,
                stride=configs.stride,
                bias=False,
                padding=(configs.kernel_size // 2),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(configs.dropout),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=configs.kernel_size,
                stride=configs.stride,
                bias=False,
                padding=(configs.kernel_size // 2),
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(
                64,
                configs.final_out_channels,
                kernel_size=configs.kernel_size,
                stride=1,
                bias=False,
                padding=(configs.kernel_size // 2),
            ),
            nn.BatchNorm2d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(
            model_output_dim * configs.final_out_channels, configs.num_classes
        )

    def forward(self, x_in):
        if x_in.shape[1] != 1:
            x_in = x_in.permute(0, 3, 1, 2)
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x


class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                configs.input_channels,
                32,
                kernel_size=configs.kernel_size,
                stride=configs.stride,
                bias=False,
                padding=(configs.kernel_size // 2),
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(
                64,
                configs.final_out_channels,
                kernel_size=8,
                stride=1,
                bias=False,
                padding=4,
            ),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(
            model_output_dim * configs.final_out_channels, configs.num_classes
        )

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x

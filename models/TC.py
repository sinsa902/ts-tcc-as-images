import torch
import torch.nn as nn
import numpy as np
from .vit import ViT


class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.embed_dim
        self.timestep = int(configs.TC.timesteps)
        self.image_size = int(configs.TC.timesteps ** (1 / 2))
        self.sample_margin = int(self.image_size // 2)
        self.Wk = nn.ModuleList([nn.Linear(configs.embed_dim, configs.embed_dim) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device

        self.projection_head = nn.Sequential(
            nn.Linear(configs.embed_dim, configs.embed_dim // 2),
            nn.BatchNorm1d(configs.embed_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.logits = nn.Linear(configs.embed_dim, configs.num_classes)

        self.ViT = ViT(configs)

    def forward(self, features_aug1, features_aug2):
        z_aug1 = features_aug1  # features are (batch_size, patch, channel)
        seq_len = z_aug1.shape[1]

        z_aug2 = features_aug2

        batch = z_aug1.shape[0]
        x_samples = torch.randint(self.sample_margin, self.image_size - self.sample_margin, size=(1,)).long().to(
            self.device)  # randomly pick time stamps
        y_samples = torch.randint(self.sample_margin, self.image_size - self.sample_margin, size=(1,)).long().to(
            self.device)  # randomly pick time stamps
        first_patch = (y_samples - self.sample_margin) * self.image_size + (x_samples - self.sample_margin)

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(self.timestep):
            r, c = i // (self.timestep ** (1 / 2)), i % (self.timestep ** (1 / 2))
            patch_num = first_patch + (r * self.image_size) + c
            encode_samples[i] = z_aug2[:, patch_num.long(), :].view(batch, self.num_channels)

        c_t = self.ViT(z_aug1)

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep

        test2 = self.logits(c_t)
        return nce, self.projection_head(c_t), test2
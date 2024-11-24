import numpy as np
import torch
from notebook.terminal import initialize
from utils import origin_plot_save
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import time
import pywt

def DataTransform(sample, random_int, config, dataset_type):
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
    encoding_image(weak_aug, random_int, config.datafolder_name+"_weak", config, dataset_type)
    weak_aug = rgb_image(random_int, config.datafolder_name+"_weak", config, dataset_type)
    encoding_image(strong_aug, random_int, config.datafolder_name+"_strong", config, dataset_type)
    strong_aug = rgb_image(random_int, config.datafolder_name+"_strong", config, dataset_type)
    return weak_aug, strong_aug

def encoding_image(input_dataset, random_int, aug_type, config, dataset_type):
    dir = os.path.join("encoded_images", config.dataset, dataset_type, aug_type)
    os.makedirs(dir, exist_ok=True)
    img_list = os.listdir(dir)
    input_dataset = input_dataset.squeeze(1)
    img_ratio = min(config.img_size / input_dataset.shape[1], 1.0)
    flag_img = config.img_type
    if flag_img==0:
        vmin=-1.
        vmax=1.
        transform_img = GramianAngularField(image_size=img_ratio, overlapping=True)
    elif flag_img==1:
        vmin=0.
        vmax=1.
        transform_img = MarkovTransitionField(image_size=img_ratio,overlapping=True)
    elif flag_img==2:
        if img_ratio ==1.0:
            img_ratio = int(1.0)
        transform_img = RecurrencePlot(dimension=img_ratio, threshold='point', percentage=20)

    pbar = tqdm(random_int)  ## tqdm 객체 생성
    for idx, i in enumerate(pbar):
        if f"{i}.png" in img_list:
            continue
        if flag_img < 3:
            image = transform_img.transform(input_dataset[idx:idx + 1])
        elif flag_img == 3:
            coef, freqs = pywt.cwt(input_dataset[idx].numpy(), np.arange(1, 225), 'morl')
        initial_plot(config)
        if flag_img < 2:
            plt.imshow(image[0], cmap='rainbow', origin='lower', vmin=vmin, vmax=vmax)
        elif flag_img == 2:
            plt.imshow(image[0], cmap='binary', origin='lower')
        elif flag_img == 3:
            heatmap = cv2.resize(coef, (224, 224))
            heatmapshow = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
            heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
            #plt.imshow(coef, cmap='rainbow', aspect='auto', vmax=abs(coef).max(), vmin=-abs(coef).max())
            #plt.gca().invert_yaxis()
        cv2.imwrite(os.path.join(dir,f"{i}.png"), heatmapshow)
        #plt.savefig(os.path.join(dir, f"{i}_plt.png"), bbox_inches="tight", pad_inches=0)
        time.sleep(0.01)
    pbar.close()


def rgb_image(random_int, aug_type, config, dataset_type):
    return_image = []
    dir = os.path.join("encoded_images", config.dataset, dataset_type, aug_type)
    pbar = tqdm(random_int)  ## tqdm 객체 생성
    for idx, i in enumerate(pbar):
        rgb_image = cv2.imread(os.path.join(dir, f"{i}.png"))
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (config.img_size, config.img_size))
        rgb_image = np.transpose(rgb_image, (2, 0, 1))
        return_image.append(rgb_image)
    pbar.close()
    return_image = np.array(return_image)
    return return_image


def initial_plot(config):
    plt.close()
    plt.figure(figsize=(config.img_size / 100, config.img_size / 100))
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]  # device = {device} cpu
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return torch.from_numpy(np.concatenate((ai), axis=1))


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[:, warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)


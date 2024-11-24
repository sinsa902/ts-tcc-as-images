class Config(object):
    def __init__(self):
        # 데이터셋 정보
        self.dataset = "sleepEDF"
        self.num_classes = 5
        self.img_type = 3  # 0: gaf / 1:mark / 2:replot /3:cwt
        self.datafolder_name = "new_cwt_origin_224x224,all_cv2_500hz"
        self.datalen = 0  # 0:all
        self.train_ratio = 0.6
        self.valid_ratio = 0.2
        self.seed = 0
        self.output_type = "resnet"  # resnet or vit

        # 모델 정보
        self.num_epoch = 30
        self.batch_size = 64
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 0.0003
        self.weightdecay = 0.0003
        self.simclr = 1.0
        self.vit = 1.0


        self.modelname = 'resnet18'  # cnn은 'CNN' resnet 10t, 18, 50, resnet101
        self.pretrain = True
        self.img_size = 224
        self.input_channels = 3  # RGB
        self.kernel_size = 3
        self.div = 2 ** 4  # ** pooling
        self.final_out_channels = 1
        self.dropout = 0.3

        # for noisy labels experiment
        self.corruption_prob = 0.3
        if self.modelname == 'CNN':
            self.patch_size = 2
        else:
            self.patch_size = 1
        self.embed_dim = 16  # psize^2*channel
        self.features_len = int(self.img_size / self.div) ** 2  # (img_size/convlayers248)^2
        self.num_heads = 8
        self.depth = 8
        self.mlp_ratio = 2.
        self.qkv_bias = False
        self.drop_rate = 0.0
        self.attn_drop_rate = 0.0

        # data parameters
        self.drop_last = True
        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.5
        self.jitter_ratio = 2
        self.max_seg = 12


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.timesteps = 49  # k^2
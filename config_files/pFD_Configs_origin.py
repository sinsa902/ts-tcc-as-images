
class Config(object):
    def __init__(self):
        # model configs
        self.dataset="pFD"
        self.input_channels = 3 #RGB
        self.num_classes = 3
        self.dropout = 0.35
        # for noisy labels experiment
        self.corruption_prob = 0.3
        self.embed_dim=768 #psize^2*channel
        self.img_size=128 #자주 바뀜
        self.patch_size=16
        self.features_len = 64  #(img_size/patch_size)^2
        self.num_heads=12
        self.depth=12
        self.mlp_ratio=2.
        self.qkv_bias=False
        self.drop_rate=0.
        self.attn_drop_rate=0.

        # training configs
        self.num_epoch = 40
        self.batch_size = 32

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 2
        self.jitter_ratio = 0.1
        self.max_seg = 5


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True

class TC(object):
    def __init__(self):
        self.timesteps = 9  #k^2

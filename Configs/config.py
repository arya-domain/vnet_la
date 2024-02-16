import os
from easydict import EasyDict

C = EasyDict()
config = C
cfg = C

C.seed = 1337
C.repo_name = "medical_semi_seg"


""" Experiments setting """
C.augmentation = True
C.dataset = "LA"
C.code_path = os.path.realpath("")  # + "/Code/VnetLA/"
C.data_path = os.path.realpath("Datasets/Left_Atrium/data/".format("."))

""" Training setting """
# trainer
C.ddp_training = False
C.batch_size = 4
C.num_workers = 1
C.shuffle = True
C.drop_last = False
C.learning_rate = 5e-2
C.threshold = 0.65
C.spatial_weight = 0.3
C.hyp = 0.1

# rampup settings (per epoch)
C.rampup_type = "sigmoid"
C.rampup_length = 40
C.rampup_start = 0

""" Model setting """
C.drop_out = True  # bayesian
C.num_classes = 2
C.momentum = 0.9
C.weight_decay = 1e-4
C.ema_momentum = 0.99

# """ Wandb setting """
os.environ["WANDB_API_KEY"] = "7fb4766980dd9063b4834ff7fac76a6849ca1aa1"
C.use_wandb = False
C.project_name = "VNET"

""" Others """
C.save_ckpt = True


C.last_val_epochs = 10

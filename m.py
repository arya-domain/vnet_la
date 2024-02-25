import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from Utils.tensor_board import Tensorboard
import os
import torch
import numpy
import random
import argparse
import datetime
from train import Trainer
from Utils.logger import *

from torch.utils.data import DataLoader
from Configs.config import config
from Dataloader.dataset import LAHeartDataset
from Dataloader.dataloader import TwoStreamBatchSampler


def main(args):
    # update the default config with the args
    config.update(vars(args))

    def worker_init_fn(worker_id):
        random.seed(config.seed + worker_id)

    current_time = datetime.datetime.now()

    train_set = LAHeartDataset(
        os.path.join(config.code_path, "Datasets/Left_Atrium/labels"),
        config.data_path,
        split="train",
        config=config,
    )

    # merge both labelled & unlabelled sampler to same batch
    batch_sampler = TwoStreamBatchSampler(
        list(range(config.labeled_num)),
        list(range(config.labeled_num, 80)),
        config.batch_size,
        int(config.batch_size / 2),
    )

    train_loader = DataLoader(
        train_set,
        batch_sampler=batch_sampler,
        num_workers=config.num_workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
    )

    val_dataset = LAHeartDataset(
        os.path.join(config.code_path, "Datasets/Left_Atrium/labels"),
        config.data_path,
        split="eval",
        num=None,
        config=config,
    )

    config.iter_per_epoch = len(train_loader)
    config.n_epochs = config.max_iterations // len(train_loader) + 1
    config.unlabeled_num = len(train_set) - config.labeled_num

    logger = logging.getLogger("VNET_LA")
    logger.propagate = False
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s - %(levelname)s - %(message)s",
    #     filename=f"scores_output/app_{current_time}.log",
    # )
    file_handler = logging.FileHandler(f"scores_output/app_{current_time}.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    logger.info(
        "training with {} epochs [{} iters]".format(
            config.n_epochs, config.iter_per_epoch * config.n_epochs
        )
    )
    logger.warning(
        "running time: " + datetime.datetime.now().strftime(" [%H:%M] %d/%m/%y")
    )
    logger.warning(
        "supervised sample: {}, unsupervised sample: {}".format(
            config.labeled_num, config.unlabeled_num
        )
    )
    logger.critical(
        "architecture: {}, backbone: {}".format(
            args.architecture, "nothing" if args.backbone is None else args.backbone
        )
    )
    tensorboard = Tensorboard(config=config)
    trainer = Trainer(
        config,
        train_loader=train_loader,
        valid_set=val_dataset,
        logger=logger,
        my_wandb=tensorboard,
    )
    trainer.run()
    return


if __name__ == "__main__":

    class CmdLineVar:
        pass

    cmd_line_var = CmdLineVar()
    cmd_line_var.architecture = "vnet"
    cmd_line_var.backbone = None
    cmd_line_var.unsup_weight = 0.3
    cmd_line_var.labeled_num = 8
    cmd_line_var.max_iterations = 15000

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    random.seed(config.seed)
    numpy.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    main(cmd_line_var)

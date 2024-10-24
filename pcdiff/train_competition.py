import click
from os.path import join, dirname, abspath
import subprocess
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import yaml
import numpy as np
import torch
import ipdb

import pcdiff.models.models as models
from torch.utils.data import DataLoader
# from ipb_loaders.pointcloud.igg_fruit import IGGFruit
# from ipb_loaders.rgbd.igg_fruit import IGGFruit
from pcdiff.utils.collations import point_set_to_sparse
from pytorch_lightning import LightningDataModule


from torch.utils.data import DataLoader
from competition_tools.dataloader import ShapeCompletionDataset

def trainer(self):
    shape_completion_dataset = ShapeCompletionDataset(data_source='/export/data/s7jeyapp/shape_completion_challenge')
    dataloader = DataLoader(shape_completion_dataset)
    ipdb.set_trace()
    return dataloader

# for item in dataloader:
#     gt = item['groundtruth_pcd']
#     sem_labels = item[0]
    
@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
@click.option('--test', '-t', is_flag=True, help='test mode')


def main(config, weights, checkpoint, test):
    
    cfg = yaml.safe_load(open(config))
    cfg['git_commit_version'] = str(subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).strip())


    data = ShapeCompletionDataset('/export/data/s7jeyapp/shape_completion_challenge')


    if weights is None:
        model = models.DiffusionPoints(cfg)
    else:
        model = models.DiffusionPoints.load_from_checkpoint(weights,hparams=cfg)

    #Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(filename=cfg['experiment']['id']+'_{epoch:02d}',
                                 save_top_k=-1)

    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id'],
                                             default_hp_metric=False)

    #Setup trainer
    trainer = Trainer(gpus=cfg['train']['n_gpus'],
                      logger=tb_logger,
                      log_every_n_steps=100,
                      resume_from_checkpoint=checkpoint,
                      max_epochs= cfg['train']['max_epoch'],
                      callbacks=[lr_monitor, checkpoint_saver],
                      check_val_every_n_epoch=10,
                      num_sanity_val_steps=1,
                      limit_val_batches=0.04, #0.02
                      #gradient_clip_val=0.5,
                      )

 # Train!
    if test:
        print('TESTING MODE')
        trainer.test(model, data)
    else:
        print('TRAINING MODE')
        # ipdb.set_trace()
        trainer.fit(model, data)

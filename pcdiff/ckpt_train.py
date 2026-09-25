import click
from os.path import join, dirname, abspath
import subprocess
from pytorch_lightning import Trainer
from pytorch_lightning import LightningDataModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import yaml
import numpy as np
import torch
import ipdb
import open3d as o3d 
# import pcdiff.datasets.datasets as datasets
import pcdiff.models.models as models
from torch.utils.data import DataLoader
from pcdiff.competition_tools.dataloader import ShapeCompletionDataset
# from pcdiff.dataloader import ShapeCompletionDataset


# from ipb_loaders.pointcloud.igg_fruit import IGGFruit
# from ipb_loaders.rgbd.igg_fruit import IGGFruit
from pcdiff.utils.collations import point_set_to_sparse

class FruitCollation:
    def __init__(self):
        return

    def __call__(self, data):
        # print('entering fit')
        # "transpose" the  batch(pt, ptn) to batch(pt), batch(ptn)
        pcd_full = []
        pcd_part = []

        mean = []
        std = []
        fruit_id = []
        mask=[]
        
        # ipdb.set_trace() 
        for sample in data:
            # ipdb.set_trace()
            # gt_np = np.asarray(sample['groundtruth_pcd'].points) #use this when using validation set
            gt_np = sample['groundtruth_pcd']
            # part_np = np.asarray(sample['rgbd_pcd'].points)
            part_np= sample['rgbd_pcd']
            fid = sample['fruit_id']
            aligned_data= sample['aligned_data']
            aligned_data_length = sample['aligned_data_length']

            aligned_xyz = aligned_data[:,:3]
            aligned_rgb = aligned_data[:,3:]

            # sparse = point_set_to_sparse(gt_np, part_np, 30000, 15000, 0.01)
            sparse = point_set_to_sparse(aligned_data, part_np, 30000, 15000, 0.01)


            # Store the sparse components
            
            pcd_full.append(sparse[0])
            mean.append(sparse[1])
            std.append(sparse[2])
            pcd_part.append(sparse[3])
            mask.append(sparse[-1])

        
        #return the items we would need in the training, etc

        return {
            'pcd_full': torch.stack(pcd_full).float(),
            'pcd_part': torch.stack(pcd_part).float(),
            'mean': torch.stack(mean).float(),
            'std': torch.stack(std).float(),
            'fruit_id': fid,
            'aligned_data': aligned_data,
            'aligned_data_length': aligned_data_length,
            'aligned_xyz':aligned_xyz,
            'aligned_rgb' : aligned_rgb,
            'mask': mask

        }


class FruitDataModule(LightningDataModule):
    def __init__(self, data_source):
        self.data_source = data_source

    def train_dataloader(self):
        '''
        dataset = IGGFruit(data_source=self.data_source)#, precomputed_augmentation='/mnt/igg_fruit/processed/SweetPepper3')
        loader = DataLoader(dataset, batch_size=8, num_workers=0, collate_fn=FruitCollation())
        print('Train samples: ', len(loader))
        '''
        dataset = ShapeCompletionDataset(data_source=self.data_source, split='train')
        loader = DataLoader(dataset, batch_size=8, num_workers=0, collate_fn=FruitCollation())

        
        return loader

    def val_dataloader(self):
        '''
        dataset = IGGFruit(data_source=self.data_source)#split='val'
        loader = DataLoader(dataset, batch_size=8, num_workers=0, collate_fn=FruitCollation())
        print('Val samples: ', len(loader))
        '''
        dataset = ShapeCompletionDataset(data_source=self.data_source, split='val')#split='val'
        loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=FruitCollation())

        return loader

    def test_dataloader(self):
        '''
        dataset = IGGFruit(data_source=self.data_source) # split='train'
        loader = DataLoader(dataset, batch_size=8, num_workers=0, collate_fn=FruitCollation())
        print('Test samples: ', len(loader))    
        '''
        dataset = ShapeCompletionDataset(data_source=self.data_source,split ='test') 
        loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=FruitCollation())


        return loader

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

    #Load data and model
    #data = datasets.dataloaders[cfg['data']['dataloader']](cfg)   


    data = FruitDataModule(cfg['data']['fruit_dir'])

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
                      check_val_every_n_epoch=1,
                      num_sanity_val_steps= 1, #skips checking the val before training
                      limit_val_batches=  1 #0.02 #1 
                      #gradient_clip_val=0.5,
                      )

 # Train!
    # ipdb.set_trace()
    if test:
        print('TESTING MODE')
      
        # trainer.test(model, data)
        trainer.test(model, data.test_dataloader()) #use to swtich between validation or test data
        # trainer.test(model, data.val_dataloader())
    else:
        print('TRAINING MODE')
        
        trainer.fit(model, data)

if __name__ == "__main__":
    main()
    

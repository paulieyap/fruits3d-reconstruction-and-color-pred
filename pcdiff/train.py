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
import open3d as o3d 
# import pcdiff.datasets.datasets as datasets
import pcdiff.models.models as models
from torch.utils.data import DataLoader
# from pcdiff.compute import ComputeMeanStd
# from pcdiff.competition_tools.dataloader import ShapeCompletionDataset
from pcdiff.dataloader import ShapeCompletionDataset


# from ipb_loaders.pointcloud.igg_fruit import IGGFruit
# from ipb_loaders.rgbd.igg_fruit import IGGFruit
from pcdiff.utils.collations import point_set_to_sparse

class FruitCollation:
    def __init__(self, norm=(None, None), recentre=None):
        # (mean, std) of XYZRGB for point_set_to_sparse; (None, None) = thesis statistics
        self.norm_mean, self.norm_std = norm
        # 'partial': shift each sample so its partial observation's centroid is the origin (no fruit-centred
        # frame at deployment); None keeps the dataset frame
        self.recentre = recentre

    def __call__(self, data):
        # print('entering fit')
        # "transpose" the  batch(pt, ptn) to batch(pt), batch(ptn)
        pcd_full = []
        pcd_part = []

        mean = []
        std = []
        fruit_id = []
        mask=[]
        
        for sample in data:
            # part_np = np.asarray(sample['rgbd_pcd'].points)
            # gt_np = np.asarray(sample['groundtruth_pcd'].points) #use this when using validation set
            gt_np = sample['groundtruth_pcd']
            part_np= np.asarray(sample['rgbd_pcd'])
            fid = sample['fruit_id']
            aligned_data= sample['aligned_data']
            aligned_data_length = sample['aligned_data_length']

            if self.recentre == 'partial':
                offset = part_np[:, :3].mean(0)
                part_np = part_np.copy()
                part_np[:, :3] -= offset
                aligned_data = aligned_data.copy()
                aligned_data[:, :3] -= offset

            aligned_xyz = aligned_data[:,:3]
            aligned_rgb = aligned_data[:,3:]
            # sparse = point_set_to_sparse(gt_np, part_np, 30000, 15000, 0.01)
            sparse = point_set_to_sparse(aligned_data, part_np, 30000, 15000, 0.01, self.norm_mean, self.norm_std)
            

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
    def __init__(self, data_source, batch_size=8, num_workers=0, cache_dir=None, norm=(None, None), recentre=None):
        super().__init__()
        self.norm = norm
        self.recentre = recentre
        self.data_source = data_source
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        '''
        dataset = IGGFruit(data_source=self.data_source)#, precomputed_augmentation='/mnt/igg_fruit/processed/SweetPepper3')
        loader = DataLoader(dataset, batch_size=8, num_workers=0, collate_fn=FruitCollation(self.norm, self.recentre))
        print('Train samples: ', len(loader))
        '''
        dataset = ShapeCompletionDataset(data_source=self.data_source, split='train', cache_dir=self.cache_dir)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                            collate_fn=FruitCollation(self.norm, self.recentre), persistent_workers=self.num_workers > 0)

        
        return loader

    def val_dataloader(self):
        '''
        dataset = IGGFruit(data_source=self.data_source)#split='val'
        loader = DataLoader(dataset, batch_size=8, num_workers=0, collate_fn=FruitCollation(self.norm, self.recentre))
        print('Val samples: ', len(loader))
        '''

        dataset = ShapeCompletionDataset(data_source=self.data_source, split='val', cache_dir=self.cache_dir)#split='val'
        loader = DataLoader(dataset, batch_size=1, num_workers=self.num_workers, collate_fn=FruitCollation(self.norm, self.recentre))

        return loader

    def test_dataloader(self):
        '''
        dataset = IGGFruit(data_source=self.data_source) # split='train'
        loader = DataLoader(dataset, batch_size=8, num_workers=0, collate_fn=FruitCollation(self.norm, self.recentre))
        print('Test samples: ', len(loader))    
        '''
        dataset = ShapeCompletionDataset(data_source=self.data_source, split='test', cache_dir=self.cache_dir) 
        loader = DataLoader(dataset, batch_size=1, num_workers=self.num_workers, collate_fn=FruitCollation(self.norm, self.recentre))


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


    data = FruitDataModule(cfg['data']['fruit_dir'], cfg['train']['batch_size'], cfg['train']['num_workers'],
                           cfg['data'].get('cache_dir'), (cfg['data'].get('norm_mean'), cfg['data'].get('norm_std')),
                           cfg['data'].get('recentre'))

    if weights is None:
        model = models.DiffusionPoints(cfg)
    else:
        model = models.DiffusionPoints.load_from_checkpoint(weights,hparams=cfg)

    #Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(filename=cfg['experiment']['id']+'_{epoch:02d}',
                                 save_top_k=-1, save_last=True,
                                 every_n_epochs=cfg['train'].get('ckpt_every_n_epochs', 1))

    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id'],
                                             default_hp_metric=False)

    #Setup trainer
    trainer = Trainer(accelerator='gpu',
                      devices=cfg['train']['n_gpus'],
                      logger=tb_logger,
                      log_every_n_steps=100,
                      max_epochs= cfg['train']['max_epoch'],
                      callbacks=[lr_monitor, checkpoint_saver] + ([ModelCheckpoint(
                          filename=cfg['experiment']['id'] + '_best_{epoch:02d}', monitor='val/cd_mm', mode='min',
                          save_top_k=3)] if cfg['train'].get('val_batches', 1) > 1 else []),
                      check_val_every_n_epoch=cfg['train'].get('val_every_n_epochs', 1),
                      num_sanity_val_steps= 1, #skips checking the val before training
                      limit_val_batches=cfg['train'].get('val_batches', 1) # 8 fruits per batch
                      , limit_train_batches=cfg['train'].get('limit_train_batches', 1.0) # int = batches per epoch (large datasets)
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
        
        trainer.fit(model, datamodule=data, ckpt_path=checkpoint)

if __name__ == "__main__":
    main()
    

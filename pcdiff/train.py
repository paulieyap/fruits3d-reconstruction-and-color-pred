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
# import pcdiff.datasets.datasets as datasets
import pcdiff.models.models as models
from torch.utils.data import DataLoader
from pcdiff.competition_tools.dataloader import ShapeCompletionDataset
# from ipb_loaders.pointcloud.igg_fruit import IGGFruit
# from ipb_loaders.rgbd.igg_fruit import IGGFruit
from pcdiff.utils.collations import point_set_to_sparse, rgbd_set_to_sparse

class FruitCollation:
    def __init__(self):
        return

    def __call__(self, data):
        # print('entering fit')
        # "transpose" the  batch(pt, ptn) to batch(pt), batch(ptn)
        pcd_full = []
        pcd_part = []
        pcd_colored = []
        mean = []
        std = []
        fruit_id = []
        

        for sample in data:
            '''
            if len(sample) == 1:
                # Testing Phase: Use only RGBD point cloud data (no ground truth)
                rgbd_np = np.asarray(sample['rgbd_pcd'].points)

                # Perform sparse conversion with only RGBD data
                sparse = rgbd_set_to_sparse(rgbd_np, 30000, 15000, 0.01)

                # Store the sparse components
                pcd_part.append(sparse[-1])
                mean.append(sparse[1])
                std.append(sparse[2])

                return {
                    'mean': torch.stack(mean).float(),
                    'std': torch.stack(std).float(),
                    'pcd_part': torch.stack(pcd_part).float(),
                }

            else:
            '''
                # Training Phase: Use ground truth and RGBD data
            # bbox = sample['groundtruth_pcd'].get_axis_aligned_bounding_box()
            # cropped = sample['rgbd_pcd'].crop(bbox)

            # # Check if the cropped point cloud contains any points
            # if len(cropped.points) == 0:
            #     # If no points are within the bounding box, retain the original RGBD point cloud
            #     part_np = np.asarray(sample['rgbd_pcd'].points)
            # else:
            #     # Otherwise, use the cropped point cloud
            #     part_np = np.asarray(cropped.points)

            gt_np = np.asarray(sample['groundtruth_pcd'].points)
            part_np = np.asarray(sample['rgbd_pcd'].points)
            fid = sample['fruit_id']
            print("part_np shape: ", part_np.shape)


            # Sparse representation using ground truth and cropped point cloud
            sparse = point_set_to_sparse(gt_np, part_np, 30000, 15000, 0.01)

            # Store the sparse components
            pcd_part.append(sparse[-1])
            pcd_full.append(sparse[0])
            mean.append(sparse[1])
            std.append(sparse[2])
            
            return {
                'pcd_full': torch.stack(pcd_full).float(),
                'mean': torch.stack(mean).float(),
                'std': torch.stack(std).float(),
                'pcd_part': torch.stack(pcd_part).float(),
                'fruit_id': fid
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
        loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=FruitCollation())
        print('Train samples: ', len(loader))
        
        return loader

    def val_dataloader(self):
        '''
        dataset = IGGFruit(data_source=self.data_source)#split='val'
        loader = DataLoader(dataset, batch_size=8, num_workers=0, collate_fn=FruitCollation())
        print('Val samples: ', len(loader))
        '''
        
        dataset = ShapeCompletionDataset(data_source=self.data_source, split='val')#split='val'
        loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=FruitCollation())
        print('Val samples: ', len(loader))
      
        return loader

    def test_dataloader(self):
        '''
        dataset = IGGFruit(data_source=self.data_source) # split='train'
        loader = DataLoader(dataset, batch_size=8, num_workers=0, collate_fn=FruitCollation())
        print('Test samples: ', len(loader))    
        '''
        # ipdb.set_trace()
        dataset = ShapeCompletionDataset(data_source=self.data_source, split ='test') 

        print("Dataloader: ", len(dataset))
        loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=FruitCollation())


        print('Test samples: ', len(loader))  
        # ipdb.set_trace()
      

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


    data = FruitDataModule('/export/data/s7jeyapp/shape_completion_challenge')
    # data = FruitDataModule('/export/data/s7jeyapp/IGG/igg_fruit/SweetPepper')
    
    # data = FruitDataModule('/mnt/igg_fruit/processed/SweetPepper')
    # data = FruitDataModule('/export/data/s7jeyapp/igg_fruit/SweetPepper/p1')
    # data = FruitDataModule('/cache/s7jeyapp/igg_fruit/SweetPepper')

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
                      num_sanity_val_steps= 0, #skips checking the val before training
                      limit_val_batches=  1 #0.02 #1 
                      #gradient_clip_val=0.5,
                      )

 # Train!
    # ipdb.set_trace()
    if test:
        print('TESTING MODE')
      
        # trainer.test(model, data)
        trainer.test(model, data.test_dataloader())
    else:
        print('TRAINING MODE')
        # trainer.fit(model, data) #for val
        trainer.fit(model, data)

if __name__ == "__main__":
    main()
    

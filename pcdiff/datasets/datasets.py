import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from pcdiff.datasets.dataloader.SemanticKITTITemporal import TemporalKITTISet
# from pcdiff.datasets.dataloader.CarlaSCTemporal import TemporalCarlaSet
from pcdiff.utils.collations import SparseSegmentCollation
import warnings

warnings.filterwarnings('ignore')

__all__ = ['TemporalKittiDataModule', 'TemporalCarlaDataModule']

class TemporalKittiDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = SparseSegmentCollation()

        data_set = TemporalKITTISet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            scan_window=self.cfg['data']['scan_window'],
            split=self.cfg['data']['split'],
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loaderDataLoader

    def val_dataloader(self, pre_training=True):
        collate = SparseSegmentCollation()

        data_set = TemporalKITTISet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            scan_window=self.cfg['data']['scan_window'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'])
        loader = DataLoader(data_set, batch_size=2,#self.cfg['train']['batch_size'],
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    # def test_dataloader(self):
    #     data_set = KITTISet(
    #         data_dir=self.cfg['data']['data_dir'],
    #         seqs=self.cfg['data']['test'],
    #         scan_window=self.cfg['train']['scan_window'],
    #         split=self.cfg['data']['split'],
    #         pre_training=self.cfg['train']['pre_train'],
    #         resolution=self.cfg['data']['resolution'],
    #         percentage=self.cfg['data']['percentage'],
    #         intensity_channel=self.cfg['data']['intensity'])
    #     loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
    #                         num_workers=self.cfg['train']['num_workers'])
    #     return loader

class TemporalCarlaDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = SparseSegmentCollation()

        data_set = TemporalCarlaSet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            scan_window=self.cfg['data']['scan_window'],
            split=self.cfg['data']['split'],
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = SparseSegmentCollation()

        data_set = TemporalCarlaSet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            scan_window=self.cfg['data']['scan_window'],
            split=self.cfg['data']['split'],
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    # def test_dataloader(self):
    #     data_set = KITTISet(
    #         data_dir=self.cfg['data']['data_dir'],
    #         seqs=self.cfg['data']['test'],
    #         scan_window=self.cfg['train']['scan_window'],
    #         split=self.cfg['data']['split'],
    #         pre_training=self.cfg['train']['pre_train'],
    #         resolution=self.cfg['data']['resolution'],
    #         percentage=self.cfg['data']['percentage'],
    #         intensity_channel=self.cfg['data']['intensity'])
    #     loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
    #                         num_workers=self.cfg['train']['num_workers'])
    #     return loader

dataloaders = {
    'KITTI': TemporalKittiDataModule,
    'Carla': TemporalCarlaDataModule,
}


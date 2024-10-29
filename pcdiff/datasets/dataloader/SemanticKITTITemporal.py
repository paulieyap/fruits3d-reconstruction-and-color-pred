import torch
from torch.utils.data import Dataset
from pcdiff.utils.pcd_preprocess import clusterize_pcd, visualize_pcd_clusters, point_set_to_coord_feats, overlap_clusters, aggregate_pcds
from pcdiff.utils.pcd_transforms import *
from pcdiff.utils.data_map import learning_map
from pcdiff.utils.collations import point_set_to_sparse
import os
import numpy as np

import warnings

warnings.filterwarnings('ignore')

#################################################
################## Data loader ##################
#################################################

class TemporalKITTISet(Dataset):
    def __init__(self, data_dir, scan_window, seqs, split, resolution, num_points):
        super().__init__()
        self.data_dir = data_dir
        self.augmented_dir = 'segments_views'

        self.n_clusters = 50
        self.resolution = resolution
        self.scan_window = scan_window
        self.num_points = num_points
        self.seg_batch = True

        self.split = split
        self.seqs = seqs

        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_list()

        self.nr_data = len(self.points_datapath)

        print('The size of %s data is %d'%(self.split,len(self.points_datapath)))

    def datapath_list(self):
        self.points_datapath = []

        for seq in self.seqs:
            point_seq_path = os.path.join(self.data_dir, 'dataset', 'sequences', seq, 'velodyne')
            point_seq_bin = os.listdir(point_seq_path)
            point_seq_bin.sort()
            
            for file_num in range(0, len(point_seq_bin), self.scan_window):
                # we guarantee that the end of sequence will not generate single scans as aggregated pcds
                end_file = file_num + self.scan_window if len(point_seq_bin) - file_num > 1.5 * self.scan_window else len(point_seq_bin)
                self.points_datapath.append([os.path.join(point_seq_path, point_file) for point_file in point_seq_bin[file_num:end_file] ])
                if end_file == len(point_seq_bin):
                    break

        #self.points_datapath = self.points_datapath[:200]

    def transforms(self, points):
        points = np.expand_dims(points, axis=0)
        points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
        points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])
        points[:,:,:3] = jitter_point_cloud(points[:,:,:3])
        points = random_drop_n_cuboids(points)

        return np.squeeze(points, axis=0)

    def __getitem__(self, index):
        #index = 500
        seq_num = self.points_datapath[index][0].split('/')[-3]
        fname = self.points_datapath[index][0].split('/')[-1].split('.')[0]

        t_frame = np.random.randint(len(self.points_datapath[index]))
        p_full, p_part = aggregate_pcds(self.points_datapath[index], self.data_dir, t_frame)

        # calculate distance to the origin
        dist_full = np.power(p_full, 2)
        dist_full = np.sqrt(dist_full.sum(-1))

        dist_part = np.power(p_part, 2)
        dist_part = np.sqrt(dist_part.sum(-1))

        # just get a small pcd region
        n_part = int(self.num_points / self.scan_window)
        n_full = (self.scan_window - 1) * n_part

        return point_set_to_sparse(
            p_full[dist_full < 10.],
            p_part[dist_part < 10.],
            n_full,
            n_part,
            self.resolution,
        )

    def __len__(self):
        #print('DATA SIZE: ', np.floor(self.nr_data / self.sampling_window), self.nr_data % self.sampling_window)
        return self.nr_data

##################################################################################################

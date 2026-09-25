"""Drop-in replacement for ``ipb_loaders.pointcloud.igg_fruit.IGGFruit``.

The original loader lived in the IPB-lab-internal ``ipb_loaders`` package,
which is not publicly available. This re-implementation reads the public
shape completion challenge layout:

    <data_source>/<split>/<fruit_id>/
        gt/pcd/fruit.ply            (not present for the test split)
        input/intrinsic.json
        input/color/<frame>.png
        input/depth/<frame>.npy
        input/masks/<frame>.png
        input/poses/<frame>.txt

Each sample is a dict compatible with the pipeline's ``FruitCollation``:

    {
        'points': (N, 6) float array, partial XYZRGB observation,
        'extra': {
            'gt_points': (M, 6) float array, full XYZ with RGB transferred
                         from the partial observation (-1 where no color),
            'fruit_id': str,
        },
    }
"""
import json
import os

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from torch.utils.data import Dataset


class IGGFruit(Dataset):

    def __init__(self,
                 data_source,
                 split='train',
                 precomputed_augmentation=None,
                 fuse_frames=True,
                 color_dist_thresh=0.01,
                 remove_outliers=True):
        """
        Args:
            data_source: dataset root containing the split folders.
            split: 'train', 'val' or 'test'.
            precomputed_augmentation: optional second root with the same
                layout (e.g. augmented copies); its fruits are added to the
                training split.
            fuse_frames: if True one sample per fruit (all frames fused),
                otherwise one sample per frame.
            color_dist_thresh: max distance (m) for transferring RGB from the
                partial observation onto ground-truth points.
            remove_outliers: apply statistical outlier removal to the partial
                observation.
        """
        self.split = split
        self.fuse_frames = fuse_frames
        self.color_dist_thresh = color_dist_thresh
        self.remove_outliers = remove_outliers

        roots = [data_source]
        if precomputed_augmentation is not None and split == 'train':
            roots.append(precomputed_augmentation)

        self.samples = []
        for root in roots:
            self.samples += self.get_file_paths(os.path.join(root, split))

        if not self.samples:
            raise RuntimeError(f'No fruits found for split "{split}" in {roots}')

    def get_file_paths(self, split_root):
        samples = []
        for fruit_id in sorted(os.listdir(split_root)):
            fruit_root = os.path.join(split_root, fruit_id)
            frames = sorted(os.listdir(os.path.join(fruit_root, 'input/masks')))
            if self.fuse_frames:
                samples.append((fruit_root, fruit_id, frames))
            else:
                samples += [(fruit_root, fruit_id, [frame]) for frame in frames]
        return samples

    @staticmethod
    def load_K(path):
        with open(path, 'r') as f:
            data = json.load(f)['intrinsic_matrix']
        return np.reshape(data, (3, 3), order='F')

    @staticmethod
    def rgbd_to_pcd(rgb, depth, mask, pose, K):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image((depth * (mask > 0)).astype(np.float32)),
            depth_scale=1,
            depth_trunc=1.0,
            convert_rgb_to_intensity=False)

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(height=rgb.shape[0], width=rgb.shape[1],
                                 fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])

        return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, np.linalg.inv(pose))

    def load_partial(self, fruit_root, frames):
        K = self.load_K(os.path.join(fruit_root, 'input/intrinsic.json'))
        pcd = o3d.geometry.PointCloud()

        for frame in frames:
            name = os.path.splitext(frame)[0]
            rgb = cv2.cvtColor(cv2.imread(os.path.join(fruit_root, 'input/color', frame)), cv2.COLOR_BGR2RGB)
            depth = np.load(os.path.join(fruit_root, 'input/depth', name + '.npy'))
            mask = cv2.imread(os.path.join(fruit_root, 'input/masks', frame), cv2.IMREAD_GRAYSCALE)
            pose = np.loadtxt(os.path.join(fruit_root, 'input/poses', name + '.txt'))
            pcd += self.rgbd_to_pcd(rgb, depth, mask, pose, K)

        if self.remove_outliers and len(pcd.points) > 200:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=2.0)

        return np.concatenate((np.asarray(pcd.points), np.asarray(pcd.colors)), axis=-1)

    def transfer_colors(self, gt_xyz, partial):
        """Give each GT point the color of its nearest partial point, -1 if too far."""
        dist, idx = cKDTree(partial[:, :3]).query(gt_xyz, k=1)
        colors = np.full((gt_xyz.shape[0], 3), -1.0)
        close = dist < self.color_dist_thresh
        colors[close] = partial[idx[close], 3:]
        return np.concatenate((gt_xyz, colors), axis=-1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fruit_root, fruit_id, frames = self.samples[idx]
        partial = self.load_partial(fruit_root, frames)

        gt_path = os.path.join(fruit_root, 'gt/pcd/fruit.ply')
        if os.path.isfile(gt_path):
            gt_xyz = np.asarray(o3d.io.read_point_cloud(gt_path).points)
            gt_points = self.transfer_colors(gt_xyz, partial)
        else:
            # test split has no ground truth: fall back to the partial observation
            gt_points = partial.copy()

        return {
            'points': partial,
            'extra': {
                'gt_points': gt_points,
                'fruit_id': fruit_id,
            },
        }

import os

import click
import numpy as np
import open3d as o3d
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader

from pcdiff.datasets.igg_fruit import IGGFruit
from pcdiff.models.models import DiffusionPoints
from pcdiff.utils.collations import point_set_to_sparse, unormalize_pcd


class FruitCollation:
    """Same batch format as train.py, built from IGGFruit samples."""
    def __init__(self, n_full=30000, n_part=15000):
        self.n_full = n_full
        self.n_part = n_part

    def __call__(self, data):
        pcd_full = []
        pcd_part = []
        mean = []
        std = []

        for sample in data:
            sparse = point_set_to_sparse(sample['extra']['gt_points'], sample['points'], self.n_full, self.n_part, 0.01)
            pcd_full.append(sparse[0])
            mean.append(sparse[1])
            std.append(sparse[2])
            pcd_part.append(sparse[3])

        return {
            'pcd_full': torch.stack(pcd_full).float(),
            'pcd_part': torch.stack(pcd_part).float(),
            'mean': torch.stack(mean).float(),
            'std': torch.stack(std).float(),
            'fruit_id': [sample['extra']['fruit_id'] for sample in data],
        }


class DiffCompletion:
    """Loads a trained DiffusionPoints (XYZRGB) checkpoint and completes partial fruits."""
    def __init__(self, diff_path, denoising_steps, cond_weight):
        hparams = yaml.safe_load(open(os.path.join(diff_path.split('checkpoints')[0], 'hparams.yaml')))
        assert denoising_steps <= hparams['diff']['t_steps'], \
            f"The number of denoising steps cannot be bigger than T={hparams['diff']['t_steps']} (you've set '-T {denoising_steps}')"

        hparams['diff']['s_steps'] = denoising_steps
        hparams['train']['uncond_w'] = cond_weight

        self.model = DiffusionPoints.load_from_checkpoint(diff_path, hparams=hparams).cuda()
        self.model.eval()
        self.hparams = self.model.hparams

    @torch.no_grad()
    def complete(self, batch):
        pcd_part = batch['pcd_part'].cuda()
        mean = batch['mean'].cuda()
        std = batch['std'].cuda()
        n_full = batch['pcd_full'].shape[1]

        x_feats = torch.randn([pcd_part.shape[0], n_full, pcd_part.shape[2]], device=self.model.device)
        x_full = self.model.points_to_tensor(x_feats, mean, std)
        x_cond = self.model.points_to_tensor(pcd_part, mean, std)
        x_uncond = self.model.points_to_tensor(torch.zeros_like(pcd_part), torch.zeros_like(mean), torch.zeros_like(std))

        # p_sample_loop only uses the first dimension of gt_pts (batch size)
        x_gen = self.model.p_sample_loop(x_full, x_cond, x_uncond, np.empty((pcd_part.shape[0], 0)), mean, std, 0)
        x_gen = x_gen.F.reshape((pcd_part.shape[0], -1, 6))

        return unormalize_pcd(x_gen, mean, std).cpu().numpy()


def to_o3d(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.clip(points[:, 3:], 0., 1.))
    return pcd


@click.command()
@click.option('--diff', '-d', type=str, required=True, help='path to the diffusion checkpoint (.ckpt)')
@click.option('--denoising_steps', '-T', type=int, default=50, help='number of denoising steps (default: 50)')
@click.option('--cond_weight', '-s', type=float, default=6.0, help='conditioning weight (default: 6.0)')
@click.option('--data', type=str, default='./data/shape_completion_challenge', help='dataset root with train/ and test/ splits')
@click.option('--split', type=str, default='test', help='split to complete (default: test)')
@click.option('--vis', is_flag=True, help='show each completed fruit')
def main(diff, denoising_steps, cond_weight, data, split, vis):
    exp_dir = diff.split('/')[-1].split('.')[0].replace('=', '') + f'_T{denoising_steps}_s{cond_weight}'
    out_dir = f'./results/{exp_dir}/diff'
    os.makedirs(out_dir, exist_ok=True)

    diff_completion = DiffCompletion(diff, denoising_steps, cond_weight)
    with open(f'./results/{exp_dir}/exp_config.yaml', 'w+') as exp_config:
        yaml.dump(dict(diff_completion.hparams), exp_config)

    loader = DataLoader(IGGFruit(data_source=data, split=split), batch_size=1, num_workers=4, collate_fn=FruitCollation())

    for batch in tqdm.tqdm(loader):
        completed = diff_completion.complete(batch)

        for fruit_id, points in zip(batch['fruit_id'], completed):
            pcd_diff = to_o3d(points)
            o3d.io.write_point_cloud(os.path.join(out_dir, f'{fruit_id}.ply'), pcd_diff)
            if vis:
                o3d.visualization.draw_geometries([pcd_diff])


if __name__ == '__main__':
    main()

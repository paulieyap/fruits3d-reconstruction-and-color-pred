import os
import time

import click
import numpy as np
import open3d as o3d
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader, Dataset

from pcdiff.datasets.igg_fruit import IGGFruit
from pcdiff.models.models import DiffusionPoints
from pcdiff.utils.collations import point_set_to_sparse, unormalize_pcd


class CachedFruit(Dataset):
    """IGGFruit-style samples from the training cache (tools/build_cache.py), so inference sees the
    same fused/ICP-refined partial observation the model was trained on."""
    def __init__(self, cache_dir, split):
        root = os.path.join(cache_dir, split)
        self.paths = sorted(os.path.join(root, f) for f in os.listdir(root) if f.endswith('.npz') and '.tmp' not in f)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        c = np.load(self.paths[idx])
        return {
            'points': c['rgbd_pcd'].astype(np.float64),
            'extra': {
                'gt_points': c['aligned_data'].astype(np.float64),
                'fruit_id': os.path.basename(self.paths[idx])[:-4],
            },
        }


class Recentred(Dataset):
    """Shifts every sample so the partial observation's centroid is at the origin, as on a robot where
    no fruit-centred frame exists (the challenge data is centred on each fruit's ground truth, which a
    model can learn). The offset is kept in extra['offset'] so completions can be shifted back."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        offset = sample['points'][:, :3].mean(0)
        sample['points'] = sample['points'].copy()
        sample['points'][:, :3] -= offset
        gt = sample['extra']['gt_points'].copy()
        gt[:, :3] -= offset
        sample['extra'] = dict(sample['extra'], gt_points=gt, offset=offset)
        return sample


class FruitCollation:
    """Same batch format as train.py, built from IGGFruit samples."""
    def __init__(self, n_full=30000, n_part=15000, norm=(None, None)):
        self.n_full = n_full
        self.n_part = n_part
        self.norm_mean, self.norm_std = norm  # must match training (hparams data.norm_mean/norm_std)

    def __call__(self, data):
        pcd_full = []
        pcd_part = []
        mean = []
        std = []

        for sample in data:
            sparse = point_set_to_sparse(sample['extra']['gt_points'], sample['points'], self.n_full, self.n_part, 0.01,
                                         self.norm_mean, self.norm_std)
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
            'offset': [sample['extra'].get('offset', np.zeros(3)) for sample in data],
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
@click.option('--cache', type=str, default=None, help='read inputs from this sample cache instead of the raw data (same preprocessing as training)')
@click.option('--out', type=str, default='./results', help='output root (default: ./results)')
@click.option('--recentre', is_flag=True, help='centre each input on its partial centroid, shift completions back')
@click.option('--vis', is_flag=True, help='show each completed fruit')
def main(diff, denoising_steps, cond_weight, data, split, cache, out, recentre, vis):
    exp_dir = diff.split('/')[-1].split('.')[0].replace('=', '') + f'_T{denoising_steps}_s{cond_weight}' + ('_recentred' if recentre else '')
    out_dir = os.path.join(out, exp_dir, 'diff')
    os.makedirs(out_dir, exist_ok=True)

    diff_completion = DiffCompletion(diff, denoising_steps, cond_weight)
    with open(os.path.join(out, exp_dir, 'exp_config.yaml'), 'w+') as exp_config:
        yaml.dump(dict(diff_completion.hparams), exp_config)

    dataset = CachedFruit(cache, split) if cache else IGGFruit(data_source=data, split=split)
    # a model trained on recentred samples must see recentred inputs
    recentre = recentre or diff_completion.hparams['data'].get('recentre') == 'partial'
    if recentre:
        dataset = Recentred(dataset)
    data_cfg = diff_completion.hparams['data']
    loader = DataLoader(dataset, batch_size=1, num_workers=0,
                        collate_fn=FruitCollation(norm=(data_cfg.get('norm_mean'), data_cfg.get('norm_std'))))

    runtimes = []
    for batch in tqdm.tqdm(loader):
        torch.cuda.synchronize()
        t = time.time()
        completed = diff_completion.complete(batch)
        torch.cuda.synchronize()
        runtimes += [(fid, (time.time() - t) / len(batch['fruit_id'])) for fid in batch['fruit_id']]

        for fruit_id, points, offset in zip(batch['fruit_id'], completed, batch['offset']):
            points[:, :3] += offset  # back to the input frame
            pcd_diff = to_o3d(points)
            o3d.io.write_point_cloud(os.path.join(out_dir, f'{fruit_id}.ply'), pcd_diff)
            if vis:
                o3d.visualization.draw_geometries([pcd_diff])

    with open(os.path.join(out, exp_dir, 'runtimes.csv'), 'w') as f:
        f.write('fruit_id,runtime_s\n')
        f.writelines(f'{fid},{rt:.3f}\n' for fid, rt in runtimes)


if __name__ == '__main__':
    main()

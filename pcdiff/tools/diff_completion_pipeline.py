import numpy as np
import MinkowskiEngine as ME
import torch
import pcdiff.models.minkunet as minknet
import open3d as o3d
from diffusers import DPMSolverMultistepScheduler
from pytorch_lightning.core.lightning import LightningModule
import yaml
import os
import tqdm
from natsort import natsorted
from pcdiff.datasets.igg_fruit import IGGFruit
import click
import time
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from pcdiff.utils.collations import point_set_to_sparse, unormalize_pcd, feats_to_coord
import time

class DiffCompletion(LightningModule):
    def __init__(self, diff_path, denoising_steps, cond_weight, vis):
        super().__init__()
        hparams = yaml.safe_load(open(diff_path.split('checkpoints')[0] + '/hparams.yaml'))
        self.save_hyperparameters(hparams)
        assert denoising_steps <= self.hparams['diff']['t_steps'], \
        f"The number of denoising steps cannot be bigger than T={self.hparams['diff']['t_steps']} (you've set '-T {denoising_steps}')"

        ckpt_diff = torch.load(diff_path)
        self.partial_enc = minknet.MinkGlobalEnc(in_channels=3, out_channels=self.hparams['model']['out_dim']).cuda()
        self.model = minknet.MinkUNetDiff(in_channels=3, out_channels=self.hparams['model']['out_dim']).cuda()
        self.load_state_dict(ckpt_diff['state_dict'])

        self.partial_enc.eval()
        self.model.eval()
        self.cuda()

        # for fast sampling
        self.hparams['diff']['s_steps'] = denoising_steps
        self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.hparams['diff']['t_steps'],
                beta_start=self.hparams['diff']['beta_start'],
                beta_end=self.hparams['diff']['beta_end'],
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.dpm_scheduler.set_timesteps(self.hparams['diff']['s_steps'])
        self.scheduler_to_cuda()

        self.hparams['train']['uncond_w'] = cond_weight
        self.hparams['data']['max_range'] = 50.
        self.w_uncond = self.hparams['train']['uncond_w']
        self.vis = vis
        
        exp_dir = diff_path.split('/')[-1].split('.')[0].replace('=','')  + f'_T{denoising_steps}_s{cond_weight}'
        os.makedirs(f'./results/{exp_dir}', exist_ok=True)
        with open(f'./results/{exp_dir}/exp_config.yaml', 'w+') as exp_config:
            yaml.dump(self.hparams, exp_config)

    def scheduler_to_cuda(self):
        self.dpm_scheduler.timesteps = self.dpm_scheduler.timesteps.cuda()
        self.dpm_scheduler.betas = self.dpm_scheduler.betas.cuda()
        self.dpm_scheduler.alphas = self.dpm_scheduler.alphas.cuda()
        self.dpm_scheduler.alphas_cumprod = self.dpm_scheduler.alphas_cumprod.cuda()
        self.dpm_scheduler.alpha_t = self.dpm_scheduler.alpha_t.cuda()
        self.dpm_scheduler.sigma_t = self.dpm_scheduler.sigma_t.cuda()
        self.dpm_scheduler.lambda_t = self.dpm_scheduler.lambda_t.cuda()
        self.dpm_scheduler.sigmas = self.dpm_scheduler.sigmas.cuda()

    def points_to_tensor(self, x_feats, mean, std):
        x_feats = ME.utils.batched_coordinates(list(x_feats[:]), dtype=torch.float32, device=self.device)

        x_coord = x_feats.clone()
        x_coord[:,1:] = feats_to_coord(x_feats[:,1:], self.hparams['data']['resolution'], mean, std)

        x_t = ME.TensorField(
            features=x_feats[:,1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t

    def reset_partial_pcd(self, x_part, x_uncond, mean, std):
        x_part = self.points_to_tensor(x_part.F.reshape(1,-1,3).detach(), mean, std)
        x_uncond = self.points_to_tensor(torch.zeros_like(x_part.F.reshape(1,-1,3)), mean, std)

        return x_part, x_uncond

    def complete_scan(self, scan, mean, std):
        x_feats = torch.randn([scan.shape[0], 2*scan.shape[1], scan.shape[2]], device=self.device)
        x_full = self.points_to_tensor(x_feats, mean, std)
        x_cond = self.points_to_tensor(scan, mean, std)
        x_uncond = self.points_to_tensor(torch.zeros_like(scan), mean, std)

        completed_scan = self.completion_loop(scan, x_full, x_cond, x_uncond, mean, std)

        return completed_scan

    def forward(self, x_full, x_full_sparse, x_part, t):
        with torch.no_grad():
            part_feat = self.partial_enc(x_part)
            out = self.model(x_full, x_full_sparse, part_feat, t)

        torch.cuda.empty_cache()
        return out.reshape(t.shape[0],-1,3)

    def classfree_forward(self, x_t, x_cond, x_uncond, t):
        x_t_sparse = x_t.sparse()
        x_cond = self.forward(x_t, x_t_sparse, x_cond, t)            
        x_uncond = self.forward(x_t, x_t_sparse, x_uncond, t)

        return x_uncond + self.w_uncond * (x_cond - x_uncond)

    def prepare_pcd(self, pcd, x_t, mean, std):
        points = x_t.F.detach()
        points = points.reshape(-1,3)
        points = unormalize_pcd(points, mean, std)[0].cpu().numpy()
        pcd.points = o3d.utility.Vector3dVector(points)

    def completion_loop(self, x_init, x_t, x_cond, x_uncond, mean, std):
        if self.vis:
            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()
            pcd = o3d.geometry.PointCloud()
            self.prepare_pcd(pcd, x_t, mean, std)
            visualizer.add_geometry(pcd)

        self.scheduler_to_cuda()

        for t in tqdm.tqdm(range(len(self.dpm_scheduler.timesteps))):
            t = self.dpm_scheduler.timesteps[t].cuda()[None]

            noise_t = self.classfree_forward(x_t, x_cond, x_uncond, t)
            x_t = self.dpm_scheduler.step(noise_t, t, x_t.F.reshape(t.shape[0],-1,3))['prev_sample']
            x_t = self.points_to_tensor(x_t, mean, std)

            x_cond, x_uncond = self.reset_partial_pcd(x_cond, x_uncond, mean, std)
            torch.cuda.empty_cache()

            # update visualizer
            if self.vis:
                self.prepare_pcd(pcd, x_t, mean, std)
                visualizer.update_geometry(pcd)
                visualizer.poll_events()
                visualizer.update_renderer()
                time.sleep(0.1)

        if self.vis:
            o3d.visualization.draw_geometries([pcd])

        return unormalize_pcd(x_t.F, mean, std)[0].cpu().detach().numpy()

class FruitCollation:
    def __init__(self):
        return

    def __call__(self, data):
        # "transpose" the  batch(pt, ptn) to batch(pt), batch(ptn)
        pcd_full = []
        pcd_part = []
        mean = []
        std = []

        for sample in data:
            sparse = point_set_to_sparse(sample['extra']['gt_points'], sample['points'], 30000, 15000, 0.001)
            pcd_part.append(sparse[-1])
            pcd_full.append(sparse[0])
            mean.append(sparse[1])
            std.append(sparse[2])

        return {'pcd_full': torch.stack(pcd_full).float(),
            'mean': torch.stack(mean).float(),
            'std': torch.stack(std).float(),
            'pcd_part': torch.stack(pcd_part).float(),
        }

class FruitDataModule(LightningDataModule):
    def __init__(self, data_source):
        self.data_source = data_source

    def train_dataloader(self):
        dataset = IGGFruit(data_source=self.data_source)
        loader = DataLoader(dataset, batch_size=4, num_workers=4, collate_fn=FruitCollation())

        return loader

    def val_dataloader(self):
        dataset = IGGFruit(data_source=self.data_source, split='test')
        loader = DataLoader(dataset, batch_size=1, num_workers=4, collate_fn=FruitCollation())

        return loader

    def test_dataloader(self):
        dataset = IGGFruit(data_source=self.data_source, split='test')
        loader = DataLoader(dataset, batch_size=1, num_workers=4, collate_fn=FruitCollation())

        return loader


@click.command()
@click.option('--diff', '-d', type=str, default='', help='path to the scan sequence')
@click.option('--denoising_steps', '-T', type=int, default=50, help='number of denoising steps (default: 50)')
@click.option('--cond_weight', '-s', type=float, default=6.0, help='conditioning weight (default: 6.0)')
@click.option('--vis', is_flag=True, help="Visualize diffusion process.")
@click.option('--data', type=str, default='./data/shape_completion_challenge', help='dataset root with train/ and test/ splits')
def main(diff, denoising_steps, cond_weight, vis, data):
    exp_dir = diff.split('/')[-1].split('.')[0].replace('=','') + f'_T{denoising_steps}_s{cond_weight}'

    diff_completion = DiffCompletion(
            diff, denoising_steps, cond_weight, vis
        )

    data = FruitDataModule(data).test_dataloader()

    os.makedirs(f'./results/{exp_dir}/diff', exist_ok=True)

    for fruit_data in tqdm.tqdm(data):
        diff_fruit = diff_completion.complete_scan(fruit_data['pcd_part'].cuda(),
                                                    fruit_data['mean'].cuda(),
                                                    fruit_data['std'].cuda(),
                                                )

        pcd_diff = o3d.geometry.PointCloud()
        pcd_diff.points = o3d.utility.Vector3dVector(diff_fruit)
        pcd_diff.estimate_normals()

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(
                unormalize_pcd(fruit_data['pcd_full'],
                                fruit_data['mean'],
                                fruit_data['std']
                            )[0].cpu().detach().numpy())
        pcd_gt.estimate_normals()

        pcd_part = o3d.geometry.PointCloud()
        pcd_part.points = o3d.utility.Vector3dVector(
                unormalize_pcd(fruit_data['pcd_part'],
                                fruit_data['mean'],
                                fruit_data['std']
                            )[0].cpu().detach().numpy())
        pcd_part.estimate_normals()
        o3d.visualization.draw_geometries([pcd_diff])
        import ipdb; ipdb.set_trace()

        #o3d.io.write_point_cloud(f'./results/{exp_dir}/diff/{pcd_path.split(".")[0]}.ply', pcd_diff)

if __name__ == '__main__':
    main()

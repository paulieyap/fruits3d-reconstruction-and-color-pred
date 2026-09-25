import torch
import torch.nn as nn
import torch.nn.functional as F
import pcdiff.models.minkunet as minknet
import numpy as np
import MinkowskiEngine as ME
import open3d as o3d
from pcdiff.utils.scheduling import beta_func
from tqdm import tqdm
from os import makedirs

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from pcdiff.utils.collations import *
from pcdiff.utils.metrics import ChamferDistance, PrecisionRecall
from diffusers import DPMSolverMultistepScheduler

class DiffusionPoints(LightningModule):
    def __init__(self, hparams:dict, data_module: LightningDataModule = None):
        super().__init__()
        # name you hyperparameter hparams, then it will be saved automagically.
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        # alphas and betas for sampling
        if self.hparams['diff']['beta_func'] == 'cosine':
            self.betas = beta_func[self.hparams['diff']['beta_func']](self.hparams['diff']['t_steps'])
        else:
            self.betas = beta_func[self.hparams['diff']['beta_func']](
                    self.hparams['diff']['t_steps'],
                    self.hparams['diff']['beta_start'],
                    self.hparams['diff']['beta_end'],
            )

        self.t_steps = self.hparams['diff']['t_steps']
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.tensor(
            np.cumprod(self.alphas, axis=0), dtype=torch.float32, device=torch.device('cuda')
        )
        self.betas = torch.tensor(self.betas, device=torch.device('cuda'))
        self.alphas = torch.tensor(self.alphas, device=torch.device('cuda'))

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)


        # for fast sampling
        self.s_steps = self.hparams['diff']['s_steps']
        self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['beta_start'],
                beta_end=self.hparams['diff']['beta_end'],
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.dpm_scheduler.set_timesteps(self.s_steps)
        self.scheduler_to_cuda()


        # self.partial_enc = minknet.MinkGlobalEnc(in_channels=3, out_channels=self.hparams['model']['out_dim']) #96
        # self.model = minknet.MinkUNetDiff(in_channels=3, out_channels=self.hparams['model']['out_dim']) #96
        self.partial_enc = minknet.MinkGlobalEnc(in_channels=6, out_channels=self.hparams['model']['out_dim']) #96
        self.model = minknet.MinkUNetDiff(in_channels=6,out_channels=6) #changed to 6

        self.chamfer_distance = ChamferDistance()
        self.precision_recall = PrecisionRecall(0.001,0.01,100)

    def scheduler_to_cuda(self):
        self.dpm_scheduler.timesteps = self.dpm_scheduler.timesteps.cuda()
        self.dpm_scheduler.betas = self.dpm_scheduler.betas.cuda()
        self.dpm_scheduler.alphas = self.dpm_scheduler.alphas.cuda()
        self.dpm_scheduler.alphas_cumprod = self.dpm_scheduler.alphas_cumprod.cuda()
        self.dpm_scheduler.alpha_t = self.dpm_scheduler.alpha_t.cuda()
        self.dpm_scheduler.sigma_t = self.dpm_scheduler.sigma_t.cuda()
        self.dpm_scheduler.lambda_t = self.dpm_scheduler.lambda_t.cuda()
        self.dpm_scheduler.sigmas = self.dpm_scheduler.sigmas.cuda()
    
    def q_sample(self, x, t, noise):
        return self.sqrt_alphas_cumprod[t][:,None,None] * x + self.sqrt_one_minus_alphas_cumprod[t][:,None,None] * noise

    def visualize_step_t(self, x_t, gt_pts, pcd, pcd_mean, pcd_std, pidx=0):
        points = x_t.F.detach().cpu().numpy()
        points = points.reshape(gt_pts.shape[0],-1,3)
        obj_mean = pcd_mean[pidx][0].detach().cpu().numpy()
        points = np.concatenate((points[pidx], gt_pts[pidx]), axis=0)

        points = unormalize_pcd(points, pcd_mean[pidx].cpu().numpy(), pcd_std[pidx].cpu().numpy())
        dist_pts = np.sqrt(np.sum((points - obj_mean)**2, axis=-1))
        dist_idx = dist_pts < 20.

        full_pcd = len(points) - len(gt_pts[pidx])
        print(f'\n[{dist_idx.sum() - full_pcd}|{dist_idx.shape[0] - full_pcd }] points inside margin...')

        pcd.points = o3d.utility.Vector3dVector(points[dist_idx])
       
        colors = np.ones((len(points), 3)) * .5
        colors[:len(gt_pts[0])] = [1.,.3,.3]
        colors[-len(gt_pts[0]):] = [.3,1.,.3]
        pcd.colors = o3d.utility.Vector3dVector(colors[dist_idx])

    def classfree_forward(self, x_t, x_cond, x_uncond, t):
        x_t_sparse = x_t.sparse()
        x_cond = self.forward(x_t, x_t_sparse, x_cond, t)
        x_uncond = self.forward(x_t, x_t_sparse, x_uncond, t)
        
        #bytedance sec 3.4
        guidance_weight = self.hparams['train']['uncond_w']
        rescale_strength = self.hparams['train']['rescale_strength']

        x_cfg = x_uncond + guidance_weight * (x_cond - x_uncond) #equation 13
        #std_cond = torch.std(x_cond,dim=2,keepdim=True)
        #std_cfg = torch.std(x_cfg,dim=2,keepdim=True)    
        #x_rescaled = x_cfg * (std_cond/std_cfg)
        #x_final = rescale_strength * x_rescaled + (1-rescale_strength) * x_cfg #equation16

        #return x_final
        return x_cfg



    def reset_partial_pcd(self, x_part, x_uncond, x_mean, x_std):
        #changed to 6
        x_part = self.points_to_tensor(x_part.F.reshape(x_mean.shape[0],-1,6).detach(), x_mean, x_std)
        x_uncond = self.points_to_tensor(
                torch.zeros_like(x_part.F.reshape(x_mean.shape[0],-1,6)), torch.zeros_like(x_mean), torch.zeros_like(x_std)
        )

        return x_part, x_uncond

    def p_sample_loop(self, x_t, x_cond, x_uncond, gt_pts, x_mean, x_std, batch_idx):
        
        pcd = o3d.geometry.PointCloud()
        self.scheduler_to_cuda()

        for t in tqdm(range(len(self.dpm_scheduler.timesteps))):
            t = torch.ones(gt_pts.shape[0]).cuda().long() * self.dpm_scheduler.timesteps[t].cuda()

            noise_t = self.classfree_forward(x_t, x_cond, x_uncond, t)
            x_t = self.dpm_scheduler.step(noise_t, t[0], x_t.F.reshape(t.shape[0],-1,6))['prev_sample']
            x_t = self.points_to_tensor(x_t, x_mean, x_std)

            # this is needed otherwise minkEngine will keep "stacking" coords maps over the x_part and x_uncond
            # i.e. memory leak
            x_cond, x_uncond = self.reset_partial_pcd(x_cond, x_uncond, x_mean, x_std)
            torch.cuda.empty_cache()
        '''
        pcd = o3d.geometry.PointCloud()
        self.scheduler_to_cuda()

        # Determine batch size, either from gt_pts if it exists or x_cond
        batch_size = gt_pts.shape[0] if gt_pts is not None else x_cond.shape[0]

        for t in tqdm(range(len(self.dpm_scheduler.timesteps))):
            # Create a tensor of timesteps with the appropriate batch size
            t = torch.ones(batch_size).cuda().long() * self.dpm_scheduler.timesteps[t].cuda()

            noise_t = self.classfree_forward(x_t, x_cond, x_uncond, t)
            x_t = self.dpm_scheduler.step(noise_t, t[0], x_t.F.reshape(t.shape[0], -1, 3))['prev_sample']
            x_t = self.points_to_tensor(x_t, x_mean, x_std)

            # Reset partial point clouds to prevent memory leaks
            x_cond, x_uncond = self.reset_partial_pcd(x_cond, x_uncond, x_mean, x_std)
            torch.cuda.empty_cache()

        '''
        return x_t

    def p_losses(self, y, noise):
        return F.mse_loss(y, noise)

    def forward(self, x_full, x_full_sparse, x_part, t):

        part_feat = self.partial_enc(x_part)
        out = self.model(x_full, x_full_sparse, part_feat, t)
        torch.cuda.empty_cache()
        
        return out.reshape(t.shape[0],-1,6)
        # return out.reshape(t.shape[0],-1,3)

    def points_to_tensor(self, x_feats, mean, std):
        # ipdb.set_trace()

        x_feats = ME.utils.batched_coordinates(list(x_feats[:]), dtype=torch.float32, device=self.device)
        #creates another dimension which is the batch index

        x_coord = x_feats[:,:4].clone() #batch index, x y z


        x_coord[:,1:] = feats_to_coord(x_feats[:,1:4], self.hparams['data']['resolution'], mean[:,:3], std[:,:3])
        #multiplies everything by ~1000 - 1mm == 1m 
        x_t = ME.TensorField(
            features=x_feats[:,1:], #in mm
            coordinates=x_coord, #meter, voxel coord
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t

    def training_step(self, batch:dict, batch_idx): #use aligned_data pcd now instead of pcd_full?
        # initial random noise
        torch.cuda.empty_cache()
        noise = torch.randn(batch['pcd_full'].shape, device=self.device)
        mask = batch['mask']
        mask = torch.stack(mask)
        # ipdb.set_trace()
        #breakpoint check: batch['pcd_full'] has -1's !


        # sample step t
        t = torch.randint(0, self.t_steps, size=(batch['pcd_full'].shape[0],), device=self.device)
        # sample q at step t
        t_sample = self.q_sample(batch['pcd_full'], t, noise)

        # replace the original points with the noise sampled
        x_full = self.points_to_tensor(t_sample, batch['mean'], batch['std'])

        # for classifier-free guidance swithc between conditional and unconditional training
        if torch.rand(1) > self.hparams['train']['uncond_prob'] or batch['pcd_full'].shape[0] == 1:
            
            x_part = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
        else:
            x_part = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            )
        
        
        denoise_t = self.forward(x_full, x_full.sparse(), x_part, t) #1,3000,6
        loss_xyz = self.p_losses(denoise_t[...,:3], noise[...,:3]) #loss_xyz

        #remove the non-colored points when computing 
        loss_rgb = self.p_losses(denoise_t[mask][...,3:], noise[mask][...,3:])

        loss_sum = loss_xyz + loss_rgb 
        self.log('train/diff_loss', loss_sum)
        #log all losses individually 

        #loss = loss_xyz + loss_rgb 
        self.log('train/loss_xyz', loss_xyz)
        self.log('train/loss_rgb', loss_rgb)
        self.log('train/loss', loss_sum)
        torch.cuda.empty_cache()

        return loss_sum

    def validation_step(self, batch:dict, batch_idx):
        if batch_idx != 0:
            return

        self.model.eval()
        self.partial_enc.eval()
        seq_dir =  f'{self.logger.log_dir}/generated_pcd_validation'
        makedirs(seq_dir, exist_ok=True)

        print('validation Fruit ID: ', batch['fruit_id'])

   

        with torch.no_grad():
            gt_pts = batch['pcd_full'].detach().cpu().numpy()

            x_feats = torch.randn(batch['pcd_full'].shape, device=self.device).detach()
            x_full = self.points_to_tensor(x_feats, batch['mean'], batch['std'])
            x_cond = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
            x_uncond = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            )


            x_gen_eval = self.p_sample_loop(x_full, x_cond, x_uncond, gt_pts, batch['mean'], batch['std'], batch_idx)
            #edit to 6 output.... also check
            x_gen_eval = x_gen_eval.F.reshape((gt_pts.shape[0],-1,6)) #changed to 6

            for i in range(len(batch['pcd_full'])):

                pcd_pred = o3d.geometry.PointCloud()

                c_pred = unormalize_pcd(x_gen_eval, batch['mean'], batch['std'])[i].cpu().detach().numpy()
                pcd_pred.points = o3d.utility.Vector3dVector(c_pred[:,:3])
                pcd_pred.colors = o3d.utility.Vector3dVector(c_pred[:,3:])

                pcd_gt = o3d.geometry.PointCloud()
                g_pred = unormalize_pcd(batch['pcd_full'], batch['mean'], batch['std'])[i].cpu().detach().numpy()
                pcd_gt.points = o3d.utility.Vector3dVector(g_pred[:,:3]) 
                pcd_gt.colors = o3d.utility.Vector3dVector(g_pred[:,3:]) 

                # ipdb.set_trace()
                o3d.io.write_point_cloud(f"{seq_dir}/{batch_idx * self.hparams['train']['batch_size'] + i}_gt.ply", pcd_gt)
                o3d.io.write_point_cloud(f"{seq_dir}/{batch_idx * self.hparams['train']['batch_size'] + i}.ply", pcd_pred)

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)
                #later: metric for colors

        # ipdb.set_trace()
        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()

        self.log('val/cd_mean', cd_mean, on_step=True)
        self.log('val/cd_std', cd_std, on_step=True)
        self.log('val/precision', pr, on_step=True)
        self.log('val/recall', re, on_step=True)
        self.log('val/fscore', f1, on_step=True)
        torch.cuda.empty_cache()
        
        return {'val/cd_mean': cd_mean, 'val/cd_std': cd_std, 'val/precision': pr, 'val/recall': re, 'val/fscore': f1}
    '''
    def test_step(self, batch: dict, batch_idx):
        self.model.eval()
        self.partial_enc.eval()

        seq_dir = f'{self.logger.log_dir}/generated_pcd_test'
        makedirs(seq_dir, exist_ok=True)

        # ipdb.set_trace()
        with torch.no_grad():
            # We no longer have ground truth, so we work with the partial point cloud
            x_feats = torch.randn(batch['pcd_part'].shape, device=self.device).detach()
            x_cond = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
            x_uncond = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            )

            # Generate the point cloud
            x_gen_eval = self.p_sample_loop(x_feats, x_cond, x_uncond, None, batch['mean'], batch['std'], batch_idx)
            x_gen_eval = x_gen_eval.F.reshape((batch['pcd_part'].shape[0], -1, 3))

            for i in range(len(batch['pcd_part'])):
                # Predicted point cloud
                pcd_pred = o3d.geometry.PointCloud()
                c_pred = unormalize_pcd(x_gen_eval, batch['mean'], batch['std'])[i].cpu().detach().numpy()
                pcd_pred.points = o3d.utility.Vector3dVector(c_pred)

                # Partial input point cloud (used as a reference)
                # pcd_ref = o3d.geometry.PointCloud()
                # r_pred = unormalize_pcd(batch['pcd_part'], batch['mean'], batch['std'])[i].cpu().detach().numpy()
                # pcd_ref.points = o3d.utility.Vector3dVector(r_pred)
                

                # Save the generated point cloud
                o3d.io.write_point_cloud(f"{seq_dir}/{batch_idx * self.hparams['train']['batch_size'] + i}.ply", pcd_pred)

                # Update Chamfer Distance and Precision/Recall metrics using the partial input as a reference
            #     self.chamfer_distance.update(pcd_ref, pcd_pred)
            #     self.precision_recall.update(pcd_ref, pcd_pred)

            # # Calculate and log metrics
            # cd_mean, cd_std = self.chamfer_distance.compute()
            # pr, re, f1 = self.precision_recall.compute_auc()

            # self.log('test/cd_mean', cd_mean, on_step=True)
            # self.log('test/cd_std', cd_std, on_step=True)
            # self.log('test/precision', pr, on_step=True)
            # self.log('test/recall', re, on_step=True)
            # self.log('test/fscore', f1, on_step=True)

            # print("cd_mean: ", cd_mean)
            # print("cd_std: ", cd_std)
            # print("precision: ", pr)
            # print("recall: ", re)
            # print("f1: ", f1)

            return {'test/cd_mean': cd_mean, 'test/cd_std': cd_std, 'test/precision': pr, 'test/recall': re, 'test/fscore': f1}

    '''
    def test_step(self, batch:dict, batch_idx): #used for both testing and validation
        self.model.eval()
        self.partial_enc.eval()

      
        seq_dir =  f'{self.logger.log_dir}/generated_pcd_test'
        makedirs(seq_dir, exist_ok=True)
        print('Testing Fruit ID: ', batch['fruit_id'])


        with torch.no_grad():
            gt_pts = batch['pcd_full'].detach().cpu().numpy()

            x_feats = torch.randn(batch['pcd_full'].shape, device=self.device).detach()
            x_full = self.points_to_tensor(x_feats, batch['mean'], batch['std'])
            x_cond = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
            x_uncond = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            )
            x_gen_eval = self.p_sample_loop(x_full, x_cond, x_uncond, gt_pts, batch['mean'], batch['std'], batch_idx)
            x_gen_eval = x_gen_eval.F.reshape((gt_pts.shape[0],-1,6)) #changed to 6

            # ipdb.set_trace()
            for i in range(len(batch['pcd_full'])):

                pcd_pred = o3d.geometry.PointCloud()
                # c_pred = x_gen_eval[i].cpu().detach().numpy() #using normalized
                c_pred = unormalize_pcd(x_gen_eval, batch['mean'], batch['std'])[i].cpu().detach().numpy()
                pcd_pred.points = o3d.utility.Vector3dVector(c_pred[:,:3])
                pcd_pred.colors = o3d.utility.Vector3dVector(c_pred[:,3:])

                pcd_gt = o3d.geometry.PointCloud()
                # g_pred = batch['pcd_full'][i].cpu().detach().numpy()  #using normalized
                g_pred = unormalize_pcd(batch['pcd_full'], batch['mean'], batch['std'])[i].cpu().detach().numpy()
                pcd_gt.points = o3d.utility.Vector3dVector(g_pred[:,:3]) 
                pcd_gt.colors = o3d.utility.Vector3dVector(g_pred[:,3:])

                pcd_part = o3d.geometry.PointCloud()
                partial = unormalize_pcd(batch['pcd_part'], batch['mean'], batch['std'])[i].cpu().detach().numpy()
                pcd_part.points = o3d.utility.Vector3dVector(partial[:,:3]) 
                pcd_part.colors = o3d.utility.Vector3dVector(partial[:,3:])

                # ipdb.set_trace() #try saving aligned output

                #commented out for test set
                # o3d.io.write_point_cloud(f"{seq_dir}/{batch_idx * self.hparams['train']['batch_size'] + i}_{batch['fruit_id']}_partial.ply", pcd_part)
                # o3d.io.write_point_cloud(f"{seq_dir}/{batch_idx * self.hparams['train']['batch_size'] + i}_{batch['fruit_id']}_gt.ply", pcd_gt)


                o3d.io.write_point_cloud(f"{seq_dir}/{batch['fruit_id']}.ply", pcd_pred)

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)
            # x_gen_eval = self.p_sample_loop(x_full, x_cond, x_uncond, gt_pts, batch['mean'], batch['std'], batch_idx)
            # x_gen_eval = x_gen_eval.F.reshape((gt_pts.shape[0],-1,3))



            
            # for i in range(len(batch['pcd_full'])):
            #     pcd_pred = o3d.geometry.PointCloud()
            #     c_pred = unormalize_pcd(x_gen_eval, batch['mean'], batch['std'])[i].cpu().detach().numpy()
            #     pcd_pred.points = o3d.utility.Vector3dVector(c_pred)

            #     pcd_gt = o3d.geometry.PointCloud()
            #     g_pred = unormalize_pcd(batch['pcd_full'], batch['mean'], batch['std'])[i].cpu().detach().numpy()
            #     pcd_gt.points = o3d.utility.Vector3dVector(g_pred)
            #     fid = batch['fruit_id']
            #     print("Fruit ID: ", fid)
            #     ipdb.set_trace()

                
            #     o3d.io.write_point_cloud(f"{seq_dir}/{fid}.ply", pcd_pred)

            #     self.chamfer_distance.update(pcd_gt, pcd_pred)
            #     self.precision_recall.update(pcd_gt, pcd_pred)

        
        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()

        self.log('test/cd_mean', cd_mean, on_step=True)
        self.log('test/cd_std', cd_std, on_step=True)
        self.log('test/precision', pr, on_step=True)
        self.log('test/recall', re, on_step=True)
        self.log('test/fscore', f1, on_step=True)
        
        print("cd_mean: ", cd_mean)
        print("cd_std: ", cd_std)
        print("precision: ", pr)
        print("recall: ", re)
        print("f1: ", f1) 


        return {'test/cd_mean': cd_mean, 'test/cd_std': cd_std, 'test/precision': pr, 'test/recall': re, 'test/fscore': f1}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['train']['lr'], betas=(0.9, 0.999))
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.998)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        #                                                        self.hparams['train']['max_epoch'],
        #                                                        eta_min=self.hparams['train']['lr'] / 1000)

        return optimizer#[optimizer], [scheduler]

#######################################
# Modules
#######################################

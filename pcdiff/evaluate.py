import numpy as np
import open3d as o3d
from utils.metrics import ChamferDistance, PrecisionRecall
import os
import click
import tqdm

@click.command()
@click.option('--path', '-p', type=str, help='path to pcd')
def main(path):
    chamfer_distance = ChamferDistance()
    precision_recall = PrecisionRecall(0.001,0.01,100)
   
    for pcd_name in tqdm.tqdm(os.listdir(path)):
        pcd = o3d.io.read_point_cloud(os.path.join(path, pcd_name))
        
        points = np.array(pcd.points)
        colors = np.array(pcd.colors)
        gt = points[colors[:,1] == 1.]
        pred = points[colors[:,1] < 1.]
        
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(pred)
        
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt)
        
        chamfer_distance.update(pcd_gt, pcd_pred)
        precision_recall.update(pcd_gt, pcd_pred)
    
    cd_mean, cd_std = chamfer_distance.compute()
    print(f'ChamferDistance(mean/std): {cd_mean}/{cd_std}')
        
    pr, re, f1 = precision_recall.compute_auc()
    print(f'Precision/Recall/f1-Score: {pr}/{re}/{f1}')

if __name__ == '__main__':
    main()


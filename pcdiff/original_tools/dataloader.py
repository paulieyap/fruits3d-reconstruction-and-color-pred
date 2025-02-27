import os
import json
import open3d as o3d
import numpy as np
import cv2
import ipdb
from pytorch_lightning.loggers import TensorBoardLogger

class ShapeCompletionDataset():

    def __init__(self,
                 data_source=None,
                 split='train',
                 return_pcd = True,
                 return_rgbd = True
                 ):

        assert return_pcd or return_rgbd, "return_pcd and return_rgbd are set to False. Set at least one to True"

        self.data_source = data_source
        self.split = split
        self.return_pcd = return_pcd
        self.return_rgbd = return_rgbd
        
        self.fruit_list = self.get_file_paths()

        # Initialize TensorBoard logger inside the dataset class
        self.tb_logger = TensorBoardLogger("logs", name=f"ShapeCompletion_{split}")


        # Log dataset size
        self.tb_logger.experiment.add_text("Dataset Info", f"Loaded {len(self.fruit_list)} samples for {self.split} set.")


    def get_file_paths(self):
        fruit_list = {}
        for fid in os.listdir(os.path.join(self.data_source, self.split)):
            fruit_list[fid] = {
                'path': os.path.join(self.data_source, self.split, fid),
            }
        return fruit_list

    def get_gt(self, fid):
        return o3d.io.read_point_cloud(os.path.join(self.fruit_list[fid]['path'],'gt/pcd/fruit.ply'))

    def get_rgbd(self, fid):
        fid_root = self.fruit_list[fid]['path']

        intrinsic_path = os.path.join(fid_root,'input/intrinsic.json')
        intrinsic = self.load_K(intrinsic_path)
        
        rgbd_data = {
            'intrinsic':intrinsic,
            'pcd': o3d.geometry.PointCloud(),
            'frames':{}
            }

        frames = os.listdir(os.path.join(fid_root,'input/masks/'))
        for frameid in frames:
            
            pose_path = os.path.join(fid_root,'input/poses/',frameid.replace('png','txt'))
            pose = np.loadtxt(pose_path)
            
            rgb_path = os.path.join(fid_root,'input/color/',frameid)
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

            depth_path = os.path.join(fid_root,'input/depth/',frameid.replace('png','npy'))
            depth = np.load(depth_path)

            mask_path = os.path.join(fid_root,'input/masks/',frameid)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            frame_key = frameid.replace('png','')

            if self.return_pcd:
                # ipdb.set_trace()
                frame_pcd = self.rgbd_to_pcd(rgb, depth, mask, pose, intrinsic)
                frame_pcd_outlierRemoved, ind = frame_pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=2.0)
                rgbd_data['pcd'] += frame_pcd_outlierRemoved


                # #original below:
                # frame_pcd = self.rgbd_to_pcd(rgb, depth, mask, pose, intrinsic)
                # rgbd_data['pcd'] += frame_pcd #aggregation of frames?
            # ipdb.set_trace()

            rgbd_data['frames'][frame_key] = {
                'rgb': rgb,
                'depth': depth,
                'mask': mask,
                'pose': pose
            }

        # ipdb.set_trace() 

        return rgbd_data

    @staticmethod
    def load_K(path):
        with open(path,'r') as f:
            data = json.load(f)['intrinsic_matrix']
        k = np.reshape(data, (3, 3), order='F') 
        return k

    @staticmethod
    def rgbd_to_pcd(rgb, depth, mask, pose, K):

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb),
                                                                  o3d.geometry.Image(depth*mask),
                                                                  depth_scale = 1,
                                                                  depth_trunc=1.0,
                                                                  convert_rgb_to_intensity=False)

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(height=rgb.shape[0],
                                 width=rgb.shape[1],
                                 fx=K[0,0],
                                 fy=K[1,1],
                                 cx=K[0,2],
                                 cy=K[1,2],
                                 )

        extrinsic = np.linalg.inv(pose)
        
        frame_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)
        return frame_pcd




    @staticmethod
    def align_pointcloud_with_rgb(pointcloud_xyz, rgb_points, rgb_colors):

        # Step 1: Create point cloud from XYZ data
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pointcloud_xyz)

        # Step 2: Initialize output with -1 for RGB values
        num_points = pointcloud_xyz.shape[0]
        output = np.hstack((pointcloud_xyz, np.full((num_points, 3), -1)))

        # Step 3: Find closest points and assign RGB values
        rgb_pc = o3d.geometry.PointCloud()
        rgb_pc.points = o3d.utility.Vector3dVector(rgb_points)

        # Build a KDTree to find nearest neighbors
        pc_tree = o3d.geometry.KDTreeFlann(rgb_pc)

        for i, point in enumerate(pointcloud_xyz):
            # Use the KDTree to find the nearest RGB point
            [_, idx, _] = pc_tree.search_knn_vector_3d(point, 1)
            nearest_point = np.asarray(rgb_pc.points)[idx[0]]

            # Calculate the distance and assign RGB if close enough

            if np.linalg.norm(point - nearest_point) < 0.01:  # Threshold distance
                output[i, 3:] = rgb_colors[idx[0]]

 
        #print no of colored points
        length = output.shape[0]
        return output, length




    def __len__(self):
        return len(self.fruit_list)

    def __getitem__(self, idx):
        

        keys = list(self.fruit_list.keys())
        fid = keys[idx]
        
        input_data = self.get_rgbd(fid)

        # Remove outliers from the partial point cloud
        partial_without_outliers, ind = input_data['pcd'].remove_statistical_outlier(nb_neighbors=200, std_ratio=2.0)
        if self.split != 'test':
            # gt_pcd = o3d.io.read_point_cloud(os.path.join(fid_root, 'gt/pcd/fruit.ply'))  # Load external ground truth
            gt_pcd = self.get_gt(fid)
        else:
            gt_pcd = partial_without_outliers  # Use partial point cloud as ground truth for test
        
        
        # Extract XYZ coordinates and RGB colors from the filtered point cloud
        frame_pts = np.array(partial_without_outliers.points)  # XYZ coordinates
        frame_rgb = np.array(partial_without_outliers.colors)  # RGB colors

        rgbd_pcd_ = np.concatenate((frame_pts, frame_rgb), axis=-1)

    
        pointcloud_xyz = np.asarray(gt_pcd.points)
        aligned_output, length_ = self.align_pointcloud_with_rgb(pointcloud_xyz, frame_pts, frame_rgb)
        
        aligned_pcd_o3d = o3d.geometry.PointCloud()
        aligned_pcd_o3d.points = o3d.utility.Vector3dVector(aligned_output[:, :3])
        aligned_pcd_o3d.colors = o3d.utility.Vector3dVector(aligned_output[:, 3:])


        # Define directory for saving PLY files and ensure it exists
        seq_dir = 'generated_pcd_Aggregated'
        os.makedirs(seq_dir, exist_ok=True)

        o3d.io.write_point_cloud(f"{seq_dir}/{fid}_groundtruth.ply", gt_pcd)
        o3d.io.write_point_cloud(f"{seq_dir}/{fid}_rgbd.ply", partial_without_outliers)
        o3d.io.write_point_cloud(f"{seq_dir}/{fid}_aligned_rgbd.ply", aligned_pcd_o3d)

        # Return structured data
        item = {
            'groundtruth_pcd': gt_pcd,            # Original ground truth point cloud
            'rgbd_pcd': partial_without_outliers,  # Partial RGBD point cloud (outliers removed)
            'aligned_data': aligned_output,        # Aligned colored partial point cloud as an array (Nx6)
            'fruit_id': fid,
            'aligned_data_length': length_
        }

        return item

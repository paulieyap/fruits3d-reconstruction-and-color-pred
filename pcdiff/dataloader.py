import os
import json
import open3d as o3d
import numpy as np
import cv2
import ipdb
from pytorch_lightning.loggers import TensorBoardLogger
import copy
import pickle
import yaml
from os.path import join, dirname, abspath

class ShapeCompletionDataset():

    def __init__(self,
                 
                 data_source=None,
                 split='train',
                 return_pcd = True,
                 return_rgbd = True
                
                 ):

        assert return_pcd or return_rgbd, "return_pcd and return_rgbd are set to False. Set at least one to True"


        config = join(dirname(abspath(__file__)),'config/config.yaml')
       
        self.hparams = yaml.safe_load(open(config))
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



    def p2p_icp(self, source, target, trans_init):
        threshold = self.hparams['train']['icp_threshold']

        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
                                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                    o3d.pipelines.registration.ICPConvergenceCriteria(self.hparams['train']['icp_convergence_max_iter']))


        transformation = reg_p2p.transformation

        print("Transformation is:")
        print(transformation)

        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.transform(transformation)
        # self.draw_registration_result(source, target, reg_p2p.transformation)

        return source_temp, transformation


    def draw_registration_result(self,source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])


    def get_all_gt(self, keys):
        all_means = []
        all_stds = []
       
        # Iterate through the list of sweet peppers (keys)
        for fid in keys:  # Assuming `keys` is a list of sweet peppers' ids
            # Assuming the 'path' field in fruit_list points to the directory containing the 'gt/pcd/fruit.ply'
            pcd_path = os.path.join(self.fruit_list[fid]['path'], 'gt/pcd/fruit.ply')
            
            if os.path.exists(pcd_path):  # Check if the file exists
                pcd = o3d.io.read_point_cloud(pcd_path)
                points = np.asarray(pcd.points)  # Convert Open3D point cloud to NumPy array
                
                # Compute mean and std for the current sweet pepper (point cloud)
                mean = np.mean(points, axis=0)
                std = np.std(points, axis=0)
                
                # Append the results to the lists
                all_means.append(mean)
                all_stds.append(std)
            else:
                print(f"Point cloud file {pcd_path} not found for sample {fid}. Skipping.")
        ipdb.set_trace()
        # Convert the lists of means and stds into arrays (if needed for later processing)
        all_means = (np.array(all_means)).mean(axis=0)
        all_stds = (np.array(all_stds)).std(axis=0)
        
        return all_means, all_stds


    def get_rgbd(self, fid):
        # print("Fruit List:", self.fruit_list)
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
            print("FRAMEID:", frameid)
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
                valid_pcds = [] 
                frame_pcd = self.rgbd_to_pcd(rgb, depth, mask, pose, intrinsic)
                frame_pcd_outlierRemoved, ind = frame_pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=2.0)
                #add calc of mean and std 
                #Calculate- Calculating the Mean and Standard Deviation of the points
                # - if the x-coordinate of the standard deviation is too high, drop/ignore this sweet pepper
                # - “too bad” standard deviation used - 1 cm

                points_array = np.asarray(frame_pcd_outlierRemoved.points)
                mean_xyz = np.mean(points_array, axis=0)
                std_xyz = np.std(points_array, axis=0)

                print(f"Frame {frame_key} - Mean: {mean_xyz}, Std Dev: {std_xyz}")

                if std_xyz[0] < self.hparams['train']['get_rgbd_thresh']:  # 3 cm threshold
                    valid_pcds.append(frame_pcd_outlierRemoved)
                else:
                    print(f"Skipping frame {frame_key}. x-coord std is: {std_xyz[0]}")


                #accepted pcds are used if the list will contain an element
            if valid_pcds:
                rgbd_data['pcd'] = valid_pcds[0]  
                for i in range(1, len(valid_pcds)):
                    icp_transformation_pcd, icp_trans = self.p2p_icp(valid_pcds[i], rgbd_data['pcd'], np.eye(4))
                    rgbd_data['pcd'] += icp_transformation_pcd
            else:
                rgbd_data['pcd'] += frame_pcd_outlierRemoved #just use the "ugly" pcd lol 
                print("No valid point clouds found. Aggregation skipped.")


                # original below:
                # frame_pcd = self.rgbd_to_pcd(rgb, depth, mask, pose, intrinsic)
                # rgbd_data['pcd'] += frame_pcd #aggregation of frames

            rgbd_data['frames'][frame_key] = {
                'rgb': rgb,
                'depth': depth,
                'mask': mask,
                'pose': pose
            }

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
        
        keys = list(self.fruit_list.keys()) #all sweet pepper keys in dataset
        fid = keys[idx]
        
        input_data = self.get_rgbd(fid)
        partial_pcd = input_data['pcd']

        # partial_without_outliers, ind = input_data['pcd'].remove_statistical_outlier(nb_neighbors=200, std_ratio=2.0)


        # Remove outliers from the partial point cloud
        if self.split != 'test':
            # gt_pcd = o3d.io.read_point_cloud(os.path.join(fid_root, 'gt/pcd/fruit.ply'))  # Load external ground truth

            gt_pcd = self.get_gt(fid)
            # mean, std = self.get_all_gt(keys) #used to pre-compute mean adn std
   
        else:
            gt_pcd = partial_pcd#partial_without_outliers  # Use partial point cloud as ground truth for test
        
        #using input_data['pcd'] instead of partial without outliers, because if with ICP, we will already remove the outliers in get_rgbd. 

        # Extract XYZ coordinates and RGB colors from the filtered point cloud
        frame_pts = np.array(partial_pcd.points)  # XYZ coordinates
        frame_rgb = np.array(partial_pcd.colors)  # RGB colors

        rgbd_pcd_ = np.concatenate((frame_pts, frame_rgb), axis=-1)


        pointcloud_xyz = np.asarray(gt_pcd.points)
        aligned_output, length_ = self.align_pointcloud_with_rgb(pointcloud_xyz, frame_pts, frame_rgb)
    
        aligned_pcd_o3d = o3d.geometry.PointCloud()
        aligned_pcd_o3d.points = o3d.utility.Vector3dVector(aligned_output[:, :3])
        aligned_pcd_o3d.colors = o3d.utility.Vector3dVector(aligned_output[:, 3:])


        # Define directory for saving PLY files and ensure it exists
        seq_dir = 'generated_pcd_Aggregated'
        os.makedirs(seq_dir, exist_ok=True)

        # o3d.io.write_point_cloud(f"{seq_dir}/{fid}_groundtruth.ply", gt_pcd)
        # o3d.io.write_point_cloud(f"{seq_dir}/{fid}_rgbd.ply", partial_pcd)
        # o3d.io.write_point_cloud(f"{seq_dir}/{fid}_aligned_rgbd.ply", aligned_pcd_o3d)


        item = {
            'groundtruth_pcd': gt_pcd,            # Original ground truth point cloud
            'rgbd_pcd': rgbd_pcd_,  # Partial RGBD point cloud (outliers removed)
            'aligned_data': aligned_output,        # Aligned colored partial point cloud as an array (Nx6)
            'fruit_id': fid,
            'aligned_data_length': length_
        }

        return item

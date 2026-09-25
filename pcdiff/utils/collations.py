import numpy as np
import MinkowskiEngine as ME
import torch
import torch.nn.functional as F
import numpy as np
import ipdb

def feats_to_coord(p_feats, resolution, mean, std):
    p_feats = unormalize_pcd(p_feats.reshape(mean.shape[0],-1,mean.shape[-1]), mean, std) 
    #feats = is the shape of the last dimension of the mean
    p_coord = torch.round(p_feats / resolution)
    #p_coord -= p_coord.min(1, keepdim=True).values

    return p_coord.reshape(-1,mean.shape[-1])

def normalize_pcd(points, mean, std):
    return (points - mean[:,None,:]) / std[:,None,:] if len(mean.shape) == 2 else (points - mean) / std

def unormalize_pcd(points, mean, std):
    return (points * std[:,None,:]) + mean[:,None,:] if len(mean.shape) == 2 else (points * std) + mean

def point_set_to_sparse(p_full, p_part, n_full, n_part, resolution): #replace p_full with rgbd
    # ipdb.set_trace()
    concat_full = np.ceil(n_full / p_full.shape[0])
    concat_part = np.ceil(n_part / p_part.shape[0])

    p_full = p_full[torch.randperm(p_full.shape[0])]
    p_full = torch.tensor(p_full.repeat(concat_full, 0)[:n_full])

    p_part = p_part[torch.randperm(p_part.shape[0])]
    p_part = torch.tensor(p_part.repeat(concat_part, 0)[:n_part])
    #p_feats = ME.utils.batched_coordinates([p_feats], dtype=torch.float32)[:2000]
    
    # after creating the voxel coordinates we normalize the floating coordinates towards mean=0 and std=1
    #p_full = torch.cat((p_full, p_part), axis=0)
    #replace with pre-computed value
    # p_mean, p_std = p_full.mean(axis=0), p_full.std(axis=0)
    precomp_mean = np.array([-0.04177155,  0.03841237, -0.03472447,  0.39953194,  0.39472064,0.37596458])
    precomp_std  = np.array([0.02371479, 0.01226978, 0.01996409, 0.15754811, 0.17013572,0.19951918])
    p_mean, p_std = precomp_mean, precomp_std
    

    
    p_full[:,:3] = normalize_pcd(p_full[:,:3], p_mean[:3], p_std[:3]) 

    #no need to adjust mask even when we're using precomputed mean and std w.r.t. GT
    #create the mask
    mask = p_full[:,4] != -1 #-1 for all r g b 
    #normalize RGB where it's not (-1)
    p_mean[3:], p_std[3:] = p_full[mask,3:].mean(axis=0), p_full[mask,3:].std(axis=0)

    p_full[mask,3:] = normalize_pcd(p_full[mask,3:],p_mean[3:], p_std[3:])


    p_part = normalize_pcd(p_part, p_mean, p_std)
    p_mean = torch.tensor(precomp_mean, dtype=torch.float32)
    p_std = torch.tensor(precomp_std, dtype=torch.float32)
    
    # ipdb.set_trace()
    return [p_full, p_mean, p_std, p_part, mask]

#based off of points set to sparse
'''
def rgbd_set_to_sparse(rgbd_pcd, n_full, n_part, resolution): #replace p_full with rgbd
    """
    Converts a point cloud to sparse representation, handling XYZRGB in p_full and p_part.

    Parameters:
    - p_full: Full point cloud (n x 6, XYZRGB)
    - p_part: Partial point cloud (m x 6, XYZRGB)
    - n_full: Number of points to sample for p_full
    - n_part: Number of points to sample for p_part
    - resolution: Placeholder resolution parameter (not used in this implementation)

    Returns:
    - A list containing:
        - p_full (normalized, sampled full point cloud)
        - p_mean (mean of full point cloud)
        - p_std (standard deviation of full point cloud)
        - p_part (normalized, sampled partial point cloud)
    """
    # Ensure both inputs have the same number of features (XYZRGB)
    if p_full.shape[1] != 6 or p_part.shape[1] != 6:
        raise ValueError("Both p_full and p_part must have 6 features (XYZRGB).")

    # Shuffle and sample p_full
    concat_full = int(np.ceil(n_full / p_full.shape[0]))
    p_full = p_full[torch.randperm(p_full.shape[0])]
    p_full = torch.tensor(p_full.repeat(concat_full, 0)[:n_full])

    # Shuffle and sample p_part
    concat_part = int(np.ceil(n_part / p_part.shape[0]))
    p_part = p_part[torch.randperm(p_part.shape[0])]
    p_part = torch.tensor(p_part.repeat(concat_part, 0)[:n_part])

    # Split XYZ and RGB for normalization
    xyz_full = p_full[:, :3]
    rgb_full = p_full[:, 3:]
    xyz_part = p_part[:, :3]
    rgb_part = p_part[:, 3:]

    # Normalize XYZ (mean=0, std=1) and leave RGB as-is
    xyz_mean, xyz_std = xyz_full.mean(dim=0), xyz_full.std(dim=0)
    xyz_full_normalized = normalize_pcd(xyz_full, xyz_mean, xyz_std)
    xyz_part_normalized = normalize_pcd(xyz_part, xyz_mean, xyz_std)

    # Recombine normalized XYZ with original RGB
    p_full_normalized = torch.cat((xyz_full_normalized, rgb_full), dim=1)
    p_part_normalized = torch.cat((xyz_part_normalized, rgb_part), dim=1)

    return [p_full_normalized, xyz_mean, xyz_std, p_part_normalized]



def normalize_pcd(pcd, mean, std):
    """
    Normalize point cloud by subtracting the mean and dividing by the standard deviation.

    Args:
        pcd (torch.Tensor): The input point cloud.
        mean (torch.Tensor): The mean to subtract.
        std (torch.Tensor): The standard deviation to divide.

    Returns:
        torch.Tensor: The normalized point cloud.
    """
    return (pcd - mean) / std

'''
def list_segments_points(p_coord, labels, resolution=0.05):
    # labels are batch_id segs_id and we now concatenate the point index (the same as the coords and feats)
    idx_labels = np.concatenate((labels, np.arange(len(labels))[:,None]), axis=-1)

    # we hash the segs_id to be unique over all the batch and the remove the ground points
    idx_labels = idx_labels[idx_labels[:,0] != -1]

    # sort to "group" together the points belonging to the same segment
    idx_labels = idx_labels[idx_labels[:,0].argsort()]
    idx_labels = np.split(idx_labels[:,1], np.unique(idx_labels[:, 0], return_index=True)[1][1:])

    seg_info = [ point_set_to_sparse(p_coord[seg.astype(int)], resolution) for seg in idx_labels ]

    #import ipdb; ipdb.set_trace()
    #seg_info = point_set_to_sparse(p_coord[idx_labels[20].astype(int)], resolution)

    #p_coord = np.array([[1,0,0,0],[1,0,0,1],
    #                    [1,0,1,0],[1,1,0,0],
    #                    [1,1,1,1],[1,1,1,0],
    #                    [1,1,0,1],[1,0,1,1]]).astype(float)

    #p_feats = np.array([[0,0,0],[0,0,1],
    #                    [0,1,0],[1,0,0],
    #                    [1,1,1],[1,1,0],
    #                    [1,0,1],[0,1,1]]).astype(float)

    #p_feats = p_feats - .5

    #p_coord = ME.utils.batched_coordinates([ torch.from_numpy(p_coord[:,1:]) ], dtype=torch.float32)
    #p_feats = ME.utils.batched_coordinates([ torch.from_numpy(p_feats).float() ], dtype=torch.float32)[:, 1:]

    return seg_info

def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = ME.utils.batched_coordinates(p_coord, dtype=torch.float32)
    p_feats = torch.vstack(p_feats).float()

    if p_label is not None:
        # we batch the segs id to later have unique labels per point
        p_label = ME.utils.batched_coordinates(p_label, device=torch.device('cpu')).numpy()
    
        return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            ), p_label

    return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            )

class SparseSegmentCollation:
    def __init__(self):
        return

    def __call__(self, data):
        # "transpose" the  batch(pt, ptn) to batch(pt), batch(ptn)
        batch = list(zip(*data))
        #import ipdb; ipdb.set_trace()

        return {'pcd_full': torch.stack(batch[0]).float(),
            'mean': torch.stack(batch[1]).float(),
            'std': torch.stack(batch[2]).float(),
            'pcd_part': torch.stack(batch[3]).float(),
        }

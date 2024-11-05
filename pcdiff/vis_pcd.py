import open3d as o3d
import numpy as np
import click
        
                                                                                                                                     
@click.command()
@click.option('--path', '-p', type=str, help='path to pcd')
@click.option('--mesh', '-m', is_flag=True, help='draw mesh')
def main(path, mesh):
    pcd = o3d.io.read_point_cloud(path)

    if not mesh:
        o3d.visualization.draw_geometries([pcd])
    else:
        points = np.array(pcd.points)
        colors = np.array(pcd.colors)
        gt = points[colors[:,1] == 1.]
        pred = points[colors[:,1] < 1.]
        alpha = 0.005

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt)
        mesh_gt = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_gt, alpha)
        mesh_gt.compute_vertex_normals()

        pcd_pred = o3d.geometry.PointCloud()
        pred[:,0] += .1
        pcd_pred.points = o3d.utility.Vector3dVector(pred)
        mesh_pred = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_pred, alpha)
        mesh_pred.compute_vertex_normals()

        o3d.visualization.draw_geometries([mesh_pred, mesh_gt], mesh_show_back_face=True)
                                                                                                                                     
if __name__ == '__main__':
    main()



import open3d as o3d
import numpy as np

def visualize_aligned_point_cloud(points, colors):
    """
    Visualizes an aligned point cloud where only the colored points retain their colors.

    Args:
        points (np.ndarray): An (n, 3) array of point coordinates.
        colors (np.ndarray): An (n, 3) array of RGB values for each point.
    """
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()

    # Set points
    pcd.points = o3d.utility.Vector3dVector(points)

    # Create a mask to apply colors only where valid RGB values exist
    valid_colors = np.all(colors != -1, axis=1)  # Assume -1 indicates no color
    rgb_values = np.zeros_like(colors)  # Initialize an array for RGB values

    # Apply colors only where valid
    rgb_values[valid_colors] = colors[valid_colors]  # Retain original colors
    rgb_values[~valid_colors] = [0, 0, 0]  # Set invalid points to black (or any color of your choice)

    # Set colors to point cloud
    pcd.colors = o3d.utility.Vector3dVector(rgb_values / 255.0)  # Normalize RGB values to [0, 1]

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name="Aligned Point Cloud Visualization")

# Example usage:
# Assuming you have the following arrays from your alignment process
# points: an nx3 numpy array of point coordinates (XYZ)
# colors: an nx3 numpy array of RGB values, with -1 for points without color

# Replace the following with your actual data
points = np.random.rand(1000, 3)  # Example points
colors = np.random.randint(0, 255, (1000, 3))  # Example colors
colors[::2] = -1  # Simulate some points without color

visualize_aligned_point_cloud(points, colors)

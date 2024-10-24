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

import open3d as o3d
import numpy as np
import argparse

def surface_reconstrcution_bpa(args):
    pcd = o3d.io.read_point_cloud(args.file_path)
    a = np.asarray(pcd.points)
    cp = np.mean(a, axis=0)
    f = np.max(np.sqrt(np.sum((a - cp.reshape(1, 3))**2, axis=1)))
    pcd.points = o3d.utility.Vector3dVector((a-cp)/f)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=25))
    # use camera_matrix to orient normals
    camera_matrix = np.asarray([1, 1, -1])
    pcd.orient_normals_towards_camera_location(camera_matrix)
    scale = 1.0 
    radii = [0.005*scale, 0.01*scale, 0.02*scale, 0.04*scale]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    xyz_mesh = mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.io.write_triangle_mesh(args.save_path, rec_mesh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='surface reconstrction arguments')
    parser.add_argument('--file_path', default='xxx.xyz', type=str, help='upsampled point cloud')
    parser.add_argument('--save_path', default='xxx.obj', type=str, help='reconstructed mesh')
    args = parser.parse_args()
    surface_reconstrcution_bpa(args)

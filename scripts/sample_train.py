import os
import numpy as np
from multiprocessing import Pool
import trimesh
import point_cloud_utils as pcu


num_sample = 600000
num_gt = 10000

base_path = 'data/train/scaled_off'
save_path = 'data/train/npz'
os.makedirs(save_path, exist_ok=True)

scaled_off_files = [f for f in os.listdir(base_path) if f.endswith('.off')]
path_list = [os.path.join(base_path, f) for f in scaled_off_files]
idx_list = list(range(len(path_list)))


def multiprocess(func):
    p = Pool(20)
    p.map(func, idx_list)
    p.close()
    p.join()


ratio_list = [0.1, 0.4, 0.5]
std_list = [2.5, 1, 0.15]

def sample(idx):
    boundary_points_list = []
    closest_points_list = []
    mesh = trimesh.load(path_list[idx])

    for i in range(3):    
        ratio = ratio_list[i]
        std = std_list[i]
        
        points = mesh.sample(int(num_sample * ratio))
        noise = np.random.randn(*points.shape) * std
        boundary_points = points + noise

        _, fi, bc = pcu.closest_points_on_mesh(boundary_points, mesh.vertices, mesh.faces)
        closest_points = pcu.interpolate_barycentric_coords(mesh.faces, fi, bc, mesh.vertices)

        boundary_points_list.append(boundary_points)
        closest_points_list.append(closest_points)

    boundary_points_list = np.concatenate(boundary_points_list, axis=0)
    closest_points_list = np.concatenate(closest_points_list, axis=0)
    points_gt = mesh.sample(int(num_gt))

    np.savez(os.path.join(save_path, os.path.splitext(os.path.basename(path_list[idx]))[0] + '.npz'), 
             samples=boundary_points_list.astype(np.float32), 
             closest_points=closest_points_list.astype(np.float32),
             points=points_gt.astype(np.float32))
        
    print(f"Saved sampled data to: {os.path.join(save_path, os.path.splitext(os.path.basename(path_list[idx]))[0] + '.npz')}")


if __name__ == '__main__':
    multiprocess(sample)

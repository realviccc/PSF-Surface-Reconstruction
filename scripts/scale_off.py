import os
import trimesh
from multiprocessing import Pool


base_path = 'PUNet/train'
save_path = 'data/train/scaled_off'
os.makedirs(save_path, exist_ok=True)

off_files = [f for f in os.listdir(base_path) if f.endswith('.off')]
base_path_off = [os.path.join(base_path, f) for f in off_files]
idx_list = list(range(len(base_path_off)))


def scalled_off(idx):
    mesh = trimesh.load(base_path_off[idx])
    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    centers = (mesh.bounds[1] + mesh.bounds[0]) / 2
    mesh.apply_translation(-centers)
    mesh.apply_scale(50 / total_size)    # [50 * 50 * 50]
    mesh.export(os.path.join(save_path, os.path.basename(base_path_off[idx])))

    print(f'Saved scaled model to {os.path.join(save_path, os.path.basename(base_path_off[idx]))}')


def multiprocess(func):
    p = Pool(16)
    p.map(func, idx_list)
    p.close()
    p.join()


if __name__ == '__main__':
    multiprocess(scalled_off)

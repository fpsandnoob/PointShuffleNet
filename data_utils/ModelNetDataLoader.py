import os
import warnings
from multiprocessing import Pool

import h5py
import numpy as np
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def rotate_point_cloud(data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    rotated_data = np.zeros(data.shape, dtype=np.float32)

    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(data.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


# def clusterfps(point, npoint):
#
#     return s.forward(point)


class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel
        self.aug_data = True if split == 'train' else False
        # self.s = ClusterFPS(npoint, iteration=10, num_centers=8, max_neighbors=150, fixed=True, device=torch.device("cuda"))

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
                # point_set = self.s.forward(point_set)
            else:
                point_set = point_set[0:self.npoints, :]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


# class ModelNeth5DataLoader(Dataset):
#     def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_dir=None):
#         self.root = root
#         self.npoints = npoint
#         self.uniform = uniform
#         self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
#
#         self.cat = [line.rstrip() for line in open(self.catfile)]
#         self.classes = dict(zip(self.cat, range(len(self.cat))))
#         self.normal_channel = normal_channel
#         self.split = split
#         # self.sampling_algo = ClusterFPS(npoint, 10, 8, 200, True, device=torch.device("cuda:0"))
#
#         shape_ids = {}
#         shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
#         shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
#
#         assert (split == 'train' or split == 'test')
#         shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
#         # list of (shape_name, shape_txt_file_path) tuple
#         self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
#                          in range(len(shape_ids[split]))]
#         print('The size of %s data is %d' % (split, len(self.datapath)))
#
#         if not os.path.exists(
#                 os.path.join(self.root, "cache_{}_{}_normal_{}.meta".format(split, npoint, normal_channel))):
#             self.cache = False
#             print(f"cache_{split}_{npoint}_normal_{normal_channel}")
#             print(f"Cache missing, generating {split} cache...")
#             self.points, self.labels = self.gene_cache()
#             print("Cache generating Complete!")
#         else:
#             self.cache = True
#             print("Found cache")
#             self.points, self.labels = self.read_cache()
#         self.length = len(self.datapath)
#         del self.datapath
#
#     def __len__(self):
#         return self.length
#
#     def read_file(self, fn):
#         cls = self.classes[fn[0]]
#         cls = np.array([cls]).astype(np.int32)
#         point = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
#         # point = self.sampling_algo.forward(point)
#         point = farthest_point_sample(point, self.npoints)
#         point[:, 0:3] = pc_normalize(point[:, 0:3])
#         if not self.normal_channel:
#             point = point[:, 0:3]
#         return point, cls
#
#     def gene_cache(self):
#         point_sets = np.empty([len(self.datapath), self.npoints, 6 if self.normal_channel else 3])
#         labels = np.empty([len(self.datapath), 1])
#         with Pool(17) as p:
#             data = p.map(self.read_file, self.datapath)
#         for i, (p, l) in enumerate(data):
#             point_sets[i] = p
#             labels[i] = l
#         point_sets = np.asarray(point_sets)
#         labels = np.asarray(labels)
#         h5f = h5py.File(
#             os.path.join(self.root, "cache_{}_{}_normal_{}.h5".format(self.split, self.npoints, self.normal_channel)),
#             'w')
#         h5f.create_dataset("points", data=point_sets)
#         h5f.create_dataset("labels", data=labels)
#         h5f.close()
#         with open(os.path.join(self.root,
#                                "cache_{}_{}_normal_{}.meta".format(self.split, self.npoints, self.normal_channel)),
#                   'w') as f:
#             f.write("200")
#         return point_sets, labels
#
#     def read_cache(self):
#         h5f = h5py.File(
#             os.path.join(self.root, "cache_{}_{}_normal_{}.h5".format(self.split, self.npoints, self.normal_channel)),
#             'r')
#         points = h5f['points'][:]
#         labels = h5f['labels'][:]
#         h5f.close()
#         return points, labels
#
#     def _get_item(self, index):
#         point_set = self.points[index]
#         cls = self.labels[index]
#
#         return point_set.astype(np.float32), cls.astype(np.int32)
#
#     def __getitem__(self, index):
#         return self._get_item(index)

class ModelNeth5DataLoader(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=True, normal_channel=True, cache_dir=None):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel
        self.split = split
        if uniform:
            print("use uniform")
        # self.sampling_algo = ClusterFPS(npoint, 10, 8, 200, True, device=torch.device("cuda:0"))

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if not os.path.exists(
                os.path.join(self.root, "cache_{}_{}_normal_{}_uniform_{}.meta".format(split, npoint, normal_channel,
                                                                                       self.uniform))):
            self.cache = False
            print("Cache missing, generating cache...")
            self.points, self.labels = self.gene_cache()
            print("Cache generating Complete!")
        else:
            self.cache = True
            print("Found cache")
            self.points, self.labels = self.read_cache()
        self.length = len(self.datapath)
        del self.datapath

    def __len__(self):
        return self.length

    def read_file(self, fn):
        cls = self.classes[fn[0]]
        cls = np.array([cls]).astype(np.int32)
        point = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        # point = self.sampling_algo.forward(point)
        if not self.uniform:
            point = farthest_point_sample(point, self.npoints)
        else:
            point = point[:self.npoints, :]
        point[:, 0:3] = pc_normalize(point[:, 0:3])
        if not self.normal_channel:
            point = point[:, 0:3]
        return point, cls

    def gene_cache(self):
        point_sets = np.empty([len(self.datapath), self.npoints, 6 if self.normal_channel else 3])
        labels = np.empty([len(self.datapath), 1])
        with Pool(17) as p:
            data = p.map(self.read_file, self.datapath)
        for i, (p, l) in enumerate(data):
            point_sets[i] = p
            labels[i] = l
        point_sets = np.asarray(point_sets)
        labels = np.asarray(labels)
        h5f = h5py.File(
            os.path.join(self.root,
                         "cache_{}_{}_normal_{}_uniform_{}.h5".format(self.split, self.npoints, self.normal_channel,
                                                                      self.uniform)),
            'w')
        h5f.create_dataset("points", data=point_sets)
        h5f.create_dataset("labels", data=labels)
        h5f.close()
        with open(os.path.join(self.root,
                               "cache_{}_{}_normal_{}_uniform_{}.meta".format(self.split, self.npoints,
                                                                              self.normal_channel, self.uniform)),
                  'w') as f:
            f.write("200")
        return point_sets, labels

    def read_cache(self):
        h5f = h5py.File(
            os.path.join(self.root,
                         "cache_{}_{}_normal_{}_uniform_{}.h5".format(self.split, self.npoints, self.normal_channel,
                                                                      self.uniform)),
            'r')
        points = h5f['points'][:]
        labels = h5f['labels'][:]
        h5f.close()
        return points, labels

    def _get_item(self, index):
        point_set = self.points[index]
        cls = self.labels[index]

        return point_set.astype(np.float32), cls.astype(np.int32)

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNeth5DataLoader('/home/hi2080ti/hlc/data', split='train', uniform=False, normal_channel=True, )
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)

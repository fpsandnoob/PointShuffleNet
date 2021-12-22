import time

import torch
from torch_cluster import fps


def batch_fps_sampling(xyz, npoint):
    """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def fps_torch(dataset, npoint):
    index = batch_fps_sampling(dataset, npoint)
    centers = index_points(dataset, index)
    return centers


device_gpu = torch.device('cuda')
device_cpu = torch.device('cpu')


class ClusterFPS(torch.nn.Module):
    num_centers: int

    def __init__(self, npoint: int, iteration: int, num_centers: int, fixed: bool, device: torch.device,
                 cuda_fps=True):
        super(ClusterFPS, self).__init__()
        self.cuda_fps = cuda_fps
        self.device = device
        self.fixed = fixed
        self.iteration = iteration
        self.npoint = npoint

        self.num_centers: int = num_centers
        self.max_neighbors = npoint // num_centers * 2

    def quick_gpu_tensor(self, dim: list, value: int):
        if self.device == torch.device("cuda"):
            return torch.cuda.FloatTensor(*dim).fill_(value)
        else:
            return torch.FloatTensor(*dim).fill_(value)

    def forward(self, x):
        t1 = time.time()
        _, centers = KMeans(x, K=self.num_centers)
        t2 = time.time()
        neighbor = self.query_neighbor(x, centers)

        if self.cuda_fps:
            new_xyz = self.cuda_batch_fps_sampling(neighbor, npoint=self.npoint // self.num_centers)
            return new_xyz.reshape(neighbor.size(0), -1, 3), centers, t1, t2, time.time()
        else:
            index = self.batch_fps_sampling(neighbor, npoint=self.npoint // self.num_centers)
            new_xyz = self.index_points(neighbor, index)
            return new_xyz.reshape(neighbor.size(0), -1, 3), centers, t1, t2, time.time()
        # new_xyz = self.index_points(neighbor, index)
        # del index

    def batch_fps_sampling(self, xyz, npoint):
        """
            Input:
                xyz: pointcloud data, [B, N, 3]
                npoint: number of samples
            Return:
                centroids: sampled pointcloud index, [B, npoint]
            """
        device = xyz.device
        B, G, N, C = xyz.shape
        # centroids = torch.zeros(B, G, npoint, dtype=torch.long, device=self.device)
        # distance = torch.ones(B, G, N, device=self.device) * 1e10
        # farthest = torch.randint(0, N, (B, G,), dtype=torch.long, device=self.device)
        centroids = self.quick_gpu_tensor([B, G, npoint], 0)
        distance = self.quick_gpu_tensor([B, G, N], 1) * 1e10
        farthest = torch.randint(0, N, (B, G,), dtype=torch.long, device=self.device)
        # batch_indices = torch.arange(G, dtype=torch.long).to(device).unsqueeze(0).repeat(farthest.size(0), 1)
        for i in range(npoint):
            centroids[:, :, i] = farthest
            # centroid = xyz[:, farthest, :].view(B, G, 1, 3)
            centroid = torch.gather(xyz, 2,
                                    farthest.view(farthest.size(0), farthest.size(1), 1, 1).repeat(1, 1, xyz.size(2),
                                                                                                   xyz.size(3)))
            # centroid = torch.gather(xyz, )
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        # print(npoint)
        return centroids

    @staticmethod
    # @torch.jit.script
    def cuda_batch_fps_sampling(xyz, npoint: int):
        """
            Input:
                xyz: pointcloud data, [B, N, 3]
                npoint: number of samples
            Return:
                centroids: sampled pointcloud index, [B, npoint]
            """
        device = xyz.device
        B, G, N, C = xyz.shape
        flatten_xyz = torch.reshape(xyz, (-1, 3))
        batch_idx = torch.arange(0, B * G, device=flatten_xyz.device)
        batch_idx = torch.repeat_interleave(batch_idx, N, dim=0)
        index = fps(flatten_xyz, batch_idx, ratio=npoint / N)
        new_xyz = flatten_xyz[index]
        new_xyz = torch.reshape(new_xyz, (B, -1, 3))
        # print(xyz.shape)

        return new_xyz

    @staticmethod
    def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """

        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        new_xyz = torch.gather(points, 2, idx.long().unsqueeze(-1).repeat(1, 1, 1, 3))
        return new_xyz

    @staticmethod
    @torch.jit.script
    def _query_neighbor(dataset, centers, dist, max_neighbors: int):
        _, indices = torch.sort(dist, dim=1, descending=False)
        neighbor_indices = indices[:, :max_neighbors].permute(0, 2, 1)
        neighbor = torch.gather(dataset.unsqueeze(1).repeat(1, centers.size(1), 1, 1), 2,
                                neighbor_indices.unsqueeze(-1).repeat(1, 1, 1, 3))
        return neighbor

    def query_neighbor(self, dataset: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        dist = self.square_distance(dataset, centers)
        neighbor = self._query_neighbor(dataset, centers, dist, self.max_neighbors)
        return neighbor

    @staticmethod
    @torch.jit.script
    def square_distance(src, dst):
        """
        Calculate Euclid distance between each two points.

        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, C = src.shape
        _, M, C = dst.shape
        # src = LazyTensor(src)
        # dst = LazyTensor(dst)

        x_i = src.view(B, N, 1, C)  # (N, 1, D) samples
        c_j = dst.view(B, 1, M, C)  # (1, K, D) centroids
        D_ij = ((x_i - c_j) ** 2).sum(-1)

        return D_ij

    def cluster(self, dataset: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        centers = self.random_init(dataset)
        codes = self.compute_codes(dataset, centers)
        num_iterations = 0
        while True:
            num_iterations += 1
            centers = self.update_centers(dataset, codes)
            new_codes = self.compute_codes(dataset, centers)
            # Waiting until the clustering stops updating altogether
            # This is too strict in practice
            if num_iterations == self.iteration:
                break
            codes = new_codes
        return centers, codes

    def cluster_cuda(self):
        return NotImplemented

    def update_centers(self, dataset: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        batch_size = dataset.size(0)
        num_points = dataset.size(1)
        dimension = dataset.size(2)
        # centers = torch.zeros(batch_size, self.num_centers, dimension, dtype=torch.float, device=self.device)
        # cnt = torch.zeros(batch_size, self.num_centers, dtype=torch.float, device=self.device)
        centers = self.quick_gpu_tensor([batch_size, self.num_centers, dimension], 0)
        cnt = self.quick_gpu_tensor([batch_size, self.num_centers], 0)
        centers.scatter_add_(1, codes.view(batch_size, -1, 1).expand(-1, -1, dimension), dataset)
        cnt.scatter_add_(1, codes, torch.ones(batch_size, num_points, dtype=torch.float, device=self.device))
        # Avoiding division by zero
        # Not necessary if there are no duplicates among the data points
        cnt = torch.where(cnt > 0.5, cnt,
                          torch.ones(batch_size, self.num_centers, dtype=torch.float, device=self.device))
        centers /= cnt.view(batch_size, -1, 1)
        return centers

    def compute_codes(self, dataset: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        dist = self.square_distance(dataset, centers)
        codes = torch.min(dist, dim=2)[1]
        return codes

    def random_init(self, dataset: torch.Tensor) -> torch.Tensor:
        num_points = dataset.size(1)
        centers = dataset[:, torch.randint(num_points, (self.num_centers,))]
        return centers


def batched_bincount(x, dim: int, max_value: int):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


@torch.jit.script
def KMeans(x: torch.Tensor, K: int = 10, Niter: int = 10):
    """Implements Lloyd's algorithm for the Euclidean metric."""
    B, N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:, :K, :].clone()  # Simplistic initialization for the centroids

    # x_i = LazyTensor(x.view(B, N, 1, D))  # (N, 1, D) samples
    # c_j = LazyTensor(c.view(B, 1, K, D))  # (1, K, D) centroids

    x_i = x.view(B, N, 1, D)  # (N, 1, D) samples
    c_j = c.view(B, 1, K, D)  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    cl = torch.empty(1)
    for i in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=2).long().view(B, -1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(1, cl[:, :, None].repeat(1, 1, D), x)

        # Divide by the number of points per cluster:
        # Ncl = torch.bincount(cl, minlength=K).type_as(c).view(B, K, 1)
        Ncl = batched_bincount(cl, 1, K).view(B, K, 1)
        c /= Ncl  # in-place division to compute the average

    return cl, c


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    dtype = torch.float32 if use_cuda else torch.float64
    B, N, D, K = 2, 10000, 3, 50
    x = 0.7 * torch.randn(B, N, D, dtype=dtype) + 0.3
    cl, c = KMeans(x, K)
    print(cl.shape, c.shape)
# if __name__ == '__main__':
#     import open3d as o3d
#     import timeit
#     def visualization_numpy(data):
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(data)
#         o3d.visualization.draw_geometries([pcd])
#
#     n = 10000
#     d = 100
#     # num_centers = 100
#     # It's (much) better to use 32-bit floats, for the sake of performance
#     device = device_gpu
#     # xyz1 = np.ascontiguousarray(
#     #     np.loadtxt("../data/guitar_0174.txt", delimiter=",", dtype='float32'))[:512, :3]
#     # xyz2 = np.ascontiguousarray(
#     #     np.loadtxt("../data/airplane_0722.txt", delimiter=",", dtype='float32'))[:512, :3]
#     # visualization_numpy(xyz2[np.random.randint(0, 1024, (1024, ))])
#     # dataset = torch.from_numpy(np.array([xyz1, xyz2])).to(device)
#     dataset = torch.from_numpy(np.random.random([1, 1000000, 3]).astype(dtype='float32')).to(device)
#     print('Starting clustering')
#     algo = ClusterFPS(npoint=100000, iteration=100, num_centers=100, fixed=True, device=device)
#
#     %timeit centers = algo.forward(dataset)
#     # %timeit _c = fps_torch(dataset, 100000)
#     #
#     # centers = algo.forward(dataset)
#     # with torch.autograd.profiler.profile(profile_memory=True, use_cuda=True) as prof:
#         # with torch.autograd.profiler.record_function("index_points"):
#         # with torch.no_grad():
#         #     centers = algo.forward(dataset)
#     # print(prof.table())
#     # prof.export_chrome_trace("./torch_cluster_fps")
#     # print(str(torch.cuda.max_memory_allocated("cuda") / 1024 /1024) + 'MB')
#     # #
#     # torch.cuda.reset_max_memory_allocated("cuda")
#
#     # _c = fps_torch(dataset, 128)
#     # with torch.autograd.profiler.profile(profile_memory=True, use_cuda=True) as prof:
#     #     _c = fps_torch(dataset, 128)
#     # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
#     # prof.export_chrome_trace("./torch_fps")
#     # print(str(torch.cuda.max_memory_allocated("cuda") / 1024 /1024) + 'MB')
#
#     # centers = algo.forward(dataset)
#     # _c = fps_torch(dataset, 256)
#     # visualization_numpy(centers[0].cpu().numpy().astype('float32')[1])
#     # visualization_numpy(_c[1].cpu().numpy().astype('float32'))

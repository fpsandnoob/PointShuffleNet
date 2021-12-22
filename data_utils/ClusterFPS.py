import numpy as np
import torch

device_gpu = torch.device('cuda')
device_cpu = torch.device('cpu')


class ClusterFPS(torch.nn.Module):
    def __init__(self, npoint: int, iteration: int, num_centers: int, fixed: bool, device: torch.device):
        super(ClusterFPS, self).__init__()
        self.device = device
        self.fixed = fixed
        self.iteration = iteration
        self.npoint = npoint

        self.num_centers = num_centers
        self.max_neighbors = npoint // num_centers * 2

    def quick_gpu_tensor(self, dim: list, value: int):
        if self.device == torch.device("cuda"):
            return torch.cuda.FloatTensor(*dim).fill_(value)
        else:
            return torch.FloatTensor(*dim).fill_(value)

    def forward(self, x):
        centers, codes = self.cluster(x)
        neighbor = self.query_neighbor(x, centers, codes)
        del codes, x, centers
        index = self.batch_fps_sampling(neighbor, npoint=self.npoint // self.num_centers)
        new_xyz = self.index_points(neighbor, index)
        del index
        return new_xyz.reshape(neighbor.size(0), -1, 3)

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

    def index_points(self, points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """

        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        new_xyz = torch.gather(points, 2, idx.long().unsqueeze(-1).repeat(1, 1, 1, 3))
        return new_xyz

    def query_neighbor(self, dataset: torch.Tensor, centers: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        dist = self.square_distance(dataset, centers)
        # dist.scatter_(-1, code.view(code.size(0), -1, 1), torch.zeros_like(dist))
        _, indices = torch.sort(dist, dim=1, descending=False)
        del dist
        neighbor_indices = indices[:, :self.max_neighbors].permute(0, 2, 1)
        del _, indices
        neighbor = torch.gather(dataset.unsqueeze(1).repeat(1, centers.size(1), 1, 1), 2,
                                neighbor_indices.unsqueeze(-1).repeat(1, 1, 1, 3))
        return neighbor

    @staticmethod
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
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

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


if __name__ == '__main__':
    import open3d as o3d


    def visualization_numpy(data):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data)
        o3d.visualization.draw_geometries([pcd])


    n = 10000
    d = 100

    device = device_gpu
    xyz1 = np.ascontiguousarray(
        np.loadtxt("../data/guitar_0174.txt", delimiter=",", dtype='float32'))[:512, :3]
    xyz2 = np.ascontiguousarray(
        np.loadtxt("../data/bookshelf_0456.txt", delimiter=",", dtype='float32'))[:512, :3]
    dataset = torch.from_numpy(np.array([xyz1, xyz2])).to(device)
    print('Starting clustering')
    algo = ClusterFPS(npoint=256, iteration=10, num_centers=8, fixed=True, device=device)

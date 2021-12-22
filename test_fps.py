import numpy as np
import torch

from models.ClusterFPS import ClusterFPS

num_point = 1000000
downsample_num_point = int(100000)
iteration = 10

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

clusterfps = ClusterFPS(downsample_num_point, 5, 128, False, device).cuda()
test_data = torch.randn(1, num_point, 3).to(device)

# t1 = time.time()
# for i in trange(iteration):
#     a = fps(test_data.squeeze(), ratio=0.1)
# print('CUDA FPS', (time.time() - t1) / iteration)
#
# t1 = time.time()
# clusterfps.cuda_fps = False
# for i in trange(iteration):
#     clusterfps(test_data)
# print('Pytorch ClusterFPS', (time.time() - t1) / iteration)

clusterfps.cuda_fps = True
t1 = []
t2 = []
t3 = []
for i in range(iteration):
    _, __, _t1, _t2, _t3 = clusterfps(test_data)
    t1.append(_t1)
    t2.append(_t2)
    t3.append(_t3)
total_kmeans = list(map(lambda tt1, tt2: tt2 - tt1, t1, t2))
total_fps = list(map(lambda tt1, tt2: tt2 - tt1, t2, t3))
total_ = list(map(lambda tt1, tt2: tt2 - tt1, t1, t3))
print(len(total_))

print(np.min(total_kmeans) * 1000, np.min(total_fps) * 1000, np.min(total_) * 1000)

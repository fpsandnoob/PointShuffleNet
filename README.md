# Pytorch Implementation of PointShuffleNet

This code is for classification testing on ModelNet40

## Environments

Ubuntu 18.04 or Windows10 <br>
Python 3.6 or above <br>
CUDA 10.2 <br>
Pytorch 1.4 or above

## Package

Pytorch <br>
torchsummary<br>
tqdm<br>
prefetch_generator<br>
h5py<br>

### Data Preparation

You need to download the test data file
from [GoogleDrive](https://drive.google.com/file/d/15RslcbHPfNCC18aHAZxKkRQLBo1OfDUo/view?usp=sharing) and put it in
the `data` directory as `data/cache_test_1024_normal_True.h5`. The data is generated
from [ModelNet](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and downsampled to 1024 points
with normal vectors. The details of generating is in class `ModelNeth5DataLoader` of `data_utils/ModelNetDataLoader.py`.

### Model

The default model in `test_cls.py` is `models/PN2_cls_ssg_shuffle_info.py` which is our ***PointShuffleNet*** with ***HER*** and ***LMIR***. For stable evaluation result, we don't use ***ClusterFPS*** in default model. But you can try it
by changing the `use_cluster` parameter in `models/PN2_cls_ssg_shuffle_info.py`.
<br>
The code of ***HER*** and ***LMIR*** is in `models/ShufflePointNet_util.py` line 61.
<br>
The code of ***ClusterFPS***  is in `models/ClusterFPS.py`

### Run

```
python test_cls.py
```

### Performance

You should get the result as

```
Test Instance Accuracy: 0.931048, Class Accuracy: 0.910952
```


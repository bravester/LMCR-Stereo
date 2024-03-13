# LMCR-Stereo
 Implementation of the LMCR-Stereo model. There are extra codes for our end, you can ignore them.

# Dataset
 Please download the dataset (SceneFlow, Middleburry, etc.) on your side, and change the method in stereo_datasets.py in \data\data_load_datasetall_aug_all folder.
```python
class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='/SceneFlow_all',...):
```
change the root to the path where you store the dataset. There are different methods for different dataset.

# Training
 Modify the configurations in train_stereo_re1-loss-N-net_feature_2to1_add_pre_zeroresize_grade_dw32_16_8_loop_end_sample_datasetall_aug_all.py in \train\pre_train_model_sceneflow folder.

# Licences:
Since the project include codes with 2 different licences, we include both of them.

- CREStereo (Apache License 2.0): https://github.com/megvii-research/CREStereo/blob/master/LICENSE
- RAFT (BSD 3-Clause):https://github.com/princeton-vl/RAFT/blob/master/LICENSE
- LoFTR (Apache License 2.0):https://github.com/zju3dv/LoFTR/blob/master/LICENSE

# References:
- CREStereo: https://github.com/megvii-research/CREStereo
- CREStereo-Pytorch: https://github.com/ibaiGorordo/CREStereo-Pytorch

# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp

from data.data_load_datasetall_aug_all.data_utils import frame_utils
# from .core.utils import frame_utils
from data.data_load_datasetall_aug_all.data_utils.augmentor import FlowAugmentor, SparseFlowAugmentor,Augmentor,SparseAugmentor


class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None,augsize='I',datatype = 'SceneFlow',test=False):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if datatype == 'SceneFlow':
            self.max_disp = 256
        if datatype == 'kitti':
            self.max_disp = 230
        if datatype == 'middlebury':
            self.max_disp = 660
        elif datatype == 'eth3d':
            self.max_disp = 62
        if augsize in 'IMR':
            if augsize =="I":
                self.height=384
                self.width=512
            if augsize == 'M':
                self.height=480
                self.width= 704
            if augsize == "MR":
                self.height = random.choice([448,480,512])
                self.width = random.choice([672,704,736])
            if augsize == "IR":
                self.height = random.choice([352, 384, 416])
                self.width = random.choice([480, 512, 544])
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseAugmentor(
                    image_height=self.height,
                    image_width=self.width,
                    max_disp=self.max_disp,
                    scale_min=0.6,
                    scale_max=1.0,
                    seed=0,
                )
            else:
                self.augmentor = Augmentor(
                    image_height=self.height,
                    image_width=self.width,
                    max_disp=self.max_disp,
                    scale_min=0.6,
                    scale_max=1.0,
                    seed=0,
                )

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader        

        self.is_test = test
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            # return self.extra_info[index],img1, img2
            return self.image_list[index] + [self.image_list[index]], img1, img2, img1,img2

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disparity_list[index])
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        disp = np.array(disp).astype(np.float32)

        flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        # valid = torch.from_numpy(valid).permute(2, 0, 1).float()

        if self.sparse:
            # valid = torch.from_numpy(valid)
            # valid=torch.squeeze(valid)
            valid = (flow[0].abs() > 0)
        else:
            valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)
        a = flow[0].abs()
        b = flow[0].abs()
        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        flow = flow[:1]
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()

        # return {
        #     'image_list': self.image_list[index] + [self.disparity_list[index]],
        #     "left": img1,
        #     "right": img2,
        #     "disparity": flow,
        #     "mask": valid.float()
        # }


    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='/SceneFlow_all', dstype='frames_cleanpass',sum_sample=400, worklog=logging,things_test=False,randon_sample=False):
        super(SceneFlowDatasets, self).__init__(aug_params)
        self.root = root
        self.dstype = dstype
        self.randon_sample = randon_sample
        self.sample = 0.1
        self.worklog = worklog
        self.sum_sample = sum_sample


        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa()
            self._add_driving()

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'FlyingThings3d')
        left_images = sorted(glob(osp.join(root, self.dstype, split, '*/*/left/*.png')))
        right_images = [im.replace('left', 'right') for im in left_images]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

        # Choose a random subset of 400 images for validation
        state = np.random.get_state()
        np.random.seed(1000)
        val_idxs = set(np.random.permutation(len(left_images))[:self.sum_sample])
        np.random.set_state(state)

        if self.randon_sample == False:
            state = np.random.get_state()
            np.random.seed(0)
            train_idxs = set(np.random.permutation(len(left_images)))
            np.random.set_state(state)
        else:
            state = np.random.get_state()
            sum = int(len(left_images) * self.sample)
            np.random.seed(len(left_images))
            train_idxs = set(np.random.permutation(len(left_images))[:sum])
            np.random.set_state(state)

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (split == 'TEST' and idx in val_idxs) or (split == 'TRAIN' and idx in train_idxs):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        a=len(self.disparity_list) - original_length
        (self.worklog).info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Monkaa')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        (self.worklog).info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")


    def _add_driving(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Driving')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        (self.worklog).info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")

class SceneFlowDatasets_testall(StereoDataset):
    def __init__(self, aug_params=None, root='/root/autodl-tmp/SceneFlow_all', dstype='frames_cleanpass', things_test=False):
        super(SceneFlowDatasets_testall, self).__init__(aug_params)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa()
            self._add_driving()

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'FlyingThings3d')
        left_images = sorted(glob(osp.join(root, self.dstype, split, '*/*/left/*.png')))
        right_images = [im.replace('left', 'right') for im in left_images]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

        # Choose a random subset of 400 images for validation
        # state = np.random.get_state()
        # np.random.seed(1000)
        # val_idxs = set(np.random.permutation(len(left_images))[:400])
        # np.random.set_state(state)

        state = np.random.get_state()
        np.random.seed(0)
        val_idxs = set(np.random.permutation(len(left_images)))
        np.random.set_state(state)

        # val_idxs = set(len(left_images))

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        a=len(self.disparity_list) - original_length
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Monkaa')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")


    def _add_driving(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Driving')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")



class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='/ETH3D', split='training'):
        super(ETH3D, self).__init__(aug_params, sparse=True,datatype = 'eth3d')

        image1_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im0.png')) )
        image2_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im1.png')) )
        disp_list = sorted( glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm')) ) if split == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')]*len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class ETH3D_test(StereoDataset):
    def __init__(self, aug_params=None, root='/ETH3D', split='test'):
        super(ETH3D_test, self).__init__(aug_params, sparse=True,datatype = 'eth3d',test=True)

        image1_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im0.png')) )
        image2_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im1.png')) )
        # disp_list = sorted( glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm')) ) if split == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')]*len(image1_list)

        for img1, img2 in zip(image1_list, image2_list):
            self.image_list += [ [img1, img2] ]
            # self.disparity_list += [ disp ]
class Shiyan_test(StereoDataset):
    def __init__(self, aug_params=None, root='/ETH3D', split='test'):
        super(Shiyan_test, self).__init__(aug_params, sparse=True,datatype = 'eth3d',test=True)

        image1_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im0.png')) )
        image2_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im1.png')) )
        # disp_list = sorted( glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm')) ) if split == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')]*len(image1_list)

        for img1, img2 in zip(image1_list, image2_list):
            self.image_list += [ [img1, img2] ]
            # self.disparity_list += [ disp ]

class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='/SintelStereo'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo)

        image1_list = sorted( glob(osp.join(root, 'training/*_left/*/frame_*.png')) )
        image2_list = sorted( glob(osp.join(root, 'training/*_right/*/frame_*.png')) )
        disp_list = sorted( glob(osp.join(root, 'training/disparities/*/frame_*.png')) ) * 2

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='/FallingThings'):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)

        with open(os.path.join(root, 'filenames.txt'), 'r') as f:
            filenames = sorted(f.read().splitlines())

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('left.jpg', 'right.jpg')) for e in filenames]
        disp_list = [osp.join(root, e.replace('left.jpg', 'left.depth.png')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='datasets', keywords=[]):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)

        with open(os.path.join(root, 'tartanair_filenames.txt'), 'r') as f:
            filenames = sorted(list(filter(lambda s: 'seasonsforest_winter/Easy' not in s, f.read().splitlines())))
            for kw in keywords:
                filenames = sorted(list(filter(lambda s: kw in s.lower(), filenames)))

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('_left', '_right')) for e in filenames]
        disp_list = [osp.join(root, e.replace('image_left', 'depth_left').replace('left.png', 'left_depth.npy')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]


class CRESetereo2(StereoDataset):
    def __init__(self, aug_params=None, root='/FallingThings'):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)

        with open(os.path.join(root, 'filenames.txt'), 'r') as f:
            filenames = sorted(f.read().splitlines())

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('left.jpg', 'right.jpg')) for e in filenames]
        disp_list = [osp.join(root, e.replace('left.jpg', 'left.depth.png')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class CRES(StereoDataset):
    def __init__(self, aug_params=None, root='/cres_result'):
        super(CRES, self).__init__(aug_params, sparse=False, reader=frame_utils.readDispCRES)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, '*/*_left.jpg')))
        image2_list = sorted(glob(os.path.join(root, '*/*_right.jpg')))
        disp_list = sorted(glob(os.path.join(root,'*/*_left.disp.png')))
        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='/KITTI', image_set='training'):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, image_set, 'image_2/*_10.png')))
        image2_list = sorted(glob(os.path.join(root, image_set, 'image_3/*_10.png')))
        disp_list = sorted(glob(os.path.join(root, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ_0/000085_10.png')]*len(image1_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='/Middlebury', split='F'):
        super(Middlebury, self).__init__(aug_params, sparse=False, reader=frame_utils.readDispMiddlebury,datatype = 'middlebury')
        assert os.path.exists(root)
        assert split in "FHQ"
        lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/trainingF/*"))))
        lines = list(filter(lambda p: any(s in p.split('/') for s in Path(os.path.join(root, "official_train_F.txt")).read_text().splitlines()), lines))
        image1_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im0.png') for name in lines])
        image2_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im1.png') for name in lines])
        disp_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/disp0GT.pfm') for name in lines])

        assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class Middlebury_test(StereoDataset):
    def __init__(self, aug_params=None, root='/Middlebury', split='F'):
        super(Middlebury_test, self).__init__(aug_params, sparse=False, reader=frame_utils.readDispMiddlebury,datatype = 'middlebury',test=True)
        assert os.path.exists(root)
        assert split in "FHQ"
        lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/testF/*"))))
        lines = list(filter(lambda p: any(s in p.split('/') for s in Path(os.path.join(root, "MiddEval3","official_test_F.txt")).read_text().splitlines()), lines))
        image1_list = sorted([os.path.join(root, "MiddEval3", f'test{split}', f'{name}/im0.png') for name in lines])
        image2_list = sorted([os.path.join(root, "MiddEval3", f'test{split}', f'{name}/im1.png') for name in lines])
        # disp_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/disp0GT.pfm') for name in lines])

        assert len(image1_list) == len(image2_list)  > 0, [image1_list, split]
        for img1, img2 in zip(image1_list, image2_list):
            self.image_list += [ [img1, img2] ]



class Middlebury_2021(StereoDataset):
    def __init__(self, aug_params=None, root='/Middlebury', split='F'):
        super(Middlebury_2021, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury,datatype = 'middlebury')
        assert os.path.exists(root)
        assert split in "FHQ"
        lines = list(map(osp.basename, glob(os.path.join(root, "data/*"))))
        lines = list(filter(lambda p: any(s in p.split('/') for s in Path(os.path.join(root, "data/data_2021.txt")).read_text().splitlines()), lines))
        image1_list = sorted([os.path.join(root, f'data', f'{name}/im0.png') for name in lines])
        image2_list = sorted([os.path.join(root,  f'data', f'{name}/im1.png') for name in lines])
        disp_list = sorted([os.path.join(root,  f'data', f'{name}/disp0GT.pfm') for name in lines])

        assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class Middlebury_extra(StereoDataset):
    def __init__(self, aug_params=None, root='/Middlebury', split='F',augsize="M"):
        super(Middlebury_extra, self).__init__(aug_params, sparse=False, reader=frame_utils.readDispMiddlebury_extra,augsize=augsize,datatype = 'middlebury')
        assert os.path.exists(root)
        assert split in "FHQ"
        lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/imperfect/*"))))
        lines = list(filter(lambda p: any(s in p.split('/') for s in Path(os.path.join(root, "official_train_extra.txt")).read_text().splitlines()), lines))
        image1_list = sorted([os.path.join(root, "MiddEval3", 'imperfect', f'{name}/im0.png') for name in lines])
        image2_list = sorted([os.path.join(root,"MiddEval3", 'imperfect', f'{name}/im1.png') for name in lines])
        disp_list = sorted([os.path.join(root,"MiddEval3", f'imperfect', f'{name}/disp0.pfm') for name in lines])

        assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class Middlebury_extra_r(StereoDataset):
    def __init__(self, aug_params=None, root='/Middlebury', split='F',augsize="I",im0='im0.png',im1='im1.png'):
        super(Middlebury_extra_r, self).__init__(aug_params, sparse=False, reader=frame_utils.readDispMiddlebury_extra,augsize=augsize,datatype = 'middlebury')
        assert os.path.exists(root)
        assert split in "FHQ"
        lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/imperfect/*"))))
        lines = list(filter(lambda p: any(s in p.split('/') for s in Path(os.path.join(root, "official_train_extra.txt")).read_text().splitlines()), lines))
        image1_list = sorted([os.path.join(root, "MiddEval3", 'imperfect', f'{name}/{im0}') for name in lines])
        image2_list = sorted([os.path.join(root,"MiddEval3", 'imperfect', f'{name}/{im1}') for name in lines])
        disp_list = sorted([os.path.join(root,"MiddEval3", f'imperfect', f'{name}/disp0.pfm') for name in lines])

        assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class Middlebury_trainQ(StereoDataset):
    def __init__(self, aug_params=None, root='/Middlebury', split='Q'):
        super(Middlebury_trainQ, self).__init__(aug_params, sparse=False, reader=frame_utils.readDispMiddlebury,datatype = 'middlebury')
        assert os.path.exists(root)
        assert split in "FHQ"
        lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/trainingQ/*"))))
        lines = list(filter(lambda p: any(s in p.split('/') for s in Path(os.path.join(root, "official_train_Q.txt")).read_text().splitlines()), lines))
        image1_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im0.png') for name in lines])
        image2_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im1.png') for name in lines])
        disp_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/disp0GT.pfm') for name in lines])

        assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

  
def fetch_dataloader(args,worklog):
    """ Create the data loader for the corresponding trainign set """
    # 为相应的trainign集合创建数据加载器

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None
    for dataset_name in args.train_datasets:
        if re.compile("middlebury_.*").fullmatch(dataset_name):
            new_dataset = Middlebury(aug_params, split=dataset_name.replace('middlebury_',''))
        elif dataset_name == 'sceneflow':
            # clean_dataset = SceneFlowDatasets(aug_params, dstype='frames_cleanpass')
            final_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
            # new_dataset = (clean_dataset*4) + (final_dataset*4)
            new_dataset = final_dataset
            worklog.info(f"Adding {len(new_dataset)} samples from SceneFlow")
        elif dataset_name == 'sceneflow_f+c':
            clean_dataset = SceneFlowDatasets(aug_params, dstype='frames_cleanpass',worklog=worklog)
            final_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass',worklog=worklog)
            # new_dataset = (clean_dataset*4) + (final_dataset*4)
            new_dataset = final_dataset + clean_dataset
            worklog.info(f"Adding {len(new_dataset)} samples from SceneFlow")

        elif dataset_name == 'sceneflow_f+c+mid':
            clean_dataset = SceneFlowDatasets(aug_params, dstype='frames_cleanpass',worklog=worklog)
            final_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass',worklog=worklog)
            # new_dataset = (clean_dataset*4) + (final_dataset*4)
            new_dataset_fc = final_dataset + clean_dataset
            worklog.info(f"Adding {len(new_dataset_fc)} samples from SceneFlow")
            middlebury_train_extra_F = Middlebury_extra_r(aug_params, split='F', im0='im0.png', im1='im1.png')
            worklog.info(f"Adding {len(middlebury_train_extra_F)} samples from middlebury_train_F")
            k = 616
            new_dataset_mid = middlebury_train_extra_F * k
            worklog.info(f"Adding {k} samples from middlebury Magnification")
            worklog.info(f"Adding {len(new_dataset_mid)} samples from middlebury ")
            new_dataset = new_dataset_fc + new_dataset_mid
            # new_dataset = new_dataset_fc
            worklog.info(f"Adding {len(new_dataset)} samples from fc+middlebury ")

        elif dataset_name == "ETH3D":
            new_dataset_cres = CRES(aug_params)
            worklog.info(f"Adding {len(new_dataset_cres)} samples from CRES")
            eth3d_train_extra_F = ETH3D(aug_params)
            worklog.info(f"Adding {len(eth3d_train_extra_F)} samples from middlebury_train_F")
            k = 200
            # new_dataset_eth3d = eth3d_train_extra_F * k
            # worklog.info(f"Adding {k} samples from ETH3D Magnification")
            # worklog.info(f"Adding {len(new_dataset_eth3d)} samples from ETH3D ")
            new_dataset = new_dataset_cres
            # new_dataset = new_dataset_fc
            worklog.info(f"Adding {len(new_dataset)} samples from CRES+eth3d ")

        elif dataset_name == "CRES+ETH3D":
            new_dataset_cres = CRES(aug_params)
            worklog.info(f"Adding {len(new_dataset_cres)} samples from CRES")
            eth3d_train_extra_F = ETH3D(aug_params)
            worklog.info(f"Adding {len(eth3d_train_extra_F)} samples from middlebury_train_F")
            k = 144
            new_dataset_eth3d = eth3d_train_extra_F * k
            worklog.info(f"Adding {k} samples from ETH3D Magnification")
            worklog.info(f"Adding {len(new_dataset_eth3d)} samples from ETH3D ")
            new_dataset = new_dataset_cres + new_dataset_eth3d
            # new_dataset = new_dataset_fc
            worklog.info(f"Adding {len(new_dataset)} samples from CRES+eth3d ")


        elif dataset_name == 'CRES+mid':
            new_dataset_cres=CRES(aug_params)
            worklog.info(f"Adding {len(new_dataset_cres)} samples from CRES")
            middlebury_train_extra_F = Middlebury_extra_r(aug_params, split='F', im0='im0.png', im1='im1.png')
            worklog.info(f"Adding {len(middlebury_train_extra_F)} samples from middlebury_train_F")
            # k = 1522
            k = 4
            new_dataset_mid = middlebury_train_extra_F * k
            worklog.info(f"Adding {k} samples from middlebury Magnification")
            worklog.info(f"Adding {len(new_dataset_mid)} samples from middlebury ")
            # new_dataset =new_dataset_cres + new_dataset_mid
            new_dataset = new_dataset_mid
            worklog.info(f"Adding {len(new_dataset)} samples from CRES+middlebury ")

        elif dataset_name == 'kitti2015':
            k = 200
            new_dataset = KITTI(aug_params) * k
            worklog.info(f"Adding {k} samples from middlebury Magnification")
            worklog.info(f"Adding {len(new_dataset)} samples from KITTI")
        elif dataset_name == 'sintel_stereo':
            new_dataset = SintelStereo(aug_params)*140
            worklog.info(f"Adding {len(new_dataset)} samples from Sintel Stereo")
        elif dataset_name == 'falling_things':
            new_dataset = FallingThings(aug_params)*5
            worklog.info(f"Adding {len(new_dataset)} samples from FallingThings")
        elif dataset_name.startswith('tartan_air'):
            new_dataset = TartanAir(aug_params, keywords=dataset_name.split('_')[2:])
            logging.info(f"Adding {len(new_dataset)} samples from Tartain Air")
        elif dataset_name == 'SF+Mid':
            # final_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
            # logging.info(f"Adding {len(final_dataset)} samples from SceneFlow")
            # new_dataset=final_dataset
            middlebury_train_F = Middlebury(aug_params,split='F')
            worklog.info(f"Adding {len( middlebury_train_F)} samples from middlebury_train_F")
            k=4
            new_dataset = middlebury_train_F * k
            # new_dataset = final_dataset + middlebury_train_Q * k + middlebury_extra * k
            # new_dataset=middlebury_train_Q*k
            worklog.info(f"Adding {k} samples from middlebury Magnification")
            worklog.info(f"Adding {len( new_dataset)} samples from middlebury ")
        elif dataset_name == 'SF+Mid_extra':
            middlebury_train_extra_F = Middlebury_extra_r(aug_params,split='F',im0='im0.png',im1='im1.png')
            worklog.info(f"Adding {len( middlebury_train_extra_F)} samples from middlebury_train_F")
            k= 200
            new_dataset = middlebury_train_extra_F * k
            # new_dataset = final_dataset + middlebury_train_Q * k + middlebury_extra * k
            # new_dataset=middlebury_train_Q*k
            worklog.info(f"Adding {k} samples from middlebury Magnification")
            worklog.info(f"Adding {len( new_dataset)} samples from middlebury ")
        elif dataset_name == 'SF_fc+Mid_extra':
            clean_dataset = SceneFlowDatasets(aug_params, dstype='frames_cleanpass',worklog=worklog)
            final_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass',worklog=worklog)
            # new_dataset = (clean_dataset*4) + (final_dataset*4)
            new_dataset_SF = final_dataset + clean_dataset
            worklog.info(f"Adding {len(new_dataset_SF)} samples from SceneFlow")
            middlebury_train_extra_F = Middlebury_extra_r(aug_params,split='F',im0='im0.png',im1='im1.png')
            worklog.info(f"Adding {len( middlebury_train_extra_F)} samples from middlebury_train_F")
            k= 308
            new_dataset_Mid = middlebury_train_extra_F * k
            # new_dataset = final_dataset + middlebury_train_Q * k + middlebury_extra * k
            # new_dataset=middlebury_train_Q*k
            worklog.info(f"Adding {k} samples from middlebury Magnification")
            worklog.info(f"Adding {len( new_dataset_Mid)} samples from middlebury ")
            new_dataset=new_dataset_SF+new_dataset_Mid
            worklog.info(f"Adding {len(new_dataset_Mid)} samples from SF_fc+Mid ")

        elif dataset_name == 'SF+Mid_extra_r':
            middlebury_train_extra_F_01 = Middlebury_extra_r(aug_params,split='F',im0='im0.png',im1='im1.png')
            middlebury_train_extra_F_0E = Middlebury_extra_r(aug_params, split='F', im0='im0.png', im1='im1E.png')
            middlebury_train_extra_F_0L = Middlebury_extra_r(aug_params, split='F', im0='im0.png', im1='im1L.png')
            logging.info(f"Adding {len( middlebury_train_extra_F_01)} samples from middlebury_train_F_01")
            logging.info(f"Adding {len(middlebury_train_extra_F_0E)} samples from middlebury_train_F_0E")
            logging.info(f"Adding {len(middlebury_train_extra_F_0L)} samples from middlebury_train_F_0L")
            k= 200
            new_dataset = middlebury_train_extra_F_01 * k+middlebury_train_extra_F_0E * k+middlebury_train_extra_F_0L * k
            # new_dataset = final_dataset + middlebury_train_Q * k + middlebury_extra * k
            # new_dataset=middlebury_train_Q*k
            logging.info(f"Adding {k} samples from middlebury Magnification")
            logging.info(f"Adding {len( new_dataset)} samples from middlebury ")
        elif dataset_name == 'SF+Mid_extra_rk':
            middlebury_train_extra_F_01 = Middlebury_extra_r(aug_params,split='F',im0='im0.png',im1='im1.png')
            middlebury_train_extra_F_0E = Middlebury_extra_r(aug_params, split='F', im0='im0.png', im1='im1E.png')
            middlebury_train_extra_F_0L = Middlebury_extra_r(aug_params, split='F', im0='im0.png', im1='im1L.png')
            logging.info(f"Adding {len( middlebury_train_extra_F_01)} samples from middlebury_train_F_01")
            logging.info(f"Adding {len(middlebury_train_extra_F_0E)} samples from middlebury_train_F_0E")
            logging.info(f"Adding {len(middlebury_train_extra_F_0L)} samples from middlebury_train_F_0L")
            k1=100
            k2=50
            k3=50
            new_dataset = middlebury_train_extra_F_01 * k1+middlebury_train_extra_F_0E * k2+middlebury_train_extra_F_0L * k3
            # new_dataset = final_dataset + middlebury_train_Q * k + middlebury_extra * k
            # new_dataset=middlebury_train_Q*k
            logging.info(f"Adding k1:{k1},k2:{k2},k3:{k3} samples from middlebury Magnification")
            logging.info(f"Adding {len( new_dataset)} samples from middlebury ")

        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    # train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
    #     pin_memory=True, shuffle=False, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=True)

    worklog.info('Training with %d image pairs' % len(train_dataset))
    return train_dataset


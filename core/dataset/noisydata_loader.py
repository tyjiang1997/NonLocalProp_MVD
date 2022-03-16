import imp
import cv2
import numpy as np
from .data_loader import SequenceFolder, load_as_float
from copy import deepcopy as dp
class NoisySequenceFolder(SequenceFolder):


    def __getitem__(self, index):

        sample = dp(self.samples[index])
        depth_keys = ['tgt_depth', 'ref_depths']

        if 'train' in self.ttype:
            for key in depth_keys:
                if isinstance(sample[key], list):
                    sample[key] = [item.replace('depth', 'depth_noisy') for item in sample[key]]
                else:
                    sample[key] = sample[key].replace('depth', 'depth_noisy')
                
        # from pdb import set_trace; set_trace()
        tgt_img = load_as_float(sample['tgt'])
        if 'test' in self.ttype:
            tgt_depth = cv2.imread(sample['tgt_depth'],-1).astype(np.float32) / 1000.0
            tgt_normal = np.tile(np.expand_dims(np.ones_like(tgt_depth), -1), (1,1,3))
        else:
            tgt_depth = np.load(sample['tgt_depth']).astype(np.float32) / 1000.0
            tgt_normal = np.load(sample['tgt_normal']).astype(np.float32)
            tgt_normal = 1.0 - tgt_normal * 2.0 # [-1, 1]
            tgt_normal[:,:,2] = np.abs(tgt_normal[:,:,2]) * -1.0

        ref_poses = sample['ref_poses']

        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if 'test' in self.ttype:
            ref_depths = [cv2.imread(depth_img,-1).astype(np.float32)/1000.0 for depth_img in sample['ref_depths']]
        else:
            ref_depths = [np.load(depth_img).astype(np.float32)/1000.0 for depth_img in sample['ref_depths']]

        if self.transform is not None:
            imgs, depths, normals, intrinsics = self.transform([tgt_img] + ref_imgs, [tgt_depth] + ref_depths, [tgt_normal], np.copy(sample['intrinsics']))
            tgt_img = imgs[0]	 
            tgt_depth = depths[0]
            tgt_normal = normals[0]
            ref_imgs = imgs[1:]
            ref_depths = depths[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        intrinsics_inv = np.linalg.inv(intrinsics)
        tgt_id = sample['tgt'].split('/')[-1][:4]
        return tgt_img, ref_imgs, tgt_normal, ref_poses, intrinsics, intrinsics_inv, tgt_depth, ref_depths, tgt_id

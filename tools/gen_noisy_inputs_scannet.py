from genericpath import exists
import imp
import os
from glob import glob
import cv2
import numpy as np
import copy 
from tqdm import tqdm
import sys

region_nums = 20
random_polygon_num = 10
radius_range = [20, 50]

class Gen_inputs():
    def __init__(self, root_path, save_pth) -> None:
        self.depth_path = root_path
        self.deoth_list = self.include_depth_list()
        self.save_path = save_pth
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

        copy_dep_list = copy.deepcopy(self.deoth_list)
        np.random.shuffle(copy_dep_list)
        self.copy_dep_list = copy_dep_list
        
    def include_depth_list(self,):
        
        list_dep = glob( self.depth_path + '*')
       
        return list_dep

    def get_depth_from_file(self, path):
        dep = np.load(path).astype(np.float32) 
        # from pdb import set_trace; set_trace()
        # dep =  np.array(read_pfm(path)[0], dtype=np.float32)
        # print(path, dep.shape)
        return dep

    def random_polygon(self, dep, region_nums=20,random_polygon_num=10, dep_cp=None, plus_rule='other_ins'):

        def uniform_random(left, right, size=None):
            rand_nums = (right - left) * np.random.random(size) + left
            return rand_nums

        def random_single_plogon(edge_num, center, radius_range):
            angles = uniform_random(0, 2 * np.pi, edge_num)
            angles = np.sort(angles)
            random_radius = uniform_random(radius_range[0], radius_range[1], edge_num)
            x = np.cos(angles) * random_radius
            y = np.sin(angles) * random_radius
            x = np.expand_dims(x, 1)
            y = np.expand_dims(y, 1)
            points = np.concatenate([x, y], axis=1)
            points += np.array(center)
            points = np.round(points).astype(np.int32)
            return points

        def draw_polygon(image_size, points, color):
            image = np.zeros(image_size, dtype=np.uint8)
            if type(points) is np.ndarray and points.ndim == 2:
                image = cv2.fillPoly(image, [points], color)
            else:
                image = cv2.fillPoly(image, points, color)
            return image

        h, w = dep.shape
        mask = np.ones([h,w]).astype(np.float)
        plus_mask = np.zeros([h,w]).astype(np.float)
        for i in range(region_nums):
            edge_num = np.random.randint(3, 10, 1)
            center_x = np.random.randint(0, w-1, 1)
            center_y = np.random.randint(0, h-1, 1)
            center = np.concatenate([center_x, center_y])
            radius = np.random.randint(radius_range[0], radius_range[1], 2)
            polygon_ver = random_single_plogon(edge_num, center, radius)
            image1 = draw_polygon((h, w), polygon_ver, (255, 255, 255))
            if i < random_polygon_num:
                mask[image1 == 255] = 0 # 直接丢弃
            else:
                mask_tmp = image1 == 255
                coordsx, coodsy = mask_tmp.nonzero()
                coods = np.stack([coordsx, coodsy]).transpose(1,0)
                np.random.shuffle(coods)
                # 选择一半作为透射噪声
                coods_len_noise = int(len(coods) / 2)
                coods = coods[:coods_len_noise]
                if plus_rule=='random':
                    # 透射参数为a + beta 
                    dep_min, dep_max = dep.min(), dep.max()
                    alpha = np.random.uniform(1,2.5,1) # 1 - 2.5
                    beta = np.random.uniform(dep_min, dep_min) * 0.1
                    mask[coods[:,0], coods[:,1]] = alpha
                    plus_mask[coods[:,0], coods[:,1]] = beta
                elif plus_rule == 'other_ins':
                    assert not dep_cp is None, 'None is not right!'
                    try:
                        mask[coods[:,0], coods[:,1]] = 1
                        plus_mask[coods[:,0], coods[:,1]] = dep_cp[coods[:,0], coods[:,1]]
                    except:
                        from pdb import set_trace; set_trace()
                    
                    
            # from pdb import set_trace; set_trace()
            dep  = dep * mask + plus_mask

        return dep

    def save_as_txt(self, dep, path):
        # from pdb import set_trace; set_trace()
        # save_path = os.path.join(self.save_path, path.split('/')[-1])
        scene, path_name =  path.split('/')[-2:]
        save_dir =  os.path.join(self.save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, path_name)
        np.save(save_path, dep.astype(np.float32))


    def gen(self, methods={
        'random_polygon':True,
        'random_region_penetrate':True},
        region_nums = 20,
    ):
        for i, (path, path_cp) in tqdm(enumerate(zip(self.deoth_list, self.copy_dep_list))):
            dep_inial = self.get_depth_from_file(path)
            dep_cp = self.get_depth_from_file(path_cp)
            
            dep = self.random_polygon(dep_inial, region_nums=region_nums, random_polygon_num=random_polygon_num, dep_cp=dep_cp)
            self.save_as_txt(dep, path)
            # max_ = dep_inial.max()
            # dep = dep * 255 / (max_ + 1e-8)
            # id =  path.split('/')[-1]
            # dep_inial = dep_inial * 255 / (max_ + 1e-8)
            # cv2.imwrite(f'/home/jty/mvs/idn-solver/tools/noisy/{id}.png', dep)
            # cv2.imwrite(f'/home/jty/mvs/idn-solver/tools/noisy/{id}_inital.png', dep_inial)
            # from pdb import set_trace; set_trace()

           

if __name__ == '__main__':

    valid_scenes = ['scene0000_00', 'scene0000_01']
    root_pth = '/6t/jty/scannet/scannet_nas/train/'
    dirs = os.listdir(root_pth)
    for dir_ in dirs:
        if dir_ not in valid_scenes: continue
        depth_path = root_pth + f'{dir_}/' + 'depth/'
        
        save_pth = depth_path.replace('depth', 'depth_noisy')
        office_gts = Gen_inputs(depth_path, save_pth)
        office_gts.gen(region_nums=region_nums)


    
   

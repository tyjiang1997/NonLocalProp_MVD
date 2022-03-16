import os
from tkinter.messagebox import NO
import torch
import torch.nn as nn
import numpy as np
import pdb
import cv2
import torch.nn.functional as F
import sys
sys.path.append('/home/jty/mvs/idn-solver/')
from utils import depth2normal

class Solver(nn.Module):
    def __init__(self, h, w, alpha1=10.0, alpha2=10.0, sigma1=20.0, sigma2=3.0, with_sommth=False):
        # check_offset: The checkerboard size to fetch depth
        # alpha1, alpha2: The weights of data term in depth update and normal update
        # sigma1, sigma2: The threshold value in color and distance weighting
        super(Solver, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.h, self.w = h, w
        yy, xx = torch.meshgrid(torch.arange(self.h), torch.arange(self.w))
        self.xy = torch.stack([xx.float(),yy.float()], 0)
        self.xy_homo = torch.cat([self.xy, torch.ones((1,h,w))], 0).cuda() # [3,h,w]
        self.xy_homo = self.xy_homo.reshape(3,-1).unsqueeze(0) # [1,3,-1]

        self.with_sommth = with_sommth


    
    def propagate_axis(self, depth, normal, rgb, conf, offset, K_inv):
        # depth & conf: [b, 1, h, w], rgb & normal: [b, 3, h, w], K_inv: [b, 3, 3]
       
        b = depth.shape[0]
        xy_homo_ = self.xy_homo.repeat(b,1,1) # [b, 3, h*w]
        xy_3d = torch.matmul(K_inv, xy_homo_).reshape(b, 3, self.h, self.w)
        nom = depth * torch.sum(normal * xy_3d, dim=1, keepdim=True)

        xy_homo_ref = xy_homo_.clone()
        # 点 i上存放点j在 i平面上的深度值
        _b, c, h, w = offset.shape
        offset = offset.reshape(_b, c, h * w)
        xy_homo_ref[:,:2,:] = xy_homo_ref[:,:2,:] + offset
        
        denom = torch.sum(normal * torch.matmul(K_inv, xy_homo_ref).reshape(b,3,self.h,self.w), dim=1, keepdim=True)
        d1_p =  (nom / (denom + 1e-18))
        # if self.with_sommth:
        #     # p = 0.7
        #     # from pdb  import set_trace; set_trace()
        #     d1_p = (1-conf )* d1_p + conf * depth
        d1_p =  d1_p.clamp(0.01, 10.0)
        # 点 i存档的变换后的点j的深度值变换回到 j原始储存点

        xy_homo_ref = xy_homo_.clone()
        xy_back = xy_homo_ref[:,:2,:] - offset
        xy_back = xy_back.reshape(-1, 2, self.h, self.w).permute(0,2,3,1)
        xy_back_mask = (((xy_back[..., 0] < 0) | (xy_back[..., 0] > self.w)) |  \
            ((xy_back[..., 1] < 0) | (xy_back[..., 1] > self.h))).unsqueeze(1)

        xy_back[..., 0] = (xy_back[..., 0] - (self.w-1)/2 ) / ((self.w-1)/2)
        xy_back[..., 1] = (xy_back[..., 1] - (self.h-1)/2 ) / ((self.h-1)/2)

        d1 = F.grid_sample(d1_p, xy_back ,align_corners=True)

        # 若是在索引之外的点用原始深度代替
        # from pdb import set_trace; set_trace()
        d1[xy_back_mask] = depth[xy_back_mask]

        # confidence 需要j点位置储存i点的
        conf1 = F.grid_sample(conf, xy_back, padding_mode='zeros',mode='bilinear',align_corners=True)
        # 若是在索引之外的点用conf为0
        conf1[xy_back_mask] = 0
        
        rgb1 = F.grid_sample(rgb, xy_back, padding_mode='zeros',mode='bilinear',align_corners=True)
        xy_back_mask = xy_back_mask.repeat_interleave(dim=1, repeats=3)
        rgb1[xy_back_mask] = 0


        return d1, conf1, rgb1
    
    def checkerboard_propagate(self, depth, normal, rgb, conf, K_inv, profiler=None, offsets=None):
        b, _, h, w = depth.shape
        if profiler is not None:
            profiler.report_process('before zeros')
        
        propagated_depth, propagated_conf, propagated_rgb = [], [], []
        distance = []
        if profiler is not None:
            profiler.report_process('checkerboard propagate prev')
        
        _b, _c, _h, _w = offsets.shape 
        offsets = offsets.reshape( _b, -1, 2, _h, _w)
        nums_offsets = offsets.shape[1]

        for i in range(nums_offsets):
            offset = offsets[:, i]
            d1, conf1, rgb1 = self.propagate_axis(depth, normal, rgb, conf, offset, K_inv)
            propagated_depth.append(d1)
            propagated_conf.append(conf1)
            propagated_rgb.append(rgb1)
            distance.append(offset)

        propagated_depth, propagated_conf, propagated_rgb = torch.stack(propagated_depth, 1), torch.stack(propagated_conf, 1), torch.stack(propagated_rgb, 1)
        distance = torch.stack(distance, 1).abs().norm(dim=2, keepdim=True)

        return propagated_depth, propagated_conf, propagated_rgb, distance
    
    def propagate_axis_less(self, points, conf, offset):
        # Up, down, left and right directions
        b = points.shape[0]
        xy_homo_ = self.xy_homo.repeat(b,1,1) # [b, 3, h*w]
        xy_homo_ref = xy_homo_.clone()
        _b, c, h, w = offset.shape
        offset = offset.reshape(_b, c, h * w)
        xy_back = xy_homo_ref[:,:2,:] - offset
        xy_back = xy_back.reshape(-1, 2, self.h, self.w).permute(0,2,3,1)
        xy_back_mask = (((xy_back[..., 0] < 0) | (xy_back[..., 0] > self.w)) | \
            ((xy_back[..., 1] < 0) | (xy_back[..., 1] > self.h))).unsqueeze(1)

        xy_back[..., 0] = (xy_back[..., 0] - (self.w-1)/2 ) / ((self.w-1)/2)
        xy_back[..., 1] = (xy_back[..., 1] - (self.h-1)/2 ) / ((self.h-1)/2)

        # confidence 需要j点位置储存i点的
        conf1 = F.grid_sample(conf, xy_back, padding_mode='zeros',mode='bilinear',align_corners=True)
        # 若是在索引之外的点用conf为0
        conf1[xy_back_mask] = 0
        
        points1 = F.grid_sample(points, xy_back, padding_mode='zeros',mode='bilinear',align_corners=True)
        xy_back_mask = xy_back_mask.repeat_interleave(dim=1, repeats=3)
        points1[xy_back_mask] = 0

        return points1, conf1
    
    def checkerboard_propagate_less(self, points, conf, offsets=None):
        b, _, h, w = conf.shape
        propagated_conf, propagated_points = [], []

        _b, _c, _h, _w = offsets.shape 
        offsets = offsets.reshape( _b, -1, 2, _h, _w)
        nums_offsets = offsets.shape[1]

        for i in range(nums_offsets):

            offset = offsets[:, i]
            points1, conf1 = self.propagate_axis_less(points, conf, offset)
            propagated_points.append(points1)
            propagated_conf.append(conf1)

        propagated_points = torch.stack(propagated_points, 1)
        propagated_conf = torch.stack(propagated_conf, 1)
        return propagated_points, propagated_conf

    def forward(self, depth, normal, image, conf, confN, K, profiler=None, offsets=None, affweights=None):
        # Update the depth and normal value. 
        # depth: [B, 1, H, W], normal: [B, 3, H, W], image: [B, 3, H, W], conf: depth confidence [B, 1, H, W], confN: normal confidence
        # Update the depth
        b = depth.shape[0]
        K_inv = torch.inverse(K.cpu()).to(depth.get_device())
        checkerboard_depth, checkerboard_conf, checkerboard_rgb, checkerboard_dis = \
            self.checkerboard_propagate(depth, normal, image, conf*confN, K_inv, profiler=profiler, offsets=offsets)
        
        if affweights is None :
            colorweights = image.unsqueeze(1).repeat(1,checkerboard_rgb.shape[1],1,1,1) - checkerboard_rgb
            colorweights = torch.exp(-torch.sum(torch.pow(colorweights, 2), 2) / self.sigma1).unsqueeze(2).detach()
            spatialweights = torch.exp(-checkerboard_dis / self.sigma2).detach()
            affweights = colorweights * spatialweights


        if affweights.shape.__len__() == 4:
            affweights = affweights.unsqueeze(2)

        checkerboard_conf = checkerboard_conf * affweights
        
        # if self.with_sommth:
        #     # from pdb import set_trace; set_trace()
        #     denom =  2 * checkerboard_conf.sum(1) + self.alpha1 * conf
        # else:
        denom = checkerboard_conf.sum(1) + self.alpha1 * conf
        nom = (checkerboard_depth * checkerboard_conf).sum(1) + self.alpha1 * conf * depth
        updated_depth = nom / (denom + 1e-16)
        updated_depth = torch.clamp(updated_depth, 0.1, 10.0)
        
        # Update the normal
        # Using the plane fitting loss
        xy_homo_ = self.xy_homo.repeat(b,1,1)
        points = torch.matmul(K_inv, xy_homo_)
        points = updated_depth * points.reshape(b, 3, self.h, self.w)
        normal_1 = -normal / normal[:,2:,:,:] # [a,b,-1]

        checkerboard_points, checkerboard_dconf = self.checkerboard_propagate_less(points, conf, offsets=offsets)

        checkerboard_dconf = checkerboard_dconf * affweights
        residual = points.unsqueeze(1) - checkerboard_points
        checkerboard_dconf = checkerboard_dconf.squeeze(2) * conf

        A11 = self.alpha2 * confN + (checkerboard_dconf * (residual[:,:,0,:,:])**2).sum(1, keepdim=True)
        A12 = (checkerboard_dconf * residual[:,:,0,:,:] * residual[:,:,1,:,:]).sum(1, keepdim=True)
        A21 = (checkerboard_dconf * residual[:,:,0,:,:] * residual[:,:,1,:,:]).sum(1, keepdim=True)
        A22 = self.alpha2 * confN + (checkerboard_dconf * (residual[:,:,1,:,:])**2).sum(1, keepdim=True)
        b1 = self.alpha2 * confN * normal_1[:,0,:,:].unsqueeze(1) + (checkerboard_dconf * residual[:,:,2,:,:] * residual[:,:,0,:,:]).sum(1, keepdim=True)
        b2 = self.alpha2 * confN * normal_1[:,1,:,:].unsqueeze(1) + (checkerboard_dconf * residual[:,:,2,:,:] * residual[:,:,1,:,:]).sum(1, keepdim=True)
        det = A11 * A22 - A12 * A21
        n1 = (b1 * A22 - b2 * A12) / (det + 1e-6)
        n2 = (b2 * A11 - b1 * A21) / (det + 1e-6)
        n1 = torch.clamp(n1, -20.0, 20.0)
        n2 = torch.clamp(n2, -20.0, 20.0)
        filler = -1.0 * torch.ones((b,1,self.h,self.w)).to(n1.get_device())
        updated_normal = torch.cat([n1,n2,filler], dim=1)
        updated_normal = -updated_normal / torch.norm(updated_normal, p=2, dim=1, keepdim=True)
        
        return updated_depth, updated_normal


def normal2color(normal_map):
    """
    colorize normal map
    :param normal_map: range(-1, 1)
    :return:
    """
    tmp = normal_map / 2. + 0.5  # mapping to (0, 1)
    color_normal = (tmp * 255).astype(np.uint8)

    return color_normal

if __name__ == '__main__':
    print('Test case...')
    # A whole plain 3D plane filling the incorrect values
    img = cv2.imread('/6t/jty/scannet/scannet_nas/train/scene0000_00/color/0000.jpg')
    depth = np.load('/6t/jty/scannet/scannet_nas/train/scene0000_00/depth/0000.npy').astype(np.float32)
    depth_f = cv2.medianBlur(depth, 5)
    depth_f = depth_f / 5000.0
    K = np.array([[535.4, 0.0, 320.1], [0.0, 539.2, 247.6], [0.0, 0.0, 1.0]])
    K = K[np.newaxis, ...]
    depth_f = depth_f[np.newaxis, ...]
    normal = depth2normal(torch.from_numpy(depth_f).float().unsqueeze(0), torch.from_numpy(np.linalg.inv(K)).float())
    normal = normal + 1e-16
    normal_img = normal[0].numpy()
    cv2.imwrite('/home/jty/mvs/idn-solver/vis/test_normal.png', normal2color(normal_img).transpose(1,2,0))
    # 

    solver = Solver(h=480, w=640, check_offsets=[1,2,3,5,7,20], alpha1=5.0, alpha2=5.0, sigma1=1000.0, sigma2=10.0)
    iter_ = 5
    conf = np.where(depth_f > 0.001, 1.0, 0.01)
    depth_var = torch.from_numpy(depth_f).unsqueeze(1).float().cuda()
    normal_var = normal.float().cuda()
    img_var = torch.from_numpy(img).float().cuda().permute(2,0,1).unsqueeze(0)
    conf_var = torch.from_numpy(conf).float().cuda().unsqueeze(-1).permute(0,3,1,2)
    K_var = torch.from_numpy(K).float().cuda()

   
    x, y = np.arange(-1,2), np.arange(-1,2)
    x, y = np.meshgrid(x,y)
    xy = np.stack([x,y],-1).reshape(-1,1)
    xy = torch.tensor(xy).cuda(depth_var.device)
    
    tmp = depth_var.new_ones(18, 480 *640) * xy
    check_offsets = [1,2,3,5,7,20]
    offsets = []
    for offset in check_offsets:
        offsets.append(offset * tmp)
    offsets = torch.cat(offsets, 0)
    c = offsets.shape[0]
    offsets = offsets.reshape(1, c, 480 , 640)

    for i in range(iter_):
        if i == 0:
            
            # (depth, normal, image, conf, confN, K, profiler=None):
            # depth: [B, 1, H, W], normal: [B, 3, H, W], image: [B, 3, H, W], conf: depth confidence [B, 1, H, W], confN: normal confidence
            updated_depth, updated_normal = solver(depth_var, normal_var, img_var, conf_var, conf_var, K_var, offsets=offsets)
        else:
            updated_depth, updated_normal = solver(updated_depth, updated_normal, img_var, conf_var, conf_var, K_var, offsets=offsets)
        # from pdb import set_trace; set_trace()
        cv2.imwrite('/home/jty/mvs/idn-solver/vis/test_depth'+str(i)+'.png', (updated_depth.detach().cpu().numpy()*100.0)[0,0].astype(np.uint8))
        cv2.imwrite('/home/jty/mvs/idn-solver/vis/test_normal'+str(i)+'.png', normal2color(updated_normal.detach().cpu().numpy())[0].transpose(1,2,0))


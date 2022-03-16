import os
import torch
import torch.nn as nn
import numpy as np
import pdb
import cv2
import sys
sys.path.append('/home/jty/mvs/idn-solver/')
from utils import depth2normal

class Solver(nn.Module):
    def __init__(self, h, w, check_offsets=[1,3,5], alpha1=10.0, alpha2=10.0, sigma1=20.0, sigma2=3.0):
        # check_offset: The checkerboard size to fetch depth
        # alpha1, alpha2: The weights of data term in depth update and normal update
        # sigma1, sigma2: The threshold value in color and distance weighting
        super(Solver, self).__init__()
        self.check_offsets = check_offsets
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.h, self.w = h, w
        yy, xx = torch.meshgrid(torch.arange(self.h), torch.arange(self.w))
        self.xy = torch.stack([xx.float(),yy.float()], 0)
        self.xy_homo = torch.cat([self.xy, torch.ones((1,h,w))], 0).cuda() # [3,h,w]
        self.xy_homo = self.xy_homo.reshape(3,-1).unsqueeze(0) # [1,3,-1]
    
    def propagate_axis(self, depth, normal, rgb, conf, offset, K_inv, axis='ud'):
        # Up, down, left and right directions
        # depth & conf: [b, 1, h, w], rgb & normal: [b, 3, h, w], K_inv: [b, 3, 3]
       
        b = depth.shape[0]
        xy_homo_ = self.xy_homo.repeat(b,1,1) # [b, 3, h*w]
        xy_3d = torch.matmul(K_inv, xy_homo_).reshape(b, 3, self.h, self.w)
        nom = depth * torch.sum(normal * xy_3d, dim=1, keepdim=True)
        if axis == 'ud':
            filler1,   filler2 = torch.zeros((b, 1, offset, self.w)), torch.zeros((b, 3, offset, self.w))
            filler1, filler2 = filler1.to(depth.get_device()), filler2.to(depth.get_device())
            # from pdb import set_trace; set_trace()
            # Up
            xy_homo_ref = xy_homo_.clone()
            
            xy_homo_ref[:,1,:] = xy_homo_ref[:,1,:] + offset
            denom = torch.sum(normal * torch.matmul(K_inv, xy_homo_ref).reshape(b,3,self.h,self.w), dim=1, keepdim=True)
            d1_p =  (nom / (denom + 1e-18)).clamp(0.01, 10.0)
            d1 = torch.cat([depth[:,:,:offset,:], d1_p[:,:,:-offset,:]], 2)
            
            # Down
            xy_homo_ref = xy_homo_.clone()
            xy_homo_ref[:,1,:] = xy_homo_ref[:,1,:] - offset
            denom = torch.sum(normal * torch.matmul(K_inv, xy_homo_ref).reshape(b,3,self.h,self.w), dim=1, keepdim=True)
            d2_p =  (nom / (denom + 1e-18)).clamp(0.01, 10.0)
            d2 = torch.cat([d2_p[:,:,offset:, :], depth[:,:,-offset:,:]], 2)
            conf1, conf2 = torch.cat([filler1, conf[:,:,:-offset,:]], 2), torch.cat([conf[:,:,offset:, :], filler1], 2)
            rgb1, rgb2 = torch.cat([filler2, rgb[:,:,:-offset,:]], 2), torch.cat([rgb[:,:,offset:, :], filler2], 2)
            # from pdb import set_trace; set_trace()
        elif axis == 'lr':
            filler1, filler2 = torch.zeros((b, 1, self.h, offset)), torch.zeros((b, 3, self.h, offset))
            filler1, filler2 = filler1.to(depth.get_device()), filler2.to(depth.get_device())
            # Left
            xy_homo_ref = xy_homo_.clone()
            xy_homo_ref[:,0,:] = xy_homo_ref[:,0,:] + offset
            denom = torch.sum(normal * torch.matmul(K_inv, xy_homo_ref).reshape(b,3,self.h,self.w), dim=1, keepdim=True)
            d1_p =  (nom / (denom + 1e-18)).clamp(0.01, 10.0)
            d1 = torch.cat([depth[:,:,:,:offset], d1_p[:,:,:,:-offset]], 3)
            # Right
            xy_homo_ref = xy_homo_.clone()
            xy_homo_ref[:,0,:] = xy_homo_ref[:,0,:] - offset
            denom = torch.sum(normal * torch.matmul(K_inv, xy_homo_ref).reshape(b,3,self.h,self.w), dim=1, keepdim=True)
            d2_p =  (nom / (denom + 1e-18)).clamp(0.01, 10.0)
            d2 = torch.cat([d2_p[:,:,:,offset:], depth[:,:,:,-offset:]], 3)
            conf1, conf2 = torch.cat([filler1, conf[:,:,:,:-offset]], 3), torch.cat([conf[:,:,:,offset:], filler1], 3)
            rgb1, rgb2 = torch.cat([filler2, rgb[:,:,:,:-offset]], 3), torch.cat([rgb[:,:,:,offset:], filler2], 3)
        
        elif axis == 'lu':
            filler1, filler2 = torch.zeros((b, 1, self.h, self.w)), torch.zeros((b, 3, self.h, self.w))
            filler1, filler2 = filler1.to(depth.get_device()), filler2.to(depth.get_device())
            # up Left
            xy_homo_ref = xy_homo_.clone()
            offset_t = xy_homo_ref.new_tensor([offset,offset])
            xy_homo_ref[:,:2,:] = (xy_homo_ref[:,:2,:].permute(2,0,1) + offset_t).permute(1,2,0)
            denom = torch.sum(normal * torch.matmul(K_inv, xy_homo_ref).reshape(b,3,self.h,self.w), dim=1, keepdim=True)
            d1_p =  (nom / (denom + 1e-18)).clamp(0.01, 10.0)
            # from pdb import set_trace; set_trace()
            d1t = torch.cat([depth[:,:,offset:,:offset], d1_p[:,:,:-offset,:-offset]], 3)
            d1 = torch.cat([depth[:,:,:offset,:], d1t], 2)

            # down Right
            xy_homo_ref = xy_homo_.clone()
            xy_homo_ref[:,:2,:] = (xy_homo_ref[:,:2,:].permute(2,0,1) - offset_t).permute(1,2,0)
            denom = torch.sum(normal * torch.matmul(K_inv, xy_homo_ref).reshape(b,3,self.h,self.w), dim=1, keepdim=True)
            d2_p =  (nom / (denom + 1e-18)).clamp(0.01, 10.0)
            # from pdb import set_trace; set_trace()
            d2t = torch.cat([d2_p[:,:,offset:,offset:], depth[:,:,:-offset,-offset:]], 3)
            d2 = torch.cat([d2t, depth[:,:,-offset:,:]], 2)

            conf1t = torch.cat([filler1[:,:,offset:,:offset], conf[:,:,:-offset,:-offset]], 3)
            conf1 = torch.cat([filler1[:,:,:offset,:], conf1t], 2)

            conf2t = torch.cat([conf[:,:,offset:,offset:], filler1[:,:,:-offset,-offset:]], 3)
            conf2 = torch.cat([conf2t, filler1[:,:,-offset:,:]], 2)

            rgb1t = torch.cat([filler2[:,:,offset:,:offset], rgb[:,:,:-offset,:-offset]], 3)
            rgb1 = torch.cat([filler2[:,:,:offset,:], rgb1t], 2)

            rgb2t = torch.cat([rgb[:,:,offset:,offset:], filler2[:,:,:-offset,-offset:]], 3)
            rgb2 = torch.cat([rgb2t, filler2[:,:,-offset:,:]], 2)

        elif axis == 'ld':
            filler1, filler2 = torch.zeros((b, 1, self.h, self.w)), torch.zeros((b, 3, self.h, self.w))
            filler1, filler2 = filler1.to(depth.get_device()), filler2.to(depth.get_device())
            # up right
            xy_homo_ref = xy_homo_.clone()
            offset_t = xy_homo_ref.new_tensor([-offset, offset])
            xy_homo_ref[:,:2,:] = (xy_homo_ref[:,:2,:].permute(2,0,1) + offset_t).permute(1,2,0)
            denom = torch.sum(normal * torch.matmul(K_inv, xy_homo_ref).reshape(b,3,self.h,self.w), dim=1, keepdim=True)
            d1_p =  (nom / (denom + 1e-18)).clamp(0.01, 10.0)
            # from pdb import set_trace; set_trace()
            d1t = torch.cat([depth[:,:,:offset,:-offset], d1_p[:,:,:-offset,offset:]], 2)
            d1 = torch.cat([d1t, depth[:,:,:,-offset:]], 3)

            # down Right
            xy_homo_ref = xy_homo_.clone()
            offset_t = xy_homo_ref.new_tensor([offset, -offset])
            xy_homo_ref[:,:2,:] = (xy_homo_ref[:,:2,:].permute(2,0,1) + offset_t).permute(1,2,0)
            denom = torch.sum(normal * torch.matmul(K_inv, xy_homo_ref).reshape(b,3,self.h,self.w), dim=1, keepdim=True)
            d2_p =  (nom / (denom + 1e-18)).clamp(0.01, 10.0)

            d2t = torch.cat([d2_p[:,:,offset:,:-offset], depth[:,:,-offset:,:-offset]], 2)
            d2 = torch.cat([ depth[:,:,:,:offset], d2t], 3)

            conf1t = torch.cat([filler1[:,:,:offset,:-offset], conf[:,:,:-offset,offset:]], 2)
            conf1 = torch.cat([conf1t, filler1[:,:,:,-offset:]], 3)

            conf2t = torch.cat([conf[:,:,offset:,:-offset], filler1[:,:,-offset:,:-offset]], 2)
            conf2 = torch.cat([ filler1[:,:,:,:offset], conf2t], 3)

            rgb1t = torch.cat([filler2[:,:,:offset,:-offset], rgb[:,:,:-offset,offset:]], 2)
            rgb1 = torch.cat([rgb1t, filler2[:,:,:,-offset:]], 3)

            rgb2t = torch.cat([rgb[:,:,offset:,:-offset], filler2[:,:,-offset:,:-offset]], 2)
            rgb2 = torch.cat([ filler2[:,:,:,:offset], rgb2t], 3)


        return d1, d2, conf1, conf2, rgb1, rgb2
    
    def checkerboard_propagate(self, depth, normal, rgb, conf, K_inv, profiler=None):
        b, _, h, w = depth.shape
        if profiler is not None:
            profiler.report_process('before zeros')
        
        propagated_depth, propagated_conf, propagated_rgb = [], [], []
        distance = []
        if profiler is not None:
            profiler.report_process('checkerboard propagate prev')
        
        # from pdb import set_trace; set_trace()
        for i in range(len(self.check_offsets)):
            
            offset = self.check_offsets[i]
            d1, d2, conf1, conf2, rgb1, rgb2 = self.propagate_axis(depth, normal, rgb, conf, offset, K_inv, axis='ud')
            propagated_depth.append(d1)
            propagated_depth.append(d2)
            propagated_conf.append(conf1)
            propagated_conf.append(conf2)
            propagated_rgb.append(rgb1)
            propagated_rgb.append(rgb2)
            distance.append(offset)
            distance.append(offset)

            d1, d2, conf1, conf2, rgb1, rgb2 = self.propagate_axis(depth, normal, rgb, conf, offset, K_inv, axis='lr')
            propagated_depth.append(d1)
            propagated_depth.append(d2)
            propagated_conf.append(conf1)
            propagated_conf.append(conf2)
            propagated_rgb.append(rgb1)
            propagated_rgb.append(rgb2)
            distance.append(offset)
            distance.append(offset)

            d1, d2, conf1, conf2, rgb1, rgb2 = self.propagate_axis(depth, normal, rgb, conf, offset, K_inv, axis='lu')
            propagated_depth.append(d1)
            propagated_depth.append(d2)
            propagated_conf.append(conf1)
            propagated_conf.append(conf2)
            propagated_rgb.append(rgb1)
            propagated_rgb.append(rgb2)
            distance.append(offset**0.5)
            distance.append(offset**0.5)

            d1, d2, conf1, conf2, rgb1, rgb2 = self.propagate_axis(depth, normal, rgb, conf, offset, K_inv, axis='ld')
            propagated_depth.append(d1)
            propagated_depth.append(d2)
            propagated_conf.append(conf1)
            propagated_conf.append(conf2)
            propagated_rgb.append(rgb1)
            propagated_rgb.append(rgb2)
            distance.append(offset**0.5)
            distance.append(offset**0.5)


        # from pdb import set_trace; set_trace()
        propagated_depth, propagated_conf, propagated_rgb = torch.stack(propagated_depth, 1), torch.stack(propagated_conf, 1), torch.stack(propagated_rgb, 1)
        distance = torch.from_numpy(np.stack(distance, -1)).float().to(depth.get_device())
        # from pdb import set_trace; set_trace()
        return propagated_depth, propagated_conf, propagated_rgb, distance
    
    def propagate_axis_less(self, points, conf, offset, axis='ud'):
        # Up, down, left and right directions
        b = points.shape[0]
        if axis == 'ud':
            filler1, filler2 = torch.zeros((b, 1, offset, self.w)), torch.zeros((b, 3, offset, self.w))
            filler1, filler2 = filler1.to(points.get_device()), filler2.to(points.get_device())
            # Up
            conf1, conf2 = torch.cat([filler1, conf[:,:,:-offset,:]], 2), torch.cat([conf[:,:,offset:,:], filler1], 2)
            points1, points2 = torch.cat([filler2, points[:,:,:-offset,:]], 2), torch.cat([points[:,:,offset:,:], filler2], 2)
        elif axis == 'lr':
            filler1, filler2 = torch.zeros((b, 1, self.h, offset)), torch.zeros((b, 3, self.h, offset))
            filler1, filler2 = filler1.to(points.get_device()), filler2.to(points.get_device())
            # Left
            conf1, conf2 = torch.cat([filler1, conf[:,:,:,:-offset]], 3), torch.cat([conf[:,:,:,offset:], filler1], 3)
            points1, points2 = torch.cat([filler2, points[:,:,:,:-offset]], 3), torch.cat([points[:,:,:,offset:], filler2], 3)
        elif axis == 'lu':
            filler1, filler2 = torch.zeros((b, 1, self.h, self.w)), torch.zeros((b, 3, self.h, self.w))
            filler1, filler2 = filler1.to(points.get_device()), filler2.to(points.get_device())
            # up Left
            conf1t = torch.cat([filler1[:,:,offset:,:offset], conf[:,:,:-offset,:-offset]], 3)
            conf1 = torch.cat([filler1[:,:,:offset,:], conf1t], 2)

            conf2t = torch.cat([conf[:,:,offset:,offset:], filler1[:,:,:-offset,-offset:]], 3)
            conf2 = torch.cat([conf2t, filler1[:,:,-offset:,:]], 2)

            points1t = torch.cat([filler2[:,:,offset:,:offset], points[:,:,:-offset,:-offset]], 3)
            points1 = torch.cat([filler2[:,:,:offset,:], points1t], 2)

            points2t = torch.cat([points[:,:,offset:,offset:], filler2[:,:,:-offset,-offset:]], 3)
            points2 = torch.cat([points2t, filler2[:,:,-offset:,:]], 2)

        elif axis == 'ld': 
            filler1, filler2 = torch.zeros((b, 1, self.h, self.w)), torch.zeros((b, 3, self.h, self.w))
            filler1, filler2 = filler1.to(points.get_device()), filler2.to(points.get_device())
            # up right
            conf1t = torch.cat([filler1[:,:,:offset,:-offset], conf[:,:,:-offset,offset:]], 2)
            conf1 = torch.cat([conf1t, filler1[:,:,:,-offset:]], 3)

            conf2t = torch.cat([conf[:,:,offset:,:-offset], filler1[:,:,-offset:,:-offset]], 2)
            conf2 = torch.cat([ filler1[:,:,:,:offset], conf2t], 3)

            points1t = torch.cat([filler2[:,:,:offset,:-offset], points[:,:,:-offset,offset:]], 2)
            points1 = torch.cat([points1t, filler2[:,:,:,-offset:]], 3)

            points2t = torch.cat([points[:,:,offset:,:-offset], filler2[:,:,-offset:,:-offset]], 2)
            points2 = torch.cat([ filler2[:,:,:,:offset], points2t], 3)

        return points1, points2, conf1, conf2
    
    def checkerboard_propagate_less(self, points, conf):
        b, _, h, w = conf.shape
        propagated_conf, propagated_points = [], []

        for i in range(len(self.check_offsets)):

            offset = self.check_offsets[i]
            points1, points2, conf1, conf2 = self.propagate_axis_less(points, conf, offset, axis='ud')
            propagated_points.append(points1)
            propagated_points.append(points2)
            propagated_conf.append(conf1)
            propagated_conf.append(conf2)

            points1, points2, conf1, conf2 = self.propagate_axis_less(points, conf, offset, axis='lr')
            propagated_points.append(points1)
            propagated_points.append(points2)
            propagated_conf.append(conf1)
            propagated_conf.append(conf2)

            points1, points2, conf1, conf2 = self.propagate_axis_less(points, conf, offset, axis='lu')
            propagated_points.append(points1)
            propagated_points.append(points2)
            propagated_conf.append(conf1)
            propagated_conf.append(conf2)

            points1, points2, conf1, conf2 = self.propagate_axis_less(points, conf, offset, axis='ld')
            propagated_points.append(points1)
            propagated_points.append(points2)
            propagated_conf.append(conf1)
            propagated_conf.append(conf2)

        propagated_points = torch.stack(propagated_points, 1)
        propagated_conf = torch.stack(propagated_conf, 1)
        return propagated_points, propagated_conf

    def forward(self, depth, normal, image, conf, confN, K, profiler=None):
        # Update the depth and normal value. 
        # depth: [B, 1, H, W], normal: [B, 3, H, W], image: [B, 3, H, W], conf: depth confidence [B, 1, H, W], confN: normal confidence
        # Update the depth
        b = depth.shape[0]
        K_inv = torch.inverse(K.cpu()).to(depth.get_device())
        checkerboard_depth, checkerboard_conf, checkerboard_rgb, checkerboard_dis = self.checkerboard_propagate(depth, normal, image, conf*confN, K_inv, profiler=profiler)
        # from pdb import set_trace; set_trace()
        colorweights = image.unsqueeze(1).repeat(1,checkerboard_rgb.shape[1],1,1,1) - checkerboard_rgb
        colorweights = torch.exp(-torch.sum(torch.pow(colorweights, 2), 2) / self.sigma1).unsqueeze(2).detach()
        # from pdb import set_trace; set_trace()
        spatialweights = torch.exp(-checkerboard_dis / self.sigma2).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(b,1,self.h,self.w).detach()
        checkerboard_conf = checkerboard_conf * colorweights * spatialweights.unsqueeze(2)
        
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

        checkerboard_points, checkerboard_dconf = self.checkerboard_propagate_less(points, conf)
        checkerboard_dconf = checkerboard_dconf * colorweights * spatialweights.unsqueeze(2)
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
    for i in range(iter_):
        if i == 0:
            
            # (depth, normal, image, conf, confN, K, profiler=None):
            # depth: [B, 1, H, W], normal: [B, 3, H, W], image: [B, 3, H, W], conf: depth confidence [B, 1, H, W], confN: normal confidence
            updated_depth, updated_normal = solver(depth_var, normal_var, img_var, conf_var, conf_var, K_var)
        else:
            updated_depth, updated_normal = solver(updated_depth, updated_normal, img_var, conf_var, conf_var, K_var)
        # from pdb import set_trace; set_trace()
        cv2.imwrite('/home/jty/mvs/idn-solver/vis/test_depth'+str(i)+'.png', (updated_depth.detach().cpu().numpy()*100.0)[0,0].astype(np.uint8))
        cv2.imwrite('/home/jty/mvs/idn-solver/vis/test_normal'+str(i)+'.png', normal2color(updated_normal.detach().cpu().numpy())[0].transpose(1,2,0))


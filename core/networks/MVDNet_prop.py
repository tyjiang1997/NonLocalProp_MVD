from re import I, S
from html5lib import serialize
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from core.networks.submodule import *
# from core.networks.solver import Solver, normal2color
from core.networks.solveroff import Solver
from core.utils.inverse_warp_d import inverse_warp_d, pixel2cam
from core.utils.inverse_warp import inverse_warp
from core.networks.loss_functions import compute_angles, cross_entropy
import matplotlib.pyplot as plt
import pdb
from core.utils.utils import Profiler
from .modulated_deform_conv_func import ModulatedDeformConvFunction
from post_processing.calConfidence import calConf

class PropPlane(nn.Module):
    def __init__(self, args, ch_g, ch_f, k_g, k_f):
        super(PropPlane, self).__init__()

        # Guidance : [B x ch_g x H x W]
        # Feature : [B x ch_f x H x W]

        # Currently only support ch_f == 1
        assert ch_f == 1, 'only tested with ch_f == 1 but {}'.format(ch_f)

        assert (k_g % 2) == 1, \
            'only odd kernel is supported but k_g = {}'.format(k_g)
        pad_g = int((k_g - 1) / 2)
        assert (k_f % 2) == 1, \
            'only odd kernel is supported but k_f = {}'.format(k_f)
        pad_f = int((k_f - 1) / 2)

        self.args = args
        self.prop_time = self.args.prop_time
        self.affinity = self.args.affinity

        self.ch_g = ch_g
        self.ch_f = ch_f
        self.k_g = k_g
        self.k_f = k_f
        # Assume zero offset for center pixels
        self.num = self.k_f * self.k_f - 1
        self.idx_ref = self.num // 2

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            if self.args.use_aff:
                self.conv_offset_aff = nn.Conv2d(
                    self.ch_g, 3 * self.num, kernel_size=self.k_g, stride=1,
                    padding=pad_g, bias=True
                )
            else:
                self.conv_offset_aff = nn.Conv2d(
                    self.ch_g, 2 * self.num, kernel_size=self.k_g, stride=1,
                    padding=pad_g, bias=True
                )
            self.conv_offset_aff.weight.data.zero_()
            self.conv_offset_aff.bias.data.zero_()

            if self.affinity == 'TC':
                self.aff_scale_const = nn.Parameter(self.num * torch.ones(1))
                self.aff_scale_const.requires_grad = False
            elif self.affinity == 'TGASS':
                self.aff_scale_const = nn.Parameter(
                    self.args.affinity_gamma * self.num * torch.ones(1))
            else:
                self.aff_scale_const = nn.Parameter(torch.ones(1))
                self.aff_scale_const.requires_grad = False
        else:
            raise NotImplementedError

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f, self.k_f)))
        self.b = nn.Parameter(torch.zeros(self.ch_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self.w_conf.requires_grad = False

        self.stride = 1
        self.padding = pad_f
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64

        # mesh-grid


    def _get_offset_affinity(self, guidance, gt_deppth_conf=None, rgb=None, tgtimg_fea=None):
        B, _, H, W = guidance.shape
        
        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            
            offset_aff = self.conv_offset_aff(guidance)
            if self.args.use_aff:
                o1, o2, aff = torch.chunk(offset_aff, 3, dim=1)
            else:
                # from pdb import set_trace; set_trace()
                o1, o2 = torch.chunk(offset_aff, 2, dim=1)

            # Add zero reference offset
            offset = torch.cat((o1, o2), dim=1).view(B, self.num, 2, H, W)
            # aff weights
            if self.args.afftype == 'cos':
                batsize, chanle, hh, ww = o1.shape
                tgtimg_fea_up = F.interpolate(tgtimg_fea, [hh, ww], mode='bilinear').detach()
                x, y = torch.arange(hh).to(tgtimg_fea_up.device), torch.arange(ww).to(tgtimg_fea_up.device)
                x, y = torch.meshgrid(x, y)
                xy = torch.stack([x, y], dim=-1).unsqueeze(0).repeat_interleave(dim=0, repeats=batsize)
                xy = xy.repeat_interleave(dim=-1, repeats=self.args.num_neighbors)
                # from pdb import set_trace; set_trace()
                xy_offsets = offset.reshape(B, -1, H, W).permute(0,2,3,1) + xy
                xy_offsets = xy_offsets.reshape(batsize, hh, ww, -1, 2)
                xy_offsets[..., 0] = (2 * xy_offsets[..., 0] / (ww-1)) - 1
                xy_offsets[..., 1] = (2 * xy_offsets[..., 1] / (hh-1)) - 1
                xy_offsets = xy_offsets.reshape(batsize, hh, -1, 2)
                offset_feas = F.grid_sample(tgtimg_fea_up, xy_offsets,  mode='bilinear',align_corners=True).reshape(batsize, hh, ww, -1, chanle)
                offset_feas = offset_feas.permute(0,3,1,2,4)

                tgtimg_fea_up = tgtimg_fea_up.unsqueeze(-1).repeat_interleave(dim=-1, repeats=self.args.num_neighbors)
                cos_weights = (offset_feas * tgtimg_fea_up).sum(dim=1).permute(0,3,1,2)

                if self.args.use_aff:
                    aff = aff * cos_weights
                else:
                    # from pdb import set_trace; set_trace()
                    aff = cos_weights

            list_offset = list(torch.chunk(offset, self.num, dim=1))
            list_offset.insert(self.idx_ref,
                               torch.zeros((B, 1, 2, H, W)).type_as(offset))
            offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

            if self.affinity in ['AS', 'ASS']:
                pass
            elif self.affinity == 'TC':
                aff = torch.tanh(aff) / self.aff_scale_const
            elif self.affinity == 'TGASS':
                aff = torch.tanh(aff) / (self.aff_scale_const + 1e-8)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Apply gt_deppth_conf
        # TODO : Need more efficient way
        if self.args.conf_prop:
            list_conf = []
            offset_each = torch.chunk(offset, self.num + 1, dim=1)

            modulation_dummy = torch.ones((B, 1, H, W)).type_as(offset).detach()

            for idx_off in range(0, self.num + 1):
                ww = idx_off % self.k_f
                hh = idx_off // self.k_f

                if ww == (self.k_f - 1) / 2 and hh == (self.k_f - 1) / 2:
                    continue

                offset_tmp = offset_each[idx_off].detach()

                conf_tmp = ModulatedDeformConvFunction.apply(
                    gt_deppth_conf, offset_tmp, modulation_dummy, self.w_conf,
                    self.b, self.stride, 0, self.dilation, self.groups,
                    self.deformable_groups, self.im2col_step)
                list_conf.append(conf_tmp)

            conf_aff = torch.cat(list_conf, dim=1)
            aff = aff * conf_aff.contiguous()

        # Affinity normalization
        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-4

        if self.affinity in ['ASS', 'TGASS']:
            aff_abs_sum[aff_abs_sum < 1.0] = 1.0

        if self.affinity in ['AS', 'ASS', 'TGASS']:
            aff = aff / aff_abs_sum

        aff_sum = torch.sum(aff, dim=1, keepdim=True)
        aff_ref = 1.0 - aff_sum

        list_aff = list(torch.chunk(aff, self.num, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)

        # from pdb import set_trace; set_trace()
        aff = F.softmax(aff, dim=1)
        return offset, aff

    def get_aff(self, depth, nmap_out, tgt_img, gt_deppth_conf=None, gt_normal_conf=None, guidance=None, intrinsics=None, tgtimg_fea=None, feat_fix=None, rgb=None):
        
        assert self.ch_g == guidance.shape[1]
        assert self.ch_f == depth.shape[1]

        if self.args.conf_prop:
            assert gt_deppth_conf is not None

        if self.args.conf_prop:
            offset, aff = self._get_offset_affinity(guidance, gt_deppth_conf, rgb, tgtimg_fea=tgtimg_fea)
        else:
            offset, aff = self._get_offset_affinity(guidance, None, rgb, tgtimg_fea=tgtimg_fea)

        # from pdb import set_trace; set_trace()
        return [offset, aff] 

    def forward(self, depth, nmap_out, tgt_img, gt_deppth_conf=None, gt_normal_conf=None, guidance=None, intrinsics=None, tgtimg_fea=None, feat_fix=None,
                rgb=None):
        assert self.ch_g == guidance.shape[1]
        assert self.ch_f == depth.shape[1]

        h, w = depth.shape[2], depth.shape[3]
        self.solver = Solver(h=h, w=w, alpha1=self.args.solver_alpha1, alpha2=self.args.solver_alpha2, with_sommth=self.args.with_sommth)

        if self.args.conf_prop:
            assert gt_deppth_conf is not None

        if self.args.conf_prop:
            offset, aff = self._get_offset_affinity(guidance, gt_deppth_conf, rgb, tgtimg_fea=tgtimg_fea)
        else:
            offset, aff = self._get_offset_affinity(guidance, None, rgb, tgtimg_fea=tgtimg_fea)


        # Propagation
        if self.args.preserve_input:
            assert depth.shape == feat_fix.shape
            mask_fix = torch.sum(feat_fix > 0.0, dim=1, keepdim=True).detach()
            mask_fix = (mask_fix > 0.0).type_as(feat_fix)

        refined_depth, refined_normal = depth.clone(), nmap_out.clone().permute(0,3,1,2)


        for k in range(1, self.prop_time + 1):
            # Input preservation for each iteration
            if self.args.preserve_input:
                refined_depth = (1.0 - mask_fix) * refined_depth \
                              + mask_fix * feat_fix

            
            refined_depth,  refined_normal = self.solver(refined_depth, refined_normal, tgt_img, gt_deppth_conf, gt_normal_conf, 
                                        intrinsics, profiler=None, offsets=offset, affweights=aff)
            
            # if self.args.with_sommth and k == self.prop_time:
            #     refined_depth = ModulatedDeformConvFunction.apply(
            #             refined_depth, offset, aff, self.w, self.b, self.stride, self.padding,
            #             self.dilation, self.groups, self.deformable_groups, self.im2col_step
            #         )


        return refined_depth, refined_normal



def convtext(in_planes, out_planes, kernel_size = 3, stride = 1, dilation = 1):

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias = False),
        nn.LeakyReLU(0.1,inplace=True)
    )

def conf_out(in_planes, out_planes, kernel_size = 3, stride = 1, dilation = 1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias = False),
        nn.Sigmoid()
    )

class MVDNet_prop(nn.Module):
    def __init__(self, cfg):
        super(MVDNet_prop, self).__init__()
        self.cfg = cfg
        self.nlabel = cfg.ndepth
        self.mindepth = cfg.mindepth
        self.no_pool = False
    
        self.feature_extraction = feature_extraction()

        self.convs = nn.Sequential(
            convtext(33, 128, 3, 1, 1),
            convtext(128, 128, 3, 1, 2),
            convtext(128, 128, 3, 1, 4),
            convtext(128, 96, 3, 1, 8),
            convtext(96, 64, 3, 1, 16),
            convtext(64, 32, 3, 1, 1),
            convtext(32, 1, 3, 1, 1)
        )

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))
 
        self.dres3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 
 
        self.classify = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.softmax = nn.Softmax(dim = -1)

        self.wc0 = nn.Sequential(convbn_3d(64 + 3, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))
        
        self.pool1 = nn.Sequential(convbn_3d(32, 32, (2,3,3), (2,1,1), (0,1,1)), nn.ReLU(inplace=True))
        self.pool2 = nn.Sequential(convbn_3d(32, 32, (2,3,3), (2,1,1), (0,1,1)), nn.ReLU(inplace=True))
        self.pool3 = nn.Sequential(convbn_3d(32, 32, (2,3,3), (2,1,1), (0,1,1)), nn.ReLU(inplace=True))
        
        self.n_convs0 = nn.Sequential(
            convtext(32, 96, 3, 1, 1),
            convtext(96, 96, 3, 1, 2),
            convtext(96, 96, 3, 1, 4),
            convtext(96, 64, 3, 1, 8),
            convtext(64, 64, 3, 1, 16)
        )
        self.n_convs1 = nn.Sequential(
            convtext(64, 32, 3, 1, 1), convtext(32, 3, 3, 1, 1)
        )
        self.cconvs_fea = nn.Sequential(
            convtext(64, 32, 3, 1, 1), convtext(32, 32, 3, 1, 1)
        )
        self.cconvs_depth = nn.Sequential(
            convtext(33, 16, 3, 1, 1), convtext(16, 16, 3, 1, 1)
        )
        self.cconvs_prob = nn.Sequential(
            convtext(128, 64, 3, 1, 1),
            convtext(64, 32, 3, 1, 1),
            convtext(32, 1, 1, 1, 1)
        )
        self.cconvs_joint = nn.Sequential(
            convtext(49, 64, 3, 1, 1),
            convtext(64, 64, 3, 1, 2),
            convtext(64, 64, 3, 1, 4),
            convtext(64, 32, 3, 1, 1),
            conf_out(32, 1, 1, 1, 1)
        )
        
        # depth-prop
        self.cconvs_offsets = nn.Sequential(
            convtext(49, 64, 3, 1, 1),
            convtext(64, 64, 3, 1, 2),
            convtext(64, 64, 3, 1, 4),
            convtext(64, 32, 3, 1, 1),
            conf_out(32, self.cfg.num_neighbors, 1, 1, 1)
        )


        self.cconvs_nfea = nn.Sequential(
            convtext(64, 32, 3, 1, 1), convtext(32, 32, 3, 1, 1)
        )
        self.cconvs_normal = nn.Sequential(
            convtext(35, 16, 3, 1, 1), convtext(16, 16, 3, 1, 1)
        )
        self.cconvs_njoint = nn.Sequential(
            convtext(48, 64, 3, 1, 1),
            convtext(64, 64, 3, 1, 2),
            convtext(64, 64, 3, 1, 4),
            convtext(64, 32, 3, 1, 1),
            conf_out(32, 1, 1, 1, 1)
        )


        self.prop_layer = PropPlane(cfg, self.cfg.num_neighbors, 1, 3,
                                self.cfg.prop_kernel)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def refine(self, depth, normal, img, dconf, nconf, K, iter=5, profiler=None, guides=None):
        h, w = depth.shape[2], depth.shape[3]
        if profiler is not None:
            profiler.report_process('before init solver')

        solver = Solver(h=h, w=w, check_offsets=self.cfg.check_offsets, alpha1=self.cfg.solver_alpha1, \
            alpha2=self.cfg.solver_alpha2, sigma1=self.cfg.solver_sigma1, sigma2=self.cfg.solver_sigma2)

        refined_depth, refined_normal = depth.clone(), normal.clone().permute(0,3,1,2)
        if profiler is not None:
            profiler.report_process('init solver')
        
        for i in range(iter):
            refined_depth, refined_normal = \
                solver(refined_depth, refined_normal, img, dconf, nconf, K, profiler=profiler, offsets=guides)

        return refined_depth, refined_normal

    def forward(self, tgt_img, refs, pose, tgt_depth, tgt_normal, intrinsics, intrinsics_inv, factor = None):

        profiler = Profiler()
        intrinsics4 = intrinsics.clone()
        intrinsics_inv4 = intrinsics_inv.clone()
        intrinsics4[:,:2,:] = intrinsics4[:,:2,:] / 4
        intrinsics_inv4[:,:2,:2] = intrinsics_inv4[:,:2,:2] * 4
        
        # extract image features
        tgtimg_fea = self.feature_extraction(tgt_img)
        _b,_ch,_h,_w = tgtimg_fea.size()
            
        disp2depth = Variable(torch.ones(_b, _h, _w)).cuda() * self.mindepth * self.nlabel
        disps = Variable(torch.linspace(0,self.nlabel-1,self.nlabel).view(1,self.nlabel,1,1).expand(_b,self.nlabel,_h,_w)).type_as(disp2depth)

        depth = disp2depth.unsqueeze(1)/(disps + 1e-16)
        if factor is not None:
            depth = depth*factor
        
        refimg_feas = []
        # for each reference image
        for j, ref in enumerate(refs):
            # build the cost volume
            cost = Variable(torch.FloatTensor(tgtimg_fea.size()[0], tgtimg_fea.size()[1]*2, self.nlabel,  tgtimg_fea.size()[2],  tgtimg_fea.size()[3]).zero_()).cuda()
            refimg_fea  = self.feature_extraction(ref)
            refimg_feas.append(refimg_fea)

            refimg_fea_warp = inverse_warp_d(refimg_fea, depth, pose[:,j], intrinsics4, intrinsics_inv4)

            cost[:, :refimg_fea_warp.size()[1],:,:,:] = tgtimg_fea.unsqueeze(2).expand(_b,_ch,self.nlabel,_h,_w)
            cost[:, refimg_fea_warp.size()[1]:,:,:,:] = refimg_fea_warp.squeeze(-1)
            
            cost = cost.contiguous()
            cost0 = self.dres0(cost)
            
            cost_in0 = cost0.clone()
            
            cost0 = self.dres1(cost0) + cost0
            cost0 = self.dres2(cost0) + cost0 
            cost0 = self.dres3(cost0) + cost0 
            cost0 = self.dres4(cost0) + cost0
            
            cost_in0 = torch.cat((cost_in0, cost0.clone()), dim = 1)
            
            cost0 = self.classify(cost0)

            if j == 0:
                costs = cost0
                cost_in = cost_in0
            else:
                costs = costs + cost0
                cost_in = cost_in + cost_in0

        costs = costs / len(refs)
        # context network
        costs_context = Variable(torch.FloatTensor(tgtimg_fea.size()[0], 1, self.nlabel,  tgtimg_fea.size()[2],  tgtimg_fea.size()[3]).zero_()).cuda()
        for i in range(self.nlabel):
            costt = costs[:, :, i, :, :]
            costs_context[:, :, i, :, :] = self.convs(torch.cat([tgtimg_fea, costt],1)) + costt

        # depth regression
        costs_up = F.interpolate(costs, [self.nlabel,tgt_img.size()[2],tgt_img.size()[3]], mode='trilinear', align_corners = True)
        costs_up = torch.squeeze(costs_up,1)
        pred0 = F.softmax(costs_up,dim=1)
        pred0_r = pred0.clone()
        pred0 = disparityregression(self.nlabel)(pred0)
        depth0 = self.mindepth*self.nlabel/(pred0.unsqueeze(1)+1e-16)

        costss_up = F.interpolate(costs_context, [self.nlabel,tgt_img.size()[2],tgt_img.size()[3]], mode='trilinear', align_corners = True)
        costss_up = torch.squeeze(costss_up,1)
        pred = F.softmax(costss_up,dim=1)
        softmax = pred.clone()
        pred_ = disparityregression(self.nlabel)(pred)
        depth1 = self.mindepth*self.nlabel/(pred_.unsqueeze(1)+1e-16)
        
        # depth confidence network
        # Warped feature, depth prediction, and probability distribution
        depth_down = F.interpolate(depth1, [tgtimg_fea.size()[2], tgtimg_fea.size()[3]], mode='bilinear', align_corners=True)
        # Detach the gradient
        depth_down = depth_down.detach()
        for j, ref in enumerate(refs):
            refimg_fea = refimg_feas[j]
            refimg_fea = inverse_warp(refimg_fea, depth_down.squeeze(1), pose[:,j], intrinsics4, intrinsics_inv4)
            concat_fea = torch.cat([tgtimg_fea, refimg_fea], 1)
            fea_conf = self.cconvs_fea(concat_fea)
            if j == 0:
                feas_conf = fea_conf.clone()
            else:
                feas_conf = feas_conf + fea_conf
        feas_conf = feas_conf / len(refs)

        depth_conf = self.cconvs_depth(torch.cat([depth_down, tgtimg_fea], 1))
        cost_cat = torch.cat([costs.squeeze(1), costs_context.squeeze(1)], 1)
        prob_conf = self.cconvs_prob(cost_cat)
        # Joint confidence fusion
        joint_fea = torch.cat([feas_conf, depth_conf, prob_conf], 1)
        joint_conf = self.cconvs_joint(joint_fea)
        joint_depth_conf = F.interpolate(joint_conf, [tgt_img.size()[2], tgt_img.size()[3]], mode='bilinear', align_corners=True)

        # deptch guidence
        joint_guide = self.cconvs_offsets(joint_fea)
        joint_depth_guides = F.interpolate(joint_guide, [tgt_img.size()[2], tgt_img.size()[3]], mode='bilinear', align_corners=True)

        b,ch,d,h,w = cost_in.size()

        # normal regression
        # concat cost volume with world coordinates
        with torch.no_grad():
            intrinsics_inv[:,:2,:2] = intrinsics_inv[:,:2,:2] * (4)
            disp2depth = Variable(torch.ones(b, h, w).cuda() * self.mindepth * self.nlabel).cuda()
            disps = Variable(torch.linspace(0,self.nlabel-1,self.nlabel).view(1,self.nlabel,1,1).expand(b,self.nlabel,h,w)).type_as(disp2depth)
            depth = disp2depth.unsqueeze(1)/(disps + 1e-16)
            if factor is not None:
                depth = depth*factor  
            
            world_coord = pixel2cam(depth, intrinsics_inv)                
            world_coord = world_coord.squeeze(-1)

        if factor is not None:
            world_coord = world_coord / (2*self.nlabel*self.mindepth*factor.unsqueeze(-1))
        else:
            world_coord = world_coord / (2*self.nlabel*self.mindepth)
        
        
        world_coord = world_coord.clamp(-1,1)
        world_coord = torch.cat((world_coord.clone(), cost_in), dim = 1) #B,ch+3,D,H,W
        world_coord = world_coord.contiguous()
        
        if self.no_pool:
            wc0 = self.pool1(self.wc0(world_coord))
        else:
            wc0 = self.pool3(self.pool2(self.pool1(self.wc0(world_coord))))

        # regress the normal from cost volume slices
        slices = []
        nmap = torch.zeros((b,3,h,w)).type_as(wc0)
        for i in range(wc0.size(2)):
            normal_fea = self.n_convs0(wc0[:,:,i])
            slices.append(self.n_convs1(normal_fea))
            if i == 0:
                nfea_conf = self.cconvs_nfea(normal_fea).clone()
            else:
                nfea_conf = nfea_conf + self.cconvs_nfea(normal_fea)
            nmap += slices[-1]        
        
        nmap_nor = F.normalize(nmap, dim=1)
        nmap_nor = nmap_nor.detach()
        
        # normal confidence network
        nfea_conf = nfea_conf / wc0.size(2)
        normal_conf = self.cconvs_normal(torch.cat([nmap_nor, tgtimg_fea], 1))
        joint_normal_conf = self.cconvs_njoint(torch.cat([nfea_conf, normal_conf], 1))
        joint_normal_conf = F.interpolate(joint_normal_conf, [tgt_img.size()[2], tgt_img.size()[3]], mode='bilinear', align_corners=True)
        joint_normal_conf = torch.clamp(joint_normal_conf, 0.01, 1.0)

        nmap_out = F.interpolate(nmap, [tgt_img.size(2), tgt_img.size(3)], mode = 'bilinear', align_corners = True)
        nmap_out = nmap_out.permute(0,2,3,1)
        nmap_out = F.normalize(nmap_out, dim=-1)
        
        # groundtruth depth and normal confidence
        gt_depth_conf = 1.0 - self.cfg.conf_dgamma * torch.abs(depth1 - tgt_depth.unsqueeze(1)) / (tgt_depth.unsqueeze(1) + 1e-6)
        gt_depth_conf = torch.clamp(gt_depth_conf, 0.01, 1.0).detach()
        total_angles_m = compute_angles(tgt_normal.permute(0,2,3,1), nmap_out, dim=-1)
        gt_nconf = 1.0 - self.cfg.conf_ngamma * total_angles_m / 180.0
        gt_normal_conf = torch.clamp(gt_nconf, 0.01, 1.0).unsqueeze(1).detach()
        
        # refine the depth and normal through solver
        # from pdb import set_trace; set_trace()
        # refined_depth, refined_normal = self.refine(depth1, nmap_out, tgt_img, gt_depth_conf, gt_normal_conf, 
        #                                 intrinsics, iter=self.iter, profiler=None, guides=joint_depth_guides)

        if self.training:
            refined_depth, refined_normal = \
                self.prop_layer(depth1, nmap_out, tgt_img, gt_depth_conf, gt_normal_conf, joint_depth_guides, intrinsics, tgtimg_fea=tgtimg_fea)
        else:
            refined_depth, refined_normal = \
                self.prop_layer(depth1, nmap_out, tgt_img, joint_depth_conf, joint_normal_conf, joint_depth_guides, intrinsics, tgtimg_fea=tgtimg_fea)


        refined_normal = refined_normal.permute(0,2,3,1)
        refined_normal = F.normalize(refined_normal, dim=-1)
        refined_normal = -torch.sign(refined_normal[:,:,:,2:]) * refined_normal
        

        # add outputs
        depth1, refined_depth = depth1.squeeze(1), refined_depth.squeeze(1)
        nmap_out, refined_normal = nmap_out.permute(0,3,1,2), refined_normal.permute(0,3,1,2)
        joint_depth_conf, joint_normal_conf = joint_depth_conf.squeeze(1), joint_normal_conf.squeeze(1)
        return_vals = []
        if self.training:
            return_vals += [depth1, refined_depth]
            return_vals += [nmap_out, refined_normal]
        else:
            return_vals += [refined_depth]
            return_vals += [refined_normal]
        
        return_vals += [joint_depth_conf, joint_normal_conf]
        return return_vals

    def inferencebatch(self, tgt_img, refs, pose, intrinsics, intrinsics_inv, factor = None):

        profiler = Profiler()
        intrinsics4 = intrinsics.clone()
        intrinsics_inv4 = intrinsics_inv.clone()
        intrinsics4[:,:2,:] = intrinsics4[:,:2,:] / 4
        intrinsics_inv4[:,:2,:2] = intrinsics_inv4[:,:2,:2] * 4
        
        # extract image features
        tgtimg_fea = self.feature_extraction(tgt_img)
        _b,_ch,_h,_w = tgtimg_fea.size()
            
        disp2depth = Variable(torch.ones(_b, _h, _w)).cuda() * self.mindepth * self.nlabel
        disps = Variable(torch.linspace(0,self.nlabel-1,self.nlabel).view(1,self.nlabel,1,1).expand(_b,self.nlabel,_h,_w)).type_as(disp2depth)

        depth = disp2depth.unsqueeze(1)/(disps + 1e-16)
        if factor is not None:
            depth = depth*factor
        
        refimg_feas = []
        # for each reference image
        for j, ref in enumerate(refs):
            # build the cost volume
            cost = Variable(torch.FloatTensor(tgtimg_fea.size()[0], tgtimg_fea.size()[1]*2, self.nlabel,  tgtimg_fea.size()[2],  tgtimg_fea.size()[3]).zero_()).cuda()
            refimg_fea  = self.feature_extraction(ref)
            refimg_feas.append(refimg_fea)

            refimg_fea_warp = inverse_warp_d(refimg_fea, depth, pose[:,j], intrinsics4, intrinsics_inv4)

            cost[:, :refimg_fea_warp.size()[1],:,:,:] = tgtimg_fea.unsqueeze(2).expand(_b,_ch,self.nlabel,_h,_w)
            cost[:, refimg_fea_warp.size()[1]:,:,:,:] = refimg_fea_warp.squeeze(-1)
            
            cost = cost.contiguous()
            cost0 = self.dres0(cost)
            
            cost_in0 = cost0.clone()
            
            cost0 = self.dres1(cost0) + cost0
            cost0 = self.dres2(cost0) + cost0 
            cost0 = self.dres3(cost0) + cost0 
            cost0 = self.dres4(cost0) + cost0
            
            cost_in0 = torch.cat((cost_in0, cost0.clone()), dim = 1)
            
            cost0 = self.classify(cost0)

            if j == 0:
                costs = cost0
                cost_in = cost_in0
            else:
                costs = costs + cost0
                cost_in = cost_in + cost_in0

        costs = costs / len(refs)
        # context network
        costs_context = Variable(torch.FloatTensor(tgtimg_fea.size()[0], 1, self.nlabel,  tgtimg_fea.size()[2],  tgtimg_fea.size()[3]).zero_()).cuda()
        for i in range(self.nlabel):
            costt = costs[:, :, i, :, :]
            costs_context[:, :, i, :, :] = self.convs(torch.cat([tgtimg_fea, costt],1)) + costt

        # depth regression
        costs_up = F.interpolate(costs, [self.nlabel,tgt_img.size()[2],tgt_img.size()[3]], mode='trilinear', align_corners = True)
        costs_up = torch.squeeze(costs_up,1)
        pred0 = F.softmax(costs_up,dim=1)
        pred0_r = pred0.clone()
        pred0 = disparityregression(self.nlabel)(pred0)
        depth0 = self.mindepth*self.nlabel/(pred0.unsqueeze(1)+1e-16)

        costss_up = F.interpolate(costs_context, [self.nlabel,tgt_img.size()[2],tgt_img.size()[3]], mode='trilinear', align_corners = True)
        costss_up = torch.squeeze(costss_up,1)
        pred = F.softmax(costss_up,dim=1)
        softmax = pred.clone()
        pred_ = disparityregression(self.nlabel)(pred)
        depth1 = self.mindepth*self.nlabel/(pred_.unsqueeze(1)+1e-16)
        
        # depth confidence network
        # Warped feature, depth prediction, and probability distribution
        depth_down = F.interpolate(depth1, [tgtimg_fea.size()[2], tgtimg_fea.size()[3]], mode='bilinear', align_corners=True)
        # Detach the gradient
        depth_down = depth_down.detach()
        for j, ref in enumerate(refs):
            refimg_fea = refimg_feas[j]
            refimg_fea = inverse_warp(refimg_fea, depth_down.squeeze(1), pose[:,j], intrinsics4, intrinsics_inv4)
            concat_fea = torch.cat([tgtimg_fea, refimg_fea], 1)
            fea_conf = self.cconvs_fea(concat_fea)
            if j == 0:
                feas_conf = fea_conf.clone()
            else:
                feas_conf = feas_conf + fea_conf
        feas_conf = feas_conf / len(refs)

        depth_conf = self.cconvs_depth(torch.cat([depth_down, tgtimg_fea], 1))
        cost_cat = torch.cat([costs.squeeze(1), costs_context.squeeze(1)], 1)
        prob_conf = self.cconvs_prob(cost_cat)
        # Joint confidence fusion
        joint_fea = torch.cat([feas_conf, depth_conf, prob_conf], 1)
        joint_conf = self.cconvs_joint(joint_fea)
        joint_depth_conf = F.interpolate(joint_conf, [tgt_img.size()[2], tgt_img.size()[3]], mode='bilinear', align_corners=True)

        # deptch guidence
        joint_guide = self.cconvs_offsets(joint_fea)
        joint_depth_guides = F.interpolate(joint_guide, [tgt_img.size()[2], tgt_img.size()[3]], mode='bilinear', align_corners=True)

        b,ch,d,h,w = cost_in.size()

        # normal regression
        # concat cost volume with world coordinates
        with torch.no_grad():
            intrinsics_inv[:,:2,:2] = intrinsics_inv[:,:2,:2] * (4)
            disp2depth = Variable(torch.ones(b, h, w).cuda() * self.mindepth * self.nlabel).cuda()
            disps = Variable(torch.linspace(0,self.nlabel-1,self.nlabel).view(1,self.nlabel,1,1).expand(b,self.nlabel,h,w)).type_as(disp2depth)
            depth = disp2depth.unsqueeze(1)/(disps + 1e-16)
            if factor is not None:
                depth = depth*factor  
            
            world_coord = pixel2cam(depth, intrinsics_inv)                
            world_coord = world_coord.squeeze(-1)

        if factor is not None:
            world_coord = world_coord / (2*self.nlabel*self.mindepth*factor.unsqueeze(-1))
        else:
            world_coord = world_coord / (2*self.nlabel*self.mindepth)
        
        
        world_coord = world_coord.clamp(-1,1)
        world_coord = torch.cat((world_coord.clone(), cost_in), dim = 1) #B,ch+3,D,H,W
        world_coord = world_coord.contiguous()
        
        if self.no_pool:
            wc0 = self.pool1(self.wc0(world_coord))
        else:
            wc0 = self.pool3(self.pool2(self.pool1(self.wc0(world_coord))))

        # regress the normal from cost volume slices
        slices = []
        nmap = torch.zeros((b,3,h,w)).type_as(wc0)
        for i in range(wc0.size(2)):
            normal_fea = self.n_convs0(wc0[:,:,i])
            slices.append(self.n_convs1(normal_fea))
            if i == 0:
                nfea_conf = self.cconvs_nfea(normal_fea).clone()
            else:
                nfea_conf = nfea_conf + self.cconvs_nfea(normal_fea)
            nmap += slices[-1]        
        
        nmap_nor = F.normalize(nmap, dim=1)
        nmap_nor = nmap_nor.detach()
        
        # normal confidence network
        nfea_conf = nfea_conf / wc0.size(2)
        normal_conf = self.cconvs_normal(torch.cat([nmap_nor, tgtimg_fea], 1))
        joint_normal_conf = self.cconvs_njoint(torch.cat([nfea_conf, normal_conf], 1))
        joint_normal_conf = F.interpolate(joint_normal_conf, [tgt_img.size()[2], tgt_img.size()[3]], mode='bilinear', align_corners=True)
        joint_normal_conf = torch.clamp(joint_normal_conf, 0.01, 1.0)

        nmap_out = F.interpolate(nmap, [tgt_img.size(2), tgt_img.size(3)], mode = 'bilinear', align_corners = True)
        nmap_out = nmap_out.permute(0,2,3,1)
        nmap_out = F.normalize(nmap_out, dim=-1)
        
        # from pdb import set_trace; set_trace()

        if False:
            refined_depth, refined_normal = \
                self.prop_layer(depth1, nmap_out, tgt_img, joint_depth_conf, joint_normal_conf, joint_depth_guides, intrinsics)

            refined_normal = refined_normal.permute(0,2,3,1)
            refined_normal = F.normalize(refined_normal, dim=-1)
            refined_normal = -torch.sign(refined_normal[:,:,:,2:]) * refined_normal
            

            # add outputs
            depth1, refined_depth = depth1.squeeze(1), refined_depth.squeeze(1)
            nmap_out, refined_normal = nmap_out.permute(0,3,1,2), refined_normal.permute(0,3,1,2)
            joint_depth_conf, joint_normal_conf = joint_depth_conf.squeeze(1), joint_normal_conf.squeeze(1)
            return_vals = []
            if self.training:
                return_vals += [depth1, refined_depth]
                return_vals += [nmap_out, refined_normal]
            else:
                return_vals += [refined_depth]
                return_vals += [refined_normal]
            
            return_vals += [joint_depth_conf, joint_normal_conf]
            return return_vals
        else:
            offsets, aff = \
                self.prop_layer.get_aff(depth1, nmap_out, tgt_img, joint_depth_conf, joint_normal_conf, joint_depth_guides, intrinsics, tgtimg_fea=tgtimg_fea)
            # from pdb import set_trace; set_trace()
            depth1 = depth1.squeeze(1)
            nmap_out = nmap_out.permute(0,3,1,2)
            joint_depth_conf, joint_normal_conf = joint_depth_conf.squeeze(1), joint_normal_conf.squeeze(1)
            return_vals = []

            return_vals += [depth1]
            return_vals += [nmap_out]
            
            return_vals += [joint_depth_conf, joint_normal_conf]
            return_vals += [offsets, aff]
            return return_vals
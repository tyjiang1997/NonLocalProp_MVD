from bdb import set_trace
import imp
from operator import is_
import os
from re import I
import time
import csv
import numpy as np
from path import Path
import argparse
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import cv2
import torch
import torch.nn.functional as F
from core.dataset import custom_transforms
from core.networks.MVDNet_conf import MVDNet_conf
from core.networks.MVDNet_joint import MVDNet_joint
from core.networks.MVDNet_nslpn import MVDNet_nslpn
from core.networks.MVDNet_prop import  MVDNet_prop
from core.utils.utils import load_config_file
from core.networks.loss_functions import compute_errors_test_batch, compute_angles, cross_entropy

from core.utils.logger import AverageMeter
from core.dataset import SequenceFolder, NoisySequenceFolder
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import *
from core.utils.utils import load_config_file, normalize_depth_for_display, vis_normal

parser = argparse.ArgumentParser(description='Iterative solver for multi-view depth and normal',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('config_file', metavar='DIR', help='path to config file')
parser.add_argument("--local_rank", default=-1)
parser.add_argument("--seed", type=int, default=1, metavar='S', help='random seed')

args = parser.parse_args()
set_random_seed(args.seed)

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1

local_rank = int(args.local_rank)
cfg = load_config_file(args.config_file)
cfg.local_rank = args.local_rank
device = torch.device(local_rank)
if is_distributed:
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()

# save writer
save_path = Path(cfg.output_dir)

def main(cfg):
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cfg.cuda) 

    global n_iter

    # Loading data
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
    if local_rank == 0 or not is_distributed:
        print("=> fetching scenes in '{}'".format(cfg.dataset_path))
    
    if cfg.dataset == 'scannet':     
        test_set = SequenceFolder(cfg.dataset_path, transform=valid_transform, ttype=cfg.test_list) 
    else:
        raise NotImplementedError

    if local_rank == 0 or not is_distributed:
        print('{} samples found in {} test scenes'.format(len(test_set), len(test_set.scenes)))

    if is_distributed:
        test_sampler = torch.utils.data.DistributedSampler(test_set, num_replicas=dist.get_world_size(),
                                                           rank=dist.get_rank())         
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True, sampler=test_sampler)
    else:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

    # create model
    if local_rank == 0 or not is_distributed:
        print("=> creating model")
    if cfg.model_name == 'MVDNet_conf':
        mvdnet = MVDNet_conf(cfg).cuda()
    elif cfg.model_name == 'MVDNet_joint':
        mvdnet = MVDNet_joint(cfg).cuda()
    elif cfg.model_name == 'MVDNet_nslpn':
        mvdnet = MVDNet_nslpn(cfg).cuda()
    elif cfg.model_name == 'MVDNet_prop':
        mvdnet = MVDNet_prop(cfg).cuda()
    else:
        raise NotImplementedError
    
    
    if not os.path.isfile(cfg.pretrained_mvdn):
        pretrained_mvdn = save_path / 'checkpoints' / sorted(os.listdir((save_path / 'checkpoints')))[-1]
    else:
        pretrained_mvdn = cfg.pretrained_mvdn
    
    if local_rank == 0 or not is_distributed:
        print(f"=> loading weights for MVDNet: {pretrained_mvdn}")
    weights = torch.load(pretrained_mvdn)   
    mvdnet.load_state_dict(weights['state_dict'], strict=True)

    torch.backends.cudnn.benchmark = True

    mvdnet.to(device)
    if is_distributed:
        mvdnet = DDP(mvdnet,device_ids=[local_rank], output_device=local_rank) 
    else:
        mvdnet = torch.nn.DataParallel(mvdnet)

    errors, error_names = validate_with_gt(cfg, test_loader, mvdnet)

    if local_rank == 0  or not is_distributed:

        decisive_error = errors[0]
        with open(save_path/'eval_log_summary.txt', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([decisive_error, errors[1], errors[2], errors[3], errors[4], errors[5], errors[6],  errors[7]])
        


def validate_with_gt(cfg, test_loader, mvdnet):
    batch_time = AverageMeter()
    test_error_names = ['abs_rel','abs_diff','sq_rel','rms','log_rms','a1','a2','a3', 'dconf', 'nconf', 'mean_angle']
    test_errors = AverageMeter(i=len(test_error_names))

    mvdnet.eval()

    end = time.time()
    with torch.no_grad(): 
        for i, (tgt_img, ref_imgs, gt_nmap, ref_poses, intrinsics, intrinsics_inv, tgt_depth, ref_depths, tgt_id) in enumerate(test_loader):
            tgt_img_var = tgt_img.cuda()
            ref_imgs_var = [img.cuda() for img in ref_imgs]
            gt_nmap_var = gt_nmap.cuda()
            ref_poses_var = [pose.cuda() for pose in ref_poses]
            intrinsics_var = intrinsics.cuda()
            intrinsics_inv_var = intrinsics_inv.cuda()
            tgt_depth_var = tgt_depth.cuda()

            pose = torch.cat(ref_poses_var,1)
            if (pose != pose).any():
                continue
            
            if cfg.model_name == 'MVDNet_conf':
                outputs = mvdnet(tgt_img_var, ref_imgs_var, pose, intrinsics_var, intrinsics_inv_var)
            elif cfg.model_name == 'MVDNet_joint':
                outputs = mvdnet(tgt_img_var, ref_imgs_var, pose, tgt_depth_var, gt_nmap_var, intrinsics_var, intrinsics_inv_var)
            elif cfg.model_name == 'MVDNet_nslpn':
                outputs = mvdnet(tgt_img_var, ref_imgs_var, pose, intrinsics_var, intrinsics_inv_var)
            elif cfg.model_name == 'MVDNet_prop':
                outputs = mvdnet(tgt_img_var, ref_imgs_var, pose, tgt_depth_var, gt_nmap_var, intrinsics_var, intrinsics_inv_var)
            else:
                raise NotImplementedError
            
            output_depth = outputs[0].data.cpu()
            nmap = outputs[1].permute(0,2,3,1)
            dconf, nconf = outputs[-2], outputs[-1]
            
            mask = (tgt_depth <= 10) & (tgt_depth >= 0.5) & (tgt_depth == tgt_depth)

            if not mask.any():
                continue
            
            n = tgt_depth.shape[0]
            test_errors_ = list(compute_errors_test_batch(tgt_depth[mask], output_depth[mask]))
            gt_dconf = 1.0 - cfg.conf_dgamma * torch.abs(tgt_depth - output_depth) / (tgt_depth + 1e-6)
            dconf_e = torch.abs(dconf.cpu()[mask] - gt_dconf[mask]).mean()
            test_errors_.append(dconf_e.item())

            n_mask = (gt_nmap_var.permute(0,2,3,1)[0,:,:] != 0)
            n_mask = n_mask[:,:,0] | n_mask[:,:,1] | n_mask[:,:,2]

            total_angles_m = compute_angles(gt_nmap_var.permute(0,2,3,1)[0], nmap[0])
            gt_nconf = 1.0 - cfg.conf_ngamma * total_angles_m / 180.0
            nconf_e = torch.abs(nconf[0][n_mask] - gt_nconf[n_mask]).mean()
            test_errors_.append(nconf_e.item())
            
            mask_angles = total_angles_m[n_mask]
            total_angles_m[~ n_mask] = 0
            test_errors_.append(torch.mean(mask_angles).item())
            
            # from pdb import set_trace; set_trace()
            test_errors.update(test_errors_, n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if (i % cfg.print_freq == 0 or i == len(test_loader)-1) and (local_rank==0 or not is_distributed):
                print('valid: Time {} Rel Error {:.4f} ({:.4f}) DConf Error {:.4f} ({:.4f}) Iter {}/{}'.format(batch_time, test_errors.val[0], test_errors.avg[0], test_errors.val[-3], test_errors.avg[-3], i, len(test_loader)))
            if cfg.save_samples:
                output_dir = Path(os.path.join(cfg.output_dir, 'evalvis'))
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                output_depth = output_depth.numpy()
                nmap_ = nmap.cpu().numpy()
                for picid, imgsave, normal in zip(tgt_id, output_depth, nmap_):
                    # from pdb import set_trace; set_trace()
                    depth_nor = normalize_depth_for_display(imgsave)
                    plt.imsave(output_dir/ f'{picid}_depth.png', depth_nor)

                    normal_n = normal / (np.linalg.norm(normal, axis=2, keepdims=True) + 1e-10)
                    normal_img = ((normal_n + 1.0) / 2.0) * 255.0
                    cv2.imwrite(output_dir/ f'{picid}_normal.png', normal_img[:,:,::-1].astype(np.uint8))

        
        if is_distributed :
            rank, world_size = get_dist_info()
            errors = merge_results_dist(test_errors, world_size, tmpdir= output_dir / 'tmpdir')
        else:
            errors = test_errors.avg
        # print(f'local{rank}',errors)
    return errors, test_error_names


if __name__ == '__main__':

    n_iter = 0
    main(cfg)

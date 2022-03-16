import torch.distributed as dist
import numpy as np
import torchvision.utils as vutils
import torch, random
import torch.nn.functional as F
from prettytable import PrettyTable
import os
import pickle
import shutil
import copy

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


    
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = copy.deepcopy(part_list[0])
    for res in (part_list[1:]):
        ordered_results._count += res._count
        for i, (_val, _sum) in enumerate(zip(res.val, res.sum)):
            ordered_results.val[i] = _val
            ordered_results.sum[i] += _sum
            ordered_results.avg[i] = ordered_results.sum[i] / ordered_results._count
    shutil.rmtree(tmpdir)
    return ordered_results.avg

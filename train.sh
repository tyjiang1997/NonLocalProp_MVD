
## 第一波实验，验证各个模块的有效性
python3 -m torch.distributed.launch --nproc_per_node 2 ddptrain.py configs/scannet_prop.yaml 
python3 -m torch.distributed.launch --nproc_per_node 2 ddptrain.py configs/scannet_stage1.yaml 
python3 -m torch.distributed.launch --nproc_per_node 2 ddptrain.py configs/scannet_propcos.yaml 
python3 -m torch.distributed.launch --nproc_per_node 2 ddptrain.py configs/scannet_propcosonly.yaml 
python3 -m torch.distributed.launch --nproc_per_node 2 ddptrain.py configs/scannet_nlspn.yaml 

# # noisy
# python3 -m torch.distributed.launch --nproc_per_node 2 ddptrain.py configs/noisy_scannet_stage1.yaml 
# python3 -m torch.distributed.launch --nproc_per_node 2 ddptrain.py configs/noisy_scannet_prop_wfliter.yaml 


# ## 第二波实验：
# #验证multi-fliter是否真有利于提出不易学习的噪声
# python3 -m torch.distributed.launch --nproc_per_node 2 ddptrain.py configs/scannet_prop_wfilter.yaml 
# #验证平滑约束项是否起作用
# python3 -m torch.distributed.launch --nproc_per_node 2 ddptrain.py configs/scannet_propnsmooth.yaml 




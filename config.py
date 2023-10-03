"""
 @Time    : 2021/8/18 22:21
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : 2023_IJCV_OPNet
 @File    : config.py
 @Function:
 
"""
import os

# download from https://github.com/pengzhiliang/Conformer
backbone_path = './Conformer_base_patch16.pth'

datasets_root = '/home/data'

cod_training_root = os.path.join(datasets_root, 'train')

chameleon_path = os.path.join(datasets_root, 'test/CHAMELEON')
camo_path = os.path.join(datasets_root, 'test/CAMO')
cod10k_path = os.path.join(datasets_root, 'test/COD10K')
nc4k_path = os.path.join(datasets_root, 'test/NC4K')

"""
 @Time    : 2021/8/18 22:20
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : 2023_IJCV_OPNet
 @File    : add_color.py
 @Function:
 
"""
import os
import sys
sys.path.append('..')
import numpy as np
from skimage import io
from skimage.transform import resize
from PIL import Image
from misc import check_mkdir

ok_method = ['B', 'GT']
ref_root_dir = '/xxx'
input_root_dir = '/xxx'
output_root_dir = '/xxx'
check_mkdir(output_root_dir)

color= [255, 0, 0]

method_list = os.listdir(input_root_dir)
for method in method_list:
    if method in ok_method:
        continue
    else:
        method_dir = os.path.join(input_root_dir, method)
        dataset_list = os.listdir(method_dir)
        for dataset in dataset_list:
            dataset_dir = os.path.join(method_dir, dataset)
            mask_list = os.listdir(dataset_dir)
            output_dir = os.path.join(output_root_dir, method, dataset)
            check_mkdir(output_dir)
            for mask_name in mask_list:
                image_path = os.path.join(ref_root_dir, dataset, 'image', mask_name[:-4] + '.jpg')
                image = io.imread(image_path)

                mask_path = os.path.join(dataset_dir, mask_name)
                mask = Image.open(mask_path).convert('L')
                mask = np.array(mask)
                if image.shape[:2] != mask.shape:
                    mask = resize(mask, image.shape[:2]) * 255.0
                    print(mask_name)

                output = np.zeros_like(image)

                for j in range(image.shape[2]):
                    if j != 3:
                        output[:, :, j] = np.where(mask >= 127.5, image[:, :, j] * 0.4 + 0.6 * color[j], image[:, :, j] * 0.5)
                    else:
                        output[:, :, j] = image[:, :, j]

                io.imsave(os.path.join(output_dir, mask_name[:-4] + "_color.jpg"), output)

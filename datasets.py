"""
 @Time    : 2021/8/18 22:57
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : 2023_IJCV_OPNet
 @File    : datasets.py
 @Function:
 
"""
import os
import os.path
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

def make_dataset(root):
    image_path = os.path.join(root, 'image')
    mask_path = os.path.join(root, 'mask')
    img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.jpg')]
    return [(os.path.join(image_path, img_name + '.jpg'), os.path.join(mask_path, img_name + '.png')) for img_name in img_list]

class ImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def collate(self, batch):
        size = [320, 352, 384, 416][np.random.randint(0, 4)]
        image, mask = [list(item) for item in zip(*batch)]

        image = torch.stack(image, dim=0)
        image = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=True)
        mask = torch.stack(mask, dim=0)
        mask = F.interpolate(mask, size=(size, size), mode="nearest")

        return image, mask

    def __len__(self):
        return len(self.imgs)


def make_dataset3(root):
    image_path = os.path.join(root, 'image')
    edge_path = os.path.join(root, 'edge')
    mask_path = os.path.join(root, 'mask')
    img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.jpg')]
    return [(os.path.join(image_path, img_name + '.jpg'), os.path.join(edge_path, img_name + '.png'), os.path.join(mask_path, img_name + '.png')) for img_name in img_list]

class ImageFolder3(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None, target_transform1=None, target_transform2=None):
        self.root = root
        self.imgs = make_dataset3(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform1 = target_transform1
        self.target_transform2 = target_transform2

    def __getitem__(self, index):
        img_path, edge_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        edge = Image.open(edge_path).convert('L')
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            img, edge, target = self.joint_transform(img, edge, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform1 is not None:
            edge = self.target_transform1(edge)
        if self.target_transform2 is not None:
            target = self.target_transform2(target)

        return img, edge, target

    def __len__(self):
        return len(self.imgs)

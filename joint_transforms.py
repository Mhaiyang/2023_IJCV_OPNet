"""
 @Time    : 2021/8/18 22:58
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : 2023_IJCV_OPNet
 @File    : joint_transforms.py
 @Function:
 
"""
import random

from PIL import Image

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)  PIL: (w, h)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


class Compose3(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, edge, mask):
        assert img.size == edge.size
        assert img.size == mask.size
        for t in self.transforms:
            img, edge, mask = t(img, edge, mask)
        return img, edge, mask

class RandomHorizontallyFlip3(object):
    def __call__(self, img, edge, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), edge.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, edge, mask

class Resize3(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)  PIL: (w, h)

    def __call__(self, img, edge, mask):
        assert img.size == edge.size
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), edge.resize(self.size, Image.NEAREST), mask.resize(self.size, Image.NEAREST)

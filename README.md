# 2023_IJCV_OPNet

## Camouflaged Object Segmentation with Omni Perception
[Haiyang Mei](https://mhaiyang.github.io/), [Ke Xu](https://kkbless.github.io/), Yunduo Zhou, Yang Wang, Haiyin Piao, Xiaopeng Wei, [Xin Yang](https://xinyangdut.github.io/)

[[Paper](https://link.springer.com/article/10.1007/s11263-023-01838-2)] [[Project Page](https://mhaiyang.github.io/IJCV2023-OPNet/index.html)]

### Abstract
Camouflaged object segmentation (COS) is a very challenging task due to the deceitful appearances of the candidate objects to the noisy backgrounds. Most existing state-of-the-art methods mimic the first-positioning-then-focus mechanism of predators, but still fail in positioning camouflaged objects in cluttered scenes or delineating their boundaries. The key reason is that their methods do not have a comprehensive understanding of the scene when they spot and focus on the objects, so that they are easily attracted by local surroundings. An ideal COS model should be able to process local and global information at the same time, i.e., to have omni perception of the scene through the whole process of camouflaged object segmentation. To this end, we propose to learn the omni perception for the first-positioning-then-focus COS scheme. Specifically, we propose an omni perception network (OPNet) with two novel modules, i.e., the pyramid positioning module (PPM) and dual focus module (DFM). They are proposed to integrate local features and global representations for accurate positioning of the camouflaged objects and focus on their boundaries, respectively. Extensive experiments demonstrate that our method, which runs at 54 fps, significantly outperforms 15 cutting-edge models on 4 challenging datasets under 4 standard metrics. 

### Citation
If you use this code, please cite:

```
@InProceedings{Haiyang:OPNet:2023,
    author = {Mei, Haiyang and Xu, Ke and Zhou, Yunduo and Wang, Yang and Piao, Haiyin and Wei, Xiaopeng and Yang, Xin.},
    title = {Camouflaged Object Segmentation with Omni Perception},
    booktitle = {International Journal of Computer Vision (IJCV)},
    month = {June},
    year = {2023}
} 
```

### License
Please see `LICENSE`

### Contact
E-Mail: haiyang.mei@outlook.com
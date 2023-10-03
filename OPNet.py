"""
 @Time    : 2021/9/27 16:51
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : 2023_IJCV_OPNet
 @File    : OPNet.py
 @Function:
 
"""
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import numpy as np

###################################################################
# ########################## Conformer ############################
###################################################################
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), out_features=None,
                 skip=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_features, act_layer=act_layer,
                       drop=drop)
        self.skip = skip

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.skip:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1,
                               bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]

        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), ):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        H = int(H)
        W = int(W)
        H_T = int(self.up_stride * H)
        W_T = int(self.up_stride * W)
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H_T, W_T))


class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """

    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride,
                                   groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True,
                                          groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)

        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t)

        x_t = self.trans_block(x_st + x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t


class Conformer(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):

        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.trans_cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )

        # 2~4 stage
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )

        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8 stage
        init_stage = fin_stage  # 5
        fin_stage = fin_stage + depth // 3  # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block, last_fusion=last_fusion
                            )
                            )
        self.fin_stage = fin_stage

        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def forward(self, x):
        conv_features = []
        tran_features = []
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # pdb.set_trace()
        # stem stage [N, 3, h, w] -> [N, 64, h / 4, w / 4]
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        # 1 stage
        x = self.conv_1(x_base, return_x_2=False)

        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t = self.trans_1(x_t)

        conv_features.append(x)
        tran_features.append(x_t)

        # 2 ~ final
        for i in range(2, self.fin_stage):
            x, x_t = eval('self.conv_trans_' + str(i))(x, x_t)

            conv_features.append(x)
            tran_features.append(x_t)

        # conv classification
        # x_p = self.pooling(x).flatten(1)
        # conv_cls = self.conv_cls_head(x_p)

        # trans classification
        # x_t = self.trans_norm(x_t)
        # tran_cls = self.trans_cls_head(x_t[:, 0])

        # return [conv_cls, tran_cls]
        return conv_features, tran_features


###################################################################
# #################### Pyramid Positioning ########################
###################################################################
class Conv_Down(nn.Module):
    def __init__(self, conv_channel, tran_channel, scale):
        super(Conv_Down, self).__init__()
        self.conv_project = nn.Conv2d(conv_channel, tran_channel, 1, 1, 0)
        self.pooling = nn.AdaptiveAvgPool2d(scale)
        self.norm = partial(nn.LayerNorm, eps=1e-6)(tran_channel)
        self.act = nn.GELU()

    def forward(self, x, t_all):
        conv_project = self.conv_project(x)
        pooling = self.pooling(conv_project).flatten(2).transpose(1, 2)
        norm = self.norm(pooling)
        act = self.act(norm)

        new = torch.cat([t_all[:, 0, :][:, None, :], act], 1)

        return new


class Tran_Down(nn.Module):
    def __init__(self, scale):
        super(Tran_Down, self).__init__()
        self.scale = scale
        self.pooling = nn.AdaptiveAvgPool2d(self.scale)

    def forward(self, t_all):
        t_seg = t_all[:, 0, :][:, None, :]
        t = t_all[:, 1:, :]
        B, N, C = t.shape
        # [B, N, C] --> [B, C, n, n] --> [B, C, n_pool, n_pool] --> [B, N_pool, C]
        t_reshape = t.transpose(1, 2).reshape(B, C, int(np.sqrt(N)), int(np.sqrt(N)))
        t_pool = self.pooling(t_reshape).reshape(B, C, pow(self.scale, 2)).transpose(1, 2)
        t_new = torch.cat([t_seg, t_pool], 1)

        return t_new


class Fusion_Down(nn.Module):
    def __init__(self):
        super(Fusion_Down, self).__init__()

    def forward(self, t_conv, t_tran):
        t_fusion = t_conv + t_tran

        return t_fusion


class Conv_Up(nn.Module):
    def __init__(self, tran_scale_channel, conv_scale_channel):
        super(Conv_Up, self).__init__()
        self.conv_project = nn.Conv2d(tran_scale_channel, conv_scale_channel, 1, 1, 0)
        self.bn = partial(nn.BatchNorm2d, eps=1e-6)(conv_scale_channel)
        self.relu = nn.ReLU()

    def forward(self, t_all, side_length):
        t = t_all[:, 1:, :]
        B, N, C = t.shape
        # [B, N, C] --> [B, C, n, n] --> [B, C_conv, n, n] --> [B, C_conv, n_up, n_up]
        t_reshape = t.transpose(1, 2).reshape(B, C, int(np.sqrt(N)), int(np.sqrt(N)))
        t_conv_project = self.relu(self.bn(self.conv_project(t_reshape)))

        return F.interpolate(t_conv_project, size=(side_length, side_length), mode='bilinear')


class Tran_Up(nn.Module):
    def __init__(self):
        super(Tran_Up, self).__init__()

    def forward(self, t_all, Num):
        t_seg = t_all[:, 0, :][:, None, :]
        t = t_all[:, 1:, :]
        target_side_length = int(np.sqrt(Num - 1))
        B, N, C = t.shape
        # [B, N, C] --> [B, C, n, n] --> [B, C, n_up, n_up] --> [B, N_up, C]
        t_reshape = t.transpose(1, 2).reshape(B, C, int(np.sqrt(N)), int(np.sqrt(N)))
        t_up = nn.UpsamplingBilinear2d(size=target_side_length)(t_reshape).reshape(B, C, pow(target_side_length,
                                                                                             2)).transpose(1, 2)
        t_new = torch.cat([t_seg, t_up], 1)

        return t_new


class Conv_Positioning(nn.Module):
    def __init__(self, channel):
        super(Conv_Positioning, self).__init__()
        self.fusion = nn.Sequential(nn.Conv2d(2 * channel, channel, 1, 1, 0), nn.BatchNorm2d(channel), nn.ReLU())

    def forward(self, x, x0, x1, x2, x3):
        concat = torch.cat([x, x0, x1, x2, x3], 1)
        fusion = self.fusion(concat)

        return fusion


class Tran_Positioning(nn.Module):
    def __init__(self, channel):
        super(Tran_Positioning, self).__init__()
        self.fusion = Mlp(in_features=int(2 * channel), hidden_features=channel, out_features=channel)

    def forward(self, t, t0, t1, t2, t3):
        concat = torch.cat([t, t0, t1, t2, t3], 2)
        fusion = self.fusion(concat)

        return fusion


class PPM(nn.Module):
    def __init__(self, conv_channel, tran_channel, scale=(5, 7, 9, 11), num_heads=9):
        super(PPM, self).__init__()
        self.conv_channel = conv_channel
        self.conv_scale_channel = conv_channel // 4
        self.tran_channel = tran_channel
        self.tran_scale_channel = tran_channel // 4
        self.scale = scale

        self.conv_down_scale0 = Conv_Down(self.conv_channel, self.tran_channel, self.scale[0])
        self.conv_down_scale1 = Conv_Down(self.conv_channel, self.tran_channel, self.scale[1])
        self.conv_down_scale2 = Conv_Down(self.conv_channel, self.tran_channel, self.scale[2])
        self.conv_down_scale3 = Conv_Down(self.conv_channel, self.tran_channel, self.scale[3])

        self.tran_down_scale0 = Tran_Down(self.scale[0])
        self.tran_down_scale1 = Tran_Down(self.scale[1])
        self.tran_down_scale2 = Tran_Down(self.scale[2])
        self.tran_down_scale3 = Tran_Down(self.scale[3])

        self.fusion_down_scale0 = Fusion_Down()
        self.fusion_down_scale1 = Fusion_Down()
        self.fusion_down_scale2 = Fusion_Down()
        self.fusion_down_scale3 = Fusion_Down()

        self.mhsa_mlp_scale0 = Block(self.tran_channel, num_heads=num_heads, mlp_ratio=1,
                                     out_features=self.tran_scale_channel, skip=False)
        self.mhsa_mlp_scale1 = Block(self.tran_channel, num_heads=num_heads, mlp_ratio=1,
                                     out_features=self.tran_scale_channel, skip=False)
        self.mhsa_mlp_scale2 = Block(self.tran_channel, num_heads=num_heads, mlp_ratio=1,
                                     out_features=self.tran_scale_channel, skip=False)
        self.mhsa_mlp_scale3 = Block(self.tran_channel, num_heads=num_heads, mlp_ratio=1,
                                     out_features=self.tran_scale_channel, skip=False)

        self.conv_up_scale0 = Conv_Up(self.tran_scale_channel, self.conv_scale_channel)
        self.conv_up_scale1 = Conv_Up(self.tran_scale_channel, self.conv_scale_channel)
        self.conv_up_scale2 = Conv_Up(self.tran_scale_channel, self.conv_scale_channel)
        self.conv_up_scale3 = Conv_Up(self.tran_scale_channel, self.conv_scale_channel)

        self.tran_up_scale0 = Tran_Up()
        self.tran_up_scale1 = Tran_Up()
        self.tran_up_scale2 = Tran_Up()
        self.tran_up_scale3 = Tran_Up()

        self.conv_positioning = Conv_Positioning(self.conv_channel)
        self.tran_positioning = Tran_Positioning(self.tran_channel)

    def forward(self, x, x_t):
        _, _, side_length, _ = x.shape
        _, Num, _ = x_t.shape

        conv_down_scale0 = self.conv_down_scale0(x, x_t)
        tran_down_scale0 = self.tran_down_scale0(x_t)
        fusion_down_scale0 = self.fusion_down_scale0(conv_down_scale0, tran_down_scale0)
        scale0 = self.mhsa_mlp_scale0(fusion_down_scale0)
        conv_up_scale0 = self.conv_up_scale0(scale0, side_length)
        tran_up_scale0 = self.tran_up_scale0(scale0, Num)

        conv_down_scale1 = self.conv_down_scale1(x, x_t)
        tran_down_scale1 = self.tran_down_scale1(x_t)
        fusion_down_scale1 = self.fusion_down_scale1(conv_down_scale1, tran_down_scale1)
        scale1 = self.mhsa_mlp_scale1(fusion_down_scale1)
        conv_up_scale1 = self.conv_up_scale1(scale1, side_length)
        tran_up_scale1 = self.tran_up_scale1(scale1, Num)

        conv_down_scale2 = self.conv_down_scale2(x, x_t)
        tran_down_scale2 = self.tran_down_scale2(x_t)
        fusion_down_scale2 = self.fusion_down_scale2(conv_down_scale2, tran_down_scale2)
        scale2 = self.mhsa_mlp_scale2(fusion_down_scale2)
        conv_up_scale2 = self.conv_up_scale2(scale2, side_length)
        tran_up_scale2 = self.tran_up_scale2(scale2, Num)

        conv_down_scale3 = self.conv_down_scale3(x, x_t)
        tran_down_scale3 = self.tran_down_scale3(x_t)
        fusion_down_scale3 = self.fusion_down_scale3(conv_down_scale3, tran_down_scale3)
        scale3 = self.mhsa_mlp_scale3(fusion_down_scale3)
        conv_up_scale3 = self.conv_up_scale3(scale3, side_length)
        tran_up_scale3 = self.tran_up_scale3(scale3, Num)

        conv_positioning = self.conv_positioning(x, conv_up_scale0, conv_up_scale1, conv_up_scale2, conv_up_scale3)
        tran_positioning = self.tran_positioning(x_t, tran_up_scale0, tran_up_scale1, tran_up_scale2, tran_up_scale3)

        return conv_positioning, tran_positioning


###################################################################
# ########################## Focus ################################
###################################################################
class Decoder(nn.Module):
    def __init__(self, conv_channel, embed_dim=576, num_heads=9, qkv_bias=True, mlp_ratio=4., act_layer=nn.GELU, if_tran=True):
        super(Decoder, self).__init__()
        self.conv_channel = conv_channel
        self.conv_mid_channel = int(self.conv_channel / 4)
        self.conv_output_channel = int(self.conv_channel / 2)
        self.if_tran = if_tran

        self.vector_conv = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                         nn.Conv2d(self.conv_channel, self.conv_mid_channel, 1, 1, 0), nn.ReLU())
        self.vector_tran = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                         nn.Conv2d(embed_dim, self.conv_mid_channel, 1, 1, 0), nn.ReLU())
        self.channel_attention_conv = nn.Sequential(nn.Conv2d(self.conv_output_channel, self.conv_output_channel, 1, 1, 0),
                                                    nn.Sigmoid())

        self.attention_map = nn.Sequential(nn.Conv2d(2, 1, 7, 1, 3), nn.Sigmoid())
        self.spatial_attention_conv = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_up = nn.Sequential(nn.Conv2d(self.conv_channel, self.conv_output_channel, 3, 1, 1),
                                     nn.BatchNorm2d(self.conv_output_channel), nn.ReLU(),
                                     nn.UpsamplingBilinear2d(scale_factor=2))
        self.conv_fusion = nn.Sequential(nn.Conv2d(self.conv_output_channel, self.conv_output_channel, 3, 1, 1),
                                         nn.BatchNorm2d(self.conv_output_channel), nn.ReLU())
        if self.if_tran:
            self.channel_attention_tran = nn.Sequential(nn.Conv2d(self.conv_output_channel, embed_dim, 1, 1, 0),
                                                        nn.Sigmoid())
            self.tran_channel = embed_dim
            self.tran_fusion = Block(self.tran_channel, num_heads=num_heads, qkv_bias=qkv_bias, mlp_ratio=mlp_ratio,
                                     out_features=self.tran_channel, skip=True, act_layer=act_layer)

    def forward(self, x, x_t, x_enc, x_t_enc, map_x, map_x_t):
        # conv channel attention
        vector_conv = self.vector_conv(x)
        vector_tran = self.vector_tran(x_t[:, 1:, :][:, :, :, None].transpose(1, 2))
        vector = torch.cat([vector_conv, vector_tran], 1)
        channel_attention_conv = self.channel_attention_conv(vector)
        conv_add = self.conv_up(x) + x_enc
        conv_add_channel_attention = conv_add * channel_attention_conv

        # conv spatial attention
        attention_map = self.attention_map(torch.cat([map_x, map_x_t], 1))
        spatial_attention_conv = self.spatial_attention_conv(attention_map)
        conv_add_spatial_attention = conv_add_channel_attention * spatial_attention_conv
        conv_fusion = self.conv_fusion(conv_add_spatial_attention)

        if self.if_tran:
            # tran channel attention
            channel_attention_tran = self.channel_attention_tran(vector).squeeze(-1).transpose(1, 2)
            # tran_add = x_t + x_t_enc
            tran_add = x_t
            tran_add_channel_attention = torch.cat([tran_add[:, 0, :][:, None, :], tran_add[:, 1:, :] * channel_attention_tran], 1)

            # tran spatial attention
            _, N, _ = x_t.shape
            tran_side_length = int(np.sqrt(N - 1))
            spatial_attention_tran = nn.UpsamplingBilinear2d((tran_side_length, tran_side_length))(attention_map).flatten(2).transpose(1, 2)
            tran_add_spatial_attention = torch.cat([tran_add_channel_attention[:, 0, :][:, None, :],
                                                    tran_add_channel_attention[:, 1:, :] * spatial_attention_tran], 1)
            tran_fusion = self.tran_fusion(tran_add_spatial_attention)

            return conv_fusion, tran_fusion
        else:
            return conv_fusion


###################################################################
# ############################ Predict ############################
###################################################################
class segmentation_token_inference(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, fea):
        B, N, C = fea.shape
        t_all = self.norm(fea)
        t_seg, t = t_all[:, 0, :][:, None, :], t_all[:, 1:, :]
        # t_seg [B, 1, 576]  t [B, 26*26, 576]

        q = self.q(t).reshape(B, N-1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(t_seg).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(t_seg).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        infer_fea = (attn @ v).transpose(1, 2).reshape(B, N-1, C)
        infer_fea = self.proj(infer_fea)
        infer_fea = self.proj_drop(infer_fea)

        infer_fea = infer_fea + fea[:, 1:, :]
        return infer_fea

class Predict(nn.Module):
    def __init__(self, conv_channel, embed_dim=576):
        super(Predict, self).__init__()
        self.conv_channel = conv_channel
        self.embed_dim = embed_dim
        self.map_x = nn.Conv2d(self.conv_channel, 1, 7, 1, 3)
        self.segmentation_token_inference = segmentation_token_inference(self.embed_dim)
        self.map_x_t = nn.Linear(self.embed_dim, 1)

    def forward(self, x, x_t):
        B, _, h, w = x.shape
        _, N, C = x_t.shape

        map_x = self.map_x(x)

        t_features = self.segmentation_token_inference(x_t)
        map_x_t = self.map_x_t(t_features).transpose(1, 2).reshape(B, 1, int(np.sqrt(N)), int(np.sqrt(N)))
        map_x_t = F.interpolate(map_x_t, size=(h, w), mode='bilinear', align_corners=True)

        return map_x, map_x_t


###################################################################
# ########################## NETWORK ##############################
###################################################################
class OPNet(nn.Module):
    def __init__(self, backbone_path=None):
        super(OPNet, self).__init__()
        # params

        # backbone
        self.conformer = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                                   num_heads=9, mlp_ratio=4, qkv_bias=True)
        if backbone_path is not None:
            self.conformer.load_state_dict(torch.load(backbone_path))
            print("From {} Load Weights Succeed!".format(backbone_path))

        # channel reduction
        self.cr4 = nn.Sequential(nn.Conv2d(1536, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv2d(1536, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv2d(768, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.cr1 = nn.Sequential(nn.Conv2d(384, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())

        # pyramid positioning
        self.ppm = PPM(512, 576, scale=(5, 7, 9, 11))

        # predict
        self.predict4 = Predict(512, embed_dim=576)
        self.predict3 = Predict(256, embed_dim=576)
        self.predict2 = Predict(128, embed_dim=576)
        self.predict1 = nn.Conv2d(64, 1, 7, 1, 3)

        # decoder
        self.decoder43 = Decoder(512, embed_dim=576)
        self.decoder32 = Decoder(256, embed_dim=576)
        self.decoder21 = Decoder(128, embed_dim=576, if_tran=False)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        # x: [batch_size, channel=3, h, w]
        conv_features, tran_features = self.conformer(x)
        layer0 = conv_features[0]  # [-1, 384, h/4, w/4]
        layer1 = conv_features[3]  # [-1, 384, h/4, w/4]
        layer2 = conv_features[7]  # [-1, 768, h/8, w/8]
        layer3 = conv_features[10]  # [-1, 1536, h/16, w/16]
        layer4 = conv_features[11]  # [-1, 1536, h/32, w/32]
        t0 = tran_features[0]  # [-1, (h/16)^2+1, 576] 416-->677
        t1 = tran_features[3]  # [-1, (h/16)^2+1, 576]
        t2 = tran_features[7]  # [-1, (h/16)^2+1, 576]
        t3 = tran_features[10]  # [-1, (h/16)^2+1, 576]
        t4 = tran_features[11]  # [-1, (h/16)^2+1, 576]

        # channel reduction
        cr4 = self.cr4(layer4)
        cr3 = self.cr3(layer3)
        cr2 = self.cr2(layer2)
        cr1 = self.cr1(layer1)

        # pyramid positioning module
        conv_ppm, tran_ppm = self.ppm(cr4, t4)
        # predict 4
        conv_predict4, tran_predict4 = self.predict4(conv_ppm, tran_ppm)

        # decoder 4 to 3
        conv_decoder43, tran_decoder43 = self.decoder43(conv_ppm, tran_ppm, cr3, t3, conv_predict4, tran_predict4)
        # predict 3
        conv_predict3, tran_predict3 = self.predict3(conv_decoder43, tran_decoder43)

        # decoder 3 to 2
        conv_decoder32, tran_decoder32 = self.decoder32(conv_decoder43, tran_decoder43, cr2, t2, conv_predict3, tran_predict3)
        # predict 2
        conv_predict2, tran_predict2 = self.predict2(conv_decoder32, tran_decoder32)

        # decoder 2 to 1
        conv_decoder21 = self.decoder21(conv_decoder32, tran_decoder32, cr1, t1, conv_predict2, tran_predict2)
        # predict final
        conv_predict1 = self.predict1(conv_decoder21)

        # rescale
        conv_predict4 = F.interpolate(conv_predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv_predict3 = F.interpolate(conv_predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv_predict2 = F.interpolate(conv_predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        tran_predict4 = F.interpolate(tran_predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        tran_predict3 = F.interpolate(tran_predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        tran_predict2 = F.interpolate(tran_predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv_predict1 = F.interpolate(conv_predict1, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return conv_predict4, conv_predict3, conv_predict2, tran_predict4, tran_predict3, tran_predict2, conv_predict1

        return torch.sigmoid(conv_predict4), torch.sigmoid(conv_predict3), torch.sigmoid(conv_predict2), \
               torch.sigmoid(tran_predict4), torch.sigmoid(tran_predict3), torch.sigmoid(tran_predict2), \
               torch.sigmoid(conv_predict1)

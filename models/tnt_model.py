# -*- coding: utf-8 -*-
# @Author: 二月三
# @Date:   2021-05-18 19:55:23
# @Last Modified by:   二月三
# @Last Modified time: 2021-05-18 22:40:17

from numpy.core.shape_base import block
import paddle
from paddle import nn
from paddle.fluid.dygraph.nn import LayerNorm
from paddle.fluid.layers.nn import pad
from paddle.nn import functional as F

import math
import os

from tnt_layers import TNT_Block, Pixel_Embed


class TNT(nn.Layer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                 num_classes=10, embed_dim=768, in_dim=48, depth=12,
                 out_num_heads=12, in_num_head=4, mlp_ratio=4., qkv_bias=False, 
                 drop_rate=0., attn_drop_rate=0.,drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, first_stride=4):
        super(TNT, self).__init__()
        '''TNT
            params_list:
                img_size:               输入图片大小（提前明确）
                patch_size:             patch的大小（提前明确）
                in_chans:               输入图片通道数
                num_classes:            分类类别
                embed_dim:              嵌入维度大小  -- 也是总的feature大小
                in_dim:                 每个patch的维度大小 -- 
                                        但不是每个patch对应的实际全部元素，
                                        全部元素还要加上in_dim[x]*每一个大小对应划分pixel的数量
                depth:                  深度(block数量)
                out_num_heads:          outer transformer的header数
                in_num_head:            inner transformer的header数
                mlp_ratio:              outer中mlp的隐藏层缩放比例
                qkv_bias:               question、keys、values是否启用bias
                drop_rate:              mlp、一般层（比如:映射输出、位置编码输出时）丢弃率
                attn_drop_rate:         注意力丢弃率
                drop_path_rate:         路径丢弃率
                norm_layer:             归一化层
                first_stride:           图片输入提取分块的步长--img --> patch
        '''

        self.num_classes = num_classes                      # 分类数
        self.embed_dim = embed_dim                          # 嵌入维度 == 特征数
        self.num_features = self.embed_dim                  # 嵌入维度 == 特征数

        # 先完成pixel-level的嵌入
        self.pixel_embeb = Pixel_Embed(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_chans, in_dim=in_dim, stride=first_stride)
        self.num_patches = self.pixel_embeb.num_patches        # 当前pixel等效的实际的patch个数
        self.new_patch_size = self.pixel_embeb.new_patch_size  # 当前等效的每一个patch对应的pixel的分辨率(w == h)
        self.num_pixel = self.new_patch_size ** 2  # 当前每个patch实际划分的分辨率，w*h = w**2 得到patch2pixel的序列大小
        
        # 在进行patch-level嵌入
        # 从pixel映射到patch上，要对每一个patch展开为pixel下的数据通过层归一化
        self.first_proj_norm_start = norm_layer(self.num_pixel * in_dim)      # self.nwe_pixel * in_dim, 即每一个patch对应的全部元素
        # 然后映射到指定嵌入维度上
        self.first_proj = nn.Linear(self.num_pixel * in_dim, self.embed_dim)  # 将全部每一个patch对应的pixel都映射到指定嵌入维度大小的空间
        # 在经过一次归一化，输出
        self.first_proj_norm_end = norm_layer(self.embed_dim)

        # 分类标记
        # 截断正态分布来填充初始化cls_token、patch_pos、pixel_pos
        self.cls_token = paddle.create_parameter(shape=(1, 1, self.embed_dim), dtype='float32', attr=nn.initializer.TruncatedNormal(std=0.02))
        # 位置编码
        # patch_position_encode: self.num_patches + 1对应实际patch数目+上边的分类标记
        self.patch_pos = paddle.create_parameter(shape=(1, self.num_patches + 1, self.embed_dim), dtype='float32', attr=nn.initializer.TruncatedNormal(std=0.02))
        # pixel_position_encode: in_dim对应每一个patch的大小, self.new_patch_size对应patch划分为pixel的分辨率
        self.pixel_pos = paddle.create_parameter(shape=(1, in_dim, self.new_patch_size, self.new_patch_size), dtype='float32', attr=nn.initializer.TruncatedNormal(std=0.02))
        # 位置编码的丢弃
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 在TNT中使用了path_drop, 进行路径丢弃
        # 为了丢弃更具随机性，提高鲁棒性，进行随机丢弃率的制作 -- 根据深度生成对应数量的丢弃率
        drop_path_random_rates = [r.numpy().item() for r in paddle.linspace(0, drop_path_rate, depth)]
        # 相同块，采用迭代生成
        tnt_blocks = []
        for i in range(depth):  # 根据深度添加TNT块
            tnt_blocks.append(
                TNT_Block(patch_embeb_dim=self.embed_dim, in_dim=in_dim, num_pixel=self.num_pixel,       # 嵌入大小， patch-level大小，patch2pixel大小
                          out_num_heads=out_num_heads, in_num_head=in_num_head, mlp_ratio=mlp_ratio,     # outer transformer头数，inner transformer头数，感知机隐藏层缩放比
                          qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,                   # transormer中bias启动情况，映射层的丢弃率， 注意力层的丢弃率
                          drop_path=drop_path_random_rates[i], norm_layer=norm_layer)                    # 路径丢弃的丢弃率，归一化层
            )
        # 放入顺序层结构中
        self.tnt_blocks = nn.Sequential(*tnt_blocks)                # 输入前后不发生shape变化
        # tnt_blocks最后的输出还要经过一层归一化
        self.tnt_block_end_norm = norm_layer(self.embed_dim)             # 沿用前边最初归一化层的嵌入大小进行归一化设置

        # 输出任务结果 -- 这里是利用cls_token进行分类，embed_dim是整个模型的嵌入维度大小，也是cls_token的最后1维度的大小
        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else TNT_Block.Identity()
        
        # 初始化网络参数
        self._init_weights()


    def _init_weights(self):
        '''
            完成整个网络的初始化工作
        '''
        for l in self.sublayers():
            if isinstance(l, nn.Linear):
                # 随即截断正态分布填充初始化
                l.weight = paddle.create_parameter(shape = (l.weight.shape), dtype=l.weight.dtype,
                                                   name=l.weight.name, attr=nn.initializer.TruncatedNormal(std=0.02))
                # bias 默认开启就初始化为0  -- 不指定其它初始化方式时
            elif isinstance(l, nn.LayerNorm):
                # bias 默认开启就初始化为0  -- 不指定其它初始化方式时
                # 常量1.0初始化
                l.weight = paddle.create_parameter(shape = (l.weight.shape), dtype=l.weight.dtype,
                                                   name=l.weight.name, attr=nn.initializer.Constant(value=1.0))

    def get_classfier(self):
        '''
            用于获取分类任务头进行相应的任务预测、输出等
            [在当前任务不是必要的]
        '''
        return self.head


    def reset_classfier(self, num_classes):
        '''
            用于修改分类任务头的分类数目
        '''
        self.num_classes = num_classes  # 修改模型分类参数
        self.head = nn.Linear(self.embeb_dim, self.num_classes) if self.num_classes > 0 else TNT_Block.Identity()
        
    
    def upstream_forward(self, inputs):
        '''
            上有任务前向运算：TNT特征提取部分的网络运算
        '''
        x = inputs 
        B = x.shape[0]   # batch_size

        # 1. 先进行pixel-level的嵌入
        pixel_embeb = self.pixel_embeb(x, self.pixel_pos)

        # 2. 再从pixel-level上升到patch-level的嵌入(即向上映射)
        # 依次通过: pixel2patch的layer_norm, 然后进行映射，最后再通过一层layer_norm完成整个映射过程
        # 其中输入的pixel_embeb要经过shape变换，将散布在不同in_dim下的参数进行拼接到对应patch下
        patch_embeb = self.first_proj_norm_end(self.first_proj(self.first_proj_norm_start(pixel_embeb.reshape(shape=(B, self.num_patches, -1)))))
        patch_embeb = paddle.concat([self.cls_token, patch_embeb], axis=1)    # 将分类任务标记拼接到patch的嵌入空间中
        patch_embeb = patch_embeb + self.patch_pos   # 加上位置编码
        patch_embeb = self.pos_drop(patch_embeb)     # 丢弃一部分编码结果

        for tnt_block in self.tnt_blocks:            # 将前期处理好的嵌入信息，进行迭代，进行信息地进一步提取
            pixel_embeb, patch_embeb = tnt_block(pixel_embeb, patch_embeb)
        
        patch_embeb = self.tnt_block_end_norm(patch_embeb)
        
        return patch_embeb[:, 0]        # 0号位置为cls_token对应的位置

    @paddle.jit.to_static
    def forward(self, inputs):
        x = inputs
        
        # 主体前向传播
        x = self.upstream_forward(x)  # 返回cls_token，用于分类用
        # 分类任务
        x = self.head(x)  # 执行分类

        return x



def tnt_s_patch16_224(**kwargs):
    '''TNT Small Model
        basic params list(can set):
            img_size: 输入图片大小         -- 必须指定相应大小（默认为224）
            in_chans: 输入图片通道         -- 必须指定相应通道（默认为3）
            num_classes: 分类数           -- 必须指定分类数（默认为10）
            patch_size: 每一个patch的大小
            embed_dim: 嵌入空间的大小
            in_dim: 每一个patch的维度大小(不包含再往下划分的pixel时的大小)
            depth:  深度--即TNT_Block叠加块的数量
            out_num_heads:   outer_transformer中注意力头的数量
            in_num_head: inner_transformer中注意力头的数量
            qkv_bias:    在进行映射时，是否启用bias--主要应用在question、keys、values的映射上
    '''
    model = TNT(patch_size=16, embed_dim=384, in_dim=24, depth=12,
                out_num_heads=6, in_num_head=4, qkv_bias=False, **kwargs)
    return model


def tnt_b_patch16_224(**kwargs):
    '''TNT Big Model
        basic params list(can set):
            img_size: 输入图片大小         -- 必须指定相应大小（默认为224）
            in_chans: 输入图片通道         -- 必须指定相应通道（默认为3）
            num_classes: 分类数           -- 必须指定分类数（默认为10）
            patch_size: 每一个patch的大小
            embed_dim: 嵌入空间的大小
            in_dim: 每一个patch的维度大小(不包含再往下划分的pixel时的大小)
            depth:  深度--即TNT_Block叠加块的数量
            out_num_heads:   outer_transformer中注意力头的数量
            in_num_head: inner_transformer中注意力头的数量
            qkv_bias:    在进行映射时，是否启用bias--主要应用在question、keys、values的映射上
    '''
    model = TNT(patch_size=16, embed_dim=640, in_dim=40, depth=12,
                out_num_heads=10, in_num_head=4, qkv_bias=False, **kwargs)
    return model

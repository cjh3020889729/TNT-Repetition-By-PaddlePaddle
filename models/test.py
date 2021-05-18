# -*- coding: utf-8 -*-
# @Author: 喝鱼汤的鱼儿
# @Date:   2021-05-18 22:02:17
# @Last Modified by:   喝鱼汤的鱼儿
# @Last Modified time: 2021-05-18 22:45:33

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.fluid.layers.nn import pad

from tnt_model import tnt_s_patch16_224, tnt_b_patch16_224, TNT

'''PS:
    1. 输入大小: (batch_size, num_classes), 值不是0~1.0的，具体预测还需通过一个softmax
    (eg: F.softmax(out))
    
    2. 修改模型预测分类数，除了直接配置参数初始化以外，还可以在模型创建后，训练前通过：
    model.reset_classfier(new_num_classes) 进行分类数修改， 暂未添加img_size修改方式

    3. 配置参数时，调节embed_dim、in_dim必须为偶数，否则在模型创建中会发生报错
'''

# 目前的配置按照tnt_small进行设置，具体可自行修改
cfg_ = {
    'img_size': 224, 
    'patch_size': 16,        
    'in_chans': 3,
    'num_classes': 10,
    'embed_dim': 384,            # 嵌入空间维度大小
    'in_dim': 24,                # 每个patch对应的直接维度，不包含pixel-level的维度大小
    'depth': 12,                 # TNT-Block叠加块数量
    'out_num_heads': 6,          # outer transformer的头数
    'in_num_head': 4,            # inner transformer的头数
    'mlp_ratio': 4,              # Attention内outer部分的MLP隐藏层缩放比大小
    'qkv_bias': False,           # 在进行question 、 keys 、 values映射时，是否启用bias
    'drop_rate': 0.,             # 普通丢弃率--用在普通的Linear上
    'attn_drop_rate': 0.,        # 注意力部分的丢弃率
    'drop_path_rate': 0.,        # 路径丢弃率 -- 即一定概率丢弃沿第1维的所有元素：eg: shape[4, 2, 3, 4] 则沿第一维-4-进行丢弃
    'norm_layer': nn.LayerNorm,  # 归一化层参数
    'first_stride': 4            # Image到pixel、patch的滑动提取步长（卷积 + unfold）
}


def create_tnt_by_cfg(cfg_, on_start_test = False):
    '''
        利用cfg_ 和 TNT进行模型创建
        cfg_: params_dicts
        on_start_test: 是否开启启动测试
    '''

    model = TNT(**cfg_)

    if on_start_test is True:     # 模型启动测试 -- 验证当前参数配置下模型是否运行
        with paddle.no_grad():
            test_data = paddle.ones(shape=(1, 3, 224, 224), dtype='float32')
            out = model(test_data)
            print(out.numpy().shape)

    return model


def create_tnt_by_basecfg(num_classes = 10, img_size = 224, in_chans = 3, 
                          choice_big = False, on_start_test = False):
    '''
        利用默认基础配置模型small and big 进行微调模型创建
        num_classes:   类别数
        img_size:      输入图片大小
        in_chans:      输入图片通道数
        choice_big:    是否选择big模型配置
        on_start_test: 是否开启启动测试
    '''
    if choice_big is True:
        model = tnt_b_patch16_224(num_classes = num_classes, img_size = img_size, in_chans = in_chans)
    else:
        model = tnt_s_patch16_224(num_classes = num_classes, img_size = img_size, in_chans = in_chans)
    
    if on_start_test is True:      # 模型启动测试 -- 验证当前参数配置下模型是否运行
        with paddle.no_grad():
            test_data = paddle.ones(shape=(1, 3, 224, 224), dtype='float32')
            out = model(test_data)
            print(out.numpy().shape)
            
    return model

# 可解注释查看运行结果
# if __name__ == '__main__':
    
#     create_tnt_by_basecfg(choice_big=False)  # 创建small tnt
#     create_tnt_by_basecfg(choice_big=True)   # 创建big   tnt

#     # 开启启动测试，确保当前配置下模型可运行
#     create_tnt_by_cfg(cfg_, on_start_test=True)                  # 创建自定义配置模型

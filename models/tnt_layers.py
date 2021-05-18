# -*- coding: utf-8 -*-
# @Author: 喝鱼汤的鱼儿
# @Date:   2021-05-18 10:11:30
# @Last Modified by:   喝鱼汤的鱼儿
# @Last Modified time: 2021-05-18 22:24:34
import paddle
from paddle import nn
from paddle.nn import functional as F

import math
import os

from paddle.tensor.creation import ones_like


class Attention(nn.Layer):
    '''
        Multi-Head Attention
    '''
    def __init__(self, in_dims, hidden_dims, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., logging=False):
        super(Attention, self).__init__()
        ''' Attention
            params list:
                in_dims:    输入维度大小
                hidden_dim: 隐藏层维度大小
                num_heads:  注意力头数量
                qkv_bias:   是否对question、keys、values开启映射
                attn_drop:  注意力丢弃率
                proj_drop:  映射丢弃率
                logging:    是否输出Attention的init参数日志
        '''

        # 确保输入层、隐藏层维度为偶数，且不为零，否则在头划分映射时会发生大小错误
        assert in_dims % 2 == 0 and in_dims != 0, \
            'please make sure the input_dims(now: {0}) is an even number.(%2==0 and !=0)'.format(in_dims)
        assert hidden_dims % 2 == 0 and hidden_dims != 0, \
            'please make sure the hidden_dims(now: {0}) is an even number.(%2==0 and !=0)'.format(hidden_dims)

        self.in_dims     = in_dims                           # ATT输入大小
        self.hidden_dims = hidden_dims                       # ATT隐藏层大小
        self.num_heads   = num_heads                         # ATT的头数目
        self.head_dims   = hidden_dims // num_heads          # 将ATT隐藏层大小按照头数平分，作为ATT-head的维度大小
        self.scale       = self.head_dims ** -0.5            # 缩放比例按照头的唯独大小进行开(-0.5次幂)
        self.qkv_bias    = qkv_bias
        self.attn_drop   = attn_drop
        self.proj_drop   = proj_drop

        # 输出日志信息
        if logging:
            print('\n—— Attention Init-Logging ——')
            print('{0:20}'.format(list(dict(in_dims=self.in_dims).keys())[0]),         ': {0:12}'.format(self.in_dims))
            print('{0:20}'.format(list(dict(hidden_dims=self.hidden_dims).keys())[0]), ': {0:12}'.format(self.hidden_dims))
            print('{0:20}'.format(list(dict(num_heads=self.num_heads).keys())[0]),     ': {0:12}'.format(self.num_heads))
            print('{0:20}'.format(list(dict(head_dims=self.head_dims).keys())[0]),     ': {0:12}'.format(self.head_dims))
            print('{0:20}'.format(list(dict(scale=self.scale).keys())[0]),             ': {0:12}'.format(self.scale))
            print('{0:20}'.format(list(dict(qkv_bias=self.qkv_bias).keys())[0]),       ': {0:12}'.format(self.qkv_bias))
            print('{0:20}'.format(list(dict(attn_drop=self.attn_drop).keys())[0]),     ': {0:12}'.format(self.attn_drop))
            print('{0:20}'.format(list(dict(proj_drop=self.proj_drop).keys())[0]),     ': {0:12}'.format(self.proj_drop))

        '''
            questions + keys  |  values
                    project layers
        '''
        # questions + keys 的映射层: *2 就是在一次操作下将两层一同映射
        # qkv_bias：是否开启bias --> 开启默认全零初始化
        self.qk = nn.Linear(self.in_dims, self.hidden_dims*2, bias_attr=qkv_bias)
        # values 的映射层
        # qkv_bias：是否开启bias --> 开启默认全零初始化
        self.v  = nn.Linear(self.in_dims, self.in_dims, bias_attr=qkv_bias)
        # ATT的丢弃层
        self.attn_drop = nn.Dropout(attn_drop)

        '''
            注意力结果映射层
        '''
        self.proj = nn.Linear(self.in_dims, self.in_dims)
        self.proj_drop = nn.Dropout(proj_drop)
    

    @paddle.jit.to_static
    def forward(self, inputs):
        x = inputs
        B, N, C= x.shape           # B:batch_size, N:patch_number, C:input_channel

        # print('\n—— Attention Forward-Logging ——')
        # print('{0:20}'.format(list(dict(B=B).keys())[0]),                      ': {0:12}'.format(B))
        # print('{0:20}'.format(list(dict(N=N).keys())[0]),                      ': {0:12}'.format(N))
        # print('{0:20}'.format(list(dict(C=C).keys())[0]),                      ': {0:12}'.format(C))

        # 利用输入映射question 和 keys维度的特征
        # print('input(x): ', x.numpy().shape)
        qk = self.qk(x)          # 将输入映射到question + keys上
        # print('qk_project: ', qk.numpy().shape)
        qk = paddle.reshape(qk, shape=(B, N, 2, self.num_heads, self.head_dims))   # 将question + keys分离
        # print('qk_reshape: ', qk.numpy().shape)
        qk = paddle.transpose(qk, perm=[2, 0, 3, 1, 4])   # 重新排列question和keys的数据排布
        # print('qk_transpose: ', qk.numpy().shape)
        '''
            ①上面实现的划分，正好对应: head_dims = hidden_dims // num_heads
            ②排布更新为: 映射类别(question+keys)，batch_size, head_number, patch_number, head_dims
        '''
        q, k = qk[0], qk[1]          # 分离question 和 keys
        # print('q: ', q.numpy().shape)
        # print('k: ', k.numpy().shape)

        # 利用输入映射 values 维度的特征
        v = self.v(x).reshape(shape=(B, N, self.num_heads, -1)).transpose(perm=(0, 2, 1, 3))
        # print('v: ', v.numpy().shape)

        # 通过question 与 keys矩阵积，计算patch的注意力结果
        # @ : 矩阵乘法
        attn = paddle.matmul(q, k.transpose(perm=(0, 1, 3, 2))) * self.scale
        # attn = (q @ k.transpose(perm=(0, 1, 3, 2))) * self.scale
        # print('attn_matrix*: ', attn.numpy().shape)
        '''
            k.transpose(perm=(0, 1, 3, 2)) : 最后两维发生转置 --> 用于矩阵乘法，实现注意力大小计算(question 对 keys)
            * self.scale : 针对注意力头数进行一定的缩放，稳定值
        '''
        attn = F.softmax(attn, axis=-1)          # 通过softmax整体估算注意力 -- 对每一个patch上的hidden_dim进行注意力计算
        # print('attn_softmax: ', attn.numpy().shape)
        attn = self.attn_drop(attn)              # 丢弃部分注意力结果

        # 将注意力结果与value进行矩阵乘法结合
        x = paddle.matmul(attn, v).transpose(perm=(0, 2, 1, 3)).reshape(shape=(B, N, -1))
        # x = (attn @ v).transpose(perm=(0, 2, 1, 3)).reshape(shape=(B, N, -1))
        # print('x_matrix*: ', x.numpy().shape)
        ''' 
            attn @ v: 实现注意力叠加
            transpose(perm=(0, 2, 1, 3)): 将patch与head维度互换(转置) -- 保证reshape不发生错误合并
            reshape(shape=(B, N, -1)): 转换回:batch_size, patch_num, out_dims形式 -- out_dims = num_head * head_dims
        '''
        x = self.proj(x)          # 将注意力叠加完成的结果进行再映射，将其映射回输入大小
        # print('x_proj: ', x.numpy().shape)
        x = self.proj_drop(x)     # 丢弃部分结果
        return x

def test_att():
    '''
        Attention测试函数
    '''
    batch_size = 2
    patch_num  = 64
    in_dims = 64
    hidden_dims = 128

    test_data = paddle.randn(shape=(batch_size, patch_num, in_dims))

    test = Attention(in_dims, hidden_dims)

    out = test(test_data)

    print('out: ', out.numpy().shape)


class MLP(nn.Layer):
    '''
        两层fc的感知机
    '''
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
        ''' MLP
            params list:
                in_features:      输入大小
                hidden_features:  隐藏层大小
                out_features:     输出大小
                act_layer:        激活层
                drop:             丢弃率
        '''

        # 如果前项为None，则返回后向作为赋值内容
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1  = nn.Linear(in_features,     hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout()
    

    @paddle.jit.to_static
    def forward(self, inputs):
        x = inputs
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

def test_mlp():
    '''
        MLP测试函数
    '''
    batch_size = 2
    in_features  = 64
    hidden_features = 256
    out_features    = 128

    test_data = paddle.randn(shape=(batch_size, in_features))

    test = MLP(in_features, hidden_features=hidden_features, out_features=out_features)

    out = test(test_data)

    print('out: ', out.numpy().shape)


class GluMLP(nn.Layer):
    '''
        两层的（门控感知机）Glu感知机 -- 可以尝试改入模型，查看效果
    '''
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Sigmoid, drop=0.):
        super(GluMLP, self).__init__()
        ''' GluMLP
            params list:
                in_features:      输入大小
                hidden_features:  隐藏层大小
                out_features:     输出大小
                act_layer:        激活层
                drop:             丢弃率
        '''

        # 如果前项为None，则返回后向作为赋值内容
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1  = nn.Linear(in_features,     hidden_features * 2)    # 后期切分用于sigmoid门控
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout()
    

    @paddle.jit.to_static
    def forward(self, inputs):
        x = inputs
        
        x = self.fc1(x)
        # print('x_chunk_before: ', x.numpy().shape)
        x, gate = paddle.chunk(x, chunks=2, axis=-1)
        # print('x_chunk_after: ', x.numpy().shape)
        x = x * self.act(gate)     # 对应元素相乘，进行sigmoid门控输出
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

def test_glumlp():
    '''
        GluMLP测试函数
    '''
    batch_size = 2
    in_features  = 64
    hidden_features = 256
    out_features    = 128

    test_data = paddle.randn(shape=(batch_size, in_features))

    test = GluMLP(in_features, hidden_features=hidden_features, out_features=out_features)

    out = test(test_data)

    print('out: ', out.numpy().shape)


class DropPath(nn.Layer):
    '''删除路径数据
        延一个路径进行丢弃(沿数据第一个维度进行丢弃)
        丢弃的对应path下，所有数据置为0
    '''
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        ''' DropPath
            params list:
                drop_prob:      丢弃率
        '''
        self.drop_prob = drop_prob


    @paddle.jit.to_static
    def forward(self, inputs):
        x = inputs
        return self.drop_path(x)  # self.training是否在训练模型下


    def drop_path(self, x):
        '''
            具体的path丢弃操作: 改变对应的值，不改变数据形状
        '''
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = paddle.to_tensor([1 - self.drop_prob])
        # 作batch_size维度大小的shape结构--(batch_size, 1, 1, ...)
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)                        # batch_size, 1*原ndim减去batch_size维度的大小
        random_tensor = keep_prob + paddle.rand(shape=shape, dtype=x.dtype)  # 按照划分的shape创建一个[0, 1)均匀分布随机tensor
        # 利用[0,1)均匀分布产生的值 + 保持率，就可以实现等比例的保留和丢弃
        # 由于随机性，可以保证丢弃的随机性
        # 由于值总是在[0,1)间，所以只要得到的值 + keep_prob大于一个阈值，就保留
        # 但是因为值时均匀分布的，虽然每一个位置上值时随机取到的，但是确实均匀划分的，
        # 因此这样相加后可以实现对应丢弃概率下的丢弃path，并非一定会执行丢弃
        # print(keep_prob + paddle.rand(shape=[2,]))
        # print(paddle.floor(keep_prob + paddle.rand(shape=[2,])))
        random_tensor = paddle.floor(random_tensor)           # 将1作为阈值，从而floor向下取整筛选满足的数据
        # 仅仅留下 0, 1
        # print(random_tensor)

        # print(x[0, 0, 0])
        # print(keep_prob)
        # print(paddle.divide(x, keep_prob)[0, 0, 0])

        # print(random_tensor.shape)
        # print('x: ', x.numpy())
        output = paddle.divide(x, keep_prob) * random_tensor
        # print('output: ', output.numpy())
        return output

def test_droppath():
    '''
        DropPath测试函数
    '''
    test_data = paddle.randint(low=-2, high=2, shape=(3, 2, 2, 2)).astype('float32')
    
    test = DropPath(0.2)

    out = test(test_data)


class TNT_Block(nn.Layer):
    '''
        实现inner transfromer 和 outer transformer, 从pixel-level 和 patch-level进行数据特征提取

        特性：
            输入输出前后，tensor不发生shape变化（中间过程存在部分映射有shape变化）
    '''
    def __init__(self, patch_embeb_dim, in_dim, num_pixel, out_num_heads=12, 
                 in_num_head=4, mlp_ratio=4.,qkv_bias=False, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        '''TNT_Block
            params list:
                patch_embeb_dim     : patch的嵌入维度大小(也是实际输入数据的映射空间大小)
                in_dim              : 单个patch的维度大小(不包含pixel-level维度)
                num_pixel           : patch下的(in_dim维度)元素对应pixel的比例 1：num_pixel，即pixel个数（也是论文中指的patch2pixel分辨率）
                out_num_heads       : 输出(outer attn)的注意力头
                in_num_head         : 输入(inner attn)的注意力头
                mlp_ratio           : outer transformer中感知机隐藏层的维度缩放率
                qkv_bias            : question 、 keys 、values对应的(线性)映射层bias启用标记
                drop                : MLP部分、proj部分的丢弃率
                attn_drop           : attn部分的丢弃率
                drop_path           : 路径丢弃的丢弃率
                act_layer           : 激活层
                norm_layer          : 归一化层
        ''' 
        super(TNT_Block, self).__init__()

        # Inner transformer
        # 输入-注意力计算 -- pixel level
        self.in_attn_norm = norm_layer(in_dim)        # 层归一化--attention的归一化层
        self.in_attn = Attention(in_dims=in_dim, hidden_dims=in_dim,
                                 num_heads=in_num_head, qkv_bias=qkv_bias,
                                 attn_drop=attn_drop, proj_drop=drop)         # attention输出，tensor的shape不变
        # 输入-多层感知机进行维度映射
        self.in_mlp_norm = norm_layer(in_dim)        # 层归一化--mlp的归一化层
        self.in_mlp = MLP(in_features=in_dim, hidden_features=int(in_dim*4),
                          out_features=in_dim, act_layer=act_layer, drop=drop) # mlp输出，tensor的shape不变
        # 输入-线性映射输出
        self.in_proj_norm = norm_layer(in_dim)        # 层归一化--proj的归一化层
        self.in_proj = nn.Linear(in_dim * num_pixel, patch_embeb_dim, bias_attr=True)           # proj输出，tensor的shape发生改变


        # outer transformer
        # 输出-注意力计算 -- patch level
        self.out_attn_norm = norm_layer(patch_embeb_dim)
        self.out_attn = Attention(in_dims=patch_embeb_dim, hidden_dims=patch_embeb_dim,
                                  num_heads=out_num_heads, qkv_bias=qkv_bias,
                                  attn_drop=attn_drop, proj_drop=drop)
        
        self.out_mlp_norm = norm_layer(patch_embeb_dim)
        self.out_mlp = MLP(in_features=patch_embeb_dim, hidden_features=int(patch_embeb_dim * mlp_ratio),
                       out_features=patch_embeb_dim, act_layer=act_layer, drop=drop)

        # 公用方法
        # 路径丢弃
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else self.Identity  # self.Identity()占位方法，不对数据做任何处理


    @paddle.jit.to_static
    def forward(self, pixel_embeb, patch_embeb):
        '''
            params list:
                pixel_embeb: 上一个block输出的pixel-level out tensor
                patch_embeb: 上一个block输出的patch-level out tensor
        '''
        # inner work
        # 1. 注意力嵌入 added
        pixel_embeb = pixel_embeb + self.drop_path(self.in_attn(self.in_attn_norm(pixel_embeb)))
        # 2. mlp嵌入   added
        pixel_embeb = pixel_embeb + self.drop_path(self.in_mlp(self.in_mlp_norm(pixel_embeb)))
        # pixel嵌入的pathc叠加，在outer中完成

        # outer work
        B, N, C = patch_embeb.shape    # B:batch_size  N:Patch_Number  C:Feature_map_channel
        # 线性映射pixel到patch维度，N-1 means；映射前后不包括class_token
        # 映射是需要完整映射，不需要路径丢弃
        pixel_embeb_proj2patch = self.in_proj(self.in_proj_norm(pixel_embeb).reshape(shape=(B, N-1, -1)))
        # patch叠加上pixel的embeb数据，从patch1 --> patchn
        # 不在这里操作class_token
        patch_embeb[:, 1:] = patch_embeb[:, 1:] + pixel_embeb_proj2patch
        # 1. 注意力嵌入 added
        patch_embeb = patch_embeb + self.drop_path(self.out_attn(self.out_attn_norm(patch_embeb)))
        # print(patch_embeb.shape)
        # 2. mlp嵌入   added
        patch_embeb = patch_embeb + self.drop_path(self.out_mlp(self.out_mlp_norm(patch_embeb)))
        
        return pixel_embeb, patch_embeb


    def Identity(self, x):
        '''
            do nothing, only return input
        '''
        return x


class Pixel_Embed(nn.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, in_dim=48, stride=4):
        super(Pixel_Embed, self).__init__()
        '''Pixel_Embed: 像素嵌入--完成后才有patch嵌入
            params_list:
                img_size:   输入图片大小
                patch_size: 当前一个patch的预置大小
                in_chans:   输入的图像通道数
                in_dim:     设定的输入维度 -- 即预定的patch的个数
                stride:     分块时，使用卷积、滑窗的步长，决定着patch向下划分pixel时的分辨率(不是patch的分辨率)
        '''
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        # 平方解释：img_size是宽时，//patch_size得到一行可划分多少个，而同样的列就有多少个
        # 这里考虑完整划分patch的个数
        self.in_dim = in_dim     # 每一个patch对应的分辨率 -- 即patch-level的分辨率
        self.new_patch_size = math.ceil(patch_size / stride)   # 向上取整 -- 确定向下划分pixel的分辨率
        self.stride = stride     # 卷积 + 滑窗的步长
        
        '''
            两步实现图像到patch的映射，与patch到pixel的分割
        '''
        self.proj = nn.Conv2D(in_channels=in_chans, out_channels=self.in_dim,
                              kernel_size=7, padding=3, stride=self.stride)
        # 7 // 2 == 3, padding = 3, conv后会保持原图大小 -- 在stride=1时
        self.unfold = F.unfold
        # 对输入提取滑动块
    

    @paddle.jit.to_static
    def forward(self, inputs, pixel_pos):
        x = inputs
        B, C, H, W = x.shape
        # 验证是否与所需大小一致
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
            
        x = self.proj(x)
        x = self.unfold(x, kernel_sizes=self.new_patch_size, strides=self.stride)   # 提取滑块，获得对应的滑块结果
        # unfold将img转换为(B, Cout, Lout)
        # Cout = Channel * kernel_sizes[0] * kernel_sizes[1]  , 即每一次滑窗在图片上得到参数个数
        # Lout = hout * wout      —— 滑动block的个数
        # hout，wout 类似卷积在图片对应h，w上的滑动次数
        x = paddle.transpose(x, perm=[0, 2, 1])   # to be shape: (B, Lout, Cout)
        x = paddle.reshape(x, shape=(B * self.num_patches, self.in_dim, self.new_patch_size, self.new_patch_size))
        # 再分解为需要的编码形式: (Batch_size * patch_number, in_dim, new_patch_size, new_patch_size)
        # Batch_size * patch_number：将每个batch得到的patch数乘以batch_size得到总的patch数量
        # in_dim: 当前设定的输入维度大小
        # new_patch_size: 由stride确定的 patch 的分辨率 -- 对应patch下feature map的w，h
        x = x + pixel_pos           # 加上位置编码
        x = paddle.reshape(x, shape=(B * self.num_patches, self.in_dim, -1))  # 拼接pixel-level的元素
        x = paddle.transpose(x, perm=[0, 2, 1])
        # 转换为(B * self.num_patches, patch_dim2pixel_size, self.in_dim)
        # patch_dim2pixel_size: 即原in_dim下所有序列元素的拼接大小
        # 原来是，in_dim对应dim下的pixels
        # 现在是，每一个pixel对应in_dim的情况
        
        return x


# if __name__ == '__main__':

#     # test_att()
#     # test_mlp()
#     # test_glumlp()
#     # test_droppath()
#     pass

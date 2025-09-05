import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from torch.utils.checkpoint import checkpoint

# from mmcv.cnn import CONV_LAYERS  # 保留为注释占位：与 MMCV 的卷积注册器相关，当前未使用
from torch import Tensor
import torch.nn.functional as F
import math


# from timm.models.layers import trunc_normal_  # 保留为注释占位：timm 截断正态初始化，当前未使用


class StarReLU(nn.Module):
    """
    StarReLU 激活函数：对输入执行 ReLU 后进行平方，并施加可学习的缩放与偏置。
    数学形式：y = s * ReLU(x)^2 + b，其中 s 与 b 为可学习参数。
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace  # 指定 ReLU 是否原地操作，节省内存
        self.relu = nn.ReLU(inplace=inplace)  # 标准 ReLU 激活
        # 可学习的缩放参数 s，初始为 scale_value
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        # 可学习的偏置参数 b，初始为 bias_value
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        # 计算 y = s * (ReLU(x))^2 + b；平方放大正值并保持非负性
        return self.scale * self.relu(x) ** 2 + self.bias


class KernelSpatialModulation_Global(nn.Module):
    """
    全局核-空间调制模块（KSM-Global）：
    根据输入特征的全局统计生成四类注意力/调制因子：
    1) 通道注意力 channel_attention：逐通道缩放输入或权重；
    2) 过滤器注意力 filter_attention：逐输出通道缩放卷积结果；
    3) 空间注意力 spatial_attention：逐空间位置生成核权值矩阵（k×k）；
    4) 核混合注意力 kernel_attention：在多组频域参数之间做权重分配。
    """

    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16,
                 temp=1.0, kernel_temp=None, kernel_att_init='dyconv_as_extra', att_multi=2.0,
                 ksm_only_kernel_att=False, att_grid=1, stride=1, spatial_freq_decompose=False,
                 act_type='sigmoid'):
        super(KernelSpatialModulation_Global, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)  # 注意力瓶颈通道数，至少为 min_channel
        self.act_type = act_type  # 注意力激活类型：'sigmoid'、'tanh' 或 'softmax'（仅 kernel 用）
        self.kernel_size = kernel_size  # 卷积核尺寸（方形核）
        self.kernel_num = kernel_num  # 频域/核组的数量，用于混合

        self.temperature = temp  # 温度系数，平滑注意力分布
        self.kernel_temp = kernel_temp  # 核注意力专用温度

        self.ksm_only_kernel_att = ksm_only_kernel_att  # 仅生成核注意力时为 True

        # 初始化方式相关标志（此处作为行为配置，不直接影响权重初始化逻辑）
        self.kernel_att_init = kernel_att_init
        self.att_multi = att_multi  # 注意力幅值放大系数

        # 全局平均池化，用于提取全局统计；此版本未直接使用 avgpool 的输出，但保留结构一致性
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.att_grid = att_grid  # 空间注意力网格尺度（当前未改变行为）
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)  # 通道降维的 1×1 卷积
        self.bn = nn.BatchNorm2d(attention_channel)  # 规范化以稳定统计
        self.relu = StarReLU()  # 使用 StarReLU 增强非线性与梯度特性

        self.spatial_freq_decompose = spatial_freq_decompose  # 是否在通道/过滤器分支输出频域上做分解（影响输出维度）

        # 通道注意力分支：输出形状与 in_planes 匹配，用于逐通道缩放
        if ksm_only_kernel_att:
            self.func_channel = self.skip  # 当仅使用核注意力时，通道注意力恒等为 1
        else:
            if spatial_freq_decompose:
                # 若启用空间频域分解，则输出两倍通道数以分别作用于低/高频分量
                self.channel_fc = nn.Conv2d(attention_channel, in_planes * 2 if self.kernel_size > 1 else in_planes, 1,
                                            bias=True)
            else:
                self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
            self.func_channel = self.get_channel_attention

        # 过滤器注意力分支：输出与 out_planes 对齐，用于逐输出通道缩放
        if (in_planes == groups and in_planes == out_planes) or self.ksm_only_kernel_att:  # 深度可分离或仅核注意力时跳过
            self.func_filter = self.skip
        else:
            if spatial_freq_decompose:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes * 2, 1, stride=stride, bias=True)
            else:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, stride=stride, bias=True)
            self.func_filter = self.get_filter_attention

        # 空间注意力分支：当核尺寸为 1 或仅核注意力时跳过；否则输出 k×k 的空间权重
        if kernel_size == 1 or self.ksm_only_kernel_att:
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        # 核混合注意力分支：当 kernel_num=1 时跳过；否则输出 kernel_num 维的权重
        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()  # 按照 Kaiming/正态等策略进行权重初始化

    def _initialize_weights(self):
        # 统一初始化：Conv2d 采用 Kaiming 正态，BN 权重=1、偏置=0，细分分支权重采用小方差正态
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if hasattr(self, 'channel_spatial'):
            nn.init.normal_(self.channel_spatial.conv.weight, std=1e-6)  # 若存在通道-空间卷积则初始化为极小值
        if hasattr(self, 'filter_spatial'):
            nn.init.normal_(self.filter_spatial.conv.weight, std=1e-6)

        if hasattr(self, 'spatial_fc') and isinstance(self.spatial_fc, nn.Conv2d):
            nn.init.normal_(self.spatial_fc.weight, std=1e-6)  # 空间注意力的核权重初始化为极小值

        if hasattr(self, 'func_filter') and isinstance(self.func_filter, nn.Conv2d):
            nn.init.normal_(self.func_filter.weight, std=1e-6)  # 过滤器分支的权重初始化为极小值

        if hasattr(self, 'kernel_fc') and isinstance(self.kernel_fc, nn.Conv2d):
            nn.init.normal_(self.kernel_fc.weight, std=1e-6)  # 核注意力分支权重初始化为极小值

        if hasattr(self, 'channel_fc') and isinstance(self.channel_fc, nn.Conv2d):
            nn.init.normal_(self.channel_fc.weight, std=1e-6)  # 通道注意力分支权重初始化为极小值

    def update_temperature(self, temperature):
        # 动态更新温度参数以控制注意力的平滑度
        self.temperature = temperature

    @staticmethod
    def skip(_):
        # 返回标量 1.0，表明跳过该注意力分支
        return 1.0

    def get_channel_attention(self, x):
        # 生成通道注意力；输出形状与输入通道对齐，并扩展到卷积核空间以便广播相乘
        if self.act_type == 'sigmoid':
            channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2),
                                                                      x.size(-1)) / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            channel_attention = 1 + torch.tanh_(
                self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1)) / self.temperature)
        else:
            raise NotImplementedError
        return channel_attention

    def get_filter_attention(self, x):
        # 生成过滤器注意力；输出形状与输出通道匹配，并扩展到卷积核空间以便广播相乘
        if self.act_type == 'sigmoid':
            filter_attention = torch.sigmoid(
                self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            filter_attention = 1 + torch.tanh_(
                self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) / self.temperature)
        else:
            raise NotImplementedError
        return filter_attention

    def get_spatial_attention(self, x):
        # 生成空间注意力；输出为 k×k 的二维核权重，并按指定激活进行缩放
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        if self.act_type == 'sigmoid':
            spatial_attention = torch.sigmoid(spatial_attention / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            spatial_attention = 1 + torch.tanh_(spatial_attention / self.temperature)
        else:
            raise NotImplementedError
        return spatial_attention

    def get_kernel_attention(self, x):
        # 生成核混合注意力；在 kernel_num 个核组之间分配权重
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        if self.act_type == 'softmax':
            kernel_attention = F.softmax(kernel_attention / self.kernel_temp, dim=1)
        elif self.act_type == 'sigmoid':
            kernel_attention = torch.sigmoid(kernel_attention / self.kernel_temp) * 2 / kernel_attention.size(1)
        elif self.act_type == 'tanh':
            kernel_attention = (1 + torch.tanh(kernel_attention / self.kernel_temp)) / kernel_attention.size(1)
        else:
            raise NotImplementedError
        return kernel_attention

    def forward(self, x, use_checkpoint=False):
        # 根据 use_checkpoint 决定是否启用检查点以节省显存
        if use_checkpoint:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x):
        # 先通过 1×1 卷积 + BN + StarReLU 提取全局条件特征，再分别生成四类注意力/调制因子
        avg_x = self.relu(self.bn(self.fc(x)))
        return self.func_channel(avg_x), self.func_filter(avg_x), self.func_spatial(avg_x), self.func_kernel(avg_x)


class KernelSpatialModulation_Local(nn.Module):
    """局部核-空间调制模块（KSM-Local）

    功能：
    基于通道维度的一维卷积在频域邻域内进行交互，生成针对 (cin × k×k) 的细粒度注意力，
    可选地利用全局频域信息（rFFT）进行补充增强。
    """

    def __init__(self, channel=None, kernel_num=1, out_n=1, k_size=3, use_global=False):
        super(KernelSpatialModulation_Local, self).__init__()
        self.kn = kernel_num  # 核数量，用于输出维度组织
        self.out_n = out_n  # 输出通道数（通常映射到 cout*k*k）
        self.channel = channel  # 输入通道数，用于自适应卷积核大小推断
        if channel is not None: k_size = round((math.log2(channel) / 2) + 0.5) // 2 * 2 + 1  # 基于通道数的奇数核自适应选择
        # 一维卷积在通道轴上做局部相关建模；输入视为形状 b×1×c
        self.conv = nn.Conv1d(1, kernel_num * out_n, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        nn.init.constant_(self.conv.weight, 1e-6)  # 初始化为极小值以稳定训练初期
        self.use_global = use_global  # 是否引入全局频域增强
        if self.use_global:
            # 复权重参数，对 rFFT 的实部与虚部分别施加线性缩放
            self.complex_weight = nn.Parameter(torch.randn(1, self.channel // 2 + 1, 2, dtype=torch.float32) * 1e-6)
        self.norm = nn.LayerNorm(self.channel)  # 对通道维做层归一化

    def forward(self, x, x_std=None):
        # 输入 x 形状为 b×1×c×1；去除末尾维度后转置为 b×1×c，以便一维卷积处理
        x = x.squeeze(-1).transpose(-1, -2)  # b×1×c
        b, _, c = x.shape
        if self.use_global:
            # 在通道维执行 rFFT，分别对实部与虚部进行逐频缩放，再逆变换回时域，作为增强项与原始 x 相加
            x_rfft = torch.fft.rfft(x.float(), dim=-1)  # b×1×(c//2+1)
            x_real = x_rfft.real * self.complex_weight[..., 0][None]
            x_imag = x_rfft.imag * self.complex_weight[..., 1][None]
            x = x + torch.fft.irfft(torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1)), dim=-1)
        x = self.norm(x)  # 层归一化保障数值稳定
        att_logit = self.conv(x)  # b×(kn*out_n)×c
        # 重排为 b×kn×cin×out_n，便于后续与权重张量对齐
        att_logit = att_logit.reshape(x.size(0), self.kn, self.out_n, c)
        att_logit = att_logit.permute(0, 1, 3, 2)  # b×kn×cin×out_n
        return att_logit


class FrequencyBandModulation(nn.Module):
    """
    频带调制模块（FBM）：
    在二维频域中通过 rFFT 构建低频掩码序列，逐级提取不同截止频率下的低频分量及其互补高频分量，
    再通过深度可分离卷积生成空间权重，对各频带分量进行逐像素加权并求和，得到重构特征。
    """

    def __init__(self,
                 in_channels,
                 k_list=[2, 4, 8],
                 lowfreq_att=False,
                 fs_feat='feat',
                 act='sigmoid',
                 spatial='conv',
                 spatial_group=1,
                 spatial_kernel=3,
                 init='zero',
                 **kwargs,
                 ):
        super().__init__()
        self.k_list = k_list  # 频带因子列表，值越大对应截止频率越低
        self.lp_list = nn.ModuleList()  # 预留：低通滤波器列表（当前实现通过掩码生成）
        self.freq_weight_conv_list = nn.ModuleList()  # 各频带对应的空间权重生成器
        self.fs_feat = fs_feat  # 特征选择策略标志（当前直接使用输入/att_feat）
        self.in_channels = in_channels
        if spatial_group > 64: spatial_group = in_channels  # 当分组过大时退化为逐通道分组
        self.spatial_group = spatial_group
        self.lowfreq_att = lowfreq_att  # 是否为最低频残留分量也生成权重
        if spatial == 'conv':
            self.freq_weight_conv_list = nn.ModuleList()
            _n = len(k_list)
            if lowfreq_att:  _n += 1  # 若需要对最终低频也加权则多分配一个卷积
            for i in range(_n):
                # 使用 groups=spatial_group 的深度可分离 2D 卷积，逐组生成空间权重
                freq_weight_conv = nn.Conv2d(in_channels=in_channels,
                                             out_channels=self.spatial_group,
                                             stride=1,
                                             kernel_size=spatial_kernel,
                                             groups=self.spatial_group,
                                             padding=spatial_kernel // 2,
                                             bias=True)
                if init == 'zero':
                    nn.init.normal_(freq_weight_conv.weight, std=1e-6)  # 极小权重保证初期近似恒等
                    freq_weight_conv.bias.data.zero_()  # 偏置置零
                self.freq_weight_conv_list.append(freq_weight_conv)
        else:
            raise NotImplementedError
        self.act = act  # 权重激活类型：'sigmoid'、'tanh' 或 'softmax'

    def sp_act(self, freq_weight):
        # 对空间权重进行非线性变换并进行幅值归一/放大
        if self.act == 'sigmoid':
            freq_weight = freq_weight.sigmoid() * 2
        elif self.act == 'tanh':
            freq_weight = 1 + freq_weight.tanh()
        elif self.act == 'softmax':
            freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
        else:
            raise NotImplementedError
        return freq_weight

    def forward(self, x, att_feat=None):
        """
        前向计算：
        1) 对输入执行 rFFT 得到频谱；
        2) 按 k_list 构造由低到更低的频率掩码，迭代提取低频与互补高频；
        3) 针对每个频带通过卷积生成空间权重并逐像素加权；
        4) 将各频带结果逐元素求和实现频带融合。
        """
        if att_feat is None: att_feat = x  # 若未提供外部条件特征则使用自身
        x_list = []  # 存放各频带的加权结果
        x = x.to(torch.float32)  # 保证 FFT 数值稳定性
        pre_x = x.clone()  # 保存上一轮残留以构造多级金字塔
        b, _, h, w = x.shape
        h, w = int(h), int(w)
        # 计算二维 rFFT，输出最后一维为半谱
        x_fft = torch.fft.rfft2(x, norm='ortho')

        for idx, freq in enumerate(self.k_list):
            # 针对当前频带构造圆形低通掩码（以频域半径为准），半径与 1/freq 成正比
            mask = torch.zeros_like(x_fft[:, 0:1, :, :], device=x.device)
            _, freq_indices = get_fft2freq(d1=x.size(-2), d2=x.size(-1), use_rfft=True)
            freq_indices = freq_indices.max(dim=-1, keepdims=False)[0]
            mask[:, :, freq_indices < 0.5 / freq] = 1.0  # 设置半径内为 1，其余为 0
            # 低频部分通过掩码筛选后逆变换得到；高频部分取残差 pre_x - 低频
            low_part = torch.fft.irfft2(x_fft * mask, s=(h, w), dim=(-2, -1), norm='ortho')
            try:
                low_part = low_part.real  # rFFT 的逆变换理论上为实数，转为实部以对齐数据类型
            except:
                pass
            high_part = pre_x - low_part  # 从上一阶段输入中扣除当前低频得到高频细节
            pre_x = low_part  # 更新残留，供下一频带使用
            # 生成空间权重并施加到当前高频分量（分组重排后逐组相乘）
            freq_weight = self.freq_weight_conv_list[idx](att_feat)
            freq_weight = self.sp_act(freq_weight)
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group, -1, h,
                                                                                           w)
            x_list.append(tmp.reshape(b, -1, h, w))
        if self.lowfreq_att:
            # 对最终残留低频同样生成空间权重并加权
            freq_weight = self.freq_weight_conv_list[len(x_list)](att_feat)
            freq_weight = self.sp_act(freq_weight)
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pre_x.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
        else:
            # 若不加权低频，则直接叠加原低频残留
            x_list.append(pre_x)
        x = sum(x_list)  # 各频带结果逐元素相加得到融合特征
        return x


def get_fft2freq(d1, d2, use_rfft=False):
    """
    生成二维频域网格及其到原点的欧氏距离，并返回按距离升序的坐标索引：
    - d1：高（行）方向长度；
    - d2：宽（列）方向长度；
    - use_rfft：是否为 rFFT（最后一维仅保留非冗余半谱）。
    返回：
    - sorted_coords^T（形状 [2, N]）：按距离排序后的二维索引；
    - freq_hw（形状 [d1, d2' , 2]）：频率网格，其中 d2' = d2 或 d2//2+1（rFFT）。
    """
    freq_h = torch.fft.fftfreq(d1)  # 行方向频率
    if use_rfft:
        freq_w = torch.fft.rfftfreq(d2)  # 列方向半谱频率
    else:
        freq_w = torch.fft.fftfreq(d2)
    # 生成二维频率网格，并计算到原点的欧氏距离
    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w), dim=-1)
    dist = torch.norm(freq_hw, dim=-1)
    # 将距离展平后排序，获得升序索引
    sorted_dist, indices = torch.sort(dist.view(-1))
    if use_rfft:
        d2 = d2 // 2 + 1  # rFFT 情况下列数为半谱长度
    sorted_coords = torch.stack([indices // d2, indices % d2], dim=-1)
    # 可选的可视化逻辑被关闭；函数返回排序坐标与网格
    return sorted_coords.permute(1, 0), freq_hw


# @CONV_LAYERS.register_module()  # 兼容 mmdet/mmseg 的注册接口占位，当前未启用
class FDConv(nn.Conv2d):
    """
    频域动态（FDConv）：
    以 nn.Conv2d 为基类，通过将卷积权重映射到二维 rFFT 频域进行稀疏参数化与核组混合，
    结合全局与局部两级调制（KSM-Global / KSM-Local）、以及可选的频带调制（FBM），
    在保证表达力的同时显著降低参数量与计算量。
    """

    def __init__(self,
                 *args,
                 reduction=0.0625,
                 kernel_num=4,
                 use_fdconv_if_c_gt=16,  # 仅当 min(in,out) > 该阈值时启用 FDConv
                 use_fdconv_if_k_in=[1, 3],  # 仅当核尺寸属于此集合时启用 FDConv
                 use_fbm_if_k_in=[3],  # 当核尺寸在集合内时启用 FBM
                 kernel_temp=1.0,
                 temp=None,
                 att_multi=2.0,
                 param_ratio=1,
                 param_reduction=1.0,
                 ksm_only_kernel_att=False,
                 att_grid=1,
                 use_ksm_local=True,
                 ksm_local_act='sigmoid',
                 ksm_global_act='sigmoid',
                 spatial_freq_decompose=False,
                 convert_param=True,
                 linear_mode=False,
                 fbm_cfg={
                     'k_list': [2, 4, 8],
                     'lowfreq_att': False,
                     'fs_feat': 'feat',
                     'act': 'sigmoid',
                     'spatial': 'conv',
                     'spatial_group': 1,
                     'spatial_kernel': 3,
                     'init': 'zero',
                     'global_selection': False,
                 },
                 **kwargs,
                 ):
        # 先调用父类构造，建立标准 Conv2d 的权重与超参
        super().__init__(*args, **kwargs)
        self.use_fdconv_if_c_gt = use_fdconv_if_c_gt
        self.use_fdconv_if_k_in = use_fdconv_if_k_in
        self.kernel_num = kernel_num
        self.param_ratio = param_ratio  # 频域参数复制比率，用于多组核的权重共享/扩展
        self.param_reduction = param_reduction  # 频域参数保留比例（<1 表示随机子采样频点）
        self.use_ksm_local = use_ksm_local
        self.att_multi = att_multi
        self.spatial_freq_decompose = spatial_freq_decompose
        self.use_fbm_if_k_in = use_fbm_if_k_in

        self.ksm_local_act = ksm_local_act
        self.ksm_global_act = ksm_global_act
        assert self.ksm_local_act in ['sigmoid', 'tanh']  # 本地调制激活函数限定集合
        assert self.ksm_global_act in ['softmax', 'sigmoid', 'tanh']  # 全局调制激活函数限定集合

        # 核数量与温度设定：当 kernel_num 为 None 时按输出通道的一半自动推断，并据此设置温度
        if self.kernel_num is None:
            self.kernel_num = self.out_channels // 2
            kernel_temp = math.sqrt(self.kernel_num * self.param_ratio)
        if temp is None:
            temp = kernel_temp

        print('*** kernel_num:', self.kernel_num)  # 打印当前核组数量以便调试
        # alpha 为缩放系数，按最小通道数的一半与核组/参数比率计算，用于频域权重的幅值调节
        self.alpha = min(self.out_channels,
                         self.in_channels) // 2 * self.kernel_num * self.param_ratio / param_reduction
        # 若通道数较小或核尺寸不在指定集合内，则退化为标准卷积，后续逻辑直接返回
        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt or self.kernel_size[
            0] not in self.use_fdconv_if_k_in:
            return
        # 构建全局调制模块
        self.KSM_Global = KernelSpatialModulation_Global(self.in_channels, self.out_channels, self.kernel_size[0],
                                                         groups=self.groups,
                                                         temp=temp,
                                                         kernel_temp=kernel_temp,
                                                         reduction=reduction,
                                                         kernel_num=self.kernel_num * self.param_ratio,
                                                         kernel_att_init=None, att_multi=att_multi,
                                                         ksm_only_kernel_att=ksm_only_kernel_att,
                                                         act_type=self.ksm_global_act,
                                                         att_grid=att_grid, stride=self.stride,
                                                         spatial_freq_decompose=spatial_freq_decompose)

        # 当核尺寸在 use_fbm_if_k_in 中时启用频带调制
        if self.kernel_size[0] in use_fbm_if_k_in:
            self.FBM = FrequencyBandModulation(self.in_channels, **fbm_cfg)
            # 也可在此处接入通道压缩后再做 FBM 的变体；当前直接使用原特征

        # 可选的本地调制模块：输出维度映射至 (cout * k*k)
        if self.use_ksm_local:
            self.KSM_Local = KernelSpatialModulation_Local(channel=self.in_channels, kernel_num=1, out_n=int(
                self.out_channels * self.kernel_size[0] * self.kernel_size[1]))

        self.linear_mode = linear_mode  # 是否改为线性化权重模式（保留原权重张量结构）
        self.convert2dftweight(convert_param)  # 将空间域权重转换为频域参数形式

    def convert2dftweight(self, convert_param):
        """
        将空间域卷积核权重转换至 rFFT 频域表示：
        1) 将 (out, in, k, k) 重排为 (out*k, in*k) 的 2D 网格；
        2) 计算二维 rFFT 获得复谱；
        3) 依据 param_reduction 选择全部或子集频点，并按 param_ratio 复制；
        4) 将实部与虚部分离为最后一维大小为 2 的张量并注册为可学习参数。
        同时预计算频点索引映射以便前向快速聚合。
        """
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        freq_indices, _ = get_fft2freq(d1 * k1, d2 * k2, use_rfft=True)  # 形状 [2, d1*k1*(d2*k2//2+1)]
        weight = self.weight.permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)  # 重排为二维矩阵
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1))  # 计算二维 rFFT：得到半谱
        if self.param_reduction < 1:
            # 随机重排频点索引并截取前比例对应数量，实现频域稀疏参数化
            freq_indices = freq_indices[:, torch.randperm(freq_indices.size(1), generator=torch.Generator().manual_seed(
                freq_indices.size(1)))]
            freq_indices = freq_indices[:, :int(freq_indices.size(1) * self.param_reduction)]
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)
            weight_rfft = weight_rfft[freq_indices[0, :], freq_indices[1, :]]
            weight_rfft = weight_rfft.reshape(-1, 2)[None,].repeat(self.param_ratio, 1, 1) / (
                        min(self.out_channels, self.in_channels) // 2)
        else:
            # 保留全部频点并将实部/虚部分离至最后一维；复制 param_ratio 份
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None,].repeat(self.param_ratio, 1,
                                                                                                  1, 1) / (
                                      min(self.out_channels, self.in_channels) // 2)

        if convert_param:
            # 用频域参数替代原空间域权重，并从模块中删除 self.weight 避免重复
            self.dft_weight = nn.Parameter(weight_rfft, requires_grad=True)
            del self.weight
        else:
            # 当不转换时，若 linear_mode 为真则压缩权重维度为二维；否则保持父类权重不变
            if self.linear_mode:
                self.weight = torch.nn.Parameter(self.weight.squeeze(), requires_grad=True)
        # 为每一份 param_ratio 预生成索引分块（按 kernel_num 均分频点）
        self.indices = []
        for i in range(self.param_ratio):
            self.indices.append(freq_indices.reshape(2, self.kernel_num, -1))

    def get_FDW(self, ):
        """当未进行 convert_param 时，运行时从当前空间域权重即时计算频域表示。"""
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        weight = self.weight.reshape(d1, d2, k1, k2).permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1))
        weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None,].repeat(self.param_ratio, 1, 1,
                                                                                              1) / (
                                  min(self.out_channels, self.in_channels) // 2)
        return weight_rfft

    def forward(self, x):
        # 若不满足启用条件，则直接调用父类的标准卷积实现
        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt or self.kernel_size[
            0] not in self.use_fdconv_if_k_in:
            return super().forward(x)
        # 提取全局描述特征并生成四类调制因子
        global_x = F.adaptive_avg_pool2d(x, 1)
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.KSM_Global(global_x)
        # 本地调制：生成对 (cout×cin×k×k) 的逐元素调制项
        if self.use_ksm_local:
            hr_att_logit = self.KSM_Local(global_x)  # b×1×cin×(cout*k*k)
            hr_att_logit = hr_att_logit.reshape(x.size(0), 1, self.in_channels, self.out_channels, self.kernel_size[0],
                                                self.kernel_size[1])
            hr_att_logit = hr_att_logit.permute(0, 1, 3, 2, 4, 5)
            if self.ksm_local_act == 'sigmoid':
                hr_att = hr_att_logit.sigmoid() * self.att_multi
            elif self.ksm_local_act == 'tanh':
                hr_att = 1 + hr_att_logit.tanh()
            else:
                raise NotImplementedError
        else:
            hr_att = 1
        b = x.size(0)
        batch_size, in_planes, height, width = x.size()
        # 构建频域权重聚合缓冲区：最后一维大小为 2 表示实部与虚部
        DFT_map = torch.zeros(
            (b, self.out_channels * self.kernel_size[0], self.in_channels * self.kernel_size[1] // 2 + 1, 2),
            device=x.device)
        # 重排核注意力以匹配 param_ratio 与 kernel_num 的分组结构
        kernel_attention = kernel_attention.reshape(b, self.param_ratio, self.kernel_num, -1)
        # 获取频域权重参数（静态参数或即时计算）
        if hasattr(self, 'dft_weight'):
            dft_weight = self.dft_weight
        else:
            dft_weight = self.get_FDW()

        # 逐份 param_ratio 聚合所选频点的权重：将实部与虚部分别乘以核注意力后累加到 DFT_map
        for i in range(self.param_ratio):
            indices = self.indices[i]
            if self.param_reduction < 1:
                w = dft_weight[i].reshape(self.kernel_num, -1, 2)[None]
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack(
                    [w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1)
            else:
                w = dft_weight[i][indices[0, :, :], indices[1, :, :]][None] * self.alpha  # 幅值缩放以补偿频点密度
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack(
                    [w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1)

        # 对聚合后的频域图执行 irFFT，恢复为空间域的自适应权重张量（按 b 维批次展开）
        adaptive_weights = torch.fft.irfft2(torch.view_as_complex(DFT_map), dim=(1, 2)).reshape(batch_size, 1,
                                                                                                self.out_channels,
                                                                                                self.kernel_size[0],
                                                                                                self.in_channels,
                                                                                                self.kernel_size[1])
        adaptive_weights = adaptive_weights.permute(0, 1, 2, 4, 3, 5)  # 重排为 b×1×cout×cin×k×k
        # 若启用 FBM，则先对输入特征做频带重加权以增强可分辨性
        if hasattr(self, 'FBM'):
            x = self.FBM(x)

        # 根据计算量与形状选择两种等价实现：
        if self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1] < (
                in_planes + self.out_channels) * height * width:
            # 路径 A：先聚合权重，再使用分组卷积一次性对整批数据进行卷积
            aggregate_weight = spatial_attention * channel_attention * filter_attention * adaptive_weights * hr_att
            aggregate_weight = torch.sum(aggregate_weight, dim=1)  # 在核组维度求和
            aggregate_weight = aggregate_weight.view(
                [-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]])
            x = x.reshape(1, -1, height, width)
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
            output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
        else:
            # 路径 B：先按空间注意力与核组聚合，再对输入施加通道注意力后卷积，最后施加过滤器注意力
            aggregate_weight = spatial_attention * adaptive_weights * hr_att
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            if not isinstance(channel_attention, float):
                x = x * channel_attention.view(b, -1, 1, 1)
            aggregate_weight = aggregate_weight.view(
                [-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]])
            x = x.reshape(1, -1, height, width)
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
            if isinstance(filter_attention, float):
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
            else:
                output = output.view(batch_size, self.out_channels, output.size(-2),
                                     output.size(-1)) * filter_attention.view(b, -1, 1, 1)
        # 若原始卷积定义带偏置，则在输出上逐通道加上偏置
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output

    def profile_module(
            self, input: Tensor, *args, **kwargs
    ):
        """理论复杂度估计接口：
        返回输入本体、参数量与 MACs（包含 FFT 近似项）。当前实现为占位近似。"""
        b_sz, c, h, w = input.shape
        seq_len = h * w

        # FFT 与 iFFT 近似开销（按 O(N log N) 粒度估计）
        p_ff, m_ff = 0, 5 * b_sz * seq_len * int(math.log(seq_len)) * c
        # 其他计算近似（占位项，按隐藏尺寸的组合给出）
        params = macs = self.hidden_size * self.hidden_size_factor * self.hidden_size * 2 * 2 // self.num_blocks
        macs = macs * b_sz * seq_len

        return input, params, macs + m_ff


if __name__ == '__main__':
    # 作为模块入口占位；实际使用中通过导入类并实例化
    pass


import torch  #  导入依赖模块/符号以供后续神经网络构建与工具函数使用
import torch.nn as nn  #  导入依赖模块/符号以供后续神经网络构建与工具函数使用
from timm.models.registry import register_model  #  导入依赖模块/符号以供后续神经网络构建与工具函数使用
import math  #  导入依赖模块/符号以供后续神经网络构建与工具函数使用
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d  #  导入依赖模块/符号以供后续神经网络构建与工具函数使用
from timm.models._builder import resolve_pretrained_cfg  #  导入依赖模块/符号以供后续神经网络构建与工具函数使用
try:  #  异常捕获开始，兼容不同版本/接口差异
    from timm.models._builder import _update_default_kwargs as update_args  #  导入依赖模块/符号以供后续神经网络构建与工具函数使用
except:  #  异常分支，提供降级兼容策略
    from timm.models._builder import _update_default_model_kwargs as update_args  #  导入依赖模块/符号以供后续神经网络构建与工具函数使用
from timm.models.vision_transformer import Mlp, PatchEmbed  #  导入依赖模块/符号以供后续神经网络构建与工具函数使用
from timm.models.layers import DropPath, trunc_normal_  #  导入依赖模块/符号以供后续神经网络构建与工具函数使用
from timm.models.registry import register_model  #  导入依赖模块/符号以供后续神经网络构建与工具函数使用
import torch.nn.functional as F  #  导入依赖模块/符号以供后续神经网络构建与工具函数使用
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  #  导入依赖模块/符号以供后续神经网络构建与工具函数使用
from einops import rearrange, repeat  #  导入依赖模块/符号以供后续神经网络构建与工具函数使用
from .registry import register_pip_model  #  导入依赖模块/符号以供后续神经网络构建与工具函数使用
from pathlib import Path  #  导入依赖模块/符号以供后续神经网络构建与工具函数使用


def _cfg(url='', **kwargs):  #  定义函数 _cfg，封装独立功能逻辑，便于复用与测试
    return {'url': url,  #  返回计算结果到调用方，保持接口清晰
            'num_classes': 1000,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            'input_size': (3, 224, 224),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            'pool_size': None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            'crop_pct': 0.875,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            'interpolation': 'bicubic',  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            'fixed_input_size': True,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            'mean': (0.485, 0.456, 0.406),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            'std': (0.229, 0.224, 0.225),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            **kwargs  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            }  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）


default_cfgs = {  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    'mamba_vision_T': _cfg(url='https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar',  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                           crop_pct=1.0,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                           input_size=(3, 224, 224),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                           crop_mode='center'),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    'mamba_vision_T2': _cfg(url='https://huggingface.co/nvidia/MambaVision-T2-1K/resolve/main/mambavision_tiny2_1k.pth.tar',  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                            crop_pct=0.98,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                            input_size=(3, 224, 224),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                            crop_mode='center'),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    'mamba_vision_S': _cfg(url='https://huggingface.co/nvidia/MambaVision-S-1K/resolve/main/mambavision_small_1k.pth.tar',  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                           crop_pct=0.93,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                           input_size=(3, 224, 224),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                           crop_mode='center'),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    'mamba_vision_B': _cfg(url='https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar',  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                           crop_pct=1.0,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                           input_size=(3, 224, 224),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                           crop_mode='center'),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    'mamba_vision_B_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-B-21K/resolve/main/mambavision_base_21k.pth.tar',  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                           crop_pct=1.0,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                           input_size=(3, 224, 224),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                           crop_mode='center'),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    'mamba_vision_L': _cfg(url='https://huggingface.co/nvidia/MambaVision-L-1K/resolve/main/mambavision_large_1k.pth.tar',  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                           crop_pct=1.0,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                           input_size=(3, 224, 224),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                           crop_mode='center'),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    'mamba_vision_L_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-L-21K/resolve/main/mambavision_large_21k.pth.tar',  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                           crop_pct=1.0,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                           input_size=(3, 224, 224),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                           crop_mode='center'),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    'mamba_vision_L2': _cfg(url='https://huggingface.co/nvidia/MambaVision-L2-1K/resolve/main/mambavision_large2_1k.pth.tar',  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                            crop_pct=1.0,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                            input_size=(3, 224, 224),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                            crop_mode='center'),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    'mamba_vision_L2_512_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-L2-512-21K/resolve/main/mambavision_L2_21k_240m_512.pth.tar',  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                            crop_pct=0.93,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                            input_size=(3, 512, 512),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                            crop_mode='squash'),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    'mamba_vision_L3_256_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-L3-256-21K/resolve/main/mambavision_L3_21k_740m_256.pth.tar',  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                            crop_pct=1.0,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                            input_size=(3, 256, 256),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                            crop_mode='center'),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    'mamba_vision_L3_512_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-L3-512-21K/resolve/main/mambavision_L3_21k_740m_512.pth.tar',  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                            crop_pct=0.93,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                            input_size=(3, 512, 512),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                            crop_mode='squash'),                                 #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
}  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）


def window_partition(x, window_size):  #  定义函数 window_partition，封装独立功能逻辑，便于复用与测试
    """ window_partition：中文文档字符串（将原英文说明改为中文，描述用途与参数约定） """  # 采用中文 docstring，概述该类/函数的职责与设计取舍
    B, C, H, W = x.shape  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)  #  张量维度变换：为后续算子对齐期望的形状
    return windows  #  返回计算结果到调用方，保持接口清晰


def window_reverse(windows, window_size, H, W):  #  定义函数 window_reverse，封装独立功能逻辑，便于复用与测试
    """ window_reverse：中文文档字符串（将原英文说明改为中文，描述用途与参数约定） """  # 采用中文 docstring，概述该类/函数的职责与设计取舍
    B = int(windows.shape[0] / (H * W / window_size / window_size))  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)  #  张量维度变换：为后续算子对齐期望的形状
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)  #  张量维度变换：为后续算子对齐期望的形状
    return x  #  返回计算结果到调用方，保持接口清晰


def _load_state_dict(module, state_dict, strict=False, logger=None):  #  定义函数 _load_state_dict，封装独立功能逻辑，便于复用与测试
    """ _load_state_dict：中文文档字符串（将原英文说明改为中文，描述用途与参数约定） """  # 采用中文 docstring，概述该类/函数的职责与设计取舍
    unexpected_keys = []  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    all_missing_keys = []  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    err_msg = []  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

    metadata = getattr(state_dict, '_metadata', None)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    state_dict = state_dict.copy()  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    if metadata is not None:  #  分支控制，根据条件切换不同执行路径以提升灵活性
        state_dict._metadata = metadata  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    
    def load(module, prefix=''):  #  定义函数 load，封装独立功能逻辑，便于复用与测试
        local_metadata = {} if metadata is None else metadata.get(  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            prefix[:-1], {})  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                     all_missing_keys, unexpected_keys,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                     err_msg)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        for name, child in module._modules.items():  #  循环结构，用于迭代处理序列/模块列表
            if child is not None:  #  分支控制，根据条件切换不同执行路径以提升灵活性
                load(child, prefix + name + '.')  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

    load(module)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    load = None  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    missing_keys = [  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        key for key in all_missing_keys if 'num_batches_tracked' not in key  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    ]  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

    if unexpected_keys:  #  分支控制，根据条件切换不同执行路径以提升灵活性
        err_msg.append('unexpected key in source '  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                       f'state_dict: {", ".join(unexpected_keys)}\n')  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    if missing_keys:  #  分支控制，根据条件切换不同执行路径以提升灵活性
        err_msg.append(  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

    
    if len(err_msg) > 0:  #  分支控制，根据条件切换不同执行路径以提升灵活性
        err_msg.insert(  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            0, 'The model and loaded state dict do not match exactly\n')  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        err_msg = '\n'.join(err_msg)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        if strict:  #  分支控制，根据条件切换不同执行路径以提升灵活性
            raise RuntimeError(err_msg)  #  主动抛出异常，提示非法配置或未实现功能
        elif logger is not None:  #  分支控制，根据条件切换不同执行路径以提升灵活性
            logger.warning(err_msg)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        else:  #  分支控制，根据条件切换不同执行路径以提升灵活性
            print(err_msg)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）


def _load_checkpoint(model,  #  定义函数 _load_checkpoint，封装独立功能逻辑，便于复用与测试
                    filename,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                    map_location='cpu',  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                    strict=False,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                    logger=None):  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    """ _load_checkpoint：中文文档字符串（将原英文说明改为中文，描述用途与参数约定） """  # 采用中文 docstring，概述该类/函数的职责与设计取舍
    checkpoint = torch.load(filename, map_location=map_location)  #  读取权重检查点到内存，支持 CPU/GPU 映射
    if not isinstance(checkpoint, dict):  #  分支控制，根据条件切换不同执行路径以提升灵活性
        raise RuntimeError(  #  主动抛出异常，提示非法配置或未实现功能
            f'No state_dict found in checkpoint file {filename}')  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    if 'state_dict' in checkpoint:  #  分支控制，根据条件切换不同执行路径以提升灵活性
        state_dict = checkpoint['state_dict']  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    elif 'model' in checkpoint:  #  分支控制，根据条件切换不同执行路径以提升灵活性
        state_dict = checkpoint['model']  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    else:  #  分支控制，根据条件切换不同执行路径以提升灵活性
        state_dict = checkpoint  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    if list(state_dict.keys())[0].startswith('module.'):  #  分支控制，根据条件切换不同执行路径以提升灵活性
        state_dict = {k[7:]: v for k, v in state_dict.items()}  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

    if sorted(list(state_dict.keys()))[0].startswith('encoder'):  #  分支控制，根据条件切换不同执行路径以提升灵活性
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

    _load_state_dict(model, state_dict, strict, logger)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    return checkpoint  #  返回计算结果到调用方，保持接口清晰


class Downsample(nn.Module):  #  定义类 Downsample，作为可复用的网络组件/模型结构
    """ Downsample：中文文档字符串（将原英文说明改为中文，描述用途与参数约定） """  # 采用中文 docstring，概述该类/函数的职责与设计取舍

    def __init__(self,  #  定义函数 __init__，封装独立功能逻辑，便于复用与测试
                 dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 keep_dim=False,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 ):  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        """ __init__：中文文档字符串（将原英文说明改为中文，描述用途与参数约定） """  # 采用中文 docstring，概述该类/函数的职责与设计取舍

        super().__init__()  #  调用父类构造，完成基础 Module 初始化
        if keep_dim:  #  分支控制，根据条件切换不同执行路径以提升灵活性
            dim_out = dim  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        else:  #  分支控制，根据条件切换不同执行路径以提升灵活性
            dim_out = 2 * dim  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.reduction = nn.Sequential(  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),  #  卷积层：提取局部空间特征，保留通道维
        )  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

    def forward(self, x):  #  定义函数 forward，封装独立功能逻辑，便于复用与测试
        x = self.reduction(x)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        return x  #  返回计算结果到调用方，保持接口清晰


class PatchEmbed(nn.Module):  #  定义类 PatchEmbed，作为可复用的网络组件/模型结构
    """ PatchEmbed：中文文档字符串（将原英文说明改为中文，描述用途与参数约定） """  # 采用中文 docstring，概述该类/函数的职责与设计取舍

    def __init__(self, in_chans=3, in_dim=64, dim=96):  #  定义函数 __init__，封装独立功能逻辑，便于复用与测试
        """ __init__：中文文档字符串（将原英文说明改为中文，描述用途与参数约定） """  # 采用中文 docstring，概述该类/函数的职责与设计取舍
        # 中文注释：保留原意并本地化说明
        super().__init__()  #  调用父类构造，完成基础 Module 初始化
        self.proj = nn.Identity()  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.conv_down = nn.Sequential(  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),  #  卷积层：提取局部空间特征，保留通道维
            nn.BatchNorm2d(in_dim, eps=1e-4),  #  批归一化：稳定训练并加速收敛
            nn.ReLU(),  #  非线性激活：引入表示能力、缓解线性可分限制
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),  #  卷积层：提取局部空间特征，保留通道维
            nn.BatchNorm2d(dim, eps=1e-4),  #  批归一化：稳定训练并加速收敛
            nn.ReLU()  #  非线性激活：引入表示能力、缓解线性可分限制
            )  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

    def forward(self, x):  #  定义函数 forward，封装独立功能逻辑，便于复用与测试
        x = self.proj(x)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        x = self.conv_down(x)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        return x  #  返回计算结果到调用方，保持接口清晰


class ConvBlock(nn.Module):  #  定义类 ConvBlock，作为可复用的网络组件/模型结构

    def __init__(self, dim,  #  定义函数 __init__，封装独立功能逻辑，便于复用与测试
                 drop_path=0.,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 layer_scale=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 kernel_size=3):  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        super().__init__()  #  调用父类构造，完成基础 Module 初始化

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)  #  卷积层：提取局部空间特征，保留通道维
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)  #  批归一化：稳定训练并加速收敛
        self.act1 = nn.GELU(approximate= 'tanh')  #  非线性激活：引入表示能力、缓解线性可分限制
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)  #  卷积层：提取局部空间特征，保留通道维
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)  #  批归一化：稳定训练并加速收敛
        self.layer_scale = layer_scale  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        if layer_scale is not None and type(layer_scale) in [int, float]:  #  分支控制，根据条件切换不同执行路径以提升灵活性
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            self.layer_scale = True  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        else:  #  分支控制，根据条件切换不同执行路径以提升灵活性
            self.layer_scale = False  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  #  随机深度（Stochastic Depth）：正则化深层网络

    def forward(self, x):  #  定义函数 forward，封装独立功能逻辑，便于复用与测试
        input = x  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        x = self.conv1(x)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        x = self.norm1(x)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        x = self.act1(x)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        x = self.conv2(x)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        x = self.norm2(x)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        if self.layer_scale:  #  分支控制，根据条件切换不同执行路径以提升灵活性
            x = x * self.gamma.view(1, -1, 1, 1)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        x = input + self.drop_path(x)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        return x  #  返回计算结果到调用方，保持接口清晰


class MambaVisionMixer(nn.Module):  #  定义类 MambaVisionMixer，作为可复用的网络组件/模型结构
    def __init__(  #  定义函数 __init__，封装独立功能逻辑，便于复用与测试
        self,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        d_model,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        d_state=16,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        d_conv=4,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        expand=2,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        dt_rank="auto",  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        dt_min=0.001,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        dt_max=0.1,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        dt_init="random",  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        dt_scale=1.0,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        dt_init_floor=1e-4,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        conv_bias=True,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        bias=False,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        use_fast_path=True,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        layer_idx=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        device=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        dtype=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    ):  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        factory_kwargs = {"device": device, "dtype": dtype}  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        super().__init__()  #  调用父类构造，完成基础 Module 初始化
        self.d_model = d_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.d_state = d_state  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.d_conv = d_conv  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.expand = expand  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.d_inner = int(self.expand * self.d_model)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.use_fast_path = use_fast_path  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.layer_idx = layer_idx  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)      #  线性层：特征映射到目标维度/类别空间
        self.x_proj = nn.Linear(  #  线性层：特征映射到目标维度/类别空间
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        )  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)  #  线性层：特征映射到目标维度/类别空间
        dt_init_std = self.dt_rank**-0.5 * dt_scale  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        if dt_init == "constant":  #  分支控制，根据条件切换不同执行路径以提升灵活性
            nn.init.constant_(self.dt_proj.weight, dt_init_std)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        elif dt_init == "random":  #  分支控制，根据条件切换不同执行路径以提升灵活性
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        else:  #  分支控制，根据条件切换不同执行路径以提升灵活性
            raise NotImplementedError  #  主动抛出异常，提示非法配置或未实现功能
        dt = torch.exp(  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            + math.log(dt_min)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        ).clamp(min=dt_init_floor)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        inv_dt = dt + torch.log(-torch.expm1(-dt))  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        with torch.no_grad():  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            self.dt_proj.bias.copy_(inv_dt)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.dt_proj.bias._no_reinit = True  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        A = repeat(  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            "n -> d n",  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            d=self.d_inner//2,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        ).contiguous()  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        A_log = torch.log(A)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.A_log = nn.Parameter(A_log)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.A_log._no_weight_decay = True  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.D._no_weight_decay = True  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)  #  线性层：特征映射到目标维度/类别空间
        self.conv1d_x = nn.Conv1d(  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            in_channels=self.d_inner//2,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            out_channels=self.d_inner//2,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            bias=conv_bias//2,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            kernel_size=d_conv,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            groups=self.d_inner//2,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            **factory_kwargs,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        )  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.conv1d_z = nn.Conv1d(  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            in_channels=self.d_inner//2,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            out_channels=self.d_inner//2,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            bias=conv_bias//2,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            kernel_size=d_conv,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            groups=self.d_inner//2,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            **factory_kwargs,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        )  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

    def forward(self, hidden_states):  #  定义函数 forward，封装独立功能逻辑，便于复用与测试
        """ forward：中文文档字符串（将原英文说明改为中文，描述用途与参数约定） """  # 采用中文 docstring，概述该类/函数的职责与设计取舍
        _, seqlen, _ = hidden_states.shape  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        xz = self.in_proj(hidden_states)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        xz = rearrange(xz, "b l d -> b d l")  #  张量维度变换：为后续算子对齐期望的形状
        x, z = xz.chunk(2, dim=1)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        A = -torch.exp(self.A_log.float())  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))  #  非线性激活：引入表示能力、缓解线性可分限制
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))  #  非线性激活：引入表示能力、缓解线性可分限制
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  #  张量维度变换：为后续算子对齐期望的形状
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)  #  张量维度变换：为后续算子对齐期望的形状
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()  #  张量维度变换：为后续算子对齐期望的形状
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()  #  张量维度变换：为后续算子对齐期望的形状
        y = selective_scan_fn(x,   #  调用选择性扫描 SSM 内核，实现高效长序列建模
                              dt,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                              A,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                              B,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                              C,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                              self.D.float(),   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                              z=None,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                              delta_bias=self.dt_proj.bias.float(),   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                              delta_softplus=True,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                              return_last_state=None)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        
        y = torch.cat([y, z], dim=1)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        y = rearrange(y, "b d l -> b l d")  #  张量维度变换：为后续算子对齐期望的形状
        out = self.out_proj(y)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        return out  #  返回计算结果到调用方，保持接口清晰
    

class Attention(nn.Module):  #  定义类 Attention，作为可复用的网络组件/模型结构

    def __init__(  #  定义函数 __init__，封装独立功能逻辑，便于复用与测试
            self,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            num_heads=8,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            qkv_bias=False,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            qk_norm=False,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            attn_drop=0.,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            proj_drop=0.,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            norm_layer=nn.LayerNorm,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    ):  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        super().__init__()  #  调用父类构造，完成基础 Module 初始化
        assert dim % num_heads == 0  #  断言输入有效性，早期发现配置错误
        self.num_heads = num_heads  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.head_dim = dim // num_heads  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.scale = self.head_dim ** -0.5  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.fused_attn = True  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  #  线性层：特征映射到目标维度/类别空间
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.attn_drop = nn.Dropout(attn_drop)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.proj = nn.Linear(dim, dim)  #  线性层：特征映射到目标维度/类别空间
        self.proj_drop = nn.Dropout(proj_drop)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

    def forward(self, x):  #  定义函数 forward，封装独立功能逻辑，便于复用与测试
        B, N, C = x.shape  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  #  张量维度变换：为后续算子对齐期望的形状
        q, k, v = qkv.unbind(0)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        q, k = self.q_norm(q), self.k_norm(k)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

        if self.fused_attn:  #  分支控制，根据条件切换不同执行路径以提升灵活性
            x = F.scaled_dot_product_attention(  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
             q, k, v,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                dropout_p=self.attn_drop.p,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            )  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        else:  #  分支控制，根据条件切换不同执行路径以提升灵活性
            q = q * self.scale  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            attn = q @ k.transpose(-2, -1)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            attn = attn.softmax(dim=-1)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            attn = self.attn_drop(attn)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            x = attn @ v  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

        x = x.transpose(1, 2).reshape(B, N, C)  #  张量维度变换：为后续算子对齐期望的形状
        x = self.proj(x)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        x = self.proj_drop(x)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        return x  #  返回计算结果到调用方，保持接口清晰


class Block(nn.Module):  #  定义类 Block，作为可复用的网络组件/模型结构
    def __init__(self,   #  定义函数 __init__，封装独立功能逻辑，便于复用与测试
                 dim,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 num_heads,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 counter,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 transformer_blocks,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 mlp_ratio=4.,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 qkv_bias=False,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 qk_scale=False,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 drop=0.,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 attn_drop=0.,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 drop_path=0.,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 act_layer=nn.GELU,   #  非线性激活：引入表示能力、缓解线性可分限制
                 norm_layer=nn.LayerNorm,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 Mlp_block=Mlp,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 layer_scale=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 ):  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        super().__init__()  #  调用父类构造，完成基础 Module 初始化
        self.norm1 = norm_layer(dim)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        if counter in transformer_blocks:  #  分支控制，根据条件切换不同执行路径以提升灵活性
            self.mixer = Attention(  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            num_heads=num_heads,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            qkv_bias=qkv_bias,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            qk_norm=qk_scale,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            attn_drop=attn_drop,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            proj_drop=drop,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            norm_layer=norm_layer,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        )  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        else:  #  分支控制，根据条件切换不同执行路径以提升灵活性
            self.mixer = MambaVisionMixer(d_model=dim,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                          d_state=8,    #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                          d_conv=3,      #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                          expand=1  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                          )  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  #  随机深度（Stochastic Depth）：正则化深层网络
        self.norm2 = norm_layer(dim)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        mlp_hidden_dim = int(dim * mlp_ratio)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

    def forward(self, x):  #  定义函数 forward，封装独立功能逻辑，便于复用与测试
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        return x  #  返回计算结果到调用方，保持接口清晰


class MambaVisionLayer(nn.Module):  #  定义类 MambaVisionLayer，作为可复用的网络组件/模型结构
    """ MambaVisionLayer：中文文档字符串（将原英文说明改为中文，描述用途与参数约定） """  # 采用中文 docstring，概述该类/函数的职责与设计取舍

    def __init__(self,  #  定义函数 __init__，封装独立功能逻辑，便于复用与测试
                 dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 depth,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 num_heads,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 window_size,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 conv=False,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 downsample=True,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 mlp_ratio=4.,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 qkv_bias=True,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 qk_scale=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 drop=0.,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 attn_drop=0.,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 drop_path=0.,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 layer_scale=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 layer_scale_conv=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 transformer_blocks = [],  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    ):  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        """ __init__：中文文档字符串（将原英文说明改为中文，描述用途与参数约定） """  # 采用中文 docstring，概述该类/函数的职责与设计取舍

        super().__init__()  #  调用父类构造，完成基础 Module 初始化
        self.conv = conv  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.transformer_block = False  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        if conv:  #  分支控制，根据条件切换不同执行路径以提升灵活性
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,  #  模块列表容器：注册可迭代的子模块序列
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                                   layer_scale=layer_scale_conv)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                                   for i in range(depth)])  #  循环结构，用于迭代处理序列/模块列表
            self.transformer_block = False  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        else:  #  分支控制，根据条件切换不同执行路径以提升灵活性
            self.blocks = nn.ModuleList([Block(dim=dim,  #  模块列表容器：注册可迭代的子模块序列
                                               counter=i,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                               transformer_blocks=transformer_blocks,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                               num_heads=num_heads,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                               mlp_ratio=mlp_ratio,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                               qkv_bias=qkv_bias,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                               qk_scale=qk_scale,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                               drop=drop,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                               attn_drop=attn_drop,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                               layer_scale=layer_scale)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                               for i in range(depth)])  #  循环结构，用于迭代处理序列/模块列表
            self.transformer_block = True  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

        self.downsample = None if not downsample else Downsample(dim=dim)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.do_gt = False  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.window_size = window_size  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

    def forward(self, x):  #  定义函数 forward，封装独立功能逻辑，便于复用与测试
        _, _, H, W = x.shape  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

        if self.transformer_block:  #  分支控制，根据条件切换不同执行路径以提升灵活性
            pad_r = (self.window_size - W % self.window_size) % self.window_size  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            pad_b = (self.window_size - H % self.window_size) % self.window_size  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            if pad_r > 0 or pad_b > 0:  #  分支控制，根据条件切换不同执行路径以提升灵活性
                x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))  #  边界填充以适配窗口大小，避免尺寸不整除
                _, _, Hp, Wp = x.shape  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            else:  #  分支控制，根据条件切换不同执行路径以提升灵活性
                Hp, Wp = H, W  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            x = window_partition(x, self.window_size)  #  将特征图分块成局部窗口以提升注意力/混合器效率

        for _, blk in enumerate(self.blocks):  #  循环结构，用于迭代处理序列/模块列表
            x = blk(x)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        if self.transformer_block:  #  分支控制，根据条件切换不同执行路径以提升灵活性
            x = window_reverse(x, self.window_size, Hp, Wp)  #  将局部窗口重组回原始空间布局
            if pad_r > 0 or pad_b > 0:  #  分支控制，根据条件切换不同执行路径以提升灵活性
                x = x[:, :, :H, :W].contiguous()  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        if self.downsample is None:  #  分支控制，根据条件切换不同执行路径以提升灵活性
            return x  #  返回计算结果到调用方，保持接口清晰
        return self.downsample(x)  #  返回计算结果到调用方，保持接口清晰


class MambaVision(nn.Module):  #  定义类 MambaVision，作为可复用的网络组件/模型结构
    """ MambaVision：中文文档字符串（将原英文说明改为中文，描述用途与参数约定） """  # 采用中文 docstring，概述该类/函数的职责与设计取舍

    def __init__(self,  #  定义函数 __init__，封装独立功能逻辑，便于复用与测试
                 dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 in_dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 depths,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 window_size,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 mlp_ratio,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 num_heads,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 drop_path_rate=0.2,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 in_chans=3,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 num_classes=1000,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 qkv_bias=True,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 qk_scale=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 drop_rate=0.,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 attn_drop_rate=0.,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 layer_scale=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 layer_scale_conv=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                 **kwargs):  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        """ __init__：中文文档字符串（将原英文说明改为中文，描述用途与参数约定） """  # 采用中文 docstring，概述该类/函数的职责与设计取舍
        super().__init__()  #  调用父类构造，完成基础 Module 初始化
        num_features = int(dim * 2 ** (len(depths) - 1))  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.num_classes = num_classes  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.levels = nn.ModuleList()  #  模块列表容器：注册可迭代的子模块序列
        for i in range(len(depths)):  #  循环结构，用于迭代处理序列/模块列表
            conv = True if (i == 0 or i == 1) else False  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            level = MambaVisionLayer(dim=int(dim * 2 ** i),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                     depth=depths[i],  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                     num_heads=num_heads[i],  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                     window_size=window_size[i],  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                     mlp_ratio=mlp_ratio,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                     qkv_bias=qkv_bias,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                     qk_scale=qk_scale,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                     conv=conv,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                     drop=drop_rate,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                     attn_drop=attn_drop_rate,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                     downsample=(i < 3),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                     layer_scale=layer_scale,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                     layer_scale_conv=layer_scale_conv,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                     transformer_blocks=list(range(depths[i]//2+1, depths[i])) if depths[i]%2!=0 else list(range(depths[i]//2, depths[i])),  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                                     )  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            self.levels.append(level)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        self.norm = nn.BatchNorm2d(num_features)  #  批归一化：稳定训练并加速收敛
        self.avgpool = nn.AdaptiveAvgPool2d(1)  #  自适应全局平均池化：将任意空间尺寸压缩为 1×1
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()  #  线性层：特征映射到目标维度/类别空间
        self.apply(self._init_weights)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

    def _init_weights(self, m):  #  定义函数 _init_weights，封装独立功能逻辑，便于复用与测试
        if isinstance(m, nn.Linear):  #  分支控制，根据条件切换不同执行路径以提升灵活性
            trunc_normal_(m.weight, std=.02)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            if isinstance(m, nn.Linear) and m.bias is not None:  #  分支控制，根据条件切换不同执行路径以提升灵活性
                nn.init.constant_(m.bias, 0)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        elif isinstance(m, nn.LayerNorm):  #  分支控制，根据条件切换不同执行路径以提升灵活性
            nn.init.constant_(m.bias, 0)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            nn.init.constant_(m.weight, 1.0)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        elif isinstance(m, LayerNorm2d):  #  分支控制，根据条件切换不同执行路径以提升灵活性
            nn.init.constant_(m.bias, 0)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            nn.init.constant_(m.weight, 1.0)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        elif isinstance(m, nn.BatchNorm2d):  #  分支控制，根据条件切换不同执行路径以提升灵活性
            nn.init.ones_(m.weight)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            nn.init.zeros_(m.bias)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）

    @torch.jit.ignore  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    def no_weight_decay_keywords(self):  #  定义函数 no_weight_decay_keywords，封装独立功能逻辑，便于复用与测试
        return {'rpb'}  #  返回计算结果到调用方，保持接口清晰

    def forward_features(self, x):  #  定义函数 forward_features，封装独立功能逻辑，便于复用与测试
        x = self.patch_embed(x)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        for level in self.levels:  #  循环结构，用于迭代处理序列/模块列表
            x = level(x)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        x = self.norm(x)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        x = self.avgpool(x)  #  分类头前的汇聚与摊平，连接到全连接层
        x = torch.flatten(x, 1)  #  分类头前的汇聚与摊平，连接到全连接层
        return x  #  返回计算结果到调用方，保持接口清晰

    def forward(self, x):  #  定义函数 forward，封装独立功能逻辑，便于复用与测试
        x = self.forward_features(x)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        x = self.head(x)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        return x  #  返回计算结果到调用方，保持接口清晰

    def _load_state_dict(self,   #  定义函数 _load_state_dict，封装独立功能逻辑，便于复用与测试
                         pretrained,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                         strict: bool = False):  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
        _load_checkpoint(self,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                         pretrained,   #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                         strict=strict)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）


@register_pip_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
@register_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
def mamba_vision_T(pretrained=False, **kwargs):  #  定义函数 mamba_vision_T，封装独立功能逻辑，便于复用与测试
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_T.pth.tar")  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    depths = kwargs.pop("depths", [1, 3, 8, 4])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    dim = kwargs.pop("dim", 80)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    in_dim = kwargs.pop("in_dim", 32)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    mlp_ratio = kwargs.pop("mlp_ratio", 4)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    resolution = kwargs.pop("resolution", 224)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_T').to_dict()  #  解析预训练配置，统一模型构建的默认参数
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)  #  合并外部传入参数覆盖默认配置，增强可定制性
    model = MambaVision(depths=depths,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        num_heads=num_heads,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        window_size=window_size,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        dim=dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        in_dim=in_dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        mlp_ratio=mlp_ratio,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        resolution=resolution,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        drop_path_rate=drop_path_rate,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        **kwargs)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.pretrained_cfg = pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.default_cfg = model.pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    if pretrained:  #  分支控制，根据条件切换不同执行路径以提升灵活性
        if not Path(model_path).is_file():  #  分支控制，根据条件切换不同执行路径以提升灵活性
            url = model.default_cfg['url']  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            torch.hub.download_url_to_file(url=url, dst=model_path)  #  如本地无权重文件，则从远端下载预训练权重以复现效果
        model._load_state_dict(model_path)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    return model  #  返回计算结果到调用方，保持接口清晰


@register_pip_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
@register_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
def mamba_vision_T2(pretrained=False, **kwargs):  #  定义函数 mamba_vision_T2，封装独立功能逻辑，便于复用与测试
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_T2.pth.tar")  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    depths = kwargs.pop("depths", [1, 3, 11, 4])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    dim = kwargs.pop("dim", 80)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    in_dim = kwargs.pop("in_dim", 32)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    mlp_ratio = kwargs.pop("mlp_ratio", 4)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    resolution = kwargs.pop("resolution", 224)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_T2').to_dict()  #  解析预训练配置，统一模型构建的默认参数
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)  #  合并外部传入参数覆盖默认配置，增强可定制性
    model = MambaVision(depths=depths,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        num_heads=num_heads,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        window_size=window_size,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        dim=dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        in_dim=in_dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        mlp_ratio=mlp_ratio,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        resolution=resolution,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        drop_path_rate=drop_path_rate,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        **kwargs)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.pretrained_cfg = pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.default_cfg = model.pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    if pretrained:  #  分支控制，根据条件切换不同执行路径以提升灵活性
        if not Path(model_path).is_file():  #  分支控制，根据条件切换不同执行路径以提升灵活性
            url = model.default_cfg['url']  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            torch.hub.download_url_to_file(url=url, dst=model_path)  #  如本地无权重文件，则从远端下载预训练权重以复现效果
        model._load_state_dict(model_path)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    return model  #  返回计算结果到调用方，保持接口清晰


@register_pip_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
@register_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
def mamba_vision_S(pretrained=False, **kwargs):  #  定义函数 mamba_vision_S，封装独立功能逻辑，便于复用与测试
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_S.pth.tar")  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    depths = kwargs.pop("depths", [3, 3, 7, 5])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    dim = kwargs.pop("dim", 96)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    in_dim = kwargs.pop("in_dim", 64)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    mlp_ratio = kwargs.pop("mlp_ratio", 4)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    resolution = kwargs.pop("resolution", 224)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_S').to_dict()  #  解析预训练配置，统一模型构建的默认参数
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)  #  合并外部传入参数覆盖默认配置，增强可定制性
    model = MambaVision(depths=depths,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        num_heads=num_heads,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        window_size=window_size,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        dim=dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        in_dim=in_dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        mlp_ratio=mlp_ratio,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        resolution=resolution,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        drop_path_rate=drop_path_rate,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        **kwargs)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.pretrained_cfg = pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.default_cfg = model.pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    if pretrained:  #  分支控制，根据条件切换不同执行路径以提升灵活性
        if not Path(model_path).is_file():  #  分支控制，根据条件切换不同执行路径以提升灵活性
            url = model.default_cfg['url']  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            torch.hub.download_url_to_file(url=url, dst=model_path)  #  如本地无权重文件，则从远端下载预训练权重以复现效果
        model._load_state_dict(model_path)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    return model  #  返回计算结果到调用方，保持接口清晰


@register_pip_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
@register_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
def mamba_vision_B(pretrained=False, **kwargs):  #  定义函数 mamba_vision_B，封装独立功能逻辑，便于复用与测试
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_B.pth.tar")  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    depths = kwargs.pop("depths", [3, 3, 10, 5])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    dim = kwargs.pop("dim", 128)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    in_dim = kwargs.pop("in_dim", 64)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    mlp_ratio = kwargs.pop("mlp_ratio", 4)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    resolution = kwargs.pop("resolution", 224)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    layer_scale = kwargs.pop("layer_scale", 1e-5)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_B').to_dict()  #  解析预训练配置，统一模型构建的默认参数
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)  #  合并外部传入参数覆盖默认配置，增强可定制性
    model = MambaVision(depths=depths,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        num_heads=num_heads,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        window_size=window_size,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        dim=dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        in_dim=in_dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        mlp_ratio=mlp_ratio,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        resolution=resolution,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        drop_path_rate=drop_path_rate,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        layer_scale=layer_scale,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        layer_scale_conv=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        **kwargs)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.pretrained_cfg = pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.default_cfg = model.pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    if pretrained:  #  分支控制，根据条件切换不同执行路径以提升灵活性
        if not Path(model_path).is_file():  #  分支控制，根据条件切换不同执行路径以提升灵活性
            url = model.default_cfg['url']  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            torch.hub.download_url_to_file(url=url, dst=model_path)  #  如本地无权重文件，则从远端下载预训练权重以复现效果
        model._load_state_dict(model_path)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    return model  #  返回计算结果到调用方，保持接口清晰


@register_pip_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
@register_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
def mamba_vision_B_21k(pretrained=False, **kwargs):  #  定义函数 mamba_vision_B_21k，封装独立功能逻辑，便于复用与测试
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_B_21k.pth.tar")  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    depths = kwargs.pop("depths", [3, 3, 10, 5])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    dim = kwargs.pop("dim", 128)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    in_dim = kwargs.pop("in_dim", 64)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    mlp_ratio = kwargs.pop("mlp_ratio", 4)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    resolution = kwargs.pop("resolution", 224)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    layer_scale = kwargs.pop("layer_scale", 1e-5)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_B_21k').to_dict()  #  解析预训练配置，统一模型构建的默认参数
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)  #  合并外部传入参数覆盖默认配置，增强可定制性
    model = MambaVision(depths=depths,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        num_heads=num_heads,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        window_size=window_size,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        dim=dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        in_dim=in_dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        mlp_ratio=mlp_ratio,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        resolution=resolution,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        drop_path_rate=drop_path_rate,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        layer_scale=layer_scale,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        layer_scale_conv=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        **kwargs)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.pretrained_cfg = pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.default_cfg = model.pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    if pretrained:  #  分支控制，根据条件切换不同执行路径以提升灵活性
        if not Path(model_path).is_file():  #  分支控制，根据条件切换不同执行路径以提升灵活性
            url = model.default_cfg['url']  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            torch.hub.download_url_to_file(url=url, dst=model_path)  #  如本地无权重文件，则从远端下载预训练权重以复现效果
        model._load_state_dict(model_path)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    return model  #  返回计算结果到调用方，保持接口清晰


@register_pip_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
@register_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
def mamba_vision_L(pretrained=False, **kwargs):  #  定义函数 mamba_vision_L，封装独立功能逻辑，便于复用与测试
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L.pth.tar")  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    depths = kwargs.pop("depths", [3, 3, 10, 5])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    dim = kwargs.pop("dim", 196)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    in_dim = kwargs.pop("in_dim", 64)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    mlp_ratio = kwargs.pop("mlp_ratio", 4)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    resolution = kwargs.pop("resolution", 224)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    layer_scale = kwargs.pop("layer_scale", 1e-5)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L').to_dict()  #  解析预训练配置，统一模型构建的默认参数
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)  #  合并外部传入参数覆盖默认配置，增强可定制性
    model = MambaVision(depths=depths,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        num_heads=num_heads,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        window_size=window_size,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        dim=dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        in_dim=in_dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        mlp_ratio=mlp_ratio,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        resolution=resolution,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        drop_path_rate=drop_path_rate,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        layer_scale=layer_scale,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        layer_scale_conv=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        **kwargs)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.pretrained_cfg = pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.default_cfg = model.pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    if pretrained:  #  分支控制，根据条件切换不同执行路径以提升灵活性
        if not Path(model_path).is_file():  #  分支控制，根据条件切换不同执行路径以提升灵活性
            url = model.default_cfg['url']  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            torch.hub.download_url_to_file(url=url, dst=model_path)  #  如本地无权重文件，则从远端下载预训练权重以复现效果
        model._load_state_dict(model_path)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    return model  #  返回计算结果到调用方，保持接口清晰


@register_pip_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
@register_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
def mamba_vision_L_21k(pretrained=False, **kwargs):  #  定义函数 mamba_vision_L_21k，封装独立功能逻辑，便于复用与测试
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L_21k.pth.tar")  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    depths = kwargs.pop("depths", [3, 3, 10, 5])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    dim = kwargs.pop("dim", 196)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    in_dim = kwargs.pop("in_dim", 64)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    mlp_ratio = kwargs.pop("mlp_ratio", 4)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    resolution = kwargs.pop("resolution", 224)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    layer_scale = kwargs.pop("layer_scale", 1e-5)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L_21k').to_dict()  #  解析预训练配置，统一模型构建的默认参数
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)  #  合并外部传入参数覆盖默认配置，增强可定制性
    model = MambaVision(depths=depths,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        num_heads=num_heads,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        window_size=window_size,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        dim=dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        in_dim=in_dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        mlp_ratio=mlp_ratio,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        resolution=resolution,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        drop_path_rate=drop_path_rate,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        layer_scale=layer_scale,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        layer_scale_conv=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        **kwargs)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.pretrained_cfg = pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.default_cfg = model.pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    if pretrained:  #  分支控制，根据条件切换不同执行路径以提升灵活性
        if not Path(model_path).is_file():  #  分支控制，根据条件切换不同执行路径以提升灵活性
            url = model.default_cfg['url']  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            torch.hub.download_url_to_file(url=url, dst=model_path)  #  如本地无权重文件，则从远端下载预训练权重以复现效果
        model._load_state_dict(model_path)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    return model  #  返回计算结果到调用方，保持接口清晰


@register_pip_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
@register_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
def mamba_vision_L2(pretrained=False, **kwargs):  #  定义函数 mamba_vision_L2，封装独立功能逻辑，便于复用与测试
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L2.pth.tar")  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    depths = kwargs.pop("depths", [3, 3, 12, 5])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    dim = kwargs.pop("dim", 196)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    in_dim = kwargs.pop("in_dim", 64)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    mlp_ratio = kwargs.pop("mlp_ratio", 4)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    resolution = kwargs.pop("resolution", 224)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    layer_scale = kwargs.pop("layer_scale", 1e-5)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L2').to_dict()  #  解析预训练配置，统一模型构建的默认参数
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)  #  合并外部传入参数覆盖默认配置，增强可定制性
    model = MambaVision(depths=depths,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        num_heads=num_heads,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        window_size=window_size,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        dim=dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        in_dim=in_dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        mlp_ratio=mlp_ratio,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        resolution=resolution,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        drop_path_rate=drop_path_rate,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        layer_scale=layer_scale,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        layer_scale_conv=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        **kwargs)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.pretrained_cfg = pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.default_cfg = model.pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    if pretrained:  #  分支控制，根据条件切换不同执行路径以提升灵活性
        if not Path(model_path).is_file():  #  分支控制，根据条件切换不同执行路径以提升灵活性
            url = model.default_cfg['url']  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            torch.hub.download_url_to_file(url=url, dst=model_path)  #  如本地无权重文件，则从远端下载预训练权重以复现效果
        model._load_state_dict(model_path)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    return model  #  返回计算结果到调用方，保持接口清晰


@register_pip_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
@register_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
def mamba_vision_L2_512_21k(pretrained=False, **kwargs):  #  定义函数 mamba_vision_L2_512_21k，封装独立功能逻辑，便于复用与测试
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L2_512_21k.pth.tar")  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    depths = kwargs.pop("depths", [3, 3, 12, 5])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    window_size = kwargs.pop("window_size", [8, 8, 32, 16])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    dim = kwargs.pop("dim", 196)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    in_dim = kwargs.pop("in_dim", 64)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    mlp_ratio = kwargs.pop("mlp_ratio", 4)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    resolution = kwargs.pop("resolution", 512)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    layer_scale = kwargs.pop("layer_scale", 1e-5)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L2_512_21k').to_dict()  #  解析预训练配置，统一模型构建的默认参数
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)  #  合并外部传入参数覆盖默认配置，增强可定制性
    model = MambaVision(depths=depths,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        num_heads=num_heads,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        window_size=window_size,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        dim=dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        in_dim=in_dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        mlp_ratio=mlp_ratio,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        resolution=resolution,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        drop_path_rate=drop_path_rate,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        layer_scale=layer_scale,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        layer_scale_conv=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        **kwargs)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.pretrained_cfg = pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.default_cfg = model.pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    if pretrained:  #  分支控制，根据条件切换不同执行路径以提升灵活性
        if not Path(model_path).is_file():  #  分支控制，根据条件切换不同执行路径以提升灵活性
            url = model.default_cfg['url']  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            torch.hub.download_url_to_file(url=url, dst=model_path)  #  如本地无权重文件，则从远端下载预训练权重以复现效果
        model._load_state_dict(model_path)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    return model  #  返回计算结果到调用方，保持接口清晰


@register_pip_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
@register_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
def mamba_vision_L3_256_21k(pretrained=False, **kwargs):  #  定义函数 mamba_vision_L3_256_21k，封装独立功能逻辑，便于复用与测试
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L3_256_21k.pth.tar")  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    depths = kwargs.pop("depths", [3, 3, 20, 10])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    window_size = kwargs.pop("window_size", [8, 8, 16, 8])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    dim = kwargs.pop("dim", 256)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    in_dim = kwargs.pop("in_dim", 64)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    mlp_ratio = kwargs.pop("mlp_ratio", 4)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    resolution = kwargs.pop("resolution", 256)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    drop_path_rate = kwargs.pop("drop_path_rate", 0.5)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    layer_scale = kwargs.pop("layer_scale", 1e-5)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L3_256_21k').to_dict()  #  解析预训练配置，统一模型构建的默认参数
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)  #  合并外部传入参数覆盖默认配置，增强可定制性
    model = MambaVision(depths=depths,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        num_heads=num_heads,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        window_size=window_size,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        dim=dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        in_dim=in_dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        mlp_ratio=mlp_ratio,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        resolution=resolution,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        drop_path_rate=drop_path_rate,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        layer_scale=layer_scale,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        layer_scale_conv=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        **kwargs)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.pretrained_cfg = pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.default_cfg = model.pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    if pretrained:  #  分支控制，根据条件切换不同执行路径以提升灵活性
        if not Path(model_path).is_file():  #  分支控制，根据条件切换不同执行路径以提升灵活性
            url = model.default_cfg['url']  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            torch.hub.download_url_to_file(url=url, dst=model_path)  #  如本地无权重文件，则从远端下载预训练权重以复现效果
        model._load_state_dict(model_path)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    return model  #  返回计算结果到调用方，保持接口清晰


@register_pip_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
@register_model  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
def mamba_vision_L3_512_21k(pretrained=False, **kwargs):  #  定义函数 mamba_vision_L3_512_21k，封装独立功能逻辑，便于复用与测试
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L3_512_21k.pth.tar")  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    depths = kwargs.pop("depths", [3, 3, 20, 10])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    window_size = kwargs.pop("window_size", [8, 8, 32, 16])  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    dim = kwargs.pop("dim", 256)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    in_dim = kwargs.pop("in_dim", 64)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    mlp_ratio = kwargs.pop("mlp_ratio", 4)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    resolution = kwargs.pop("resolution", 512)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    drop_path_rate = kwargs.pop("drop_path_rate", 0.5)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    layer_scale = kwargs.pop("layer_scale", 1e-5)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L3_512_21k').to_dict()  #  解析预训练配置，统一模型构建的默认参数
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)  #  合并外部传入参数覆盖默认配置，增强可定制性
    model = MambaVision(depths=depths,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        num_heads=num_heads,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        window_size=window_size,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        dim=dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        in_dim=in_dim,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        mlp_ratio=mlp_ratio,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        resolution=resolution,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        drop_path_rate=drop_path_rate,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        layer_scale=layer_scale,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        layer_scale_conv=None,  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
                        **kwargs)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.pretrained_cfg = pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    model.default_cfg = model.pretrained_cfg  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    if pretrained:  #  分支控制，根据条件切换不同执行路径以提升灵活性
        if not Path(model_path).is_file():  #  分支控制，根据条件切换不同执行路径以提升灵活性
            url = model.default_cfg['url']  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
            torch.hub.download_url_to_file(url=url, dst=model_path)  #  如本地无权重文件，则从远端下载预训练权重以复现效果
        model._load_state_dict(model_path)  #  实现细节：参数赋值/张量流转/层级组装等（详见本行语义）
    return model  #  返回计算结果到调用方，保持接口清晰
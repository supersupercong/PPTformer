import torch
import torch.nn.functional as F
import math

from basicsr.utils.registry import ARCH_REGISTRY

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class LayerNorm_GRN(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

# class Gated_Convolution(nn.Module):
#     """ ConvNeXtV2 Block.
#
#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#     """
#
#     def __init__(self, in_c, bias=False):
#         super().__init__()
#         dim = in_c
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
#         self.norm = LayerNorm_GRN(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(dim, dim)  # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.grn = GRN(dim)
#         self.pwconv2 = nn.Linear(dim, dim)
#         # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.grn(x)
#         x = self.pwconv2(x)
#         x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#
#         # x = input + self.drop_path(x)
#         x = input + x
#         return x

# class Gated_Convolution(nn.Module):
#     def __init__(self, in_c=24, bias=False):
#         super(Gated_Convolution, self).__init__()
#
#         self.proj = nn.Conv2d(in_c, 2 * in_c, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.LayerNorm = LayerNorm(in_c)
#
#     def forward(self, x):
#         x_ori = x
#         x = self.LayerNorm(x)
#         x1, x2 = self.proj(x).chunk(2, dim=1)
#         y = x1 * x2
#
#         return y + x_ori




class ConvNeXtV2(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, in_c, FFN_facttor, bias=False):
        super().__init__()
        dim = in_c * FFN_facttor
        self.dwconv = nn.Conv2d(in_c, dim, kernel_size=3, padding=1, groups=in_c)  # depthwise conv
        self.norm = LayerNorm_GRN(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(dim)
        self.pwconv2 = nn.Linear(dim, in_c)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        # x = input + self.drop_path(x)
        x = input + x
        return x

class Residual_Block(nn.Module):
    def __init__(self, dim, bias=False):
        super(Residual_Block, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, dim, 3, 1, 1,bias=bias),
        )

    def forward(self, x):
        out = x + self.res(x)
        return out



class Gated_Convolution(nn.Module):
    def __init__(self, in_c=24, bias=False):
        super(Gated_Convolution, self).__init__()

        self.proj = nn.Sequential(
            Residual_Block(dim=in_c, bias=bias),
            # Residual_Block(dim=in_c, bias=bias)
        )

    def forward(self, x):
        y = self.proj(x)
        return y + x


############################ our FFN ############################
class Wo_Mask_Aware_FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor,  bias=True):
        super(Wo_Mask_Aware_FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.hidden_features = hidden_features

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)

        self.dwconv_pwconv1 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=1, bias=bias)
        )

        self.dwconv_pwconv2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                      groups=hidden_features, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        )

    def forward(self, x, x_mask):
        x1 = self.dwconv1(self.project_in(x))  # .chunk(2, dim=1)

        dwconv_pwconv1 = self.dwconv_pwconv1(x1)

        dwconv_pwconv2 = self.dwconv_pwconv2(dwconv_pwconv1)

        return dwconv_pwconv2


class Dual_Direction_Fusion(nn.Module):
    def __init__(self, dim, bias=True):
        super(Dual_Direction_Fusion, self).__init__()
        self.concat = nn.Conv2d(2 * dim, 2 * dim, kernel_size=1, bias=bias)
        self.fusion_two_branches = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias)
        self.mask_transfer1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x, mask_feature):
        mask_transfer1 = self.mask_transfer1(mask_feature)
        concat1, concat2 = self.concat(torch.cat([x, mask_transfer1], dim=1)).chunk(2, dim=1)
        out1 = concat1 * x + x

        out2 = concat2 * mask_transfer1 + mask_transfer1

        fusion_two_branches = self.fusion_two_branches(torch.cat([out1, out2], dim=1))

        return fusion_two_branches+x


class Attention_wo_mask(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(Attention_wo_mask, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, x_mask):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        v = torch.nn.functional.normalize(v, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, 3*dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(3*dim, 3*dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        ######################################
        self.kv = nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.kv_dwconv = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias)
        )

        self.new_k = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias)
        self.new_v = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, x_mask):
        b, c, h, w = x.shape
        q, k, v = self.q_dwconv(self.q(x)).chunk(3, dim=1)

        ######################
        k_mask, v_mask = self.kv_dwconv(self.kv(x_mask)).chunk(2, dim=1)

        k = self.new_k(torch.cat([k, k_mask], dim=1))
        v = self.new_v(torch.cat([v, v_mask], dim=1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        v = torch.nn.functional.normalize(v, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# class Cross_TransformerBlock(nn.Module):
#     def __init__(self, dim=32,
#                  num_heads=1,
#                  ffn_expansion_factor=2,
#                  bias=True,
#                  LayerNorm_type='WithBias',
#                  ffn_mask=True,
#                  fusion_in_self_attention=True,
#                  attention_mask=True,
#                  ):
#         super(Cross_TransformerBlock, self).__init__()
#         self.dim =dim
#         self.norm1 = LayerNorm(dim=dim, LayerNorm_type=LayerNorm_type)
#         if attention_mask is True: #dim, ffn_expansion_factor, fusion_in_self_attention=True, bias=True
#             self.attn = Cross_Attention(dim=dim, num_heads=num_heads, bias=bias)
#         else:
#             self.attn = Attention_wo_mask(dim=dim, num_heads=num_heads, bias=bias)
#         self.norm2 = LayerNorm(dim=dim, LayerNorm_type=LayerNorm_type)
#         if ffn_mask is True:
#             self.ffn = Mask_Aware_FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
#         else:
#             self.ffn = Wo_Mask_Aware_FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
#
#     def forward(self, x, x_mask):
#         x = x + self.attn(self.norm1(x), x_mask)
#         x = x + self.ffn(self.norm2(x), x_mask)
#
#         return x

######## mask aware self-attention #################
class Self_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias=True, fusion_in_self_attention=True):
        super(Self_Attention, self).__init__()
        self.fusion_in_self_attention = fusion_in_self_attention
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        ####### self-attention #########
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.dual_direction_fusion = Dual_Direction_Fusion(dim=dim,bias=bias)


    def forward(self, x, x_mask):
        b, c, h, w = x.shape

        if self.fusion_in_self_attention is True:
            x = self.dual_direction_fusion(x, x_mask)
        else:
            x = x

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        v = torch.nn.functional.normalize(v, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# class Self_TransformerBlock(nn.Module):
#     def __init__(self, dim=32, num_heads=1, ffn_expansion_factor=3.14, bias=True,
#                  LayerNorm_type='WithBias',
#                  ffn_mask=True,
#                  fusion_in_self_attention=True,
#                  attention_mask=True):
#         super(Self_TransformerBlock, self).__init__()
#         self.dim = dim
#         self.norm1 = LayerNorm(dim=dim, LayerNorm_type=LayerNorm_type)
#         if attention_mask is True:
#             self.attn = Self_Attention(dim=dim, num_heads=num_heads, bias=bias, fusion_in_self_attention=fusion_in_self_attention)
#         else:
#             self.attn = Attention_wo_mask(dim=dim, num_heads=num_heads, bias=bias)
#         self.norm2 = LayerNorm(dim=dim, LayerNorm_type=LayerNorm_type)
#         if ffn_mask is True:
#             self.ffn = Mask_Aware_FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
#         else:
#             self.ffn = Wo_Mask_Aware_FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
#
#     def forward(self, x, x_mask):
#         x = x + self.attn(self.norm1(x), x_mask)
#         x = x + self.ffn(self.norm2(x), x_mask)
#
#         return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=8, stride=8, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


##########################################################################
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.dual_direction_fusion = Dual_Direction_Fusion(dim=dim, bias=bias)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, features):
        x = self.dual_direction_fusion(x, features)
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Mask_Aware_FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(Mask_Aware_FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_mask = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dual_direction_fusion = Dual_Direction_Fusion(dim=hidden_features, bias=bias)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x, feature_mask):
        x = self.project_in(x)
        feature_mask = self.project_mask(feature_mask)

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        dual_direction_fusion = self.dual_direction_fusion(x2, feature_mask)

        x = F.gelu(x1) * dual_direction_fusion
        x = self.project_out(x)
        return x

##########################################################################
class General_TransformerBlock(nn.Module):
    def __init__(self, dim=32,
                 num_heads=1,
                 ffn_expansion_factor=3,
                 propagation=True,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(General_TransformerBlock, self).__init__()
        self.dim = dim
        self.propagation = propagation
        self.norm1 = LayerNorm(dim=dim, LayerNorm_type=LayerNorm_type)
        self.attn = Attention(dim=dim, num_heads=num_heads, bias=bias)
        self.norm2 = LayerNorm(dim=dim, LayerNorm_type=LayerNorm_type)
        self.ffn = Mask_Aware_FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        self.norm_feature = LayerNorm(dim=dim, LayerNorm_type=LayerNorm_type)

    def forward(self, x, feature):
        feature = self.norm_feature(feature)
        x = x + self.attn(self.norm1(x), feature)
        x = x + self.ffn(self.norm2(x), feature)

        return x


class IntraBlock(nn.Module):
    def __init__(self, dim=32,
                 num_heads=1,
                 ffn_expansion_factor=2,
                 bias=True,
                 LayerNorm_type='WithBias',
                 in_unit_num=3,
                 ffn_mask=True,
                 fusion_in_self_attention=True,
                 attention_mask=True,
                 all_cross_attention=True,
                 all_self_attention=True,
                 all_cross_self_attention=True
                 ):
        super(IntraBlock, self).__init__()
        self.unit_num = in_unit_num
        self.unit1 = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        for i in range(self.unit_num):
            self.unit1.append(General_TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            ))
            self.conv1.append(nn.Conv2d((i+2)*dim, dim, 1, 1, bias=bias))

        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, input, feature):
        tmp = input
        cat_list = []
        cat_list.append(tmp)
        for i in range(self.unit_num):
            tmp = self.unit1[i](tmp, feature)
            cat_list.append(tmp)
            tmp = self.conv1[i](torch.cat(cat_list, dim=1))

        return self.gamma * tmp + input


class InterBlock(nn.Module):
    def __init__(self,
                 dim=32,
                 num_heads=1,
                 ffn_expansion_factor=3,
                 bias=True,
                 LayerNorm_type='WithBias',
                 out_unit_num=4,
                 in_unit_num=3,
                 ffn_mask=True,
                 fusion_in_self_attention=True,
                 attention_mask=True,
                 all_cross_attention=True,
                 all_self_attention=True,
                 all_cross_self_attention=True):
        super(InterBlock, self).__init__()
        self.unit_num = out_unit_num
        self.channel = dim
        self.Block = nn.ModuleList()
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.Cross_Attention = Cross_Attention(dim=dim, num_heads=num_heads, bias=bias)
        self.Self_Attention = Self_Attention(dim=dim, num_heads=num_heads, bias=bias, fusion_in_self_attention=fusion_in_self_attention,)
        self.Mask_Aware_FFN = Mask_Aware_FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        ###########################################
        self.concat = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias)
        ###########################################
        self.conv1 = nn.ModuleList()
        for i in range(self.unit_num):
            self.Block.append(IntraBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                in_unit_num=in_unit_num,
                ffn_mask=ffn_mask,
                fusion_in_self_attention=fusion_in_self_attention,
                attention_mask=attention_mask,
                all_cross_attention=all_cross_attention,
                all_self_attention=all_self_attention,
                all_cross_self_attention=all_cross_self_attention,
            ))
            self.conv1.append(nn.Conv2d((i + 2) * dim, dim, 1, 1, bias=bias))

        self.norm_cross_attention = LayerNorm(dim=dim, LayerNorm_type=LayerNorm_type)
        self.norm_self_attention = LayerNorm(dim=dim, LayerNorm_type=LayerNorm_type)
        self.norm_ffn = LayerNorm(dim=dim, LayerNorm_type=LayerNorm_type)
        self.norm_mask = LayerNorm(dim=dim, LayerNorm_type=LayerNorm_type)

    def fusion(self, x, x_mask):
        # print('x_mask',x_mask.size())
        x_mask = self.norm_mask(x_mask)
        Cross_Attention = self.Cross_Attention(self.norm_cross_attention(x), x_mask)
        Self_Attention = self.Self_Attention(self.norm_cross_attention(x), x_mask)
        fusion = self.concat(torch.cat([Cross_Attention, Self_Attention], dim=1)) + x
        Mask_Aware_FFN = self.Mask_Aware_FFN(self.norm_ffn(fusion), x_mask) + fusion

        return Mask_Aware_FFN

    def forward(self, x, x_mask):
        fusion = self.fusion(x, x_mask)
        tmp = x
        cat_list=[]
        cat_list.append(tmp)
        for i in range(self.unit_num):
            tmp = self.Block[i](tmp, fusion)
            cat_list.append(tmp)
            tmp = self.conv1[i](torch.cat(cat_list, dim=1))
        return self.gamma * tmp + x



class Net(nn.Module):
    def __init__(self, channel_query_dict,
                 out_list_block,
                 in_list_block,
                 list_heads,
                 ffn_expansion_factor=3,
                 bias=True,
                 LayerNorm_type='WithBias',
                 ffn_mask=True,
                 fusion_in_self_attention=True,
                 attention_mask=True,
                 all_cross_attention=True,
                 all_self_attention=True,
                 all_cross_self_attention=True,
                 ):
        super().__init__()
        self.channel_query_dict = channel_query_dict
        self.enter = nn.Sequential(nn.Conv2d(3, channel_query_dict[128], 3, 1, 1))
        self.en_block1 = InterBlock(dim=channel_query_dict[128],
                                    num_heads=list_heads[0],
                                    ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias,
                                    LayerNorm_type=LayerNorm_type,
                                    out_unit_num=out_list_block[0],
                                    in_unit_num=in_list_block[0],
                                    ffn_mask=ffn_mask,
                                    fusion_in_self_attention=fusion_in_self_attention,
                                    attention_mask=attention_mask,
                                    all_cross_attention=all_cross_attention,
                                    all_self_attention=all_self_attention,
                                    all_cross_self_attention=all_cross_self_attention,
                                    )

        self.c_128to64 = nn.Sequential(nn.Conv2d(channel_query_dict[128], channel_query_dict[128] // 2, kernel_size=3,
                                                 stride=1, padding=1, bias=False),
                                       nn.PixelUnshuffle(2))
        self.en_block2 = InterBlock(dim=channel_query_dict[64],
                                            num_heads=list_heads[1],
                                            ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias,
                                            LayerNorm_type=LayerNorm_type,
                                            out_unit_num=out_list_block[1],
                                            in_unit_num=in_list_block[1],
                                            ffn_mask=ffn_mask,
                                            fusion_in_self_attention=fusion_in_self_attention,
                                            attention_mask=attention_mask,
                                            all_cross_attention=all_cross_attention,
                                            all_self_attention=all_self_attention,
                                            all_cross_self_attention=all_cross_self_attention,
                                            )

        self.c_64to32 = nn.Sequential(nn.Conv2d(channel_query_dict[64], channel_query_dict[64] // 2, kernel_size=3,
                                                stride=1, padding=1, bias=False),
                                      nn.PixelUnshuffle(2))
        self.bottom_block3 = InterBlock(dim=channel_query_dict[32],
                                                num_heads=list_heads[2],
                                                ffn_expansion_factor=ffn_expansion_factor,
                                                bias=bias,
                                                LayerNorm_type=LayerNorm_type,
                                                out_unit_num=out_list_block[2],
                                                in_unit_num=in_list_block[2],
                                                ffn_mask=ffn_mask,
                                                fusion_in_self_attention=fusion_in_self_attention,
                                                attention_mask=attention_mask,
                                                all_cross_attention=all_cross_attention,
                                                all_self_attention=all_self_attention,
                                                all_cross_self_attention=all_cross_self_attention,
                                        )

        self.c_32to64 = nn.Sequential(nn.Conv2d(channel_query_dict[32], channel_query_dict[32] * 2, kernel_size=3,
                                                stride=1, padding=1, bias=False),
                                      nn.PixelShuffle(2))
        self.skip1 = nn.Conv2d(channel_query_dict[64]*2, channel_query_dict[64], kernel_size=1,
                                                stride=1, bias=False)
        self.de_block2 = InterBlock(dim=channel_query_dict[64],
                                            num_heads=list_heads[1],
                                            ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias,
                                            LayerNorm_type=LayerNorm_type,
                                            out_unit_num=out_list_block[1],
                                            in_unit_num=in_list_block[1],
                                            ffn_mask=ffn_mask,
                                            fusion_in_self_attention=fusion_in_self_attention,
                                            attention_mask=attention_mask,
                                            all_cross_attention=all_cross_attention,
                                            all_self_attention=all_self_attention,
                                            all_cross_self_attention=all_cross_self_attention,
                                            )

        self.c_64to128 = nn.Sequential(nn.Conv2d(channel_query_dict[64], channel_query_dict[64] * 2, kernel_size=3,
                                                 stride=1, padding=1, bias=False),
                                       nn.PixelShuffle(2))
        self.skip2 = nn.Conv2d(channel_query_dict[128] * 2, channel_query_dict[128], kernel_size=1,
                               stride=1, bias=False)

        self.de_block3 = InterBlock(dim=channel_query_dict[128],
                                            num_heads=list_heads[0],
                                            ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias,
                                            LayerNorm_type=LayerNorm_type,
                                            out_unit_num=out_list_block[0],
                                            in_unit_num=in_list_block[0],
                                            ffn_mask=ffn_mask,
                                            fusion_in_self_attention=fusion_in_self_attention,
                                            attention_mask=attention_mask,
                                            all_cross_attention=all_cross_attention,
                                            all_self_attention=all_self_attention,
                                            all_cross_self_attention=all_cross_self_attention,
                                            )

        self.exit = nn.Sequential(nn.Conv2d(channel_query_dict[128], 3, 3, 1, 1))

        # self.mask_input_1_1 = nn.Conv2d(3, channel_query_dict[128], 3, 1, 1)
        # self.mask_input_1_2 = nn.Conv2d(3, channel_query_dict[128], 3, 1, 1)
        #
        # self.mask_input_2_1 = nn.Conv2d(3, channel_query_dict[64], 3, 2, 1)
        # self.mask_input_2_2 = nn.Conv2d(3, channel_query_dict[64], 3, 2, 1)
        #
        # self.mask_input_4_1 = nn.Conv2d(3, channel_query_dict[32], 6, 4, 1)
        # self.mask_input_2_2 = nn.Conv2d(3, channel_query_dict[64], 3, 1, 1)

    def forward(self, x, mask_representatin_features):
        ori = x
        features = []
        enter = self.enter(x)
        en_block1 = self.en_block1(enter, mask_representatin_features[0])
        features.append(en_block1)

        c_128to64 = self.c_128to64(en_block1)
        # en_block2 = self.en_block2(c_128to64, F.upsample(x_mask, (c_128to64.size()[2:])))
        en_block2 = self.en_block2(c_128to64, mask_representatin_features[1])
        features.append(en_block2)

        c_64to32 = self.c_64to32(en_block2)
        bottom_block3 = self.bottom_block3(c_64to32, mask_representatin_features[2])
        features.append(bottom_block3)
        c_32to64 = self.c_32to64(bottom_block3)

        skip1 = self.skip1(torch.cat([c_32to64, en_block2], dim=1))
        de_block2 = self.de_block2(skip1, mask_representatin_features[1])
        features.append(de_block2)
        c_64to128 = self.c_64to128(de_block2)

        skip2 = self.skip2(torch.cat([c_64to128, en_block1], dim=1))
        de_block3 = self.de_block3(skip2, mask_representatin_features[0])
        features.append(de_block3)
        exit = self.exit(de_block3)

        return exit + ori


class Mask_Feature_Representation(nn.Module):
    def __init__(self,
                 in_channel=3,
                 max_depth=3,
                 input_res=128,
                 channel_query_dict=None,
                 bias=True,
                 ):
        super().__init__()

        ksz = 3

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channel, channel_query_dict[input_res], 3, padding=1),
            Residual_Block(dim=channel_query_dict[input_res], bias=bias),
            Residual_Block(dim=channel_query_dict[input_res], bias=bias),
        )

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.max_depth = max_depth
        res = input_res
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                Residual_Block(dim=out_ch, bias=bias),
                Residual_Block(dim=out_ch, bias=bias),
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
            res = res // 2

    def forward(self, input):
        outputs = []
        x = self.in_conv(input)
        outputs.append(x)
        for idx, m in enumerate(self.blocks):
            x = m(x)
            outputs.append(x)

        return outputs

@ARCH_REGISTRY.register()
class FeMaSRNet(nn.Module):
    def __init__(self,
                 *,
                 out_list_block=[1, 1, 1],
                 in_list_block=[3, 3, 3],
                 list_heads=[2, 4, 8],
                 ffn_mask=True,
                 fusion_in_self_attention=True,
                 attention_mask=True,
                 all_cross_self_attention=True,
                 all_cross_attention=False,
                 all_self_attention=False,
                 num_refinement=4,
                 num_heads_refinement=1,
                 ffn_expansion_factor=3,
                 bias=True,
                 LayerNorm_type='WithBias',
                 **ignore_kwargs):
        super().__init__()
        channel_query_dict = {32: 128, 64: 64, 128: 32}

        self.Mask_Feature_Representation = Mask_Feature_Representation(
            in_channel=3,
            max_depth=2,
            input_res=128,
            channel_query_dict=channel_query_dict,
            bias=bias)

        self.restoration_network = Net(channel_query_dict=channel_query_dict,
                                       out_list_block=out_list_block,
                                       in_list_block=in_list_block,
                                       list_heads=list_heads,
                                       ffn_expansion_factor=ffn_expansion_factor,
                                       bias=bias,
                                       LayerNorm_type=LayerNorm_type,
                                       ffn_mask=ffn_mask,
                                       fusion_in_self_attention=fusion_in_self_attention,
                                       attention_mask=attention_mask,
                                       all_cross_attention=all_cross_attention,
                                       all_self_attention=all_self_attention,
                                       all_cross_self_attention=all_cross_self_attention,
                                       )


    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def encode_and_decode(self, input, input_mask, current_iter=None):
        Mask_Feature_Representation = self.Mask_Feature_Representation(input_mask)

        # print('Mask_Feature_Representation',Mask_Feature_Representation)

        restoration = self.restoration_network(input, Mask_Feature_Representation)
        return restoration


    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16):
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height * self.scale_factor
        output_width = width * self.scale_factor
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x * self.scale_factor
                output_end_x = input_end_x * self.scale_factor
                output_start_y = input_start_y * self.scale_factor
                output_end_y = input_end_y * self.scale_factor

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale_factor
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale_factor
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale_factor
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale_factor

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]
        return output

    def check_image_size(self, x, window_size=16):
        _, _, h, w = x.size()
        mod_pad_h = (window_size - h % (window_size)) % (
            window_size)
        mod_pad_w = (window_size - w % (window_size)) % (
            window_size)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # print('F.pad(x, (0, mod_pad_w, 0, mod_pad_h)', x.size())
        return x

    @torch.no_grad()
    def test(self, input):
        _, _, h_old, w_old = input.shape

        # input = self.check_image_size(input)
        # if lq_equalize is not None:
        #     lq_equalize = self.check_image_size(lq_equalize)

        restoration = self.encode_and_decode(input)

        output = restoration
        # output = output[:,:, :h_old, :w_old]

        # self.use_semantic_loss = org_use_semantic_loss
        return output

    def forward(self, input, input_mask):
        restoration = self.encode_and_decode(input, input_mask)

        return restoration

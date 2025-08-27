# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from DINO and timm library:
https://github.com/facebookresearch/dino
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import math
import torch
import torch.nn as nn

from functools import partial
from utils import trunc_normal_

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


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


class MaskedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., separate_obj_attn=False):
        super().__init__()
        self.separate_obj_attn = separate_obj_attn
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    # TODO: Make this work with any mask, not just the object mask
    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B, H, N, C')
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, C')

        # print("Shapes for Attention:")
        # print("Q: ", q.shape)
        # print("K: ", k.shape)
        # print("V: ", v.shape)
        # print("Attention Mask: ", attn_mask.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, C') @ (B, H, C', N) = (B, H, N, N)
        if attn_mask is not None:  # (B, O, 1+O+H*W)
            # Apply the object mask to the object token attention vector
            # Question: Should [OBJ] token attend to the [CLS] token or not? Currently, [OBJ] does attend to [CLS]!
            num_object_tokens = attn_mask.shape[1]  # (B, O, 1+O+H*W)
            # Objects can attend to self and assgined patches
            attn[:, :, 1:1+num_object_tokens, :] = attn[:, :, 1:1+num_object_tokens, :].masked_fill(~attn_mask[:, None, :, :], float('-inf'))
            if self.separate_obj_attn:
                # CLS cannot attend to object tokens
                attn[:, :, 0, 1:1+num_object_tokens] = float('-inf')
                # Patches cannot attend to non-assigned objects
                attn[:, :, 1+num_object_tokens:, 1:1+num_object_tokens] = attn[:, :, 1+num_object_tokens:, 1:1+num_object_tokens].masked_fill(~attn_mask.transpose(-2, -1)[:, None, 1+num_object_tokens:, :], float('-inf'))

            # attn[:, :, :, 1:1+num_object_tokens] = attn[:, :, :, 1:1+num_object_tokens].masked_fill(~attn_mask.transpose(-2,-1)[:, None, :, :], float('-inf'))  # v9 -> degraded performance on 4 objects
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, H, N, N) @ (B, H, N, C') = (B, H, N, C') -> (B, N, H, C') -> (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print("Number of patches: ", N)
        # print("Is [CLS] finite:", torch.isfinite(x[:, 0]).all().item())
        # print("Is [OBJ] finite:", torch.isfinite(x[:, 1:5]).all().item())
        # print("Is [PATCH] finite:", torch.isfinite(x[:, 5:]).all().item())
        # print("Is X finite:", torch.isfinite(x).all().item())
        # Find for which image the object token is not finite
        # finite_mask = torch.isfinite(x[:, 1:5]).all(dim=(1, 2))
        # print("For which image, [OBJ] finite: ", finite_mask)
        # for i in range(4):
        #     print("Object attention mask for non-finite [OBJ]: ", attn_mask[~finite_mask, i])
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=0, separate_obj_attn=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MaskedAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, separate_obj_attn=separate_obj_attn)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, return_attention=False, obj_attn_mask=None):
        y, attn = self.attn(self.norm1(x), attn_mask=obj_attn_mask)
        if return_attention:
            return attn
        if self.gamma_1 is None:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            
    def forward(self, x):
        B, C, H, W = x.shape
        return self.proj(x)

class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, num_object_tokens=1, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), return_all_tokens=False, 
                 init_values=0, use_mean_pooling=False, masked_im_modeling=False, single_embed_obj_pos=False, 
                 use_different_obj_tokens=False, separate_obj_attn=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_object_tokens = num_object_tokens
        self.return_all_tokens = return_all_tokens
        self.use_different_obj_tokens = use_different_obj_tokens

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.obj_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_obj_pos_embed = 1 if single_embed_obj_pos else num_object_tokens
        num_obj_token_allocate = num_object_tokens if self.use_different_obj_tokens else 1
        self.obj_token = nn.Parameter(torch.zeros(1, num_obj_token_allocate, embed_dim)) # -> CRIBO
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_obj_pos_embed + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                init_values=init_values, separate_obj_attn=separate_obj_attn)
            for i in range(depth)])

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.obj_token, std=.02)
        self.apply(self._init_weights)

        # masked image modeling
        self.masked_im_modeling = masked_im_modeling
        if masked_im_modeling:
            self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_object_tokens - 1
        N = self.pos_embed.shape[1] - self.num_obj_pos_embed - 1
        class_pos_embed = self.pos_embed[:, 0]
        obj_pos_embed = self.pos_embed[:, 1:1+self.num_obj_pos_embed]
        if self.num_obj_pos_embed == 1 and self.num_object_tokens > 1:
            obj_pos_embed = obj_pos_embed.expand(1, self.num_object_tokens, -1)
        patch_pos_embed = self.pos_embed[:, 1+self.num_obj_pos_embed:]
        if npatch == N and w == h:
            return torch.cat((class_pos_embed.unsqueeze(0), obj_pos_embed, patch_pos_embed), dim=1)
        
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), obj_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x, mask=None):
        B, nc, w, h = x.shape
        # patch linear embedding
        x = self.patch_embed(x)  # (B, embed_dim, W, H)

        # mask image modeling
        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(1, 2)  # (B, embed_dim, W, H) -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)

        # add the [OBJ] token to the embed patch tokens
        if self.use_different_obj_tokens:
            obj_tokens = self.obj_token.expand(B, -1, -1)
        else:
            obj_tokens = self.obj_token.expand(B, self.num_object_tokens, -1)  # (B, num_object_tokens, embed_dim)
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, obj_tokens, x), dim=1)  # (B, 1 + num_object_tokens + num_patches, embed_dim)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x, return_all_tokens=None, mask=None, obj_attn_mask=None):
        # mim
        if self.masked_im_modeling:
            assert mask is not None
            x = self.prepare_tokens(x, mask=mask)
        else:
            x = self.prepare_tokens(x)

        # Masked Attention for [OBJ] token
        for blk in self.blocks:
            x = blk(x, obj_attn_mask=obj_attn_mask)

        x = self.norm(x)
        if self.fc_norm is not None:
            x[:, 0] = self.fc_norm(x[:, 2:, :].mean(1))
        
        return_all_tokens = self.return_all_tokens if \
            return_all_tokens is None else return_all_tokens
        if return_all_tokens:
            return x
        return x[:, 1]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1, obj_attn_mask=None):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, obj_attn_mask=obj_attn_mask)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output
        
    def get_num_layers(self):
        return len(self.blocks)

    def mask_model(self, x, mask):
        x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        return x


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model

def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model

def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model

def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model

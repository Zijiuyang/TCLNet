# Modifications Copyright 2026 Zhejiang University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_

from compressai.layers import ResidualBlock

__all__ = ["WMSA", "Block", "ConvTransBlock", "SwinBlock", "ConvBN", "TCBlock"]


class WMSA(nn.Module):
    """Window multi-head self-attention (W-MSA / SW-MSA)."""

    def __init__(self, input_dim, output_dim, head_dim, window_size, attn_type):
        super().__init__()
        assert attn_type in ("W", "SW")
        assert input_dim % head_dim == 0, "input_dim must be divisible by head_dim"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = attn_type

        self.embedding_layer = nn.Linear(input_dim, 3 * input_dim, bias=True)
        self.linear = nn.Linear(input_dim, output_dim)

        rel_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_params = nn.Parameter(torch.zeros(rel_size, self.n_heads))
        trunc_normal_(self.relative_position_params, std=0.02)

        self.relative_position_params = nn.Parameter(
            self.relative_position_params.view(2 * window_size - 1, 2 * window_size - 1, self.n_heads)
            .transpose(1, 2)
            .transpose(0, 1)
        )

    def generate_mask(self, h, w, p, shift):
        attn_mask = torch.zeros(
            h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device
        )
        if self.type == "W":
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, "w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)")
        return attn_mask

    def forward(self, x):
        if self.type != "W":
            x = torch.roll(
                x,
                shifts=(-(self.window_size // 2), -(self.window_size // 2)),
                dims=(1, 2),
            )

        x = rearrange(
            x,
            "b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c",
            p1=self.window_size,
            p2=self.window_size,
        )
        h_windows = x.size(1)
        w_windows = x.size(2)

        x = rearrange(
            x,
            "b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c",
            p1=self.window_size,
            p2=self.window_size,
        )

        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, "b nw np (threeh c) -> threeh b nw np c", c=self.head_dim).chunk(3, dim=0)

        sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), "h p q -> h 1 1 p q")

        if self.type != "W":
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = torch.softmax(sim, dim=-1)
        output = torch.einsum("hbwij,hbwjc->hbwic", probs, v)
        output = rearrange(output, "h b w p c -> b w p (h c)")
        output = self.linear(output)

        output = rearrange(
            output,
            "b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c",
            w1=h_windows,
            p1=self.window_size,
        )

        if self.type != "W":
            output = torch.roll(
                output,
                shifts=(self.window_size // 2, self.window_size // 2),
                dims=(1, 2),
            )
        return output

    def relative_embedding(self):
        cord = torch.tensor(
            np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]),
            device=self.relative_position_params.device,
        )
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, attn_type="W"):
        super().__init__()
        assert attn_type in ("W", "SW")

        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, attn_type)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, attn_type="W"):
        super().__init__()
        assert attn_type in ("W", "SW")

        self.conv_dim = conv_dim
        self.trans_dim = trans_dim

        self.trans_block = Block(trans_dim, trans_dim, head_dim, window_size, drop_path, attn_type)
        self.conv1_1 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)

        self.conv_block = ResidualBlock(conv_dim, conv_dim)

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x

        trans_x = Rearrange("b c h w -> b h w c")(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange("b h w c -> b c h w")(trans_x)

        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        return x + res


class SwinBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path):
        super().__init__()
        self.block_1 = Block(input_dim, output_dim, head_dim, window_size, drop_path, attn_type="W")
        self.block_2 = Block(input_dim, output_dim, head_dim, window_size, drop_path, attn_type="SW")
        self.window_size = window_size

    def forward(self, x):
        b, c, h, w = x.shape
        pad_l = pad_t = 0
        pad_r = pad_b = 0

        if h <= self.window_size or w <= self.window_size:
            pad_h = max(0, self.window_size + 1 - h)
            pad_w = max(0, self.window_size + 1 - w)
            pad_t = pad_h // 2
            pad_b = pad_h - pad_t
            pad_l = pad_w // 2
            pad_r = pad_w - pad_l
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        trans_x = Rearrange("b c h w -> b h w c")(x)
        trans_x = self.block_1(trans_x)
        trans_x = self.block_2(trans_x)
        trans_x = Rearrange("b h w c -> b c h w")(trans_x)

        if pad_l or pad_r or pad_t or pad_b:
            trans_x = trans_x[:, :, pad_t : pad_t + h, pad_l : pad_l + w]

        return trans_x


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super().__init__(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=padding, groups=groups, bias=False)),
                    ("bn", nn.BatchNorm2d(out_planes)),
                ]
            )
        )


class TCBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.path1 = nn.Sequential(
            OrderedDict(
                [
                    ("conv3x3", ConvBN(2, 11, 7)),
                    ("relu5", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ("conv1x9", ConvBN(11, 11, [1, 9])),
                    ("conv9x1", ConvBN(11, 11, [9, 1])),
                    ("ConvTrans5", ConvTransBlock(conv_dim=5, trans_dim=6, head_dim=1, window_size=4, drop_path=0, attn_type="SW")),
                ]
            )
        )
        self.path2 = nn.Sequential(
            OrderedDict(
                [
                    ("conv3x3_3", ConvBN(2, 11, 5)),
                    ("relu6", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ("conv1x5", ConvBN(11, 11, [1, 11])),
                    ("conv5x1", ConvBN(11, 11, [11, 1])),
                    ("ConvTrans6", ConvTransBlock(conv_dim=5, trans_dim=6, head_dim=1, window_size=4, drop_path=0, attn_type="SW")),
                ]
            )
        )
        self.conv1x1 = ConvBN(22, 2, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = x
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)
        return self.relu(out + identity)
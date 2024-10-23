from functools import partial

import torch
import torch.nn as nn 

from einops import rearrange

from .norm import RMSNorm

class TauEncoder(nn.Module):
    def __init__(self, out_channels):
        super(TauEncoder, self).__init__()
        self.in_out = [(48, 24), (24, 12), (12, 6), (6, 3)]

        self.blocks = nn.ModuleList([])
        for idx, (in_channels, out_channels) in enumerate(self.in_out):
            self.blocks.append(nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                RMSNorm(out_channels),
                nn.GELU(),
                nn.Dropout(0.1)
            ]))
        
    def forward(self, x):
        B, L, D = x.shape
        x = rearrange(x, 'b (num_h num_w) (c_new scale_h scale_w) -> b c_new (num_h scale_h) (num_w scale_w)', 
                     num_h=16, num_w=16, scale_h=4, scale_w=4)
        # x: (B, 256, 768) -> (B, 48, 64, 64)
        for conv1, conv2, res_conv, norm, act, dropout in self.blocks:

            h = conv1(x)
            h = norm(h)
            h = act(h)
            h = dropout(h) 
            h = conv2(h)
            h = norm(h)
            h = act(h)
            h = dropout(h)

            x = h + res_conv(x)

        return x 
        # x: (B, 3, 64, 64)
            

        
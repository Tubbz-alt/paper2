"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import copy
import math
import functools
from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import core.utils as utils

from core.wing import FAN



class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1],
                                    [-1, 8., -1],
                                    [-1, -1, -1]]).to(device) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)  # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, masks=None):
        # print("input x.shape", x.shape)
        x = self.from_rgb(x)
        # print("x.shape", x.shape)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)
            # print("encode x.shape", x.shape)
        for block in self.decode:
            x = block(x, s)
            # print("decode x.shape", x.shape)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])
        # print("self.to_rgb(x)", self.to_rgb(x).shape)
        return self.to_rgb(x)

# class Generator_pix2pix_unet(nn.Module):
#     def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
#         super().__init__()
#         dim_in = 64   
#         dim_out = dim_in  
#         down_c = [2,3,3,4,4,4,4]
#         up_c = [[4,4], [8,4], [8,4], [8,4], [8,3], [4,2], [3,1], [2,1]]
#         dropoff = [0,0,0,1,1,1,0]
#         self.down_layers = nn.ModuleList()
#         self.up_layers = nn.ModuleList()
#         self.norm = nn.InstanceNorm2d(dim_in, affine=True)
#         self.lrelu = nn.LeakyReLU(0.2)
#         self.relu = nn.ReLU(0.2)
#         self.conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0)
#         self.deconv = nn.ConvTranspose2d(dim_in, dim_out, 1, 1, 0)
#         self.down = nn.Sequential(
#                 nn.LeakyReLU(0.2),
#                 nn.Conv2d(dim_in, dim_out, 1, 1, 0),
#                 nn.InstanceNorm2d(dim_in, affine=True))
#         self.up = nn.Sequential(
#             nn.ReLU(0.2),
#             nn.ConvTranspose2d(dim_in, dim_out, 1, 1, 0),
#             nn.InstanceNorm2d(dim_in, affine=True))
#         self.skip = nn.ModuleList()
            
#         downconv = nn.Conv2d(3, dim_in, 4, 2, 0)) # 64*128*128
#         self.skip.append(downconv)
#         self.down_layers.append(downconv)
#         for i in down_c:
#             dim_out = style_dim*i
#             self.down_layers.append(self.down)
#             self.skip.append(self.conv)
#             dim_in = dim_out

#         index = 6
#         for i,j in up_c:
#             if(index < 0):
#                 dim_in = style_dim*i 
#                 dim_out = 3
#                 x = self.up
#             else : 
#                 dim_in = style_dim*i 
#                 dim_out = style_dim*j
#                 x = self.up
#                 x = torch.cat((x,skip[index]), 1)

#             self.up_layers.append(x)
#             index = index - 1

#        def forward(self, x, s, masks=None):
#            x  
#            return x

class Generator_pix2pix_unet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_downs=7, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Generator_pix2pix_unet, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class Generator_unet(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        repeat_num = int(np.log2(img_size)) - 3
        dim_in = 64
        dim_out = 512
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)  # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.down_layers = nn.ModuleList()
        self.down_norm_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.deconv = nn.ConvTranspose2d(64, 3, 3, 1, 1)
        skip_layer = nn.ModuleList()
        self.norm = nn.InstanceNorm2d(dim_in, affine=True)
        self.actv = nn.LeakyReLU(0.2)

        # encoder 
        for i in range(repeat_num):
            in_c = dim_in
            if(in_c*2 > max_conv_dim): 
                out_c = max_conv_dim 
                dim_in = max_conv_dim
            else: 
                out_c = in_c*2
                dim_in = in_c*2
            downnorm = nn.InstanceNorm2d(in_c, affine=True)
            self.down_norm_layers.append(downnorm)
            downconv = nn.Conv2d(in_c, out_c, 4, 2, 1)
            self.down_layers.append(downconv)
            skip_layer.append(downconv)

        # decoder 
        for i in range(repeat_num+1):
            in_c = dim_out
            if(dim_out // 2 < 8): 
                out_c = 3 
                dim_out = 3
            elif (i > 2): 
                out_c = dim_out //2
                dim_out = dim_out //2
            if(i==0):
                up = nn.ConvTranspose2d(in_c+style_dim, out_c, 3, 1, 1)
            else:
                up = nn.ConvTranspose2d((in_c*2)+style_dim, out_c, 4, 2, 1)

            self.up_layers.append(up)
            
    def forward(self, x, s, masks=None):
        i = 0 
        skip = []
        x = self.from_rgb(x)
        skip.append(x)
        for down in self.down_layers:
            x = self.down_norm_layers[i](x) 
            x = self.actv(x)
            x = down(x)
            skip.append(x)
            i = i+1

        i = 0 
        for up in self.up_layers:
            x = utils.tile_concat(x, s)
            x = self.actv(x)
            x = up(x)
            if(len(skip)-i-1>0):
                index = len(skip)-i-1
                x = torch.cat((x,skip[index]), 1)
            i = i+1

        x = self.deconv(x)
        return x

class Generator_wnet(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        repeat_num = int(np.log2(img_size)) - 3
        dim_in = 64
        dim_out = 512
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)  # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.up2_layers = nn.ModuleList()
        self.conv = nn.ConvTranspose2d(dim_in, dim_in, 3, 1, 1)
        self.deconv = nn.ConvTranspose2d(64, 3, 3, 1, 1)
        #encoder 
        for i in range(repeat_num):
            in_c = dim_in
            if(in_c*2 > max_conv_dim): 
                out_c = 512 
                dim_in = 512
            else: 
                out_c = in_c*2
                dim_in = in_c*2
            down = nn.Conv2d(in_c, out_c, 4, 2, 1)
            self.down_layers.append(down)

        #decoder 
        for i in range(repeat_num+1):
            in_c = dim_out
            print
            if(dim_out // 2 < 8): 
                out_c = 3 
                dim_out = 3
            elif (i > 2): 
                out_c = dim_out //2
                dim_out = dim_out //2

            if(i==0):
                up2 = nn.ConvTranspose2d(in_c+style_dim, out_c, 3, 1, 1)
                up = nn.ConvTranspose2d(in_c, out_c, 3, 1, 1)
            else:
                up = nn.ConvTranspose2d(in_c*2, out_c, 4, 2, 1)
                up2 = nn.ConvTranspose2d(in_c*2, out_c, 4, 2, 1)
                
            self.up_layers.append(up)
            self.up2_layers.append(up2)

    def forward(self, x, s, masks=None):
        i = 0 
        skip = []
        x = self.from_rgb(x)
        skip.append(x)
        # U1 
        for down in self.down_layers:
            x = down(x)
            skip.append(x)

        for up in self.up_layers:
            x = up(x)
            if(len(skip)-i-1>0):
                index = len(skip)-i-1
                x = torch.cat((x,skip[index]), 1)
            i = i+1
        
        # U2 
        skip2 = []
        i = 0 
        x = self.conv(x)
        skip2.append(x) 
        for down in self.down_layers:
            x = down(x)
            skip2.append(x) 

        x = utils.tile_concat(x, s)

        for up in self.up2_layers:
            x = up(x)
            if(len(skip2)-i-1>0):
                index = len(skip2)-i-1
                x = torch.cat((x,skip2[index]), 1)
            i = i+1

        x = self.deconv(x)
        return x

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out

class Discriminator_pix2pix(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(Discriminator_pix2pix, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
        
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
def build_model(args):
    # generator = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf)
    generator_pix2pix = Generator_pix2pix_unet(3, 3, 7, 64, norm_layer=nn.BatchNorm2d, use_dropout=False)
    discriminator_pix2pix = Discriminator_pix2pix(3, 64, 3, norm_layer=nn.BatchNorm2d)
    generator = Generator_unet(args.img_size, args.style_dim, w_hpf=args.w_hpf)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
    style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains)
    discriminator = Discriminator(args.img_size, args.num_domains)
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator,
                 generator_pix2pix=generator_pix2pix,
                 discriminator_pix2pix=discriminator_pix2pix)
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    if args.w_hpf > 0:
        fan = FAN(fname_pretrained=args.wing_path).eval()
        nets.fan = fan
        nets_ema.fan = fan

    return nets, nets_ema
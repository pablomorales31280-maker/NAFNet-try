# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d, FLC_Pooling, FLC_Pooling_learn_alpha, FLC_Pooling_learn_alpha_blurred_alpha_dropout, FLC_Pooling_random_alpha_blurred, TransposedUpsample
from basicsr.models.archs.local_arch import Local_Base

import numpy as np


## Resizing modules
class FLC_Downsample(nn.Module):
    def __init__(self, n_feat, use_conv, use_alpha, learn_alpha, use_blur, drop_alpha, test_wo_drop_alpha, test_drop_alpha, transpose = False, stop=False, half=False, padding="reflect"):
        super(FLC_Downsample, self).__init__()
        self.use_conv = use_conv
        self.use_alpha = use_alpha
        self.learn_alpha = learn_alpha
        self.use_blur = use_blur
        self.drop_alpha = drop_alpha
        self.test_wo_drop_alpha = test_wo_drop_alpha
        self.test_drop_alpha = test_drop_alpha
        self.stop = stop
        self.transpose = transpose
        self.channel_multiplier = 1 #if self.transpose else 1
        self.half_precision = half
        self.padding = padding

        if self.use_alpha:
            if self.learn_alpha:
                if self.use_blur:
                    if self.drop_alpha:
                        self.body = nn.Sequential(nn.Conv2d(n_feat*self.channel_multiplier, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False), FLC_Pooling_learn_alpha_blurred_alpha_dropout(channels=n_feat, transpose=self.transpose, test_wo_drop_alpha = self.test_wo_drop_alpha, test_drop_alpha=self.test_drop_alpha, stop = stop, half_precision = self.half_precision, padding=self.padding),)                        
                    else:                        
                        self.body = nn.Sequential(nn.Conv2d(n_feat*self.channel_multiplier, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False), FLC_Pooling_random_alpha_blurred(channels=n_feat, transpose=self.transpose, test_drop_alpha=self.test_drop_alpha, half_precision = self.half_precision, padding=self.padding), )
                else:
                    #self.body = nn.Sequential(FLC_Pooling_learn_alpha(transpose=self.transpose), nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),)
                    self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False), FLC_Pooling_learn_alpha(transpose=self.transpose), )
        else:
            self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False), FLC_Pooling(transpose=self.transpose), )

    def forward(self, x):
        return self.body(x)

class FreqAvgUpsample(nn.Module):
    def __init__(self, n_feat, padding='zero', transpose=False):
        super(FreqAvgUpsample, self).__init__()
        self.padding = 'constant' if padding =='zero' else 'mirror'
        self.body = nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv1 = nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, groups=n_feat//2, bias=False)
        self.beta = nn.Parameter(torch.tensor(0.3), requires_grad = True)
        self.shuffle = nn.PixelShuffle(2)
        self.transpose = transpose        

    def forward(self, x):
        dtype = x.dtype
        x = self.body(x)        
        channels = x.shape[1]
        
        if self.transpose:
            x = x.transpose(2,3)
        freq = torch.fft.fft2(x.to(torch.float32), norm='forward')

        avg_list, avg_channel_list = [], []
        for i in range(0, freq.shape[1], 4):
            avg = torch.mean(freq[:,i:i+4,:,:], dim=1)
            avg = torch.unsqueeze(avg, dim=1)
            avg_channels = torch.cat([avg]*4, dim=1)
            avg_list.append(avg)
            avg_channel_list.append(avg_channels)        
        #tmp = torch.cat(avg_list, dim=1)
        avg_list = torch.cat(avg_list, dim=1)
        avg_channel_list = torch.cat(avg_channel_list, dim=1)
                
        freq = freq - avg_channel_list
        freq = torch.fft.ifft2(freq, norm='forward').to(dtype)
        highFreq = self.shuffle(freq)

        padding = F.pad(avg_list, (x.shape[-1], 0, x.shape[-2], 0), mode=self.padding)
        freqUp = torch.fft.ifft2(padding, norm='forward').to(dtype)        

        if self.transpose:
            freqUp = freqUp.transpose(2, 3)
            highFreq = highFreq.transpose(2, 3)
        
        return freqUp*(1-self.beta) + self.beta*highFreq


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFReLUBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNetFLCReLU(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], **kwargs):
        super().__init__()

        self.use_alpha = kwargs['use_alpha']
        self.learn_alpha = kwargs['learn_alpha']
        self.use_blur = kwargs['use_blur']
        self.kernel_size = kwargs['kernel_size']
        self.para_kernel_size = kwargs['para_kernel_size']
        self.padding = kwargs['padding']
        
        self.use_conv = kwargs['use_conv']
        self.first_drop_alpha = kwargs['first_drop_alpha']
        self.drop_alpha = kwargs['drop_alpha']
        self.test_wo_drop_alpha = kwargs['test_wo_drop_alpha']
        self.test_drop_alpha = kwargs['test_drop_alpha']
        self.half_precision = kwargs['half_precision']
        self.upsampling_method = kwargs['upsampling_method']
        
        
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        transpose = True

        chan = width
        count = 0
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFReLUBlock(chan) for _ in range(num)]
                )
            )
            
            self.downs.append(FLC_Downsample(chan, self.use_conv, self.use_alpha, self.learn_alpha, self.use_blur, 
                                            drop_alpha= True if self.first_drop_alpha and count==0 else self.drop_alpha, 
                                            test_wo_drop_alpha = self.test_wo_drop_alpha, test_drop_alpha=self.test_drop_alpha, 
                                            transpose=transpose, half=self.half_precision, padding = self.padding))
            transpose = not transpose
            count +=1
            """
            if self.use_alpha:
                if self.learn_alpha:
                    if self.use_blur:
                        self.downs.append(
                        nn.Sequential(FLC_Pooling_learn_alpha_blurred(channels=chan), nn.Conv2d(chan, 2*chan, 3, 1, padding=1))#.cuda()
                        ) 
                    else:      
                        self.downs.append(
                            nn.Sequential(FLC_Pooling_learn_alpha(), nn.Conv2d(chan, 2*chan, 3, 1, padding=1))#.cuda()
                        )        
                else:
                    if self.use_blur:
                        self.downs.append(
                        nn.Sequential(FLC_Pooling_random_alpha_blurred(channels=chan), nn.Conv2d(chan, 2*chan, 3, 1, padding=1))#.cuda()
                        )    
                    else:
                        self.downs.append(
                            nn.Sequential(FLC_Pooling_random_alpha(), nn.Conv2d(chan, 2*chan, 3, 1, padding=1))#.cuda()
                        )        
            else:
                self.downs.append(
                    nn.Sequential(FLC_Pooling(), nn.Conv2d(chan, 2*chan, 3, 1, padding=1))#.cuda()
                )
            """
            #self.downs.append(                
            #    nn.Conv2d(chan, 2*chan, 2, 1)
            #)
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFReLUBlock(chan) for _ in range(middle_blk_num)]
            )

        transpose = True
        for num in dec_blk_nums:
            if self.upsampling_method =='TransposedConv' and self.kernel_size>1:
                self.ups.append(
                nn.Sequential(
                    TransposedUpsample(chan, self.kernel_size, self.para_kernel_size)
                    )
                )
            elif self.upsampling_method == 'FreqAvgUp':
                self.ups.append(FreqAvgUpsample(n_feat=chan, transpose=transpose))
                transpose = not transpose
            else:    
                self.ups.append(
                    nn.Sequential(
                        nn.Conv2d(chan, chan * 2, 1, bias=False),
                        nn.PixelShuffle(2)
                    )
                )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFReLUBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            try:
                x = x + enc_skip
            except Exception:
                x = x + enc_skip.transpose(2,3)
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNetLocalFLCReLU(Local_Base, NAFNetFLCReLU):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNetFLCReLU.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    net = NAFNetFLCReLU(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)

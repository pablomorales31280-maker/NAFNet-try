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
from basicsr.models.archs.arch_util import LayerNorm2d, FLC_Pooling, FLC_Pooling_learn_alpha, FLC_Pooling_random_alpha_blurred, TransposedUpsample
from basicsr.models.archs.local_arch import Local_Base

class BaselineReLUBlock(nn.Module):
    def __init__(self, c, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Channel Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )

        # RELU
        self.relu = nn.ReLU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        #import ipdb;ipdb.set_trace()
        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x * self.se(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.relu(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class BaselineFLCReLU(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], dw_expand=1, ffn_expand=2, **kwargs):
        super().__init__()

        self.use_alpha = kwargs['use_alpha']
        self.learn_alpha = kwargs['learn_alpha']
        self.use_blur = kwargs['use_blur']
        self.kernel_size = kwargs['kernel_size']
        self.para_kernel_size = kwargs['para_kernel_size']
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[BaselineReLUBlock(chan, dw_expand, ffn_expand) for _ in range(num)]
                )
            )
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
            #self.downs.append(                
            #    nn.Conv2d(chan, 2*chan, 2, 1)
            #)
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[BaselineReLUBlock(chan, dw_expand, ffn_expand) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            if self.kernel_size>1:
                self.ups.append(
                nn.Sequential(
                    TransposedUpsample(chan, self.kernel_size, self.para_kernel_size)
                    )
                )
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
                    *[BaselineReLUBlock(chan, dw_expand, ffn_expand) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        #import ipdb;ipdb.set_trace()
        #inp = inp.cuda()
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        #import ipdb;ipdb.set_trace()

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            #print(x.shape)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            #import ipdb;ipdb.set_trace()
            try:
                x = x + enc_skip
            except:
                #import ipdb;ipdb.set_trace()
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

class BaselineLocalFLCReLU(Local_Base, BaselineFLCReLU):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        BaselineFLCReLU.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

if __name__ == '__main__':
    img_channel = 3
    width = 32

    dw_expand = 1
    ffn_expand = 2

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    net = BaselineFLCReLU(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                 enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, dw_expand=dw_expand, ffn_expand=ffn_expand)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T


class PixelUnshuffleDown(nn.Module):
    def __init__(self, in_channels = None):
        super().__init__()
        self.body = nn.PixelUnshuffle(2)

    def forward(self, x):
        return self.body(x)
    


class ConvStride2Down(nn.Module):
    def __init__(self, in_channels, out_channels = None, bias = False):
        super().__init__()
        if  out_channels is None :
            out_channels = in_channels * 2
        self.body = nn.Conv2d(in_channels, out_channels, kernel_size = 2, stride = 2, padding = 0, bias = bias)

    def forward(self, x):
        return self.body(x)
    

class FrequencyPreservedPooling(nn.Module):
    def __init__(self, channels = None, test_wo_drop_alpha = False, transpose = True, test_drop_alpha = False, stop = False, half_precision = False, padding = "reflect"):
        super().__init__()
        self.transpose = transpose
        self.test_wo_drop_alpha = test_wo_drop_alpha
        self.test_drop_alpha = test_drop_alpha
        self.stop = stop
        self.half_precision = half_precision
        self.padding = padding
        self.alpha = nn.Parameter(torch.tensor(0.3), requires_grad = True)
        self.downsample_high = nn.PixelUnshuffle(2)

    def forward(self, x):
        orig_x_size = x.shape
        orig_x = x
        x = F.pad(x, (3*x.shape[-1]//4+1, 3*x.shape[-1]//4, 3*x.shape[-2]//4+1, 3*x.shape[-2]//4), mode = self.padding)
        if self.transpose :
            x = x.transpose(2, 3)
        in_freq = torch.fft.fftshift(torch.fft.fft2(x.to(torch.float32), norm = "forward"))
        low_part = in_freq[:, :, int(x.shape[2] / 4): int(x.shape[2] / 4 * 3), int(x.shape[3] / 4): int(x.shape[3] / 4 * 3)]
        low_part = torch.fft.ifft2(torch.fft.ifftshift(low_part), norm = "forward").real
        if self.half_precision :
            low_part = low_part.half()
        if self.transpose :
            low_part = low_part.transpose(2, 3)
        low_part = torch.cat((low_part, low_part, low_part, low_part), dim = 1)
        if self.test_drop_alpha :
            return T.CenterCrop(orig_x_size[-2]//2, orig_x_size[-1]//2)(low_part)
        zeroed_high = torch.zeros_like(in_freq)
        zeroed_high[:, :, int(x.shape[2] / 4): int(x.shape[2] / 4 * 3), int(x.shape[3] / 4): int(x.shape[3] / 4 * 3)] = in_freq[:, :, int(x.shape[2] / 4): int(x.shape[2] / 4 * 3), int(x.shape[3] / 4): int(x.shape[3] / 4 * 3)]
        zeroed_high = torch.fft.ifft2(torch.fft.ifftshift(zeroed_high), norm = "forward").real
        if self.half_precision :
            zeroed_high = zeroed_high.half()
        if self.transpose :
            zeroed_high = zeroed_high.transpose(2, 3)
        zeroed_high = T.CenterCrop((orig_x_size[-2], orig_x_size[-1]))(zeroed_high)
        high_part = orig_x - zeroed_high
        high_part = self.downsample_high(high_part)
        low_part = T.CenterCrop((orig_x_size[-2]//2, orig_x_size[-1]//2))(low_part)
        return low_part * (1-self.alpha) + high_part * self.alpha

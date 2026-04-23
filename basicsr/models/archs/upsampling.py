import torch
from torch import nn
from torch.nn import functional as F


class PixelShuffleUp(nn.Module):
    def __init__(self, in_channels, out_channels = None, bias = False):
        super().__init__()
        if out_channels is None :
            out_channels = in_channels // 2
        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels*4, kernel_size = 1, stride = 1, padding = 0, bias = bias), nn.PixelShuffle(2))
    
    def forward(self, x):
        return self.body(x)
    

class LCTCUp(nn.Module):
    def __init__(self, in_channels, out_channels = None, large_kernel = 7, small_kernel = None, bias = False):
        super().__init__()
        if out_channels is None :
           out_channels = in_channels // 2
        self.large_path = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = large_kernel, stride = 2, padding = large_kernel//2, output_padding = 1, bias = bias)
        self.small_path = None
        if small_kernel is not None :
            self.small_path = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = small_kernel, stride = 2, padding = small_kernel//2, output_padding = 1, bias = bias)
        
    def forward(self, x):
        out = self.large_path(x)
        if self.small_path is not None :
            out = out + self.small_path(x)
        return out
    

class FreqAvgUp(nn.Module):
    def __init__(self, in_channels, out_channels = None, padding = "constant", transpose = False, bias = False):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels // 2
        self.padding = "constant"
        self.transpose = transpose
        self.beta = nn.Parameter(torch.tensor(0.3, dtype = torch.float32))
        self.body = nn.Conv2d(in_channels, out_channels * 4, kernel_size = 3, stride = 1, padding = 1, bias = bias)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        dtype = x.dtype
        x = self.body(x)
        if self.transpose:
            x = x.transpose(2, 3)
        freq = torch.fft.fft2(x.to(torch.float32), norm = "forward")
        b, c, h, w = freq.shape
        freq_g = freq.view(b, c // 4, 4, h, w)
        avg = freq_g.mean(dim = 2)
        avg_channels = avg.unsqueeze(2).expand(-1, -1, 4, -1, -1).reshape(b, c, h, w)
        high_freq = freq - avg_channels
        high_spatial = torch.fft.ifft2(high_freq, norm = "forward").real.to(dtype)
        high_spatial = self.shuffle(high_spatial)
        pad_w_left = w // 2
        pad_w_right = w - pad_w_left
        pad_h_top = h // 2
        pad_h_bottom = h - pad_h_top
        low_freq = torch.fft.fftshift(avg, dim = (-2, -1))
        low_freq = F.pad(low_freq, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom), mode = self.padding, value = 0.0)
        low_freq = torch.fft.ifftshift(low_freq, dim = (-2, -1))
        low_spatial = torch.fft.ifft2(low_freq, norm = "forward").real.to(dtype)
        if self.transpose:
            low_spatial = low_spatial.transpose(2, 3)
            high_spatial = high_spatial.transpose(2, 3)
        beta = self.beta.to(device = x.device, dtype = low_spatial.dtype)
        return low_spatial * (1 - beta) + high_spatial * beta
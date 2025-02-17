import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
import torchvision.ops as ops

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class GEGLU(nn.Module):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    """

    def geglu(self, x: Tensor) -> Tensor:
        # assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=1)
        return a * F.gelu(b)

    def forward(self, x: Tensor) -> Tensor:
        return self.geglu(x)


class MotionEnhanceEstimator(nn.Module):
    def __init__(self, embed_dim=32, n_first_channels = 15):
        super().__init__()

        self.normflow = nn.LayerNorm(embed_dim)
        self.normevent = nn.LayerNorm(embed_dim)

        self.fc_flow = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim))

        self.fc_offset = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 2 * 3 * 3, kernel_size=3, stride=1, padding=1))

        self.fc_mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 3 * 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())

        self.weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim, 3, 3))

        self.spatial_filter = nn.Sequential(
            nn.Conv2d(2*embed_dim, 2*embed_dim, kernel_size=1),
            nn.Conv2d(2*embed_dim, 2*embed_dim, kernel_size=3, stride=1, padding=1, groups=2*embed_dim),
            nn.Conv2d(2*embed_dim, 2*embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*embed_dim, embed_dim, kernel_size=1),
            nn.Sigmoid())

        self.final_linear = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim),
            GEGLU(),
            nn.Conv2d(int(embed_dim/2), embed_dim, kernel_size=1))


    def forward(self, input_flow, input_event):
        # Guide the coarse-grained optical flow with fine-grained event details
        N, C, H, W = input_flow.shape
        flow_norm = self.normflow(input_flow.permute(0, 2, 3, 1).contiguous().reshape(-1, C)).reshape(N, H, W, -1).permute(0, 3, 1, 2).contiguous() #(N, C, H, W)
        event_norm = self.normevent(input_event.permute(0, 2, 3, 1).contiguous().reshape(-1, C)).reshape(N, H, W, -1).permute(0, 3, 1, 2).contiguous() #(N, C, H, W)

        x = self.fc_flow(flow_norm)

        offset = self.fc_offset(event_norm) #(N, C, H, W)
        mask = self.fc_mask(event_norm) #(N, C, H, W)
        dc = ops.deform_conv2d(x, offset, self.weight, padding=1, mask=mask)

        x_fft = torch.fft.rfft2(x) #(N, C, H, W/2)
        dc_fft = torch.fft.rfft2(dc) #(N, C, H, W/2)
        dc_real = dc_fft.real
        dc_imag = dc_fft.imag

        fft_cate = torch.cat((dc_real, dc_imag), dim=1) #(N, 2C, H, W/2)
        sp_mask = self.spatial_filter(fft_cate) #(N, C, H, W/2)
        x_spatial_fft = torch.mul(x_fft, sp_mask) #(N, C, H, W/2)
        x_irfft = torch.fft.irfft2(x_spatial_fft, s=(H, W)) #(N, C, H, W/2)

        x_output = self.final_linear(x_irfft)
        x_output = x_output + input_flow

        return x_output
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.ops as ops

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


class TemporalFusion_module(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()

        self.normx = nn.LayerNorm(embed_dim)
        self.normforward = nn.LayerNorm(embed_dim)
        self.normbackward = nn.LayerNorm(embed_dim)

        self.fc = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim))

        self.fc_fwd_offset = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 2 * 3 * 3, kernel_size=3, stride=1, padding=1))

        self.fc_fwd_mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 3 * 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())

        self.fwd_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim, 3, 3))

        self.fc_back_offset = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 2 * 3 * 3, kernel_size=3, stride=1, padding=1))

        self.fc_back_mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 3 * 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())

        self.back_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim, 3, 3))

        self.mask_forward = nn.Sequential(
            nn.Conv2d(embed_dim, 2*embed_dim, kernel_size=1),
            nn.Conv2d(2*embed_dim, 2*embed_dim, kernel_size=3, stride=1, padding=1, groups=2*embed_dim),
            nn.Conv2d(2*embed_dim, 2*embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*embed_dim, embed_dim, kernel_size=1),
            nn.Sigmoid())

        self.mask_backward = nn.Sequential(
            nn.Conv2d(embed_dim, 2*embed_dim, kernel_size=1),
            nn.Conv2d(2*embed_dim, 2*embed_dim, kernel_size=3, stride=1, padding=1, groups=2*embed_dim),
            nn.Conv2d(2*embed_dim, 2*embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*embed_dim, embed_dim, kernel_size=1),
            nn.Sigmoid())

        self.final_linear = nn.Sequential(
            nn.Conv2d(3*embed_dim, 3*embed_dim, kernel_size=3, stride=1, padding=1, groups=3*embed_dim),
            GEGLU(),
            nn.Conv2d(int(3*embed_dim/2), embed_dim, kernel_size=1))


    def forward(self, input_x, flow_forward, flow_backward):
        N, C, H, W = input_x.shape
        x_norm = self.normx(input_x.permute(0, 2, 3, 1).contiguous().reshape(-1, C)).reshape(N, H, W, -1).permute(0, 3, 1, 2).contiguous() #(N, C, H, W)
        flowf_norm = self.normforward(flow_forward.permute(0, 2, 3, 1).contiguous().reshape(-1, C)).reshape(N, H, W, -1).permute(0, 3, 1, 2).contiguous() #(N, C, H, W)
        flowb_norm = self.normbackward(flow_backward.permute(0, 2, 3, 1).contiguous().reshape(-1, C)).reshape(N, H, W, -1).permute(0, 3, 1, 2).contiguous() #(N, C, H, W)

        x_fc = self.fc(x_norm)
        fwd_offset = self.fc_fwd_offset(flowf_norm)
        fwd_mask = self.fc_fwd_mask(flowf_norm)
        back_offset = self.fc_back_offset(flowb_norm)
        back_mask = self.fc_back_mask(flowb_norm)

        fwd_dc = ops.deform_conv2d(x_fc, fwd_offset, self.fwd_weight, padding=1, mask=fwd_mask)
        back_dc = ops.deform_conv2d(x_fc, back_offset, self.back_weight, padding=1, mask=back_mask)

        flowf = self.mask_forward(fwd_dc)
        flowb = self.mask_backward(back_dc) #(N, C, H, W)

        x_forward = torch.mul(x_fc, flowf)
        x_backward = torch.mul(x_fc, flowb)

        x_bi = torch.cat((x_fc, x_forward, x_backward), dim=1) #(N, 2C, H, W/2)

        x_mlp = self.final_linear(x_bi)
        output_x = x_mlp + input_x

        return output_x
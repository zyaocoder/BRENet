import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import numbers
from einops import rearrange

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

## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Mutual_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x, y):

        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        b,c,h,w = x.shape

        q = self.q(x) # image
        k = self.k(y) # event
        v = self.v(y) # event
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

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
        
##########################################################################
## Event-Image Channel Attention (EICA)
class EventImage_ChannelAttentionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(EventImage_ChannelAttentionTransformerBlock, self).__init__()

        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_Attention(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, image, event):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = image.shape
        fused = image + self.attn(self.norm1_image(image), self.norm1_event(event)) # b, c, h, w

        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)

        return fused


# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction, bias):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=16, bias=False):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, stride=1, padding=1, bias=bias))
        modules_body.append(nn.ReLU(inplace=True))
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, stride=1, padding=1, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class FlowAggregate_module(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()

        # self.normimg = nn.BatchNorm2d(embed_dim)
        # self.normflow = nn.BatchNorm2d(embed_dim)
        self.normimg = nn.LayerNorm(embed_dim)
        self.normflow = nn.LayerNorm(embed_dim)

        self.fc_fuse = nn.Conv2d(2*embed_dim, embed_dim, kernel_size=1)
        self.fc_depth_fuse = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim)

        self.fc_img = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.fc_depth_img = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim)

        self.fc_cross = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        self.spatial_filter = nn.Sequential(
            nn.Conv2d(2*embed_dim, 2*embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*embed_dim, embed_dim, kernel_size=1),
            nn.Sigmoid())

        self.spatial_domain = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1))

        self.CAB = CAB(embed_dim)
        self.cross_attention = EventImage_ChannelAttentionTransformerBlock(embed_dim)
        self.linear = nn.Conv2d(embed_dim, int(embed_dim/2)+1, kernel_size=1)

        self.final_linear = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim),
            GEGLU(),
            nn.Conv2d(int(embed_dim/2), embed_dim, kernel_size=1))


    def forward(self, input_x, input_flow):
        N, C, H, W = input_x.shape
        x_norm = self.normimg(input_x.permute(0, 2, 3, 1).contiguous().reshape(-1, C)).reshape(N, H, W, -1).permute(0, 3, 1, 2).contiguous() #(N, C, H, W)
        flow_norm = self.normflow(input_flow.permute(0, 2, 3, 1).contiguous().reshape(-1, C)).reshape(N, H, W, -1).permute(0, 3, 1, 2).contiguous() #(N, C, H, W)
        
        fuse = torch.cat((x_norm, flow_norm), dim=1)
        x = self.fc_img(x_norm)
        x = self.fc_depth_img(x) #(N, C, H, W)
        fuse_fc = self.fc_fuse(fuse) #(N, C, H, W)
        fuse_depth = self.fc_depth_fuse(fuse_fc) #(N, C, H, W)

        # Spatial Attention
        x_f = torch.fft.rfft2(x) #(N, C, H, W/2)
        flow_f = torch.fft.rfft2(fuse_depth) #(N, C, H, W/2)
        f_real = flow_f.real
        f_imag = flow_f.imag

        flow_cate = torch.cat((f_real, f_imag), dim=1) #(N, 2C, H, W/2)
        sp_filter = self.spatial_filter(flow_cate) #(N, C, H, W/2)
        fb = torch.mul(x_f, sp_filter) #(N, C, H, W/2)
        fb_irfft = torch.fft.irfft2(fb, s=(H, W)) #(N, C, H, W/2)
        fb_irfft = self.spatial_domain(fb_irfft) #(N, C, H, W/2)

        # Channel Attention
        img_channel = fb_irfft + fuse_fc
        x_1df = torch.fft.rfft(fb_irfft, dim=1)
        ca_filter = self.CAB(img_channel)
        ca_filter = self.linear(ca_filter)
        x_channel = torch.mul(x_1df, ca_filter)
        x_channel = torch.fft.irfft(x_channel, dim=1)

        flow_cross = self.fc_cross(flow_norm)
        output_ca = self.cross_attention(x_channel, flow_cross)
        linear_x = self.final_linear(output_ca)
        output_x = linear_x + input_x

        return output_x
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .eraft.eraft import ERAFT
from .eraft.utils import visualize_optical_flow

from mmseg.models.utils import FlowAggregate_module, TemporalFusion_module, MotionEnhanceEstimator, TemporalConv_module

from IPython import embed

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x


@HEADS.register_module()
class BRENetHead(BaseDecodeHead):
    """
    BRENet: Rethinking RGB-Event Semantic Segmentation with a Novel Bidirectional Motion-enhanced Event Representation
    """
    def __init__(self, feature_strides, scale_resolution, **kwargs):
        super(BRENetHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.decoder_params = kwargs['decoder_params']
        embedding_dim = self.decoder_params['embed_dim']
        self.reduce_res = self.decoder_params['reduce_res']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.FAM = FlowAggregate_module(embedding_dim)

        self.linear_flow_forward = MLP(input_dim=2, embed_dim=embedding_dim)
        self.linear_flow_backward = MLP(input_dim=2, embed_dim=embedding_dim)

        self.TFM = TemporalFusion_module(embedding_dim)
        self.MEE = MotionEnhanceEstimator(embedding_dim, self.decoder_params['num_vovel_bin'])
        self.temporal_model = TemporalConv_module(in_channels = embedding_dim, receptive_field = 2, input_shape = (40, 40))

        self.linear_eventold = MLP(input_dim=self.decoder_params['num_vovel_bin'], embed_dim=embedding_dim)
        self.linear_eventnew = MLP(input_dim=self.decoder_params['num_vovel_bin'], embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.flownet = ERAFT(n_first_channels = self.decoder_params['num_vovel_bin'])
        self.init_flownet()

    def init_flownet(self):
        checkpoint = torch.load(self.decoder_params['checkpoint'])
        current_model_dict = self.flownet.state_dict()

        new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), checkpoint['model'].values())}

        self.flownet.load_state_dict(new_state_dict, strict = False)
        # freeze the FlowNet weights
        if self.decoder_params['freeze'] == True:
            for _, param in self.flownet.named_parameters():
                param.requires_grad = False

    def forward(self, inputs, event_new, event_old):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        #Estimate optical flow
        if self.reduce_res == True:
            im1 = torch.nn.functional.interpolate(event_old, size=c1.size()[2:],mode='bilinear',align_corners=False)
            im2 = torch.nn.functional.interpolate(event_new, size=c1.size()[2:],mode='bilinear',align_corners=False)
        else:
            im1 = torch.nn.functional.interpolate(event_old, size=[c1.size()[2]*2, c1.size()[3]*2],mode='bilinear',align_corners=False)
            im2 = torch.nn.functional.interpolate(event_new, size=[c1.size()[2]*2, c1.size()[3]*2],mode='bilinear',align_corners=False)
        
        _, flow_list_forward = self.flownet(image1=im1, image2=im2)
        flow_feature_forward = flow_list_forward[-1]

        _, flow_list_backward = self.flownet(image1=im2, image2=im1)
        flow_feature_backward = flow_list_backward[-1]

        if self.reduce_res == False:
            flow_feature_forward = torch.nn.functional.interpolate(flow_feature_forward, size=c1.size()[2:],mode='bilinear',align_corners=False)
            flow_feature_backward = torch.nn.functional.interpolate(flow_feature_backward, size=c1.size()[2:],mode='bilinear',align_corners=False)

        # visualize_optical_flow(flow_est[0].cpu().numpy())
        flow_feature_forward = self.linear_flow_forward(flow_feature_forward).permute(0,2,1).contiguous().reshape(n, -1, c1.shape[2], c1.shape[3])
        flow_feature_backward = self.linear_flow_backward(flow_feature_backward).permute(0,2,1).contiguous().reshape(n, -1, c1.shape[2], c1.shape[3])

        if self.reduce_res == True:
            event_fea1 = self.linear_eventold(im1).permute(0,2,1).contiguous().reshape(n, -1, im1.shape[2], im1.shape[3]).unsqueeze(2)
            event_fea2 = self.linear_eventnew(im2).permute(0,2,1).contiguous().reshape(n, -1, im2.shape[2], im2.shape[3]).unsqueeze(2)
        else:
            event_fea1 = self.linear_eventold(im1).permute(0,2,1).contiguous().reshape(n, -1, im1.shape[2], im1.shape[3])
            event_fea1 = resize(event_fea1, size=c1.size()[2:],mode='bilinear',align_corners=False).unsqueeze(2)
            event_fea2 = self.linear_eventnew(im2).permute(0,2,1).contiguous().reshape(n, -1, im2.shape[2], im2.shape[3])
            event_fea2 = resize(event_fea2, size=c1.size()[2:],mode='bilinear',align_corners=False).unsqueeze(2)

        temporal_seq = torch.cat([event_fea1, event_fea2], dim=2)

        temporal_event = self.temporal_model(temporal_seq).squeeze(2)

        MoFlow_forward = self.MEE(flow_feature_forward, temporal_event)
        MoFlow_backward = self.MEE(flow_feature_backward, temporal_event)

        _c4 = self.linear_c4(c4).permute(0,2,1).contiguous().reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).contiguous().reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).contiguous().reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).contiguous().reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1)) # (B, C, H/4, W/4)

        flow_forward = self.FAM(_c, MoFlow_forward)
        flow_backward = self.FAM(_c, MoFlow_backward)

        _c = self.TFM(_c, flow_forward, flow_backward)

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
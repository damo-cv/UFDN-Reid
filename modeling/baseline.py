import torch
from torch import nn

from .backbone.resnet import ResNet, BasicBlock, Bottleneck
from .backbone.resnet_sync import ResNet as ResNet_sync
from .backbone.resnet_sync import Bottleneck as Bottleneck_sync
from .backbone.vit import vit_base_patch16_224_TransReID
# from .backbone.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
# from .backbone.resnet_ibn_a import resnet50_ibn_a

def freeze(layer,freeze=True):
    for child in layer.children():
        for param in child.parameters():
            if freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def weights_init_reduction(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.constant_(m.weight, 1.0 / m.weight.size(0))
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class Baseline(nn.Module):

    def __init__(self, cfg, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0
        if model_name == 'resnet18':
            self.in_planes = 512
            self.reduction = 1
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.reduction = 1
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.in_planes = 2048
            self.reduction = 1
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50_sync':
            self.in_planes = 2048
            self.reduction = 1
            self.base = ResNet_sync(last_stride=last_stride,
                               block=Bottleneck_sync,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.in_planes = 2048
            self.reduction = 1
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.in_planes = 2048
            self.reduction = 1
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
        elif model_name == 'vit':
            self.in_planes = 768
            self.reduction = 1            
            self.base=vit_base_patch16_224_TransReID(img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        self.model_name=model_name
  

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.warm_freese=cfg.SOLVER.WARMUP_FREEZE_EPOCH

        self.reduction_layer = nn.Sequential(
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(),
            nn.Conv2d(self.in_planes, int(self.in_planes/self.reduction), kernel_size=1, padding=0, bias=True),
            # nn.Dropout(p=0.05)
        )
        self.reduction_layer.apply(weights_init_kaiming)

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.bottleneck.apply(weights_init_kaiming)

            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x,epoch=120):
        if epoch<self.warm_freese:
            # print(epoch,self.warm_freese)
            freeze(self.base)
        elif epoch==self.warm_freese:
            # print(epoch,self.warm_freese)
            freeze(self.base,freeze=False)

        base_feat=self.base(x)
        if self.model_name=='vit':
            global_feat=base_feat[:,0]
            global_feat = global_feat.view(global_feat.shape[0], -1)
        else:
            reduce_base_feat = self.reduction_layer(base_feat)

            global_feat = self.gap(reduce_base_feat)  # (b, 2048, 1, 1)
            global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    # def load_param(self, trained_path):
    #     param_dict = torch.load(trained_path)
    #     for i in param_dict:
    #         if 'classifier' in i:
    #             continue
    #         self.state_dict()[i].copy_(param_dict[i])

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['state_dict']
        # print(param_dict.keys())
        # param_dict1 = torch.load(trained_path)
        # print(param_dict1.keys())
        # exists_models
        for key in param_dict:
            key_cut = key[7:]
            if 'classifier' in key_cut:
                continue
            self.state_dict()[key_cut].copy_(param_dict[key])


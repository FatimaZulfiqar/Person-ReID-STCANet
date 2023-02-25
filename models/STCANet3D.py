from __future__ import absolute_import

import torch
import math
import copy
import torchvision
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.nn import functional as F

from .STCA3D import STCABlock3D
from .resnets1 import resnet50_s1
from models import inflate
#from .resnet import resnet50_s1

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        # init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class Bottleneck3d(nn.Module):

    def __init__(self, bottleneck2d, inflate_time=False):
        super(Bottleneck3d, self).__init__()

        if inflate_time == True:
            self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=3, time_padding=1, center=True)
        else:
            self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)
        self.conv2 = inflate.inflate_conv(bottleneck2d.conv2, time_dim=1)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)
        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = self._inflate_downsample(bottleneck2d.downsample)
        else:
            self.downsample = None

    def _inflate_downsample(self, downsample2d, time_stride=1):
        downsample3d = nn.Sequential(
            inflate.inflate_conv(downsample2d[0], time_dim=1, 
                                 time_stride=time_stride),
            inflate.inflate_batch_norm(downsample2d[1]))
        return downsample3d

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class STCANet3D(nn.Module):
    def __init__(self, num_classes, use_gpu, loss={'xent', 'htri'}):
        super(STCANet3D, self).__init__()
        
        self.loss = loss
        self.use_gpu = use_gpu
        resnet2d = resnet50_s1(pretrained=True)

        self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)

        self.layer1 = self._inflate_reslayer(resnet2d.layer1)
        self.layer2 = self._inflate_reslayer(resnet2d.layer2)
        
        self.layer3 = self._inflate_reslayer(resnet2d.layer3)
        self.layer4 = self._inflate_reslayer(resnet2d.layer4)

        
        self.STCABlock3D2 = STCABlock3D(512)
        self.feat_dim = 2048

        # fc using random initialization
        add_block = nn.BatchNorm1d(self.feat_dim)
        add_block.apply(weights_init_kaiming)
        self.bn = add_block
        
        # classifier using Random initialization
        classifier = nn.Linear(self.feat_dim, num_classes)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier
      
       
        
    def _inflate_reslayer(self, reslayer2d, enhance_idx=[], channels=512):
        reslayers3d = []
        for i, layer2d in enumerate(reslayer2d):
            layer3d = Bottleneck3d(layer2d)
            reslayers3d.append(layer3d)
            
        return nn.Sequential(*reslayers3d)
        
    def pool(self, x):
        kernel_size = x.size()[2:]
       # print("Kernel size in pool",kernel_size)
        x = F.max_pool3d(x, kernel_size = kernel_size) 
       # f = F.avg_pool3d(x4,x4.size()[2:])
        x = x.view(x.size(0), -1) #[b, c]
        return x

    def pooling(self, x):
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*t, c, h, w)
        #kernel_size = x.size()[2:]
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        return x

    def forward(self, x):
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)

      x = self.layer1(x)
      x = self.layer2(x) 


      x2,a = self.STCABlock3D2(x)
      x = self.layer3(x2)
      x = self.layer4(x)
      b, c, t, h, w = x.size()

      

      x = self.pool(x)
    
      if not self.training:
            return x

      
      f = self.bn(x)
      y = self.classifier(f)  

      if self.loss == {'xent'}:
        return y
      elif self.loss == {'xent', 'htri'}:
        return y, f
      else:
        raise KeyError("Unsupported loss: {}".format(self.loss))
       
       
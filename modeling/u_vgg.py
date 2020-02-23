import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from .utils import *


class U_VGG(nn.Module):
    def __init__(self, load_model='', downsample=1, bn=False, NL='swish', objective='dmp', sp=False, se=True, pyramid=''):
        super(U_VGG, self).__init__()
        self.downsample = downsample
        self.bn = bn
        self.NL = NL
        self.objective = objective
        self.pyramid = pyramid
        self.front0 = make_layers([64, 64], in_channels=3, batch_norm=bn, NL=self.NL)
        self.front1 = make_layers(['M', 128, 128], in_channels=64, batch_norm=bn, NL=self.NL)
        self.front2 = make_layers(['M', 256, 256, 256], in_channels=128, batch_norm=bn, NL=self.NL)
        self.front3 = make_layers(['M', 512, 512, 512], in_channels=256, batch_norm=bn, NL=self.NL)
        self.sp = sp
        if sp:
            print('use sp module')
            self.sp_module = SPModule(512)
        # basic cfg for backend is [512, 512, 256, 128, 64, 64]
        if not self.pyramid:
            self.backconv0 = make_layers([512, 512, 256], in_channels=512, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.backconv1 = make_layers([128], in_channels=512, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.backconv2 = make_layers([64], in_channels=256, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.backconv3 = make_layers([64], in_channels=128, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            
        elif self.pyramid == 'dilation':
            print('use dilation pyramid in backend')
            self.backconv0 = nn.Sequential(DilationPyramid(512, 128), DilationPyramid(512, 128), DilationPyramid(512, 128))
            self.backconv1 = nn.Sequential(DilationPyramid(512, 64))
            self.backconv2 = nn.Sequential(DilationPyramid(256, 32))
            self.backconv3 = nn.Sequential(DilationPyramid(128, 16))
            
        elif self.pyramid == 'size':
            print('use size pyramid in backend')
            self.backconv0 = nn.Sequential(SizePyramid(512, 128), SizePyramid(512, 128), SizePyramid(512, 128))
            self.backconv1 = nn.Sequential(SizePyramid(512, 64))
            self.backconv2 = nn.Sequential(SizePyramid(256, 32))
            self.backconv3 = nn.Sequential(SizePyramid(128, 16))
            
        elif self.pyramid == 'depth':
            print('use depth pyramid in backend')
            self.backconv0 = nn.Sequential(DepthPyramid(512, 128), DepthPyramid(512, 128), DepthPyramid(512, 128))
            self.backconv1 = nn.Sequential(DepthPyramid(512, 64))
            self.backconv2 = nn.Sequential(DepthPyramid(256, 32))
            self.backconv3 = nn.Sequential(DepthPyramid(128, 16))
            
        # objective is density map(dmp) and (binary) attention map(amp)
        if self.objective == 'dmp+amp':
            print('objective dmp+amp!')
            self.amp_process = make_layers([64,64], in_channels=64, dilation=True, batch_norm=bn, NL=self.NL, se=se)
            self.amp_layer = nn.Conv2d(64, 1, kernel_size=1)
            self.sgm = nn.Sigmoid()
        elif self.objective == 'dmp':
            print('objective dmp')
        else:
            raise Exception('objective must in [dmp, dmp+amp]')
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.load_model = load_model
        #self._init_weights()
        self._random_init_weights()

    def forward(self, x_in):
        x1 = self.front0(x_in)#1 size, 64
        x2 = self.front1(x1)#1/2 size, 128
        x3 = self.front2(x2)#1/4 size, 256
        x4 = self.front3(x3)#1/8 size, 512
        
        if self.sp:
            x4 = self.sp_module(x4)
        
        x = self.backconv0(x4) #1/8 size, 512
        
        x = F.interpolate(x, size=[s//4 for s in x_in.shape[2:]]) #1/4 size, 256
        
        x = torch.cat([x3, x], dim=1) #1/4 size,
        x = self.backconv1(x) #1/4 size, 
        
        x = F.interpolate(x, size=[s//2 for s in x_in.shape[2:]]) #1/2 size, 
        
        x = torch.cat([x2, x], dim=1) #1/2 size, 
        x = self.backconv2(x) #1/2 size, 
        
        x = F.interpolate(x, size=x_in.shape[2:]) #1 size, 
        
        x = torch.cat([x1, x], dim=1) #1 size, 
        x = self.backconv3(x) #1 size, 64
        
        if self.objective == 'dmp+amp':
            dmp = self.output_layer(x)
            amp = self.amp_layer(self.amp_process(x))
            amp = self.sgm(amp)
            dmp = amp * dmp
            del x
            dmp = torch.abs(dmp)
            return dmp, amp
        else:
            x = self.output_layer(x)
            x = torch.abs(x)
            return x

    def _random_init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _init_weights(self):
        if not self.load_model:
            pretrained_dict = dict()
            model_dict = self.state_dict()
            path = "/home/datamining/Models/vgg16_bn-6c64b313.pth" if self.bn else '/home/datamining/Models/vgg16-397923af.pth'
            pretrained_model = torch.load(path)
            self._random_init_weights()
            # load the pretrained vgg16 parameters
            
            for i, (k, v) in enumerate(pretrained_model.items()):
                #print(i, k)
                
                if i < 4:
                    layer_id = 0
                    module_id = k.split('.')[-2]
                elif i < 8:
                    layer_id = 1
                    module_id = int(k.split('.')[-2]) - 4
                elif i < 14:
                    layer_id = 2
                    module_id = int(k.split('.')[-2]) - 9
                elif i < 20:
                    layer_id = 3
                    module_id = int(k.split('.')[-2]) - 16
                else:
                    break
                k = 'front' + str(layer_id) + '.' + str(module_id) + '.' + k.split('.')[-1]
                
                if k in model_dict and model_dict[k].size() == v.size():
                    print(k, ' parameters loaded!')
                    pretrained_dict[k] = v
            
            print(path, 'weights loaded!')
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            self.load_state_dict(torch.load(self.load_model))
            print(self.load_model,' loaded!')


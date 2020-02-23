import torch.nn as nn
import torch.nn.functional as F

def ConvINReLU(cfg, in_planes):
    layers = []
    for c in cfg:
        if len(c)==2:
            plane, k = c
            layers += [nn.Conv2d(in_planes, plane, k, padding=(k-1)//2), nn.InstanceNorm(plane, affine=True), nn.ReLU(True)]
            in_planes = plane
        if len(c)==1:
            # im not sure 
            layers += [nn.ConvTranspose2d(in_planes, in_planes, 2, stride=2), nn.InstanceNorm(in_planes, affine=True), nn.ReLU(True)]
    return nn.Sequential(*layers)


class SAModule(nn.Module):
    def __init__(self, reduc, in_planes, out_planes):
        super(self, SAModule).__init__()
        # if there is reduction 
        self.reduc = reduc
        sub_planes = out_planes // 4
        self.branch1 = nn.Sequential(nn.Conv2d(in_planes, sub_planes, 1), nn.ReLU(True))
        if self.reduc:
            self.branch2 = ConvINReLU(((in_planes // 2, 1), (sub_planes, 3)),in_planes)
            self.branch3 = ConvINReLU(((in_planes // 2, 1), (sub_planes, 5)),in_planes)
            self.branch4 = ConvINReLU(((in_planes // 2, 1), (sub_planes, 7)),in_planes)
        else:
            self.branch2 = ConvINReLU(((sub_planes, 3)),in_planes)
            self.branch3 = ConvINReLU(((sub_planes, 5)),in_planes)
            self.branch4 = ConvINReLU(((sub_planes, 7)),in_planes)
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], 1)
        return out


class SANet(nn.Module):
    def __init__(self):
        super(self, SANet).__init__()
        #first
        self.encoder = nn.Sequential(SAModule(False, 3, 64), nn.MaxPool2d(kernel_size=2, stride=2),
                                     SAModule(True, 64, 128), nn.MaxPool2d(kernel_size=2, stride=2),
                                     SAModule(True, 128, 128), nn.MaxPool2d(kernel_size=2, stride=2),
                                     SAModule(True, 128, 64))
        self.decoder_cfg = ([64, 9], [T], [32, 7], [T], [16, 5], [T], [16, 3], [16, 5], [1, 1])
        self.decoder = ConvINReLU(cfg=self.decoder_cfg, in_planes=64)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
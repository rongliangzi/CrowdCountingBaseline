import torch.nn as nn
import torch


class conv_act(nn.Module):
    '''
    basic module for conv-(bn-)activation
    '''
    def __init__(self, in_channels, out_channels, kernel_size,  NL='relu', dilation=1, stride=1, same_padding=True, use_bn=False):
        super(conv_act, self).__init__()
        padding = (kernel_size + (dilation - 1) * (kernel_size - 1) - 1) // 2 if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True)
        if NL == 'relu' :
            self.activation = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.activation = nn.PReLU() 
        elif NL == 'swish':
            self.activation = Swish()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        return x


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False, NL='relu', se=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if NL=='prelu':
                nl_block = nn.PReLU()
            elif NL=='swish':
                nl_block = Swish()
            else:
                nl_block = nn.ReLU(inplace=True)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nl_block]
            else:
                layers += [conv2d, nl_block]
            if se:
                layers += [SEModule(v)]
            in_channels = v
    return nn.Sequential(*layers)


class DilationPyramid(nn.Module):
    '''
    aggregate different dilations
    '''
    def __init__(self, in_channels, out_channels, dilations=[1,2,3,6], NL='relu'):
        super(DilationPyramid, self).__init__()
        assert len(dilations)==4, 'length of dilations must be 4'
        self.conv1 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL), conv_act(out_channels, out_channels, 3, NL, dilation=dilations[0]))
        self.conv2 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL), conv_act(out_channels, out_channels, 3, NL, dilation=dilations[1]))
        self.conv3 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL), conv_act(out_channels, out_channels, 3, NL, dilation=dilations[2]))
        self.conv4 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL), conv_act(out_channels, out_channels, 3, NL, dilation=dilations[3]))
        self.conv5 = conv_act(4*out_channels, 4*out_channels, 1, NL)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        output = torch.cat([x1, x2, x3, x4], 1)
        output = self.conv5(output)
        return output


class SizePyramid(nn.Module):
    '''
    aggregate different filter sizes, [1,3,5,7] like ADCrowdNet's decoder
    '''
    def __init__(self, in_channels, out_channels, NL='relu'):
        super(SizePyramid, self).__init__()
        self.conv1 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL), conv_act(out_channels, out_channels, 3, NL))
        self.conv2 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL), conv_act(out_channels, out_channels, 5, NL))
        self.conv3 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL), conv_act(out_channels, out_channels, 7, NL))
        self.conv4 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL))
        self.conv5 = conv_act(4*out_channels, 4*out_channels, 1, NL)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        output = torch.cat([x1, x2, x3, x4], 1)
        output = self.conv5(output)
        return output
        
        
class DepthPyramid(nn.Module):
    '''
    aggregate different depths, like TEDNet's decoder
    '''
    def __init__(self, in_channels, out_channels, NL='relu'):
        super(DepthPyramid, self).__init__()
        self.conv1 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL), conv_act(out_channels, out_channels, 3, NL))
        self.conv2 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL), conv_act(out_channels, out_channels, 3, NL), conv_act(out_channels, out_channels, 3, NL))
        self.conv3 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL), conv_act(out_channels, out_channels, 3, NL), conv_act(out_channels, out_channels, 3, NL), conv_act(out_channels, out_channels, 3, NL))
        self.conv4 = nn.Sequential(conv_act(in_channels, out_channels, 1, NL))
        self.conv5 = conv_act(4*out_channels, 4*out_channels, 1, NL)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        output = torch.cat([x1, x2, x3, x4], 1)
        output = self.conv5(output)
        return output
                
    
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SPModule(nn.Module):
    def __init__(self, in_channels, branch_out=None):
        super(SPModule, self).__init__()
        if not branch_out:
            # ensure the in and out have the same channels.
            branch_out = in_channels
        self.dilated1 = nn.Sequential(nn.Conv2d(in_channels, branch_out,3,padding=2, dilation=2),nn.ReLU(True))
        self.dilated2 = nn.Sequential(nn.Conv2d(in_channels, branch_out,3,padding=4, dilation=4),nn.ReLU(True))
        self.dilated3 = nn.Sequential(nn.Conv2d(in_channels, branch_out,3,padding=8, dilation=8),nn.ReLU(True))
        self.dilated4 = nn.Sequential(nn.Conv2d(in_channels, branch_out,3,padding=12, dilation=12),nn.ReLU(True))
        self.down_channels = nn.Sequential(nn.Conv2d(branch_out*4, in_channels,1),nn.ReLU(True))
    def forward(self,x):
        x1 = self.dilated1(x)
        x2 = self.dilated2(x)
        x3 = self.dilated3(x)
        x4 = self.dilated4(x)
        # concat
        x = torch.cat([x1,x2,x3,x4],1)
        x = self.down_channels(x)
        return x


class SEModule(nn.Module):
    def __init__(self, in_, reduction=16):
        super().__init__()
        squeeze_ch = in_//reduction
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_, squeeze_ch, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(squeeze_ch, in_, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))
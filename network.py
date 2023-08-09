import torch
import torch.nn as nn

import torchvision.models as models
import torchvision.transforms as transforms

class TransformNetwork(nn.Module):
    def __init__(self):
        super(TransformNetwork, self).__init__()
        self.layer = nn.Sequential(
            
            #3 convlayer
            ConvLayer(3, 32, 9, 1),
            ConvLayer(32, 64, 3, 2), 
            ConvLayer(64, 128, 3, 2),

            #residual layer
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            
            #deconv layer
            DeconvLayer(128, 64, 3, 1),
            DeconvLayer(64, 32, 3, 1),
            ConvLayer(32, 3, 9, 1, activation='linear')

        )
    def forward(self, x):
        return self.layer(x)
class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, pad='reflect', activation='relu', normalization='instance'):
        super(ConvLayer, self).__init__()
        #Padding
        if pad == 'reflect':
            self.pad = nn.ReflectionPad2d(kernel_size//2)
        elif pad == 'zero':
            self.pad = nn.ZeroPad2d(kernel_size//2)
        else:
            raise NotImplementedError('Comming soon')

        # Conv
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)

        #activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'linear':
            self.activation = lambda x : x
        else:
            raise NotImplementedError('Comming soon')
        
        #normalization
        if normalization == 'instance':
            self.normalization = nn.InstanceNorm2d(out_ch, affine=True)
        else:
            raise NotImplementedError('Comming soon')
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x

class ResidualLayer(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, pad='reflect', normalization='instance'):
        super(ResidualLayer, self).__init__()

        #conv1
        self.conv1 = ConvLayer(in_ch, out_ch, kernel_size, stride, pad, activation='relu', normalization=normalization)

        #conv2
        self.conv2 = ConvLayer(in_ch, out_ch, kernel_size, stride, pad, activation='relu', normalization=normalization)

    def forward(self, x):
        y = self.conv1(x)
        return self.conv2(y) + x
class DeconvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, pad='reflect', activation = 'relu', normalization='instance', upsample='nearest'):
        super(DeconvLayer, self).__init__()
        self.up_sample = upsample
       #pad
        if pad == 'reflect':
           self.pad = nn.ReflectionPad2d(kernel_size//2)
        elif pad == 'zero':
           self.pad = nn.ZeroPad2d(kernel_size//2)
        else:
           raise NotImplementedError("Coming soon")
        
        #conv
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)

        #normalization

        if normalization == 'instance':
            self.normalization = nn.InstanceNorm2d(out_ch, affine=True)
        else:
            raise NotImplementedError("Comming soon")

        #activation

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'linear':
            self.activation = lambda x: x
        else:
            raise NotImplementedError('Comming soon')
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode=self.up_sample)
        x = self.pad(x)
        x = self.conv(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x
           
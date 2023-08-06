import sys, os

#Our project root directory. Some imports depend on this
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.pardir))
sys.path.append(PROJECT_ROOT)

import torch
import torch.nn as nn
import util.resnet as rn
import torch.nn.functional as F
from util.constants import RESNET_50_WEIGHTS

class ResNetUnet(nn.Module):
    def __init__(self, args):
        super(ResNetUnet, self).__init__()
        self.inputChannel = args.inputChannel
        self.outputChannel = args.outputChannel
        self.ConvLayer1, \
        self.ConvLayer2, \
        self.ConvLayer3, \
        self.ConvLayer4, \
        self.ConvLayer5 = rn.resnet50(RESNET_50_WEIGHTS, pretrained=args.pretrained)    

        if args.freezebackbone:
            self.__freeze_backbone__()

        self.ConvLayer6 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1) 
        self.ConvLayer7 = DoubleConv(in_channels=2048, out_channels=512)
        self.ConvLayer8 = DoubleConv(in_channels=1024, out_channels=256)
        self.ConvLayer9 = DoubleConv(in_channels=512,  out_channels=128)
        self.ConvLayer10 = DoubleConv(in_channels=256,  out_channels=128) #64

        self.ConvTranspose1 = TransposedConvolutionBlock(in_channels=1024, out_channels=1024)
        self.ConvTranspose2 = TransposedConvolutionBlock(in_channels=512, out_channels=512)
        self.ConvTranspose3 = TransposedConvolutionBlock(in_channels=256, out_channels=256)
        self.ConvTranspose4 = TransposedConvolutionBlock(in_channels=128, out_channels=128)

        self.ConvTransposex = TransposedConvolutionBlock(in_channels=64, out_channels=128)

        self.ConvTranspose5 = TransposedConvolutionBlock(in_channels=128, out_channels=64)

        self.outConv = nn.Conv2d(in_channels=64, out_channels=args.outputChannel, kernel_size=1)

    def __freeze_backbone__(self):
        for parameter_en1, parameter_en2, parameter_en3, parameter_en4, parameter_en5 in zip(self.ConvLayer1.parameters(),
                                                                                             self.ConvLayer2.parameters(),
                                                                                             self.ConvLayer3.parameters(),
                                                                                             self.ConvLayer4.parameters(),
                                                                                             self.ConvLayer5.parameters()):
            parameter_en1.requires_grad = False
            parameter_en2.requires_grad = False
            parameter_en3.requires_grad = False
            parameter_en4.requires_grad = False
            parameter_en5.requires_grad = False

    def forward(self, out0):
        out1 = self.ConvLayer1(out0) #64   x 256 x 256
        out2 = self.ConvLayer2(out1) #256  x 256 x 256
        out3 = self.ConvLayer3(out2) #512  x 128 x 128
        out4 = self.ConvLayer4(out3) #1024 x 64  x 64
        out5 = self.ConvLayer5(out4) #2048 x 32  x 32 
        
        out_to_transpose = self.ConvLayer6(out5) #1024 x 32 x 32
        out_transpose1 = self.ConvTranspose1(out_to_transpose) #1024 x 64 x 64

        concat1 = torch.cat((out_transpose1, out4), dim=1) #2048 x 64 x 64
        out_transpose2 = self.ConvTranspose2(self.ConvLayer7(concat1)) #512 x 128 x 128

        concat2 = torch.cat((out_transpose2, out3), dim=1) #1024 x 128 x 128
        out_transpose3 = self.ConvTranspose3(self.ConvLayer8(concat2)) #256 x 256 x 256

        concat3 = torch.cat((out_transpose3, out2), dim=1) #512 x 256 x 256
        out_transpose4 = self.ConvTranspose4(self.ConvLayer9(concat3)) #128 x 512 x 512

        out1 = self.ConvTransposex(out1) #upsample to 128 x 512 x 512

        concat4 = torch.cat((out_transpose4, out1), dim=1) #256 x 512 x 512
        out_transpose5 = self.ConvTranspose5(self.ConvLayer10(concat4)) #64 x 1024 x 1024

        output = self.outConv(out_transpose5) #1 x 1024 x 1024

        return output


class DoubleConv(nn.Module):
    '''conv -> [BN] -> ReLU'''
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.double_conv(x)


class TransposedConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(TransposedConvolutionBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                                 stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        x = self.conv_transpose(x)
        return x
import torch.nn as nn
import torch

class CNN_for_DBD(nn.Module):
    def __init__(self):
        super(CNN_for_DBD, self).__init__()
        self.U_Net_for_DBD = U_Net_for_DBD()
    def forward(self, input_1):
        dbd_result = self.U_Net_for_DBD(input_1)
        return dbd_result

class U_Net_for_DBD(nn.Module):
    def __init__(self):
        super(U_Net_for_DBD, self).__init__()
        self.base1 = Base_DBD_1()
        self.base2 = Base_DBD_2()
    def forward(self, input_1):
        s1, s2, s3 = self.base1(input_1)
        dbd1 = self.base2(s1, s2, s3)
        return dbd1

class Base_DBD_1(nn.Module):
    def __init__(self):
        super(Base_DBD_1, self).__init__()
        self.avgpool = nn.AvgPool2d((2, 2))
        self.conv1_1_2 = BaseConv(3, 64, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv1_2_2 = BaseConv(64, 64, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv2_1_2 = BaseConv(64, 128, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv2_2_2 = BaseConv(128, 128, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv3_1_2 = BaseConv(128, 256, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv3_2_2 = BaseConv(256, 256, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv3_3_2 = BaseConv(256, 256, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv4_1_2 = BaseConv(256, 512, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv4_2_2 = BaseConv(512, 512, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv4_3_2 = BaseConv(512, 512, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)

    def forward(self, x):
        x = self.conv1_1_2(x)
        x = self.conv1_2_2(x)
        x = self.avgpool(x)
        x = self.conv2_1_2(x)
        x = self.conv2_2_2(x)
        s1 = x
        x = self.avgpool(x)
        x = self.conv3_1_2(x)
        x = self.conv3_2_2(x)
        x = self.conv3_3_2(x)
        s2 = x
        x = self.avgpool(x)
        x = self.conv4_1_2(x)
        x = self.conv4_2_2(x)
        x = self.conv4_3_2(x)
        s3 = x
        return s1, s2, s3

class Base_DBD_2(nn.Module):
    def __init__(self):
        super(Base_DBD_2, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1_2 = BaseConv(512, 256, 1, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv2_2 = BaseConv(256, 256, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv3_2 = BaseConv(512, 128, 1, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv4_2 = BaseConv(128, 128, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv5_2 = BaseConv(256, 64, 1, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv6_2 = BaseConv(64, 64, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv7_2 = BaseConv(64, 32, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv_out_base_1 = BaseConv(32, 32, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv_out_base_2 = BaseConv(32, 32, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.conv_out_base_3 = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, s1, s2, s3):
        x = s3
        x = self.conv1_2(x)
        x = self.conv2_2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s2], 1)
        x = self.conv3_2(x)
        x = self.conv4_2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s1], 1)
        x = self.conv5_2(x)
        x = self.conv6_2(x)
        x = self.conv7_2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv_out_base_1(x)
        x = self.conv_out_base_2(x)
        x = self.conv_out_base_3(x)

        return x

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input
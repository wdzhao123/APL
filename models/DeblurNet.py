import torch.nn as nn
import torch
from math import sqrt
import torch
import torch.nn.functional as F

class CNN_for_Generator(nn.Module):
    def __init__(self):
        super(CNN_for_Generator, self).__init__()
        self.Net_for_Generator = Net_for_Generator()
    def forward(self, input_1, input_2):
        generate_result = self.Net_for_Generator(input_1, input_1, input_2)
        return generate_result

class Net_for_Generator(nn.Module):
    def __init__(self):
        super(Net_for_Generator, self).__init__()
        self.base2 = Base_Generator()
    def forward(self, input_1, input_2, mask):
        real_blur = input_1 * (1 - mask)
        real_clear = input_2 * mask
        clear_result = self.base2(real_blur, real_clear, mask)
        return clear_result

class Base_Generator(nn.Module):
    def __init__(self):
        super(Base_Generator, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.residual_layer_1 = make_layer(Conv_ReLU_Block, 2)
        self.fuse_layer_1 = ResBlock_SFT()
        self.residual_layer_2 = make_layer(Conv_ReLU_Block, 3)
        self.fuse_layer_2 = ResBlock_SFT()
        self.residual_layer_3 = make_layer(Conv_ReLU_Block, 3)
        self.fuse_layer_3 = ResBlock_SFT()
        self.residual_layer_4 = make_layer(Conv_ReLU_Block, 2)
        self.output = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, input_1, input_2, input_3):
        mask = input_3
        out = input_1
        residual = input_1
        out = self.relu(self.input(out))
        out = self.residual_layer_1(out)
        out = self.fuse_layer_1(input_1, input_2, out)
        out = self.residual_layer_2(out)
        out = self.fuse_layer_2(input_1, input_2, out)
        out = self.residual_layer_4(out)
        out = self.output(out)
        out = torch.add(out,residual)
        out = out * (1-mask)

        clear_result = out + input_2
        return clear_result

class SFT_Layer(nn.Module):
    def __init__(self):
        super(SFT_Layer, self).__init__()
        self.avgpool = nn.AvgPool2d(160, stride=1)
        self.layer1_1 = BaseConv(3, 128, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.layer1_2 = BaseConv(128, 128, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.layer1_3 = BaseConv(128, 128, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.layer2_1 = BaseConv(256, 128, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.layer2_2 = BaseConv(128, 128, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.layer2_3 = BaseConv(128, 128, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)

        self.fc1_1 = BaseConv(128, 128, 1, 1, activation=nn.LeakyReLU(0.2), use_bn=False)
        self.fc1_2 = BaseConv(128, 128, 1, 1, activation=nn.LeakyReLU(0.2), use_bn=False)

        self.layer3_1 = BaseConv(3, 128, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.layer3_2 = BaseConv(128, 128, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.layer3_3 = BaseConv(128, 128, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.layer4_1 = BaseConv(256, 128, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.layer4_2 = BaseConv(128, 128, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)
        self.layer4_3 = BaseConv(128, 128, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True)

        self.fc2_1 = BaseConv(128, 128, 1, 1, activation=nn.LeakyReLU(0.2), use_bn=False)
        self.fc2_2 = BaseConv(128, 128, 1, 1, activation=nn.LeakyReLU(0.2), use_bn=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, input_2, out):
        I1 = input_1
        I2 = input_2
        f1 = self.layer1_1(I1)
        f1 = self.layer1_2(f1)
        f1 = self.layer1_3(f1)
        f2 = self.layer1_1(I2)
        f2 = self.layer1_2(f2)
        f2 = self.layer1_3(f2)
        f3 = torch.cat([f1, f2], 1)
        f6 = f3
        f3 = self.layer2_1(f3)
        f3 = self.layer2_2(f3)
        f3 = self.layer2_3(f3)
        v_a = self.avgpool(f3)
        v_a = self.fc1_1(v_a)
        v_a = self.fc1_2(v_a)
        v_a = self.sigmoid(v_a)

        f6 = self.layer4_1(f6)
        f6 = self.layer4_2(f6)
        f6 = self.layer4_3(f6)
        v_b = self.avgpool(f6)
        v_b = self.fc2_1(v_b)
        v_b = self.fc2_2(v_b)
        v_b = self.sigmoid(v_b)
        out = v_a * out + v_b
        fused_feature = out
        return fused_feature

class ResBlock_SFT(nn.Module):
    def __init__(self):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFT_Layer()
        self.conv0 = nn.Conv2d(128, 128, 3, 1, 1)
        self.sft1 = SFT_Layer()
        self.conv1 = nn.Conv2d(128, 128, 3, 1, 1)

    def forward(self, input_1, input_2, s):
        fea = self.sft0(input_1, input_2, s)
        fea = F.leaky_relu(self.conv0(fea), inplace = True)
        fea = self.sft1(input_1, input_2, fea)
        fea = self.conv1(fea)
        fused_feature = s + fea
        return fused_feature

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

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.BN(self.conv(x)))

def make_layer(block, num_of_layer):
    layers = []
    for _ in range(num_of_layer):
        layers.append(block())
    return nn.Sequential(*layers)


##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self,opt=None):
        super(Discriminator,self).__init__()
        self.ngpu = 1
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),


            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if isinstance(x.data,torch.cuda.FloatTensor) and self.ngpu>1:
            output = nn.parallel.data_parallel(self.net,x,range(self.ngpu))
        else:
            output = self.net(x)
        return output
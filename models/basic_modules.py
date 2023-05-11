from torch import nn
from torch.nn.utils import weight_norm
from torch.nn.parameter import Parameter
from torch import norm_except_dim
import torch
import torch.nn.functional as F
import pdb


# ================Basic layers in ml_memAE_sc========================
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    '''
    inconv only changes the number of channels
    '''

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2),
            # double_conv(in_ch, out_ch)

            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
            double_conv(out_ch, out_ch),

        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False, op="none"):
        super(up, self).__init__()
        self.bilinear = bilinear
        self.op = op
        if self.bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch // 2, 1), )
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)
        assert op in ["concat", "none"]

        if op == "concat":
            self.conv = double_conv(in_ch, out_ch)
        else:
            self.conv = double_conv(out_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        if self.op == "concat":##进行skip连接
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1

        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.conv(x)
        return x


# ================Basic layers in VUNet========================

class Downsample(nn.Module):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        if out_channels == None:
            self.down = nn.Conv2d(
                channels, channels, kernel_size=3, stride=2, padding=1,
            )
        else:
            self.down = nn.Conv2d(
                channels, out_channels, kernel_size=3, stride=2, padding=1,
            )

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels == None:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=1,output_padding=1,
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1,output_padding=1,
            )
            
    def forward(self, x):
        out = self.up(x)
        return out

class IDAct(nn.Module):
    def forward(self, input):
        return input

class VUnetResnetBlock(nn.Module):
    """
    Resnet Block as utilized in the vunet publication
    """

    def __init__(
            self,
            out_channels,
            use_skip=False,
            kernel_size=3,
            activate=True,
            final_act=False,
            dropout_prob=0.0,
    ):
        """

        :param n_channels: The number of output filters
        :param process_skip: the factor between output and input nr of filters
        :param kernel_size:
        :param activate:
        """
        super().__init__()
        self.dout = nn.Dropout(p=dropout_prob)
        self.use_skip = use_skip
        if self.use_skip:
            self.conv2d1 = nn.Conv2d(
                in_channels=2 * out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,

            )
            self.pre = nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels, kernel_size=1,
            )
        else:
            self.conv2d1 = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,

            )

        self.conv2d2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
        if activate:
            self.act_fn = nn.LeakyReLU() if final_act else nn.ELU()
        else:
            self.act_fn = IDAct()#原样输出，不进行激活函数

    def forward(self, x, a=None):
        x_prc = x

        if self.use_skip:
            assert a is not None
            a = self.act_fn(a)
            a = self.pre(a)
            x_prc = torch.cat([x_prc, a], dim=1)

        x_prc = self.act_fn(x_prc)
        x_prc = self.dout(x_prc)
        x_prc = self.conv2d1(x_prc)
        #x_prc = self.conv2d2(x_prc)
        return x + x_prc

# se_atten
class channel_Attention(nn.Module):
    def __init__(self, channel, reduction=16,use_conv3x3=True):
        super(channel_Attention, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.use_conv3x3=use_conv3x3
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel,
            kernel_size=3,padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),#GAP
            nn.Conv2d(self.channel, self.channel//self.reduction,kernel_size=1),#FC
            nn.ReLU(),#RELU
            nn.Conv2d(self.channel//self.reduction,self.channel,kernel_size=1),#FC
            nn.Sigmoid()
        )
        #param initial 参数初始化
        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # se_atten
    def forward(self,x):
        if self.use_conv3x3:
            #3*3 conv
            x = self.conv1(x)  # b, c, h, w
            x = self.bn1(x)  # b, c, h, w
        se = self.se(x)  # b, c, 1, 1
        return se
    
class spatial_Attention(nn.Module):
    def __init__(self, channel,kernel_size=3,use_conv3x3=True):
        super(spatial_Attention, self).__init__()
        self.channel = channel
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.use_conv3x3=use_conv3x3
        self.conv3x3 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel,
            kernel_size=3,padding=padding,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel)
        
        self.conv1 = nn.Conv2d(self.channel, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat([avg_out, max_out], dim=1)
        # x = self.conv1(x)
        if self.use_conv3x3:
            #3*3 conv
            x=self.conv3x3(x) # b, c, h, w
            x = self.bn1(x)  # b, c, h, w
        x = self.conv1(x) #b,1,h,w
        x = self.sigmoid(x) #b,1,h,w
        return x

# if __name__ == '__main__':
#     model = Upsample(in_channels=32)
#     print(model)
#     dummy_x = torch.rand(4, 32, 32, 32)
#     dummy_out = model(dummy_x)
#     print(dummy_out.shape)
#     print(-1)

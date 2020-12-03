import numpy as np 
import torch 
import torch.nn as nn  
import torch.nn.functional as F
from torch.nn import init 

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def resnet_Add(x1,x2):
    res_add = x1 + x2 
    return res_add

def Concate(x1,x2):
    x = torch.cat((x1,x2),1)
    return x     

class conv3d_bn_relu_drop(nn.Module):
    def __init__(self,in_channel,out_channel,drop):
        super(conv3d_bn_relu_drop,self).__init__()
        self.conv3d_bn_relu_drop = nn.Sequential(
            nn.Conv3d(in_channel,out_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
            nn.Dropout3d(p=drop)
        )
    
    def forward(self,x):
        x = self.conv3d_bn_relu_drop(x)
        return x 

class conv3d_bn_relu(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(conv3d_bn_relu,self).__init__()
        self.conv3d_bn_relu = nn.Sequential(
            nn.Conv3d(in_channel,out_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(out_channel),
            nn.ReLU()
        )
    
    def forward(self,x):
        x = self.conv3d_bn_relu(x)
        return x

class conv3d_gn_relu(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(conv3d_gn_relu,self).__init__()
        self.conv3d_gn_relu = nn.Sequential(
            nn.Conv3d(in_channel,out_channel,kernel_size=3,stride=1,padding=1),
            nn.GroupNorm(16,out_channel),
            nn.ReLU()
        )
    
    def forward(self,x):
        x = self.conv3d_gn_relu(x)
        return x

class conv3d_gn_relu_drop(nn.Module):
    def __init__(self,in_channel,out_channel,drop = 0.5):
        super(conv3d_gn_relu_drop,self).__init__()
        self.conv3d_gn_relu_drop = nn.Sequential(
            nn.Conv3d(in_channel,out_channel,kernel_size=3,stride=1,padding=1),
            nn.GroupNorm(16,out_channel),
            nn.ReLU(),
            nn.Dropout3d(p=drop)
        )
    
    def forward(self,x):
        x = self.conv3d_gn_relu_drop(x)
        return x 

class conv3d_gn_relu_by1x1(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(conv3d_gn_relu_by1x1,self).__init__()
        self.conv3d_gn_relu_by1x1 = nn.Sequential(
            nn.Conv3d(in_channel,32,kernel_size=1),
            #nn.GroupNorm(16,32),
            nn.ReLU(),
            nn.Conv3d(32,32,kernel_size=3,stride=1,padding=1),
            nn.GroupNorm(16,32),
            nn.ReLU(),
            nn.Conv3d(32,out_channel,kernel_size=1),
            #nn.GroupNorm(16,out_channel),
            nn.ReLU()
        )
    
    def forward(self,x):
        x = self.conv3d_gn_relu_by1x1(x)
        return x 

class conv3d_gn_relu_drop_by1x1(nn.Module):
    def __init__(self,in_channel,out_channel,drop=0.5):
        super(conv3d_gn_relu_drop_by1x1,self).__init__()
        self.conv3d_gn_relu_drop_by1x1 = nn.Sequential(
            nn.Conv3d(in_channel,32,kernel_size=1),
            #nn.GroupNorm(16,64),
            nn.ReLU(),
            nn.Conv3d(32,32,kernel_size=3,stride=1,padding=1),
            nn.GroupNorm(16,32),
            nn.ReLU(),
            nn.Conv3d(32,out_channel,kernel_size=1),
            #nn.GroupNorm(16,out_channel),
            nn.ReLU(),
            nn.Dropout(p=drop)
        )
    
    def forward(self,x):
        x = self.conv3d_gn_relu_drop_by1x1(x)
        return x         

class conv3d_FRN_TLU(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(conv3d_FRN_TLU,self).__init__()
        self.conv3d_FRN_TLU = nn.Sequential(
            nn.Conv3d(in_channel,out_channel,kernel_size=3,padding=1),
            FRN(out_channel),
            TLU(out_channel)
        )

    def forward(self,x):
        x = self.conv3d_FRN_TLU(x)
        return x 

class down(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(down,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channel,out_channel,kernel_size=3,stride=2,padding=1),
            nn.ReLU()
        )
    
    def forward(self,x):
        x = self.conv(x)
        return x  

class up(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(up,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channel,out_channel,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.up(x)
        return x 
  
class Max_conv(nn.Module):
    def __init__(self,in_channel,out_channel,max_time = 0):
        super(Max_conv,self).__init__()
        self.maxpol = nn.MaxPool3d(kernel_size=max_time,stride = max_time,ceil_mode=True)
        self.conv = conv3d_FRN_TLU(in_channel,out_channel)

    def forward(self,x):
        x = self.maxpol(x)
        x = self.conv(x)
        return x 

class Up_conv(nn.Module):
    def __init__(self,in_channel,out_channel,up_time=0):
        super(Up_conv,self).__init__()
        self.up = nn.Upsample(scale_factor = up_time,mode='trilinear')
        self.conv = conv3d_FRN_TLU(in_channel,out_channel)
    
    def forward(self,x):
        x = self.up(x)
        x = self.conv(x)
        return x 

def ones_(tensor):
    r"""Fills the input Tensor with ones`.

    Args:
        tensor: an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.ones_(w)
    """
    with torch.no_grad():
        return tensor.fill_(1)


def zeros_(tensor):
    r"""Fills the input Tensor with zeros`.

    Args:
        tensor: an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.zeros_(w)
    """
    with torch.no_grad():
        return tensor.zero_()

class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6, is_eps_leanable=False):
        """
        weight = gamma, bias = beta
        beta, gamma:
            Variables of shape [1, 1, 1, 1, C]. if TensorFlow
            Variables of shape [1, C, 1, 1, 1]. if PyTorch
        eps: A scalar constant or learnable variable.
        """
        super(FRN, self).__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable

        self.weight = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1, 1), requires_grad=True)
        self.bias = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1, 1), requires_grad=True)
        if is_eps_leanable:
            self.eps = nn.parameter.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        ones_(self.weight)
        zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x):
        """
        0, 1, 2, 3 -> (B, H, W, Z, C) in TensorFlow
        0, 1, 2, 3 -> (B, C, Z, H, W) in PyTorch
        TensorFlow code
            nu2 = tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True)
            x = x * tf.rsqrt(nu2 + tf.abs(eps))
            # This Code include TLU function max(y, tau)
            return tf.maximum(gamma * x + beta, tau)
        """
        # Compute the mean norm of activations per channel.
        nu2 = x.pow(2).mean(dim=[2, 3, 4], keepdim=True)

        # Perform FRN.
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        # Scale and Bias
        x = self.weight * x + self.bias
        return x

class TLU(nn.Module):
    def __init__(self, num_features):
        """max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau"""
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        zeros_(self.tau)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau)

class StartResBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(StartResBlock,self).__init__()
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channel,out_channel,kernel_size=1,stride=1,bias=False),
            FRN(out_channel)
        )
        self.startallconv = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            FRN(out_channel),
            TLU(out_channel),
            nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1),
            FRN(out_channel),
            TLU(out_channel),
            nn.Conv3d(out_channel, out_channel, kernel_size=1, stride=1, bias=False),
            FRN(out_channel)
        )

    def forward(self,x):
        identity = self.shortcut(x)
        out = self.startallconv(x)
        out = out + identity
        return out 

class MiddleResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MiddleResBlock, self).__init__()
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            FRN(out_channel)
        )
        self.startallconv = nn.Sequential(
            #nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            #FRN(out_channel),
            TLU(out_channel),
            nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            FRN(out_channel),
            TLU(out_channel),
            nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1),
            FRN(out_channel),
            TLU(out_channel),
            nn.Conv3d(out_channel, out_channel, kernel_size=1, stride=1, bias=False),
            #FRN(out_channel)
        )
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.startallconv(x)
        out = out + identity
        return  out

class EndResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EndResBlock, self).__init__()
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            FRN(out_channel)
        )
        self.startallconv = nn.Sequential(
            #nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            FRN(out_channel),
            TLU(out_channel),
            nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            FRN(out_channel),
            TLU(out_channel),
            nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1),
            FRN(out_channel),
            TLU(out_channel),
            nn.Conv3d(out_channel, out_channel, kernel_size=1, stride=1, bias=False),
            #FRN(out_channel)
        )
        self.endfrn = FRN(out_channel)
        self.endtlu = TLU(out_channel)
        #self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.startallconv(x)
        out = self.endtlu(self.endfrn(out + identity))
        return  out

class MyResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MyResBlock, self).__init__()
        self.mystartresblock = StartResBlock(in_channel, out_channel)
        self.myendresblock = EndResBlock(out_channel, out_channel)
    def forward(self, x):
        x = self.mystartresblock(x)
        x = self.myendresblock(x)
        return x

class U_net_TriplePlus(nn.Module):
    def __init__(self,args):
        super(U_net_TriplePlus,self).__init__()
        #filters = [32,64,128,256,512]
        filters = [16,32,64,128,256]
        in_channel = 4
        out_channel = 3

        ## -------------Encoder--------------
        self.conv1 = MyResBlock(in_channel, filters[0])
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = MyResBlock(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = MyResBlock(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = MyResBlock(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool3d(kernel_size=2)

        self.conv5 = MyResBlock(filters[3], filters[4])

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks


        self.de1_1 = Up_conv(filters[4],filters[0],2)
        self.de1_2 = conv3d_FRN_TLU(filters[3],filters[0])
        self.de1_3 = Max_conv(filters[2],filters[0],2)
        self.de1_4 = Max_conv(filters[1],filters[0],4)
        self.de1_5 = Max_conv(filters[0],filters[0],8)
        self.de1_6 = conv3d_FRN_TLU(self.UpChannels,self.UpChannels)

        self.de2_1 = Up_conv(filters[4],filters[0],4)
        self.de2_2 = Up_conv(self.UpChannels,filters[0],2)
        self.de2_3 = conv3d_FRN_TLU(filters[2],filters[0])
        self.de2_4 = Max_conv(filters[1],filters[0],2)
        self.de2_5 = Max_conv(filters[0],filters[0],4)
        self.de2_6 = conv3d_FRN_TLU(self.UpChannels,self.UpChannels)

        self.de3_1 = Up_conv(filters[4],filters[0],8)
        self.de3_2 = Up_conv(self.UpChannels,filters[0],4)
        self.de3_3 = Up_conv(self.UpChannels,filters[0],2)
        self.de3_4 = conv3d_FRN_TLU(filters[1],filters[0])
        self.de3_5 = Max_conv(filters[0],filters[0],2)
        self.de3_6 = conv3d_FRN_TLU(self.UpChannels,self.UpChannels)

        self.de4_1 = Up_conv(filters[4],filters[0],16)
        self.de4_2 = Up_conv(self.UpChannels,filters[0],8)
        self.de4_3 = Up_conv(self.UpChannels,filters[0],4)
        self.de4_4 = Up_conv(self.UpChannels,filters[0],2)
        self.de4_5 = conv3d_FRN_TLU(filters[0],filters[0])
        self.de4_6 = conv3d_FRN_TLU(self.UpChannels,self.UpChannels)

        self.out_conv = nn.Conv3d(self.UpChannels,out_channel,kernel_size=3,padding=1)
    
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, FRN):#nn.BatchNorm2d
                init_weights(m, init_type='kaiming')

    def forward(self,x):
        ## -------------Encoder-------------
        x1 = self.conv1(x)

        x2 = self.maxpool1(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool2(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool3(x3)
        x4 = self.conv4(x4)

        x5 = self.maxpool4(x4)
        x5 = self.conv5(x5)

        ## -------------Decoder-------------
        x6_1 = self.de1_1(x5)
        x6_2 = self.de1_2(x4)
        x6_3 = self.de1_3(x3)
        x6_4 = self.de1_4(x2)
        x6_5 = self.de1_5(x1)
        x6_6 = torch.cat((x6_1,x6_2,x6_3,x6_4,x6_5),1)
        x6 = self.de1_6(x6_6)

        x7_1 = self.de2_1(x5)
        x7_2 = self.de2_2(x6)
        x7_3 = self.de2_3(x3)
        x7_4 = self.de2_4(x2)
        x7_5 = self.de2_5(x1)
        x7_6 = torch.cat((x7_1,x7_2,x7_3,x7_4,x7_5),1)
        x7 = self.de2_6(x7_6)

        x8_1 = self.de3_1(x5)
        x8_2 = self.de3_2(x6)
        x8_3 = self.de3_3(x7)
        x8_4 = self.de3_4(x2)
        x8_5 = self.de3_5(x1)
        x8_6 = torch.cat((x8_1,x8_2,x8_3,x8_4,x8_5),1)
        x8 = self.de3_6(x8_6)

        x9_1 = self.de4_1(x5)
        x9_2 = self.de4_2(x6)
        x9_3 = self.de4_3(x7)
        x9_4 = self.de4_4(x8)
        x9_5 = self.de4_5(x1)
        x9_6 = torch.cat((x9_1,x9_2,x9_3,x9_4,x9_5),1)
        x9 = self.de4_6(x9_6)

        output = self.out_conv(x9)
        return output
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from torch.nn.parameter import Parameter


class ARMLayer(nn.Module):  # attribute Feature Extract Layer
    def __init__(self, channel, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(ARMLayer, self).__init__()
        self.gmp = nn.AdaptiveMaxPool2d(1)
        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=3, bias=False,
                             groups=channel)  # 输出channel个特征
        # self.bn = nn.InstanceNorm1d(channel)

    def forward(self, x):
        b, c, _, _ = x.size()

        # attribute pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)

        # print(mean.shape)
        max = self.gmp(x).view(b, c, -1)
        # print(max.shape)

        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, max, std), -1)  # (b, c, 3)

        # attribute integration
        z = self.cfc(u)  # (b, c, 1)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)
        # z = self.bn(z)

        # print('ok')
        return x * g.expand_as(x)


class ARMConvBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ARMConvBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.arm = ARMLayer(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(self.pad1(x))
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(self.pad1(out))
        out = self.bn2(out)

        out = self.arm(out)

        out += residual
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)
        # self.srm = SRMLayer(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(self.pad1(x))
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(self.pad1(out))
        out = self.bn2(out)

        # out = self.srm(out)

        out += residual
        out = self.relu(out)
        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (
                1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out


class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)

        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (
                1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out


class AdaLIN_Morph(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()
        self.eps = 1e-6
        self.rho = nn.Parameter(torch.FloatTensor(1).fill_(1.0))
        self.gamma = nn.Linear(z_dim, z_dim)
        self.beta = nn.Linear(z_dim, z_dim)

    def forward(self, x, z):
        b, c, h, w = x.shape
        ins_mean = x.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        ins_var = x.view(b, c, -1).var(dim=2) + self.eps
        ins_std = ins_var.sqrt().view(b, c, 1, 1)

        x_ins = (x - ins_mean) / ins_std

        ln_mean = x.view(b, -1).mean(dim=1).view(b, 1, 1, 1)
        ln_val = x.view(b, -1).var(dim=1).view(b, 1, 1, 1) + self.eps
        ln_std = ln_val.sqrt()

        x_ln = (x - ln_mean) / ln_std

        rho = (self.rho - 0.1).clamp(0, 1.0)  # smoothing
        x_hat = rho * x_ins + (1 - rho) * x_ln

        gamma = self.gamma(z).view(b, c, 1, 1)
        beta = self.beta(z).view(b, c, 1, 1)
        x_hat = x_hat * gamma + beta
        return x_hat


class AdaResBlk_Morph(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0),
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0),
        )
        self.addin_1 = AdaLIN_Morph()
        self.addin_2 = AdaLIN_Morph()
        self.relu = nn.ReLU()

    def forward(self, x, z):
        x1 = self.conv1(x)
        x1 = self.relu(self.addin_1(x1, z))

        x2 = self.conv2(x1)
        x2 = self.addin_2(x2, z)
        return x + x2


class LIN_Morph(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.eps = 1e-6
        self.gamma = nn.Parameter(torch.FloatTensor(1, dim, 1, 1).fill_(1.0))
        self.beta = nn.Parameter(torch.FloatTensor(1, dim, 1, 1).fill_(0.0))
        self.rho = nn.Parameter(torch.FloatTensor(1, dim, 1, 1).fill_(0.0))

    def forward(self, x):
        b, c, h, w = x.shape
        ins_mean = x.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        ins_val = x.view(b, c, -1).var(dim=2).view(b, c, 1, 1) + self.eps
        ins_std = ins_val.sqrt()

        ln_mean = x.view(b, -1).mean(dim=1).view(b, 1, 1, 1)
        ln_val = x.view(b, -1).var(dim=1).view(b, 1, 1, 1) + self.eps
        ln_std = ln_val.sqrt()

        rho = torch.clamp(self.rho, 0, 1)
        x_ins = (x - ins_mean) / ins_std
        x_ln = (x - ln_mean) / ln_std

        x_hat = rho * x_ins + (1 - rho) * x_ln
        return x_hat * self.gamma + self.beta


class PixelNorm(nn.Module):
    def __init__(self, num_channels=None):
        super().__init__()
        # num_channels is only used to match function signature with other normalization layers
        # it has no actual use

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-5)


# Morph 中生成AdaLIn的组件
class MLP(nn.Module):
    def __init__(self, inc, dim, n_layers):
        super().__init__()
        ActFunc = nn.LeakyReLU(0.2)
        mlp = [PixelNorm(),
               nn.Linear(inc, dim),
               ActFunc,
               PixelNorm()]
        for i in range(n_layers - 2):
            mlp.extend([
                nn.Linear(dim, dim),
                ActFunc,
                PixelNorm()
            ])
        mlp.extend(
            [nn.Linear(dim, dim),
             PixelNorm()])
        self.dim = dim
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        x = x.view(b, c)
        x = self.mlp(x)
        return x


class STA(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.avg_p = nn.AdaptiveAvgPool2d(dim)  # 均值统计layer, 输出64*64
        # self.max_p = nn.AdaptiveMaxPool2d(dim)  # 最大值统计layer, 输出64*64

    def forward(self, x):
        # o1 = self.max_p(x)
        o2 = self.avg_p(x)
        # out = torch.cat((o1, o2), dim=1)
        # return out
        return o2


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class SkyLake_F(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=None, feat_nc=None):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(SkyLake_F, self).__init__()
        if gpu_ids is None:
            gpu_ids = []
        if feat_nc is None:
            feat_nc = [128, 256, 256, 256]
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

        if self.use_mlp:
            print("------------------create_mlp---------------------")
            for mlp_id, input_nc in enumerate(feat_nc):
                mlp_ = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
                setattr(self, 'mlp_%d' % mlp_id, mlp_)

    # def create_mlp(self, feats):
    #     print("------------------create_mlp---------------------")
    #     for mlp_id, feat in enumerate(feats):
    #         input_nc = feat.shape[1]
    #         mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
    #         if len(self.gpu_ids) > 0:
    #             mlp.cuda()
    #         setattr(self, 'mlp_%d' % mlp_id, mlp)
    #     init_net(self, self.init_type, self.init_gain, self.gpu_ids)
    #     self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        # if self.use_mlp and not self.mlp_init:
        #     self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # H, W flatten
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    '''
                    torch.randperm produces cudaErrorIllegalAddress for newer versions of PyTorch. 
                    https://github.com/taesungp/contrastive-unpaired-translation/issues/83
                    '''
                    # patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = np.random.permutation(feat_reshape.shape[1])  # random select
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class SkyLake_G(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=6, img_size=256, z_dim=32, lock_weight=False):
        print('------------------------SkyLake_G--------------------------')
        assert (n_blocks >= 0)
        super(SkyLake_G, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.z_dim = z_dim  # Use z to control the generated direction. The dimension of z is 32 * 2

        DownBlock = []
        # N = (W-F+2P)/S + 1, The dimension of the output dimension feature map remains unchanged, 256 * 256
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=True),
                      nn.InstanceNorm2d(ngf, affine=True),
                      nn.ReLU(True)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2 ** n_downsampling
        # 6 Attribute recalibration modules ARMResBlock #  [1, 256, 64, 64]
        for i in range(n_blocks):
            DownBlock += [ARMConvBlock(ngf * mult, ngf * mult)]

        # MLP，Convert z to hidden space and control generation
        self.mlp = MLP(self.z_dim * 2, 256, 8)
        # self.fc = nn.Conv2d(ngf * mult, self.z_dim * 2, 1, 1, 0)
        # self.avp = nn.AdaptiveAvgPool2d(1)

        # Up-Sampling Bottleneck
        adain_resblock = []
        for i in range(n_blocks):
            adain_resblock.append(AdaResBlk_Morph(ngf * mult))
        self.adain_resblocks = nn.ModuleList(adain_resblock)

        # Up-Sampling
        UpBlock = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            UpBlock += [nn.UpsamplingBilinear2d(scale_factor=2),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                        LIN_Morph(int(ngf * mult / 2)),
                        nn.ReLU()]
            mult = mult // 2
            UpBlock += [nn.ReflectionPad2d(1),
                        nn.Conv2d(ngf * mult, ngf * mult, 3, 1, 0),
                        LIN_Morph(ngf * mult),
                        nn.ReLU()]

        UpBlock += [nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                    nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)
        # Fix the encoder trunk when using features to rebuild constraints
        if lock_weight:
            print("Continue train on cyc feature map, lock the DownBlock weight")
            for p in self.DownBlock.parameters():
                p.requires_grad = False
        self.UpBlock = nn.Sequential(*UpBlock)

    def forward(self, _input, z, only_decoder=False, layers=None):  # z determines the direction of generation
        if layers is None:
            layers = []
        feats = []
        if not only_decoder:  # 不是仅仅decoder特征
            if len(layers) > 0:
                e = _input
                for layer_id, layer in enumerate(self.DownBlock):
                    # print(layer_id, layer)
                    e = layer(e)
                    if layer_id in layers:
                        # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, e.size(1)))
                        feats.append(e)
                    else:
                        # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, e.size(1)))
                        pass
                    # if layer_id == layers[-1]:
                    # print("%d: break the output of %s %d" % (layer_id, layer.__class__.__name__, e.size(1)))
                    # return feats
                return feats, e
            else:
                e = self.DownBlock(_input)  # Attribute Feature Map after DownSampling
        else:
            e = _input
        x = e
        # b, c, h, w = e.shape
        # e_sta = self.avp(x)
        # e_sta = self.fc(e_sta)
        # e_sta = e_sta.view(b, -1)
        # print("z.shape: ", z.shape, " e_out.shape:", e_out.shape)
        '''cat change start'''
        # z = torch.cat((e_sta, z), dim=1)
        # print("z.shape is :", z.shape)
        '''cat change end'''
        z = self.mlp(z)  # [batch, 256]
        # print(z.shape)
        for i in range(self.n_blocks):
            x = self.adain_resblocks[i](x, z)
        out = self.UpBlock(x)

        return out, e


##################################################################################
# Encoder and Decoders
##################################################################################

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class CAM(nn.Module):  # class attention map
    def __init__(self, dim, is_G=False):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 均值池化
        self.gmp = nn.AdaptiveMaxPool2d(1)  # 最大池化
        self.weight = nn.Parameter(torch.FloatTensor(dim, 1))  # 用Parameter表示权重
        nn.init.xavier_uniform_(self.weight)  # 初始化权重大小
        self.cam_bias = nn.Parameter(torch.FloatTensor(1))  # bias是只有一个维度的标量
        self.cam_bias.data.fill_(0)  # 初始化bias为0
        if is_G is False:
            self.conv1x1 = nn.Sequential(  # 将维度降低成原来的一半的1*1卷积
                nn.Conv2d(2 * dim, dim, 1, 1),
                nn.LeakyReLU(0.2),
            )
        else:
            self.conv1x1 = nn.Sequential(  # 将维度降低成原来的一半的1*1卷积
                nn.Conv2d(2 * dim, dim, 1, 1),
                nn.ReLU(),
            )

    def forward(self, e):
        b, c, h, w = e.shape
        gap = self.gap(e).view(b, c)
        gmp = self.gmp(e).view(b, c)

        x_a = torch.matmul(gap, self.weight) + self.cam_bias  # for classfication loss
        x_m = torch.matmul(gmp, self.weight) + self.cam_bias

        x_gap = e * (self.weight + self.cam_bias).view(1, c, 1, 1)
        x_gmp = e * (self.weight + self.cam_bias).view(1, c, 1, 1)

        x = torch.cat((x_gap, x_gmp), dim=1)
        x = self.conv1x1(x)
        x_class = torch.cat((x_a, x_m), dim=1)  # b, 2
        return x, x_class


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class SkyLake_D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=7):  # 7-Layer Multiscale Discriminator
        super(SkyLake_D, self).__init__()
        self.input_nc = input_nc
        if self.input_nc == 1:
            print('Use the gray image for train discriminator')
        # ------------------------------------------------------- Dis_0 Local
        Dis_0 = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                     nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 4):  # 1->(7-4), 3 layers
            mult = 2 ** (i - 1)
            Dis_0 += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                          nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 4 - 1)  # mult == 4

        Dis_0 += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        mult = mult * 2  # Passing through a convolution layer, number of channels*2
        self.conv0 = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))  # Dis_0

        # ------------------------------------------------------- Dis_1  Local

        Dis_1 = []
        for i in range(n_layers - 3, n_layers - 2):  # 3->(7-2), 2 layers
            mult = 2 ** (i - 1)
            Dis_1 += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                          nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)  # 2**4

        Dis_1 += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2)
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        # Class Activation Map

        self.cam = nn.utils.spectral_norm(CAM(mult * ndf))

        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))
        self.Dis_0 = nn.Sequential(*Dis_0)
        self.Dis_1 = nn.Sequential(*Dis_1)

    def forward(self, _input):

        # print(self.Dis_0)
        # print(self.Dis_1)

        if self.input_nc == 1:  # change the RGB image to gray image
            r, g, b = _input.unbind(dim=-3)
            _input = (0.2989 * r + 0.587 * g + 0.114 * b).to(_input.dtype)
            _input = _input.unsqueeze(dim=-3)

        x_0 = self.Dis_0(_input)  # 5 layers local Dis
        x_0 = self.pad(x_0)  # [1, 512, 33, 33]
        out_0 = self.conv0(x_0)
        # print(x_0.shape)
        x_1 = self.Dis_1(x_0)  # 5+2 layers local Dis
        # print(x_1.shape)
        x_1 = self.pad(x_1)  # [1, 2048, 17, 17]
        out_1 = self.conv1(x_1)

        x = x_1  # Copy an X and use it to multiply cam

        # print(x.shape)  # when n_layer=5: (1, 512, 31, 31)
        #                 # when n_layer=7: (1, 2048, 17, 17)

        x, cam_logit = self.cam(x)

        heatmap = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv2(x)

        return out, cam_logit, heatmap, out_0, out_1


# Definition of Generator
def define_G(input_nc=3, output_nc=3, ngf=64, netG='acg', norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, gpu_ids=[], lock_weight=False):
    net = SkyLake_G(lock_weight=lock_weight)
    # else:
    #     raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


# Definition of Discriminator
def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    # norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'multi':  # default PatchGAN classifier
        net = SkyLake_D(input_nc)
    elif netD == 'gray':  # use a gray image for Dis
        net = SkyLake_D(input_nc=1)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


# Definition of F, Reference code from CUT
def define_F(input_nc, netF, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, no_antialias=False,
             gpu_ids=[], opt=None, feat_nc=None):
    if netF == 'sample':
        net = SkyLake_F(use_mlp=False, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc,
                        feat_nc=feat_nc)
    elif netF == 'mlp_sample':
        net = SkyLake_F(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc,
                        feat_nc=feat_nc)

    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)
    return init_net(net, init_type, init_gain, gpu_ids)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))  # 给0和1打real和fake的标签
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)  # 展开成为生成器输出张量的size

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        return loss


class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w


# class StyleEncoder(nn.Module):
#     def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, vae=False):
#         super(StyleEncoder, self).__init__()
#         self.vae = vae
#         self.model = []
#         self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type='reflect')]
#         for i in range(2):
#             self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
#             dim *= 2
#         for i in range(n_downsample - 2):
#             self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
#         self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
#         if self.vae:
#             self.fc_mean = nn.Linear(dim, style_dim)  # , 1, 1, 0)
#             self.fc_var = nn.Linear(dim, style_dim)  # , 1, 1, 0)
#         else:
#             self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
#
#         self.model = nn.Sequential(*self.model)
#         self.output_dim = dim
#
#     def forward(self, x):
#         if self.vae:
#             output = self.model(x)
#             output = output.view(x.size(0), -1)
#             output_mean = self.fc_mean(output)
#             output_var = self.fc_var(output)
#             return output_mean, output_var
#         else:
#             output = self.model(x).view(x.size(0), -1)
#             print(output.shape)
#             return output


'''
net_G test
'''
# netG = SkyLake_G()
# print('---------- Networks initialized -------------')
#
#
# num_params = 0
# for param in netG.parameters():
#     num_params += param.numel()
#
# print('[Network %s] Total number of parameters : %.3f M' % ("netG", num_params / 1e6))
# print('-----------------------------------------------')
#
# netD = SkyLake_D(input_nc=3)
# print('---------- Networks initialized -------------')
#
#
# num_params = 0
# for param in netD.parameters():
#     num_params += param.numel()
#
# print('[Network %s] Total number of parameters : %.3f M' % ("netG", num_params / 1e6))
# print('-----------------------------------------------')

#
# input_ = torch.randn(4, 3, 256, 256)
# z_ = torch.randn(4, 64)
# netG(input_, z_)
#
# # for name, param in netG.state_dict(keep_vars=True).items():
# #     print(name, param.requires_grad)

'''
net_D test
'''

'''
net_F test
'''

# netF = SkyLake_F(use_mlp=True)
#
# init_net(netF, init_type='normal', init_gain=0.02, gpu_ids=[])
#
# # input_feats = []
# #
# # input_feat = torch.randn(4, 128, 64, 64)
# # input_feat_1 = torch.randn(4, 256, 256, 256)
# # input_feat_2 = torch.randn(4, 256, 256, 256)
# #
# # input_feats.append(input_feat)
# # input_feats.append(input_feat_1)
# # input_feats.append(input_feat_2)
# #
# # netF(input_feats)
#
# for name, param in netF.state_dict(keep_vars=True).items():
#     print(name, param.requires_grad)


'''
torch mean test
'''
# input_image = torch.randn(4, 3, 256, 256)
# output_image = torch.mean(input_image, dim=1, keepdim=True)
# print(output_image)

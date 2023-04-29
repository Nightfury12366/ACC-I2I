import torch
import itertools

from torch import nn

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .networks import RhoClipper


class ACGI2ISCModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=16.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=16.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='weight for identity')
            parser.add_argument('--lambda_attribute', type=float, default=0.0, help='')
            parser.add_argument('--cyc_layers', type=str, default='0,12,17', help='compute NCE loss on which layers')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'attri_A', 'idt_A']
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')

        self.visual_names = visual_names_A  # combine visualizations for A and B
        self.z_dim = 32
        if self.isTrain:
            self.model_names = ['G', 'D_A']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        if self.isTrain:
            self.cyc_layers = [int(i) for i in self.opt.cyc_layers.split(',')]

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.lock_weight)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.sta = networks.STA(32)
            self.sta.to(self.device)
            torch.nn.DataParallel(self.sta, self.opt.gpu_ids)

            # self.e_sta = networks.STA(1)
            # self.e_sta.to(self.device)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionAttri = torch.nn.L1Loss()  # l1 loss on the map

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # self.Rho_clipper = RhoClipper(0, 1)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        b, c, h, w = self.real_A.shape
        if AtoB:
            # 前z_dim个元素+=1, 意味着向B域变化
            z_ab = torch.randn(b, self.z_dim * 2) * 0.2
            z_ab[:, :self.z_dim] += 1

            z_bb = torch.randn(b, self.z_dim * 2) * 0.2
            z_bb[:, :self.z_dim] += 1

            z_aba = torch.randn(b, self.z_dim * 2) * 0.2
            z_aba[:, self.z_dim:] += 1
        else:
            # 后z_dim个元素+=1, 意味着向A域变化
            z_ab = torch.randn(b, self.z_dim * 2) * 0.2
            z_ab[:, self.z_dim:] += 1

            z_bb = torch.randn(b, self.z_dim * 2) * 0.2
            z_bb[:, self.z_dim:] += 1

            z_aba = torch.randn(b, self.z_dim * 2) * 0.2
            z_aba[:, :self.z_dim] += 1

        self.z_ab = z_ab.to(self.device)
        self.z_aba = z_aba.to(self.device)
        self.z_bb = z_bb.to(self.device)

    def forward(self):
        self.fake_B, self.attri_real_A = self.netG(self.real_A, self.z_ab)  # G_A(A)
        self.rec_A, self.attri_fake_B = self.netG(self.fake_B, self.z_aba)  # G_B(G_A(A))
        self.idt_B, _ = self.netG(self.real_B, self.z_bb)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real, cam_real, _, pred_real_D0, pred_real_D1 = netD(real)
        loss_D_real_C = self.criterionGAN(pred_real, True)  # 带有cam的
        loss_CAM_real = self.criterionGAN(cam_real, True)
        loss_D_real_0 = self.criterionGAN(pred_real_D0, True)
        loss_D_real_1 = self.criterionGAN(pred_real_D1, True)
        loss_D_real = loss_D_real_C + loss_CAM_real + loss_D_real_0 + loss_D_real_1

        # Fake
        pred_fake, cam_fake, _, pred_fake_D0, pred_fake_D1 = netD(fake.detach())
        loss_D_fake_C = self.criterionGAN(pred_fake, False)
        loss_CAM_fake = self.criterionGAN(cam_fake, False)
        loss_D_fake_0 = self.criterionGAN(pred_fake_D0, False)
        loss_D_fake_1 = self.criterionGAN(pred_fake_D1, False)
        loss_D_fake = loss_D_fake_C + loss_CAM_fake + loss_D_fake_0 + loss_D_fake_1
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_G(self):
        """Calculate the loss for generators G_A"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_attribute = self.opt.lambda_attribute
        # Identity loss
        if lambda_idt > 0:
            self.loss_idt_A = self.criterionIdt(self.idt_B, self.real_B) * lambda_B * lambda_idt
        else:
            self.loss_idt_A = 0

        # GAN loss D_A(G_A(A))
        loss_GA_cam, cam_GA, _, loss_GA_D0, pred_GA_D1 = self.netD_A(self.fake_B)

        self.loss_G_A = self.criterionGAN(loss_GA_cam, True) + self.criterionGAN(cam_GA, True) + self.criterionGAN(
            loss_GA_D0, True) + self.criterionGAN(pred_GA_D1, True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.sta(self.real_A), self.sta(self.rec_A)) * lambda_A

        # Cross-domain common attribute feature are similar
        if lambda_attribute > 0:
            # RealA and fakeA2B attribute feature are similar
            self.loss_attri_A = self.criterionAttri(self.attri_fake_B, self.attri_real_A) * lambda_attribute
        else:
            self.loss_attri_A = 0

        self.loss_G = self.loss_G_A + self.loss_cycle_A + self.loss_idt_A + self.loss_attri_A
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights
        # clip parameter of AdaILN and ILN, applied after optimizer step
        # self.netG.apply(self.Rho_clipper)

    def calculate_cyc_loss(self, src, tgt):
        n_layers = len(self.cyc_layers)
        feat_q = self.netG(tgt, z=None, only_decoder=False, layers=self.cyc_layers)
        feat_k = self.netG(src, z=None, only_decoder=False, layers=self.cyc_layers)

        total_cyc_loss = 0.0
        for f_q, f_k, nce_layer in zip(feat_q, feat_k, self.cyc_layers):
            loss = torch.nn.L1Loss()(f_q, f_k)
            total_cyc_loss += loss.mean()

        return total_cyc_loss / n_layers

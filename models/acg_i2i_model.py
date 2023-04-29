import torch
import itertools

from torch import nn

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .networks import RhoClipper


class ACGI2IModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='weight for identity')
            parser.add_argument('--lambda_attribute', type=float, default=0.4, help='')
            parser.add_argument('--cyc_layers', type=str, default='0,12,17', help='compute NCE loss on which layers')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'attri_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'attri_B']

        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        self.z_dim = 32
        if self.isTrain:
            self.model_names = ['G', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        if self.isTrain:
            self.cyc_layers = [int(i) for i in self.opt.cyc_layers.split(',')]

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.lock_weight)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionAttri = torch.nn.L1Loss()  # l1 loss on the map

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # self.Rho_clipper = RhoClipper(0, 1)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        pass

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        b, c, h, w = self.real_A.shape

        # 前z_dim个元素+=1, 意味着向B域变化
        z_ab = torch.randn(b, self.z_dim * 2) * 0.2
        z_ab[:, :self.z_dim] += 1

        # 后z_dim个元素+=1, 意味着向A域变化
        z_ba = torch.randn(b, self.z_dim * 2) * 0.2
        z_ba[:, self.z_dim:] += 1

        z_aba = torch.randn(b, self.z_dim * 2) * 0.2
        z_aba[:, self.z_dim:] += 1

        z_bab = torch.randn(b, self.z_dim * 2) * 0.2
        z_bab[:, :self.z_dim] += 1

        self.z_ab = z_ab.to(self.device)
        self.z_ba = z_ba.to(self.device)
        self.z_aba = z_aba.to(self.device)
        self.z_bab = z_bab.to(self.device)

    def forward(self):
        self.fake_B, self.attri_real_A = self.netG(self.real_A, self.z_ab)  # G_A(A)
        self.rec_A, self.attri_fake_B = self.netG(self.fake_B, self.z_aba)  # G_B(G_A(A))
        self.fake_A, self.attri_real_B = self.netG(self.real_B, self.z_ba)  # G_B(B)
        self.rec_B, self.attri_fake_A = self.netG(self.fake_A, self.z_bab)  # G_A(G_B(B))

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

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_attribute = self.opt.lambda_attribute
        # Identity loss
        if lambda_idt > 0:
            b, c, h, w = self.real_A.shape
            z_bb = torch.randn(b, self.z_dim * 2) * 0.2
            z_bb[:, :self.z_dim] += 1  # 前 z1
            z_aa = torch.randn(b, self.z_dim * 2) * 0.2
            z_aa[:, self.z_dim:] += 1  # 后 z2
            z_aa = z_aa.to(self.device)
            z_bb = z_bb.to(self.device)
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A, _ = self.netG(self.attri_real_B, z_bb, True)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B, _ = self.netG(self.attri_real_A, z_aa, True)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        loss_GA_cam, cam_GA, _, loss_GA_D0, pred_GA_D1 = self.netD_A(self.fake_B)

        self.loss_G_A = self.criterionGAN(loss_GA_cam, True) + self.criterionGAN(cam_GA, True) + self.criterionGAN(
            loss_GA_D0, True) + self.criterionGAN(pred_GA_D1, True)

        # GAN loss D_B(G_B(B))
        loss_GB_cam, cam_GB, _, loss_GB_D0, pred_GB_D1 = self.netD_B(self.fake_A)

        self.loss_G_B = self.criterionGAN(loss_GB_cam, True) + self.criterionGAN(cam_GB, True) + self.criterionGAN(
            loss_GB_D0, True) + self.criterionGAN(pred_GB_D1, True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.real_A, self.rec_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.real_B, self.rec_B) * lambda_B

        # Cross-domain common attribute feature are similar
        if lambda_attribute > 0:
            # RealA and fakeA2B attribute feature are similar
            self.loss_attri_A = self.criterionAttri(self.attri_fake_B,
                                                    self.attri_real_A) * lambda_attribute
            # RealB and fakeB2A attribute feature are similar
            self.loss_attri_B = self.criterionAttri(self.attri_fake_A,
                                                    self.attri_real_B) * lambda_attribute
        else:
            self.loss_attri_A = 0
            self.loss_attri_B = 0

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + \
                      self.loss_idt_B + self.loss_attri_A + self.loss_attri_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
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

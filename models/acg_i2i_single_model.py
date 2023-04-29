import torch
import itertools

from torch import nn

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
from .networks import RhoClipper


class ACGI2ISingleModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
            parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
            parser.add_argument('--nce_layers', type=str, default='5,9,12,17',
                                help='compute NCE loss on which layers')
            parser.add_argument('--idt_layers', type=str, default='0', help='compute NCE loss on which layers')
            parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                                help='how to downsample the feature map')
            parser.add_argument('--netF_nc', type=int, default=256)
            parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
            parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
            parser.add_argument('--lambda_A', type=float, default=5.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=5.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='weight for identity')
            parser.add_argument('--lambda_attribute', type=float, default=0.5, help='')

            # parser.set_defaults(pool_size=0)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['D_A', 'G_A', 'NCE', 'IDT']
        # self.loss_names = ['D_A', 'G_A', 'NCE']
        visual_names_A = ['real_A', 'fake_B', 'real_B']

        if self.isTrain:
            self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
            self.idt_layers = [int(i) for i in self.opt.idt_layers.split(',')]

        if self.isTrain:
            visual_names_A.append('idt_B')

        self.visual_names = visual_names_A  # combine visualizations for A and B
        self.z_dim = 32
        if self.isTrain:
            self.model_names = ['G', 'D_A', 'F', 'F_I']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators and F
            self.netF = networks.define_F(opt.input_nc, opt.netF, opt.norm, not opt.no_dropout, opt.init_type,
                                          opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netF_I = networks.define_F(opt.input_nc, opt.netF, opt.norm, not opt.no_dropout, opt.init_type,
                                            opt.init_gain, opt.no_antialias, self.gpu_ids, opt, feat_nc=[3])
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionNCE = []
            self.criterionIDT = []

            self.criterion_attri = torch.nn.L1Loss()
            self.criterion_idt = torch.nn.L1Loss()
            self.sta = networks.STA(32)
            self.e_sta = networks.STA(1)
            self.e_sta.to(self.device)
            self.sta.to(self.device)
            for _ in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_F)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # self.Rho_clipper = RhoClipper(0, 1)

    # def data_dependent_initialize(self, data):
    #     """
    #     The feature network netF is defined in terms of the shape of the intermediate, extracted
    #     features of the encoder portion of netG. Because of this, the weights of netF are
    #     initialized at the first feedforward pass with some input images.
    #     Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
    #     """
    #     bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
    #
    #     self.set_input(data)
    #     self.real_A = self.real_A[:bs_per_gpu]
    #     self.real_B = self.real_B[:bs_per_gpu]
    #     self.z_ab = self.z_ab[:bs_per_gpu]
    #     self.z_bb = self.z_bb[:bs_per_gpu]
    #
    #     self.forward()  # compute fake images: G(A)
    #     if self.opt.isTrain:
    #         self.backward_D_A()  # calculate gradients for D
    #         self.backward_G()  # calculate graidents for G
    #         if self.opt.lambda_NCE > 0.0:
    #             self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr,
    #                                                 betas=(self.opt.beta1, self.opt.beta2))
    #             self.optimizers.append(self.optimizer_F)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        b, c, h, w = self.real_A.shape

        # 前z_dim个元素+=1, 意味着向B域变化
        if AtoB:
            z_ab = torch.randn(b, self.z_dim * 2) * 0.2
            z_ab[:, :self.z_dim] += 1

            z_bb = torch.randn(b, self.z_dim * 2) * 0.2
            z_bb[:, :self.z_dim] += 1
        else:
            z_ab = torch.randn(b, self.z_dim * 2) * 0.2
            z_ab[:, self.z_dim:] += 1

            z_bb = torch.randn(b, self.z_dim * 2) * 0.2
            z_bb[:, self.z_dim:] += 1

        self.z_ab = z_ab.to(self.device)
        self.z_bb = z_bb.to(self.device)

    def forward(self):
        self.fake_B, _ = self.netG(self.real_A, self.z_ab)  # G_A(A)
        self.idt_B, _ = self.netG(self.real_B, self.z_bb)  # G_A(B)

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
        """Calculate the loss for generators G_A and G_B"""
        # GAN loss D_A(G_A(A))

        loss_GA_cam, cam_GA, _, loss_GA_D0, pred_GA_D1 = self.netD_A(self.fake_B)

        self.loss_G_A = self.criterionGAN(loss_GA_cam, True) + self.criterionGAN(cam_GA, True) + self.criterionGAN(
            loss_GA_D0, True) + self.criterionGAN(pred_GA_D1, True)

        self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)

        self.loss_IDT = self.criterion_idt(self.sta(self.real_B), self.sta(self.idt_B))

        loss_NCE_both = self.loss_NCE + self.loss_IDT

        self.loss_G = self.loss_G_A + loss_NCE_both
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.

        # D_A and D_B
        self.set_requires_grad([self.netD_A], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights

        # G_A and G_B
        self.set_requires_grad([self.netD_A], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        # F mlp_sample
        self.optimizer_F.zero_grad()
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

        self.optimizer_F.step()
        # clip parameter of AdaILN and ILN, applied after optimizer step
        # self.netG.apply(self.Rho_clipper)

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q, e1 = self.netG(tgt, z=None, only_decoder=False, layers=self.nce_layers)
        feat_k, e2 = self.netG(src, z=None, only_decoder=False, layers=self.nce_layers)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        attribute_loss = 0.0
        if self.opt.lambda_attribute > 0:
            attribute_loss = self.criterion_attri(self.e_sta(e1), self.e_sta(e2)) * self.opt.lambda_attribute

        return (total_nce_loss / n_layers) * 0.7 + attribute_loss

    def calculate_IDT_loss(self, src, tgt):
        n_layers = len(self.idt_layers)
        feat_q, e1 = self.netG(tgt, z=None, only_decoder=False, layers=self.idt_layers)
        feat_k, e2 = self.netG(src, z=None, only_decoder=False, layers=self.idt_layers)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        attribute_loss = 0.0
        if self.opt.lambda_attribute > 0:
            attribute_loss = self.criterion_attri(self.e_sta(e1), self.e_sta(e2)) * self.opt.lambda_attribute

        return (total_nce_loss / n_layers) * 0.7 + attribute_loss
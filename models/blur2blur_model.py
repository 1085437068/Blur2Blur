import torch
import torch.nn.functional as F
import yaml
from models.explore.kernel_encoding.kernel_wizard import KernelWizard
from . import networks
from .base_model import BaseModel
from .losses import GANLoss, VGGLoss, cal_gradient_penalty



class Blur2BlurModel(BaseModel):
    """This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
    """
    '''实现了一个基于 Pix2Pix 架构的无监督去模糊模型。该模型使用生成对抗网络（GAN）来学习从输入图像到输出图像的映射，特别适用于去模糊任务。以下是对文件内容的详细解析'''
    '''继承自：BaseModel  功能：实现了一个无监督去模糊模型，包括生成器（Generator）和判别器（Discriminator）'''

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        功能：添加新的数据集特定选项，并重写现有选项的默认值。

        Parameters:
            parser          -- original option parser 原始选项解析器
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options. 是否处于训练阶段

        Returns:
            the modified parser.修改后的解析器

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        parser.set_defaults(norm="batch", netG="unet_256")
        if is_train:
            parser.set_defaults(pool_size=0)  #图像缓冲池大小为0
            parser.add_argument("--lambda_Perc", type=float, default=0.8, help="weight for Perc loss") #添加感知损失--生成器
            parser.add_argument("--lambda_gp", type=float, default=0.0001, help="weight for GP loss")  #添加梯度惩罚权重--判别器

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.初始化类

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions 存储所有实验标志的选项类
        """
        BaseModel.__init__(self, opt)
        # breakpoint()
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses> 定义训练损失名称
        self.loss_names = ["G_GAN", "G_Perc", "D_real", "D_fake"]

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ["G", "D"] #定义需要保存到磁盘的模型名称
            self.visual_names = ["real_A", "blur_known", "sharp_known", "fake_B_"] #定义可视化名称
        else:  # during test time, only load G
            self.model_names = ["G"]
            self.visual_names = ["real_A", "fake_B_"]
        self.opt = opt

        # define networks (both generator and discriminator) 根据选项定义生成器和判别器
        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )
        self.upscale = torch.nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.device = torch.device("cuda:{}".format(self.gpu_ids[0])) if self.gpu_ids else torch.device("cpu")
        if self.isTrain: 
            # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc
            self.netD = networks.define_D(
                opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids
            )

            # define loss functions
            self.criterionGAN = GANLoss(opt.gan_mode).to(self.device) #GAN损失
            self.criterionPerc = VGGLoss().to(self.device) #感知损失

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>. 初始化优化器
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.netD.to(self.device)
            self.netD = torch.nn.DataParallel(self.netD)
        #打开模型配置
        with open("options/generate_blur/augmentation.yml", "r") as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)["KernelWizard"]
            model_path = opt["pretrained"]
        self.genblur = KernelWizard(opt)  #调用去模糊网络
        print("Loading KernelWizard...")
        self.genblur.eval()
        self.genblur.load_state_dict(torch.load(model_path))
        self.genblur = self.genblur.to(self.device)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps. 从数据加载器解包输入数据并执行必要的预处理步骤。

        Parameters:
            input (dict): include the data itself and its metadata information. nput，包含数据本身及其元数据信息。

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.sizeA = input["sizeA"]
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

        if self.isTrain:
            self.real_B = input["B" if AtoB else "A"].to(self.device)

            self.blur_known = input["C"].to(self.device)
            self.sharp_known = input["D"].to(self.device)

            kernel_mean, kernel_sigma = self.genblur(self.sharp_known, self.blur_known) #根据sharp_known和blur_known生成模糊参数
            self.kernel_real = kernel_mean + kernel_sigma * torch.randn_like(kernel_mean) #生成模糊核
            self.real_B = self.genblur.adaptKernel(self.real_B, self.kernel_real) #根据模糊核真实图片进行模糊

    def deblurring_step(self, x):
        nbatch = x.shape[0]
        chunk_size = 4
        outs = []
        with torch.no_grad():
            for idx in range(0, nbatch, chunk_size):
                pred = self.deblur(x[idx : idx + chunk_size])
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
        return torch.cat(outs, dim=0).to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>.运行前向传播；由 optimize_parameters 和 test 函数调用"""
        self.fake_B = self.netG(self.real_A)  # G(A)生成假图像fake_B
        # self.fake_B_ = self.fake_B[2]
        # 打印输出类型和形状以便调试
        #print(f"Generator output type: {type(self.fake_B)}")
        if isinstance(self.fake_B, torch.Tensor):
            #print(f"Shape of self.fake_B: {self.fake_B.shape}")

            # 上采样 fake_B 到不同分辨率
            fake_B_128 = F.interpolate(self.fake_B, size=(128, 128), mode="bilinear", align_corners=False)
            fake_B_64 = F.interpolate(self.fake_B, size=(64, 64), mode="bilinear", align_corners=False)

            # 组合成多尺度输出
            self.fake_B = [fake_B_64, fake_B_128, self.fake_B]
        elif isinstance(self.fake_B, list) and all(isinstance(item, torch.Tensor) for item in self.fake_B):
            #print(f"Generator output is a list of tensors: {[item.shape for item in self.fake_B]}")
        
            # 假设 self.fake_B 已经是多尺度输出，直接使用
            if len(self.fake_B) != 3:
                raise ValueError("Expected self.fake_B to have exactly 3 elements (multi-scale outputs)")
        
            # 如果需要进一步处理，可以在这里进行
            self.fake_B_ = self.fake_B[2]  # 取出最高分辨率的输出[1,3,256,256]
        else:
            raise ValueError("Unexpected output format from generator")
        
        self.fake_B_ = self.fake_B[2] # 取出最高分辨率的输出[1,3,256,256]

    def backward_D(self, iters):
        """Calculate GAN loss for the discriminator 计算判别器的 GAN 损失"""

        # Fake; stop backprop to the generator by detaching fake_B
        fake_B = [x.detach() for x in self.fake_B]  #从生成器生成的假样本 self.fake_B 中分离出 fake_B，使用 detach() 停止梯度传播到生成器
        pred_fake = self.netD(fake_B) #使用判别器 netD 预测 fake_B
        self.loss_D_fake = self.criterionGAN(iters, pred_fake, False, dis_update=True) #并计算假样本的损失 loss_D_fake

        # Real 对真实样本 self.real_B 进行多尺度插值，生成 real_B0 和 real_B1，并与原图一起组成 real_B
        real_B0 = F.interpolate(self.real_B, scale_factor=0.25, mode="bilinear")
        real_B1 = F.interpolate(self.real_B, scale_factor=0.5, mode="bilinear")
        real_B = [real_B0, real_B1, self.real_B]
        # 检查 real_B 和 fake_B 是否为列表并且长度至少为 3
        if len(real_B) < 3 or len(fake_B) < 3:
            raise ValueError("real_B or fake_B does not have enough elements for gradient penalty calculation")
        pred_real = self.netD(real_B) #使用判别器 netD 预测 real_B

        self.loss_D_real = self.criterionGAN(0, pred_real, True, dis_update=True) #并计算真实样本的损失 loss_D_real

        # combine loss and calculate gradients 将假样本和真实样本的损失组合，并加上梯度惩罚项，计算总损失 loss_D
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5  #GAN损失
        self.loss_D += cal_gradient_penalty(self.netD, real_B[2], fake_B[2], self.real_B.device, self.opt.lambda_gp)[0] #GAN损失 + 梯度惩罚正则化

        self.loss_D.backward()

    def backward_G(self, iters):
        """Calculate GAN and L1 loss for the generator用于计算生成器的损失并进行反向传播"""
        # First, G(A) should fake the discriminator
        pred_fake = self.netD(self.fake_B) #将生成的假图像输入判别器，得到判别器的预测值
        self.loss_G_GAN = self.criterionGAN(0, pred_fake, True, dis_update=False) #计算GAN损失

        # Second, G(A) = A
        real_A0 = F.interpolate(self.real_A, scale_factor=0.25, mode="bilinear")  #对真实图像进行 0.25 倍插值
        real_A1 = F.interpolate(self.real_A, scale_factor=0.5, mode="bilinear") #对真实图像进行 0.5 倍插值
        perc1 = self.criterionPerc.forward(self.fake_B[0], real_A0) #计算第一个尺度的感知损失
        perc2 = self.criterionPerc.forward(self.fake_B[1], real_A1) #计算第二个尺度的感知损失
        perc3 = self.criterionPerc.forward(self.fake_B[2], self.real_A) #计算第三个尺度的感知损失
        self.loss_G_Perc = (perc1 + perc2 + perc3) * self.opt.lambda_Perc #将三个尺度的感知损失相加，并乘以权重 lambda_Perc

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_Perc #将 GAN 损失和感知损失相加，得到总损失
        self.loss_G.backward() #调用 backward 方法进行反向传播，计算梯度

    def optimize_parameters(self, iters):
        '''用于优化生成器和判别器的参数'''
        self.forward()  # compute fake images: G(A)调用 self.forward() 计算生成的假图像

        # update D_kernel
        self.set_requires_grad(self.netD, True)  # enable backprop for D 启用判别器的梯度计算
        self.optimizer_D.zero_grad()  # set D's gradients to zero 清零判别器的梯度
        self.backward_D(iters)  # calculate gradients for D 计算判别器的梯度
        self.optimizer_D.step()  # update D's weights 更新判别器的权重

        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G 禁用判别器的梯度计算
        self.optimizer_G.zero_grad()  # set G's gradients to zero 清零生成器的梯度
        self.backward_G(iters)  # calculate graidents for G 计算生成器的梯度
        self.optimizer_G.step()  # update G's weights 更新生成器的权重

import argparse
import os

import data
import models
import torch
from util import util


class BaseOptions:
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        ## basic parameters 基础参数
        #--dataroot: 数据集的路径，通常包含训练和验证数据的子文件夹。
        parser.add_argument(
            "--dataroot", required=True, help="path to images (should have subfolders trainA, trainB, valA, valB, etc)"
        )
        #--name: 实验名称，决定存储样本和模型的位置。
        parser.add_argument(
            "--name",
            type=str,
            default="experiment_name",
            help="name of the experiment. It decides where to store samples and models",
        )
        # --gpu_ids: 使用的GPU ID，例如 0 或 0,1,2。使用 -1 表示使用CPU。
        parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
        # -checkpoints_dir: 模型保存的目录。
        parser.add_argument("--checkpoints_dir", type=str, default="./ckpts", help="models are saved here")
        
        ## model parameters 模型参数
        # --model: 选择使用的模型，例如 cycle_gan、pix2pix、test、colorization。
        parser.add_argument(
            "--model",
            type=str,
            default="cycle_gan",
            help="chooses which model to use. [cycle_gan | pix2pix | test | colorization]",
        )
        # --input_nc: 输入图像的通道数，3表示RGB，1表示灰度图
        parser.add_argument(
            "--input_nc", type=int, default=3, help="# of input image channels: 3 for RGB and 1 for grayscale"
        )
        # --output_nc: 输出图像的通道数，3表示RGB，1表示灰度图
        parser.add_argument(
            "--output_nc", type=int, default=3, help="# of output image channels: 3 for RGB and 1 for grayscale"
        )
        # --ngf: 生成器最后一层卷积层的滤波器数量
        parser.add_argument("--ngf", type=int, default=64, help="# of gen filters in the last conv layer")
        # --ndf: 判别器第一层卷积层的滤波器数量
        parser.add_argument("--ndf", type=int, default=64, help="# of discrim filters in the first conv layer")
        # --netD: 判别器架构，例如 basic、n_layers、pixel
        parser.add_argument(
            "--netD",
            type=str,
            default="basic",
            help="specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator",
        )
        #--num_Ds: 判别器的数量
        parser.add_argument("--num_Ds", type=int, default=2, help="number of Discrminators")
        # --netG: 生成器架构，例如 resnet_9blocks、resnet_6blocks、unet_256、unet_128
        parser.add_argument(
            "--netG",
            type=str,
            default="resnet_9blocks",
            help="specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]",
        )
        # --n_layers_D: 仅在 netD == n_layers 时使用，指定判别器的层数  
        parser.add_argument("--n_layers_D", type=int, default=3, help="only used if netD==n_layers")
        # --nz: 隐变量的维度
        parser.add_argument("--nz", type=int, default=512, help="#latent vector")
        # --norm: 归一化方法，例如 instance、batch、none
        parser.add_argument(
            "--norm",
            type=str,
            default="instance",
            help="instance normalization or batch normalization [instance | batch | none]",
        )
        # --init_type: 网络初始化方法，例如 normal、xavier、kaiming、orthogonal
        parser.add_argument(
            "--init_type",
            type=str,
            default="normal",
            help="network initialization [normal | xavier | kaiming | orthogonal]",
        )
        #--init_gain: 初始化缩放因子
        parser.add_argument(
            "--init_gain", type=float, default=0.02, help="scaling factor for normal, xavier and orthogonal."
        )
        #--no_dropout: 是否在生成器中使用dropout
        parser.add_argument("--no_dropout", action="store_true", help="no dropout for the generator")
        #--nl: 非线性激活函数，例如 relu、lrelu、elu
        parser.add_argument("--nl", type=str, default="relu", help="non-linearity activation: relu | lrelu | elu")
        # dataset parameters 数据集参数
        # --dataset_mode: 数据集加载方式，例如 unaligned、aligned、single、colorization
        parser.add_argument(
            "--dataset_mode",
            type=str,
            default="unaligned",
            help="chooses how datasets are loaded. [unaligned | aligned | single | colorization]",
        )
        # --direction: 数据转换方向，例如 AtoB 或 BtoA
        parser.add_argument("--direction", type=str, default="AtoB", help="AtoB or BtoA")
        #--serial_batches: 是否按顺序加载数据，否则随机加载
        parser.add_argument(
            "--serial_batches",
            action="store_true",
            help="if true, takes images in order to make batches, otherwise takes them randomly",
        )
        # --num_threads: 加载数据的线程数。
        
        parser.add_argument("--num_threads", default=4, type=int, help="# threads for loading data")
        # --batch_size: 输入批次大小。
        parser.add_argument("--batch_size", type=int, default=1, help="input batch size")
        # --load_size: 图像缩放大小。
        parser.add_argument("--load_size", type=int, default=256, help="scale images to this size")  #设置为256输出图片不会裁剪
        # --crop_size: 裁剪后的图像大小。
        parser.add_argument("--crop_size", type=int, default=256, help="then crop to this size")
        # --max_dataset_size: 允许的最大样本数。
        parser.add_argument(
            "--max_dataset_size",
            type=int,
            default=float("inf"),
            help="Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.",
        )
        # --preprocessA: A数据集的预处理方式，例如 resize_and_crop、crop、scale_width、scale_width_and_crop、none。
        parser.add_argument(
            "--preprocessA",
            type=str,
            default="resize_and_crop",
            help="scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]",
        )
        # --preprocessB: B数据集的预处理方式，同上。
        parser.add_argument(
            "--preprocessB",
            type=str,
            default="resize_and_crop",
            help="scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]",
        )
        # --no_flip: 是否进行数据增强翻转。
        parser.add_argument(
            "--no_flip", action="store_true", help="if specified, do not flip the images for data augmentation"
        )
        # --display_winsize: 可视化窗口大小
        parser.add_argument(
            "--display_winsize", type=int, default=500, help="display window size for both visdom and HTML"
        )
        # --ratio: 模糊与清晰图像的比例。
        parser.add_argument("--ratio", type=float, default=0.6, help="ratio of Blur:Sharp")

        ## additional parameters 其它参数
        #--where_add: 在网络G中添加z的位置，例如 input、all、middle
        parser.add_argument(
            "--where_add", type=str, default="all", help="input|all|middle; where to add z in the network G"
        )
        #--epoch: 加载的模型epoch，latest 表示加载最新的缓存模型
        parser.add_argument(
            "--epoch", type=str, default="latest", help="which epoch to load? set to latest to use latest cached model"
        )
        #--load_iter: 加载的迭代次数，大于0时加载特定迭代次数的模型
        parser.add_argument(
            "--load_iter",
            type=int,
            default="0",
            help="which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]",
        )
        #--verbose: 是否打印更多调试信息
        parser.add_argument("--verbose", action="store_true", help="if specified, print more debugging information")
        #--suffix: 自定义后缀，用于生成实验名称
        parser.add_argument(
            "--suffix",
            default="",
            type=str,
            help="customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}",
        )
        # wandb parameters
        #--use_wandb: 是否使用Wandb进行日志记录。
        # --use_tensorboard: 是否使用TensorBoard进行日志记录。
        # --wandb_project_name: Wandb项目的名称。
        parser.add_argument("--use_wandb", action="store_true", help="if specified, then init wandb logging")
        parser.add_argument("--use_tensorboard", action="store_true", help="if specified, then init wandb logging")
        parser.add_argument("--wandb_project_name", type=str, default="Blur2Blur", help="specify wandb project name")
        self.initialized = True

        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        # breakpoint()
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        # breakpoint()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, "{}_opt.txt".format(opt.phase))
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test
        opt.TestD = self.TestD

        # process opt.suffix
        if opt.suffix:
            suffix = ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt

        return self.opt

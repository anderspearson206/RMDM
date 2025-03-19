

import argparse
import os
from ssl import OP_NO_TLSv1
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.custom_dataset_loader import CustomDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from RadioUNet.lib import loaders, modules
import torchvision.transforms as transforms
from torchsummary import summary
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
import torch.nn as nn
from tqdm import tqdm
criterion = nn.MSELoss()
nmse = []
def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


import torch
import torchvision.transforms.functional as F
from skimage.metrics import structural_similarity as ssim_skimage
import numpy as np


def calculate_ssim(ss, m):
    """
    计算结构相似性指数（SSIM）

    参数:
    ss (torch.Tensor): 预测图片的张量，已经在CPU上，维度格式如 [batch_size, channels, height, width]
    m (torch.Tensor): 真实图片的张量，已经在CPU上，维度格式同预测图片

    返回:
    float: 计算得到的SSIM值
    """
    if ss.size()!= m.size():
        raise ValueError("预测图片和真实图片尺寸不一致")
    
    # 将张量转为numpy数组，并调整维度顺序以及数据类型（符合scikit-image要求）
    ss_np = ss[0][0].numpy()
    m_np = m[0][0].numpy()
    
    # 灰度化处理（如果是彩色图像，SSIM计算通常可以先转为灰度图，这里假设是单通道图像可省略这步）
    # 如果是彩色图像，可以使用例如下面的方式灰度化（示例是RGB转灰度常用算法）
    # ss_gray = np.dot(ss_np[...,:3], [0.2989, 0.5870, 0.1140])
    # m_gray = np.dot(m_np[...,:3], [0.2989, 0.5870, 0.1140])
    
    # 使用scikit-image的ssim函数计算SSIM值
    ssim_value = ssim_skimage(ss_np, m_np, data_range=m_np.max() - m_np.min())
    return ssim_value

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)
    id = 0

    

    if args.data_name not in ['Radio', 'Radio_2', 'Radio_3']:
        args.data_name = 'Radio'
        ds = loaders.RadioUNet_c(phase="test")
        args.in_ch = 3


    elif args.data_name == 'Radio_2':
        
        
        ds = loaders.RadioUNet_s(phase="test", carsSimul="yes", carsInput="yes")
        args.in_ch = 5

    elif args.data_name == 'Radio_3':


        ds = loaders.RadioUNet_s(phase="test", simulation="rand", cityMap="missing", missing=4,dir_dataset="/home/user/dxc/motion/MedSegDiff/RadioUNet/RadioMapSeer/")
        args.in_ch = 4
    else:
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor()]
        transform_test = transforms.Compose(tran_list)

        ds = CustomDataset(args, args.data_dir, transform_test, mode = 'Test')
        args.in_ch = 4

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []


    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    for b,m,path in tqdm(DataLoader(ds,batch_size=1, shuffle=True, num_workers=1)):
        #b, m, path = next(data)  #should return an image from the dataloader "data"
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)     #add a noise channel$

        x_t = img.clone().float()
        img[:, 0,...] = x_t[:, 0,...] + 10*x_t[:, 1,...]
        if args.data_name == 'ISIC':
            slice_ID=path[0].split("_")[-1].split('.')[0]
        elif args.data_name == 'BRATS':
            # slice_ID=path[0].split("_")[2] + "_" + path[0].split("_")[4]
            slice_ID=path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []
        
        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, cal, cal_out = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                step = args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            co = th.tensor(cal_out)
            if args.version == 'new':
                enslist.append(sample[:,-1,:,:])
            else:
                enslist.append(co)

            if args.debug:
                # print('sample size is',sample.size())
                # print('org size is',org.size())
                # print('cal size is',cal.size())
                if args.data_name == 'ISIC':
                    # s = th.tensor(sample)[:,-1,:,:].unsqueeze(1).repeat(1, 3, 1, 1)
                    o = th.tensor(org)[:,:-1,:,:]
                    c = th.tensor(cal).repeat(1, 3, 1, 1)
                    # co = co.repeat(1, 3, 1, 1)

                    s = sample[:,-1,:,:]
                    b,h,w = s.size()
                    ss = s.clone()
                    ss = ss.view(s.size(0), -1)
                    ss -= ss.min(1, keepdim=True)[0]
                    ss /= ss.max(1, keepdim=True)[0]
                    ss = ss.view(b, h, w)
                    ss = ss.unsqueeze(1).repeat(1, 3, 1, 1)

                    tup = (ss,o,c)
                elif args.data_name == 'BRATS':
                    s = th.tensor(sample)[:,-1,:,:].unsqueeze(1)
                    m = th.tensor(m.to(device = 'cuda:0'))[:,0,:,:].unsqueeze(1)
                    o1 = th.tensor(org)[:,0,:,:].unsqueeze(1)
                    o2 = th.tensor(org)[:,1,:,:].unsqueeze(1)
                    o3 = th.tensor(org)[:,2,:,:].unsqueeze(1)
                    o4 = th.tensor(org)[:,3,:,:].unsqueeze(1)
                    c = th.tensor(cal)

                    tup = (o1/o1.max(),o2/o2.max(),o3/o3.max(),o4/o4.max(),m,s,c,co)

                else:
                    ss = sample[0][0]
                     # 假设ss经过操作后得到了numpy数组data
                    data = ss.cpu().detach().numpy()

                    # 使用savez函数存储为.npz文件
                    np.savez(f'/home/user/dxc/motion/MedSegDiff/results/results_3/combined_{id}.npz', data = data)
                    fig, axs = plt.subplots(1, 3)  # 创建1行3列的子图布局

                    # 绘制第一张图
                    axs[0].imshow(ss.cpu().detach().numpy())
                    axs[0].set_title(f'NMSE={round(float(criterion(ss.cpu(), m.cpu()) / criterion(m, 0 * m)),4)}', fontsize=10, color='black', fontweight='bold')
                    
                    axs[0].axis('off')
                   

                    # 绘制第二张图
                    axs[1].imshow(cal[0][0].cpu().detach().numpy())
                    axs[1].set_title(f'NMSE={round(float(criterion(cal.cpu(), m.cpu()) / criterion(m, 0 * m)),4)}', fontsize=10, color='black', fontweight='bold')
                    
                    axs[1].axis('off')
                    

                    # 绘制第三张图
                    axs[2].imshow(m[0][0].cpu().detach().numpy())
                    axs[2].set_title('Ground Truth', fontsize=10, color='black', fontweight='bold')
                    
                    axs[2].axis('off')
                    print("cal",criterion(cal.cpu(), m[0][0].cpu()) / criterion(m[0][0], 0 * m[0][0]))
                    print("pre",criterion(ss.cpu(), m[0][0].cpu()) / criterion(m[0][0], 0 * m[0][0]))
                    nmse.append(float((criterion(ss.cpu(), m.cpu()) / criterion(m, 0 * m)).cpu().detach().numpy()))
                    print('nmse is ',np.array(nmse).mean())
                    print('pre ssim is ',calculate_ssim(sample.cpu(), m.cpu()))
                    print('cal ssim is ',calculate_ssim(cal.cpu(), m.cpu()))
                    
                    # 调整布局，让图片显示更合理
                    plt.tight_layout()
                    # 保存组合后的图片，可以指定合适的路径和文件名，这里示例为与前面同路径下的combined_{id}.png
                    plt.savefig(f'/home/user/dxc/motion/MedSegDiff/results/results_3/combined_{id}.png', dpi = 300)
                    id = id + 1
        #         compose = th.cat(tup,0)
        #         vutils.save_image(compose, fp = os.path.join(args.out_dir, str(slice_ID)+'_output'+str(i)+".jpg"), nrow = 1, padding = 10)
        # ensres = staple(th.stack(enslist,dim=0)).squeeze(0)
        # vutils.save_image(ensres, fp = os.path.join(args.out_dir, str(slice_ID)+'_output_ens'+".jpg"), nrow = 1, padding = 10)

def create_argparser():
    defaults = dict(
        data_name = 'BRATS',
        data_dir="../dataset/brats2020/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",         #path to pretrain model
        num_ensemble=1,      #number of samples in the ensemble
        gpu_dev = "0",
        out_dir='./results/',
        multi_gpu = None, #"0,1,2"
        debug = False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()

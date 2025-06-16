import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
from yaml import load

from basicsr.utils import img2tensor, tensor2img, imwrite
from basicsr.archs.femasr_arch import FeMaSRNet
from basicsr.utils.download_util import load_file_from_url

from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
import numpy as np
import torch
import cv2
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
import numpy as np
import torch
import cv2
import logging
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
from basicsr.data import create_dataloader, create_dataset

import torch

_ = torch.manual_seed(123)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')

# from skimage.metrics import structural_similarity as ssim_gray
# from skimage.metrics import peak_signal_noise_ratio as psnr_gray

from comput_psnr_ssim import calculate_ssim as ssim_gray
from comput_psnr_ssim import calculate_psnr as psnr_gray

# def ssim_gray(imgA, imgB, gray_scale=True):
#     if gray_scale:
#         score, diff = ssim(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY), full=True,
#                            multichannel=False)
#     # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
#     else:
#         score, diff = ssim(imgA, imgB, full=True, multichannel=True)
#     return score
#
#
# def psnr_gray(imgA, imgB, gray_scale=True):
#     if gray_scale:
#         psnr_val = psnr(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY))
#         return psnr_val
#     else:
#         psnr_val = psnr(imgA, imgB)
#         return psnr_val


pretrain_model_url = {
    'x4': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX4_model_g.pth',
    'x2': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX2_model_g.pth',
}

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def equalize_hist_color(img):
    # 使用 cv2.split() 分割 BGR 图像
    channels = cv2.split(img)
    eq_channels = []
    # 将 cv2.equalizeHist() 函数应用于每个通道
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch))
    # 使用 cv2.merge() 合并所有结果通道
    eq_image = cv2.merge(eq_channels)
    return eq_image

    # def get_residue_structure_mean(self, tensor, r_dim=1):
    #     max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  # keepdim
    #     min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    #     res_channel = (max_channel[0] - min_channel[0])
    #     mean = torch.mean(tensor, dim=r_dim, keepdim=True)
    #
    #     device = mean.device
    #     res_channel = res_channel / torch.max(mean, torch.full(size=mean.size(), fill_value=0.000001).to(device))
    #     return res_channel

def get_residue_structure_mean(tensor, r_dim=1):
    max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  # keepdim
    min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    res_channel = (max_channel[0] - min_channel[0])
    mean = torch.mean(tensor, dim=r_dim, keepdim=True)
    device = mean.device
    res_channel = res_channel / torch.max(mean, torch.full(size=mean.size(), fill_value=0.000001).to(device))
    return res_channel
import torch.nn.functional as F
def check_image_size(x,window_size=128):
    _, _, h, w = x.size()
    mod_pad_h = (window_size  - h % (window_size)) % (
                window_size )
    mod_pad_w = (window_size  - w % (window_size)) % (
                window_size)
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    # print('F.pad(x, (0, mod_pad_w, 0, mod_pad_h)', x.size())
    return x

def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import numpy as np
def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }
    if img_np is None:
        raise Exception('None .. !!!')
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder, self.lq_folder, self.lq_mask_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_lq_mask']
        self.filename_tmpl = '{}'

        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.lq_mask_folder, self.gt_folder], ['lq', 'lq_mask', 'gt'],
            self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')

        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        lq_mask_path = self.paths[index]['lq_mask_path']

        img_bytes = self.file_client.get(lq_mask_path, 'lq_mask')

        try:
            img_lq_mask = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))


        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_lq_mask = img2tensor([img_gt, img_lq, img_lq_mask],
                                    bgr2rgb=True,
                                    float32=True)

        return {
            'lq': img_lq,
            'lq_mask': img_lq_mask,
            'gt': img_gt,
            'lq_path': lq_path,
            'lq_mask_path': lq_mask_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)



from sklearn.metrics import mean_squared_error

from PIL import Image # No need for ImageChops
import math
from skimage import img_as_float
from skimage.metrics import mean_squared_error as mse

def rmsdiff(im1, im2):
    """Calculates the root mean square error (RSME) between two images"""
    return math.sqrt(mse(img_as_float(im1), img_as_float(im2)))

from skimage.color import rgb2lab
def calc_RMSE(real_img, fake_img):
    # convert to LAB color space
    real_lab = rgb2lab(real_img)
    fake_lab = rgb2lab(fake_img)
    return real_lab - fake_lab


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            # image_numpy = image_numpy.convert('L')
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return np.clip(image_numpy, 0, 255).astype(imtype)

def calculate_rmse(recovered, gt):
    # Transform into lab color space
    recovered_lab = rgb2lab(recovered)
    gt_lab = rgb2lab(gt)

    return abs((gt_lab - recovered_lab)).sum()

def main():
    """Inference demo for FeMaSR
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input', type=str, default='/data_8T1/wangcong/dataset/Rain13K/rain13ktest/Rain100H/input',
    #                     help='Input image or folder')
    # parser.add_argument('-g', '--gt', type=str, default='/data_8T1/wangcong/dataset/Rain13K/rain13ktest/Rain100H/target',
    #                     help='groundtruth image')
    # parser.add_argument('-i', '--input', type=str,
    #                     default='/data_8T1/wangcong/dataset/real-world-images/real-input',
    #                     help='Input image or folder')
    # parser.add_argument('-g', '--gt', type=str,
    #                     default='/data_8T1/wangcong/dataset/real-world-images/real-input',
    #                     help='groundtruth image')
    parser.add_argument('-i', '--input', type=str,
                        default='/data_8T1/wangcong/dataset/ISTD_Dataset/test/test_A',
                        help='Input image or folder')
    parser.add_argument('-i_mask', '--input_mask', type=str,
                        default='/data_8T1/wangcong/dataset/ISTD_Dataset/test/test_A_segment_convert',
                        help='Input image or folder')
    parser.add_argument('-g', '--gt', type=str,
                        default='/data_8T1/wangcong/dataset/ISTD_Dataset/test/test_C',
                        help='groundtruth image')
    # parser.add_argument('-i', '--input', type=str,
    #                     default='/data_8T1/wangcong/dataset/Deblur_dataset/GoPro_patch/train/input_crops',
    #                     help='Input image or folder')
    # parser.add_argument('-i_mask', '--input_mask', type=str,
    #                     default='/data_8T1/wangcong/dataset/Deblur_dataset/GoPro_patch/train/input_crops_segment',
    #                     help='Input image or folder')
    # parser.add_argument('-g', '--gt', type=str,
    #                     default='/data_8T1/wangcong/dataset/Deblur_dataset/GoPro_patch/train/target_crops',
    #                     help='groundtruth image')
    # parser.add_argument('-w_vqgan', '--weight_vqgan', type=str,
    #                     default='/data_8T1/wangcong/net_g_260000.pth',
    #                     help='path for model weights')
    parser.add_argument('-w', '--weight', type=str,
                        default='./experiments/014_FeMaSR_LQ_stage/models/net_g_300000.pth',
                        help='path for model weights')
    parser.add_argument('-o', '--output', type=str, default='results/100l', help='Output folder')
    parser.add_argument('-s', '--out_scale', type=int, default=1, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=600,
                        help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if args.weight is None:
    #     weight_path_vqgan = load_file_from_url(pretrain_model_url[f'x{args.out_scale}'])
    # else:
    #     weight_path_vqgan = args.weight_vqgan
    enhance_weight_path = args.weight
    # print('weight_path', weight_path_vqgan)
    # set up the model
    # VQGAN = FeMaSRNet(codebook_params=[[16, 1024, 256], [32, 1024, 128], [64, 1024, 64], [128, 1024, 32]], LQ_stage=False, scale_factor=args.out_scale).to(device)
    # VQGAN.load_state_dict(torch.load(weight_path_vqgan)['params'], strict=False)
    # VQGAN.eval()

    EnhanceNet = FeMaSRNet(out_list_block=[2, 2, 5],
                 in_list_block=[2, 3, 4],
                 list_heads=[1, 2, 4],
                 ffn_mask=True,
                 fusion_in_self_attention=True,
                 attention_mask=True,
                 all_cross_attention=True,
                 all_self_attention=False,
                 all_cross_self_attention=False,
                 num_refinement=4,
                 num_heads_refinement=2,
                 ffn_expansion_factor=3,
                 bias=True,
                 LayerNorm_type='WithBias').to(device)
    EnhanceNet.load_state_dict(torch.load(enhance_weight_path)['params'], strict=False)
    EnhanceNet.eval()
    print_network(EnhanceNet)
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
    ssim_all = 0
    psnr_all = 0
    rmse_all = 0
    lpips_all = 0
    num_img = 0
    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)
        pbar.set_description(f'Test {img_name}')

        gt_path = args.gt
        file_name = path.split('/')[-1]

        gt_img = cv2.imread(os.path.join(gt_path, file_name), cv2.IMREAD_UNCHANGED)

        # print(gt_img)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_tensor = img2tensor(img).to(device) / 255.
        img_tensor = img_tensor.unsqueeze(0)
        b, c, h, w = img_tensor.size()
        img_tensor = check_image_size(img_tensor)

        # lq_mask_img = cv2.imread(os.path.join(lq_mask_path, file_name), cv2.IMREAD_UNCHANGED)
        lq_mask_path = args.input_mask
        file_name = file_name#[:-4] + '.png'
        lq_mask_img = cv2.imread(os.path.join(lq_mask_path, file_name))
        # lq_mask_file_name = os.path.join(lq_mask_path, file_name)
        # lq_mask_img = cv2.imread(os.path.join(lq_mask_path, file_name))
        print('lq_mask_img', lq_mask_img.shape)
        lq_mask_tensor = img2tensor(lq_mask_img).to(device) / 255.
        lq_mask_tensor = lq_mask_tensor.unsqueeze(0)
        b, c, h, w = lq_mask_tensor.size()
        lq_mask_tensor = check_image_size(lq_mask_tensor)

        print('lq_mask_tensor', lq_mask_tensor.size())

        with torch.no_grad():
            import time
            t0 = time.time()
            print('lq_mask_tensor',lq_mask_tensor.size())
            output = EnhanceNet(img_tensor, lq_mask_tensor)
            t1 = time.time()
            print('time:', t1-t0)
        output = output

        output = output[:, :, :h, :w]
        output_img = tensor2img(output)
        lq_mask_tensor = lq_mask_tensor[:, :, :h, :w]
        output_mask = tensor2img(lq_mask_tensor)
        gray = True
        # ssim = ssim_gray(output_img, gt_img, gray_scale=gray)
        # psnr = psnr_gray(output_img, gt_img, gray_scale=gray)
        ssim = ssim_gray(output_img, gt_img)
        psnr = psnr_gray(output_img, gt_img)
        diff = calc_RMSE(output_img, gt_img)
        rmse = np.sqrt(np.power(diff, 2).mean(axis=(0, 1))).sum()
        print('rmse', rmse)
        # rmse = np.abs(
        #     cv2.cvtColor(output_img, cv2.COLOR_RGB2LAB) - cv2.cvtColor(gt_img, cv2.COLOR_RGB2LAB)).mean() * 3
        lpips_value = lpips(2 * torch.clip(img2tensor(output_img).unsqueeze(0) / 255.0, 0, 1) - 1,
                            2 * img2tensor(gt_img).unsqueeze(0) / 255.0 - 1)
        ssim_all += ssim
        psnr_all += psnr
        rmse_all += rmse
        lpips_all += lpips_value
        num_img += 1
        print('num_img', num_img)
        print('ssim', ssim)
        print('psnr', psnr)
        print('rmse_all', rmse_all)
        print('lpips_value', lpips_value)
        save_path = os.path.join(args.output, f'{img_name}')
        # save_path_first = os.path.join(args.output + 'first/', f'{img_name}')
        imwrite(output_img, save_path)

        save_path_mask = os.path.join(args.output, 'mask/', f'{img_name}')
        imwrite(output_mask, save_path_mask)

        pbar.update(1)
    pbar.close()
    print('avg_ssim:%f' % (ssim_all / num_img))
    print('avg_psnr:%f' % (psnr_all / num_img))
    print('avg_rmse:%f' % (rmse_all / num_img))
    print('avg_lpips:%f' % (lpips_all / num_img))


if __name__ == '__main__':
    main()

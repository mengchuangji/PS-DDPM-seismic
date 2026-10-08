import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from torchvision.utils import make_grid, save_image

from models.diffusion import Model
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from functions.denoising import efficient_generalized_steps
from functions.denoising import dps_sampling, pseudoinverse_guidance_noisy,pseudoinverse_guidance_noisy_funcs,pseudoinverse_guidance_noiseless_funcs,pseudoinverse_guidance_inpainting

import torchvision.utils as tvu

from guided_diffusion.unet import UNetModel
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random
import scipy.io as sio
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math


def plot_std(img,dpi,figsize):
    import matplotlib.pyplot as plt
    plt.figure(dpi=dpi, figsize=figsize)
    plt.imshow(img, vmin=0, vmax=0.1, cmap=plt.cm.Greys)
    # plt.title('fake_noisy')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    plt.tight_layout()
    plt.show()
def plot_cmap(img,dpi,figsize,data_range,cmap,cbar=False):

    plt.figure(dpi=dpi, figsize=figsize)
    plt.imshow(img, vmin=data_range[0], vmax=data_range[1], cmap=cmap)
    # plt.title('fake_noisy')
    if cbar:
        plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    plt.tight_layout()
    plt.show()

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "cosine":
        betas = betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif beta_schedule == "pow":
        betas = np.linspace(0, 1, num_diffusion_timesteps)
        betas = pow(betas, 3)
        betas = betas * (beta_end - beta_start) + beta_start
    elif beta_schedule == "pow4":
        betas = np.linspace(0, 1, num_diffusion_timesteps)
        betas = pow(betas, 4)
        betas = betas * (beta_end - beta_start) + beta_start
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self, obs, obs_GT):

        cls_fn = None
        if self.config.model.type == 'simple':
            # from ermongroup/ddim ermongroup/SDEdit
            model = Model(self.config)

            # from improved-diffusion guided_diffusion
            # model = create_model(
            #     image_size=self.config.data.image_size,
            #     num_channels=128,
            #     num_res_blocks=2,
            #     learn_sigma=True,
            #     class_cond=False,
            #     use_checkpoint=False,
            #     attention_resolutions="16,8",
            #     num_heads=4,
            #     num_heads_upsample=-1,
            #     use_scale_shift_norm=True,
            #     dropout=0.0,
            # )

            # This used the pretrained DDPM model
            if self.config.data.dataset == "marmousi":
                name = "marmousi"
            else:
                raise ValueError

            # model.load_state_dict(torch.load(os.path.join(self.args.log_path_model, f'model{self.config.sampling.ckpt_id}.pt'), map_location=self.device))
            # model.load_state_dict(
            #     torch.load(os.path.join(self.args.log_path_model, f'model{self.config.sampling.ckpt_id:06d}.pt'),
            #                map_location=self.device))

            model = torch.nn.DataParallel(model)
            states = torch.load(os.path.join(self.args.log_path_model, f'ckpt_{self.config.sampling.ckpt_id}.pth'),
                           map_location=self.device)
            model.load_state_dict(states[0], strict=False)


            model.to(self.device)
            # model = torch.nn.DataParallel(model)



        self.sample_sequence(model, obs, obs_GT,cls_fn=None)

    def sample_sequence(self, model, obs, obs_GT, cls_fn=None):

        args, config = self.args, self.config



        #get original images and corrupted y_0
        # dataset, test_dataset = get_dataset(args, config)



        print('processing single data')


        ## show stochastic variation mcj ##
        num_variations = self.args.num_variations
        stochastic_variations = torch.zeros((4 + num_variations) * self.config.sampling.batch_size,
                                            self.config.data.channels, self.config.data.image_shape[0], self.config.data.image_shape[1])
        stochastic_variations_R = torch.zeros((4 + num_variations) * self.config.sampling.batch_size,
                                              self.config.data.channels, self.config.data.image_shape[0],
                                              self.config.data.image_shape[1])


        
        def seed_worker(worker_id):
            worker_seed = args.seed % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        g = torch.Generator()
        g.manual_seed(args.seed)

        import time
        start_time = time.time()

        # ## get degradation matrix ##
        deg = args.deg
        from utils.get_degradation_matrix import get_degradation_matrix
        H_funcs = get_degradation_matrix(args, config)
        # H_funcs = None
        # if deg[:2] == 'cs':
        #     compress_by = int(deg[2:])
        #     from functions.svd_replacement import WalshHadamardCS
        #     H_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by, torch.randperm(self.config.data.image_size**2, device=self.device), self.device)
        # elif deg[:3] == 'inp':
        #     from functions.svd_replacement import Inpainting
        #     if deg == 'inp_lolcat':
        #         loaded = np.load("inp_masks/lolcat_extra.npy")
        #         mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
        #         missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        #     elif deg == 'inp_lorem':
        #         loaded = np.load("inp_masks/lorem3.npy")
        #         mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
        #         missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        #     else:
        #         missing_r = torch.randperm(config.data.image_size**2)[:config.data.image_size**2 // 2].to(self.device).long() * 3
        #     missing_g = missing_r + 1
        #     missing_b = missing_g + 1
        #     missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        #     H_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device)
        # elif deg == 'deno':
        #     from functions.svd_replacement import Denoising
        #     H_funcs = Denoising(config.data.channels, self.config.data.image_size, self.device)
        # elif deg[:10] == 'sr_bicubic':
        #     factor = int(deg[10:])
        #     from functions.svd_replacement import SRConv
        #     def bicubic_kernel(x, a=-0.5):
        #         if abs(x) <= 1:
        #             return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
        #         elif 1 < abs(x) and abs(x) < 2:
        #             return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
        #         else:
        #             return 0
        #     k = np.zeros((factor * 4))
        #     for i in range(factor * 4):
        #         x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
        #         k[i] = bicubic_kernel(x)
        #     k = k / np.sum(k)
        #     kernel = torch.from_numpy(k).float().to(self.device)
        #     H_funcs = SRConv(kernel / kernel.sum(), \
        #                      config.data.channels, self.config.data.image_size, self.device, stride = factor)
        # elif deg == 'deblur_uni':
        #     from functions.svd_replacement import Deblurring
        #     H_funcs = Deblurring(torch.Tensor([1/9] * 9).to(self.device), config.data.channels, self.config.data.image_size, self.device)
        # elif deg == 'deblur_gauss':
        #     from functions.svd_replacement import Deblurring
        #     sigma = 10
        #     pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
        #     kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
        #     H_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
        # elif deg == 'deblur_aniso':
        #     from functions.svd_replacement import Deblurring2D
        #     sigma = 20
        #     pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
        #     kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.device)
        #     sigma = 1
        #     pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
        #     kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.device)
        #     H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels, self.config.data.image_size, self.device)
        # elif deg[:2] == 'sr':
        #     blur_by = int(deg[2:])
        #     from functions.svd_replacement import SuperResolution
        #     H_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
        # elif deg == 'color':
        #     from functions.svd_replacement import Colorization
        #     H_funcs = Colorization(config.data.image_size, self.device)
        # else:
        #     print("ERROR: degradation type not supported")
        #     quit()

        # args.sigma_0 = 2 * args.sigma_0 #to account for scaling to [-1,1]
        sigma_0 = args.sigma_0


        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0

        # pbar = tqdm.tqdm(val_loader)
        # for x_orig, classes in pbar:

        classes = None
        x_orig= obs
        x_orig = x_orig.to(self.device)
        x_orig = data_transform(self.config, x_orig)

        # plot_cmap(x_orig.view(x_orig.shape[0], config.data.channels, self.config.data.image_shape[0],
        #                    self.config.data.image_shape[1]).cpu().squeeze().numpy(),
        #           300, (3.7, 3), data_range=[-1, 1], cmap=plt.cm.seismic, cbar=True)

        # y_0 = obs.to(self.device)
        # y_0 = obs.view(x_orig.shape[0], -1).to(self.device)


        # y_0 = H_funcs.H(x_orig)
        R = obs_GT.to(self.device)
        y_0 = H_funcs.H(R)

        # y_0 = y_0 + sigma_0* torch.randn_like(y_0)*abs(y_0).max() #sigma_0

        plot_cmap(y_0.view(y_0.shape[0], config.data.channels, self.config.data.image_shape[0],
                           -1).cpu().squeeze().numpy(),
                  300, (3.7, 3), data_range=[-1, 1], cmap=plt.cm.seismic, cbar=True)



        # plot_cmap(y_0.view(y_0.shape[0], config.data.channels, self.config.data.image_shape[0],
        #                    self.config.data.image_shape[1]).cpu().squeeze().numpy(),
        #           300, (3.7, 3), data_range=[-1, 1], cmap=plt.cm.seismic, cbar=True)

        # pinv_y_0=y_0.view(y_0.shape[0], config.data.channels, self.config.data.image_shape[0], self.config.data.image_shape[1])


        # if deg[:3] == 'inp':
        #     y_0=H_funcs.H(y_0)
        #     pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], config.data.channels, self.config.data.image_shape[0], self.config.data.image_shape[1])
        # else:
        if deg[:3] == 'SRI':
            pinv_y_0 = y_0.view(y_0.shape[0], config.data.channels, self.config.data.image_shape[0], self.config.data.image_shape[1])
        else:
            pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], config.data.channels, self.config.data.image_shape[0],
                                                self.config.data.image_shape[1])
        plot_cmap(pinv_y_0.cpu().squeeze().numpy(), 300, (3.0, 3), data_range=[-1, 1], cmap=plt.cm.seismic, cbar=False)

        if deg[:6] == 'deblur':
            pinv_y_0 = y_0.view(y_0.shape[0], config.data.channels, self.config.data.image_shape[0], self.config.data.image_shape[1])
        elif deg == 'color':
            pinv_y_0 = y_0.view(y_0.shape[0], 1, self.config.data.image_shape[0], self.config.data.image_shape[1]).repeat(1, 3,
                                                                                                                  1, 1)
        elif deg[:3] == 'inp':
            pinv_y_0 += H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_0))).reshape(*pinv_y_0.shape) - 1
        elif deg[:2] == 'cs':
            # 随机打乱索引
            perm = H_funcs.perm
            y_flat = obs.view(y_0.shape[0],
                                  config.data.channels * self.config.data.image_shape[0] * self.config.data.image_shape[
                                      1]).to(self.config.device)
            # 获取前 n 个元素
            n = self.config.data.channels * self.config.data.image_shape[0] * self.config.data.image_shape[
                1] // H_funcs.ratio
            sampled_elements = y_flat[:, perm[:n]]
            # 创建一个与原始矩阵相同形状的零矩阵
            y_sampled = torch.zeros_like(y_flat).to(self.config.device)
            # 将前 n 个元素放回矩阵中
            y_sampled[:, perm[:n]] = sampled_elements
            if config.data.seis_rescaled == True:
                y_sampled[:, perm[n:]] = 0.5
            else:
                y_sampled[:, perm[n:]] = 0
            pinv_y_0 = y_sampled.reshape(*obs.shape).to(self.config.device)

        # plot_cmap(pinv_y_0.cpu().squeeze().numpy(), 300, (3.0, 3), data_range=[-1, 1], cmap=plt.cm.seismic, cbar=False)

        # for i in range(len(pinv_y_0)):
        #     tvu.save_image(
        #         inverse_data_transform(config, pinv_y_0[i]),
        #         os.path.join(self.args.image_folder, f"y0_{idx_so_far + i}.png")
        #     )
        #     tvu.save_image(
        #         inverse_data_transform(config, x_orig[i]),
        #         os.path.join(self.args.image_folder, f"orig_{idx_so_far + i}.png")
        #     )
        # mcj
        sample_y_0_ = inverse_data_transform(self.config, pinv_y_0.cpu())
        stochastic_variations[0: self.config.sampling.batch_size, :, :, :] = inverse_data_transform(config, obs_GT)
        stochastic_variations[1 * self.config.sampling.batch_size: 2 * self.config.sampling.batch_size, :, :,
        :] = inverse_data_transform(config, pinv_y_0)
        stochastic_variations_R[0 * self.config.sampling.batch_size: 1 * self.config.sampling.batch_size, :, :,
        :] = 0

        ##Begin DDIM
        x = torch.randn(
            y_0.shape[0],
            config.data.channels,
            self.config.data.image_shape[0],
            self.config.data.image_shape[1],
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        # with torch.no_grad():
        #     x, _ = self.sample_image(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)
        #
        # x = [inverse_data_transform(config, y) for y in x]

        stochastic_variations_x_t = torch.zeros(
            ((1 + (args.timesteps + 1)) * num_variations) * self.config.sampling.batch_size,
            self.config.data.channels, self.config.data.image_shape[0], self.config.data.image_shape[1])

        # save x_0
        for i in range(num_variations):
            # # DDRM
            # with torch.no_grad():
            #     x_, _ = self.sample_image(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn,
            #                               classes=classes)

            # DPS guidance
            # x_ = self.DPS_sample_image(x, model, H_funcs, y_0, sigma_0, scale=0.0, scale_guidance=1, last=False,
            #                      cls_fn=None, classes=None)

            # pseudoinverse_guidance
            # x_ = self.pseudoinverse_guidance_sample_image(x, model, H_funcs, y_0, sigma_0, scale=0.4, scale_guidance=1,
            #                                               last=False, cls_fn=None, classes=None)

            # reddiff guidance and comparision method
            x_ = self.sample_image_(x, model, H_funcs, y_0, sigma_0, scale=1, scale_guidance=1, last=False, cls_fn=cls_fn,
                                      classes=classes)





            print("--- {:.2f} seconds ---".format(time.time() - start_time))

            x_ = [inverse_data_transform(config, y) for y in x_]
            stochastic_variations[
            (self.config.sampling.batch_size) * (i + 2): (self.config.sampling.batch_size) * (i + 3), :, :, :] = x_[-1]

            if deg=='deno':
                stochastic_variations_R[
                (self.config.sampling.batch_size) * (i + 2): (self.config.sampling.batch_size) * (i + 3), :, :,
                :] = sample_y_0_.cpu() - x_[-1]
            else:
                stochastic_variations_R[
                (self.config.sampling.batch_size) * (i + 2): (self.config.sampling.batch_size) * (i + 3), :, :,
                :] = inverse_data_transform(config, obs_GT).cpu() - x_[-1]



            for j, x_t in enumerate(x_[:]):
                x_t = x_t.view(x_[-1].shape[0], self.config.data.channels,
                               self.config.data.image_shape[0], self.config.data.image_shape[1]).to(self.config.device)
                if x_t.requires_grad:
                    x_t=x_t.detach()
                stochastic_variations_x_t[
                (self.config.sampling.batch_size) * (j + 1 + (1+len(x_[:])) * i): (self.config.sampling.batch_size) * ( j + 2 + (1+len(x_[:])) * i), :,:,:] \
                    = inverse_data_transform(self.config, x_t)
            stochastic_variations_x_t[i * (1 + (args.timesteps+1)): i * (1 + (args.timesteps+1)) + 1:, :, :] = inverse_data_transform(config, pinv_y_0)

        # calculate mean and std ##
        runs = stochastic_variations[
               (self.config.sampling.batch_size) * (2): (self.config.sampling.batch_size) * (
                       2 + num_variations), :, :, :]
        runs = runs.view(-1, self.config.sampling.batch_size, self.config.data.channels,
                         self.config.data.image_shape[0], self.config.data.image_shape[1])
        stochastic_variations[
        (self.config.sampling.batch_size) * (-2): (self.config.sampling.batch_size) * (-1), :, :,
        :] = torch.mean(runs, dim=0)
        stochastic_variations[(self.config.sampling.batch_size) * (-1):, :, :, :] = torch.std(runs, dim=0)
        if deg=='deno':
            stochastic_variations_R[(self.config.sampling.batch_size) * (-2): (self.config.sampling.batch_size) * (-1), :,
            :,
            :] = obs.cpu() - torch.mean(runs, dim=0)
        else:
            stochastic_variations_R[(self.config.sampling.batch_size) * (-2): (self.config.sampling.batch_size) * (-1),
            :,:,:] = inverse_data_transform(config, obs_GT).cpu() - torch.mean(runs, dim=0)
        stochastic_variations_R[(self.config.sampling.batch_size) * (-1):, :, :, :] = 0

        sio.savemat(os.path.join(self.args.image_folder, "results.mat"),
                    {'data': stochastic_variations.cpu().squeeze().numpy()})
        from utils.make_grid_h import make_grid_h
        image_grid = make_grid_h(stochastic_variations, self.config.sampling.batch_size)
        ######### plot stochastic_variations ###############
        # import matplotlib.pyplot as plt
        plt.gcf().set_size_inches(15, 10)
        # plt.gcf().set_size_inches(3*(4 + num_variations), 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
        plt.imshow(image_grid.numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.seismic, vmin=-1, vmax=1)
        # plt.colorbar()  # 添加色标
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(os.path.join(self.args.image_folder, 'stochastic_variation.png'), dpi=300, bbox_inches='tight')
        sio.savemat(os.path.join(self.args.image_folder, "results.mat"),
                    {'data': stochastic_variations.cpu().squeeze().numpy()})
        plot_cmap(((stochastic_variations.cpu()[2, :, :, :])).squeeze().numpy(), 300, (3.0, 3), data_range=[-1, 1],
                  cmap=plt.cm.seismic, cbar=False)
        plot_std(((stochastic_variations.cpu()[-1, :, :, :])).squeeze().numpy(), dpi=300, figsize=(3.7, 3))
        ######### plot stochastic_variations ###############

        ######### plot stochastic_variations_R (residual) ###############
        image_grid = make_grid_h(stochastic_variations_R, self.config.sampling.batch_size, padding=16)
        plt.gcf().set_size_inches(15, 10)
        # plt.gcf().set_size_inches(3*(4 + num_variations), 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
        plt.imshow(image_grid.numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.seismic, vmin=-1, vmax=1)
        # plt.colorbar()  # 添加色标
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(os.path.join(self.args.image_folder, 'stochastic_variation_R.png'), dpi=300, bbox_inches='tight')
        sio.savemat(os.path.join(self.args.image_folder, "results_residual.mat"),
                    {'data': stochastic_variations_R.cpu().squeeze().numpy()})
        plot_cmap(((stochastic_variations_R.cpu()[2, :, :, :])).squeeze().numpy(), 300, (3.0, 3), data_range=[-1, 1],
                  cmap=plt.cm.seismic, cbar=False)
        ######### plot stochastic_variations_R (residual) ###############


        ######### plot stochastic_variations_x_t ###############
        # Number of intermediate timesteps to display (in addition to first and last)
        num_middle_steps = 10
        # Total number of timesteps per sample: pinv_y_0 + x_t[0...T]
        total_steps = 1 + args.timesteps + 1  # Usually: timesteps + 2
        batch_size = config.sampling.batch_size
        # Total number of samples
        N_total = stochastic_variations_x_t.shape[0]
        num_samples = N_total // total_steps
        # Uniformly sample 10 timesteps from the middle (excluding first and last)
        middle_indices = np.linspace(1, total_steps - 2, num_middle_steps, dtype=int)
        # Final selected timestep indices: always include first and last
        final_indices = [0] + middle_indices.tolist() + [total_steps - 1]
        # Collect selected frames
        selected_frames = []
        for i in range(num_samples):
            for j in final_indices:
                idx = i * total_steps + j
                selected_frames.append(stochastic_variations_x_t[idx])
        # Stack selected frames into a single tensor: (num_samples * 12, C, H, W)
        selected_frames = torch.stack(selected_frames, dim=0)
        # Create image grid (12 columns: 1 + 10 + 1 timesteps)
        image_grid = make_grid(selected_frames, len(final_indices), padding=8)

        # image_grid = make_grid(stochastic_variations_x_t, 1 + (args.timesteps + 1), padding=4)
        plt.gcf().set_size_inches(15, 15)
        # plt.gcf().set_size_inches(3*(2 + len(x_t_list)*num_variations) , 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
        plt.imshow(image_grid.numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.seismic, vmin=-1, vmax=1)
        # plt.colorbar()  # 添加色标
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(os.path.join(self.args.image_folder, 'stochastic_variation_x_t.png'), dpi=300, bbox_inches='tight')
        sio.savemat(os.path.join(self.args.image_folder, "results_x_t.mat"),
                    {'data': stochastic_variations_x_t.cpu().squeeze().numpy()})
        ######### plot stochastic_variations_x_t ###############

        for i in [-1]:  # range(len(x)):
            for j in range(x_[i].size(0)):
                tvu.save_image(
                    x_[i][j], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{i}.png")
                )
                if i == len(x_) - 1 or i == -1:
                    orig = inverse_data_transform(config, x_orig[j])
                    mse = torch.mean((x_[i][j].to(self.device) - orig) ** 2)
                    psnr = 10 * torch.log10(1 / mse)
                    avg_psnr += psnr

        idx_so_far += y_0.shape[0]

        # pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))
        #
        # avg_psnr = avg_psnr / (idx_so_far - idx_init)
        # print("Total Average PSNR: %.2f" % avg_psnr)
        # print("Number of samples: %d" % (idx_so_far - idx_init))

        ## report PSNRs mcj##
        from utils.metric import batch_SNR, batch_PSNR, batch_SSIM

        clean = stochastic_variations[0 * self.config.sampling.batch_size: 1 * self.config.sampling.batch_size, :, :, :]
        obs = stochastic_variations[1 * self.config.sampling.batch_size: 2 * self.config.sampling.batch_size, :, :, :]

        mse_obs = torch.mean((obs - clean) ** 2)
        instance_mse_obs = ((obs - clean) ** 2).view(obs.shape[0], -1).mean(1)
        # psnr_obs = torch.mean(10 * torch.log10(1 / instance_mse_obs))
        # psnr_obs = 10 * torch.log10(1**2 / mse_obs)
        # print("MSE/PSNR of the observations %f, %f" % (mse_obs, psnr_obs))
        psnr_obs = batch_PSNR(obs, clean)
        ssim_obs = batch_SSIM(obs, clean)
        snr_obs = batch_SNR(obs, clean)
        print("MSE/PSNR/SNR/SSIM of the observations %f, %.2f,%.2f, %.4f" % (mse_obs, psnr_obs, snr_obs, ssim_obs))

        for i in range(num_variations):
            general = stochastic_variations[
                      (2 + i) * self.config.sampling.batch_size: (3 + i) * self.config.sampling.batch_size, :, :, :]
            mse = torch.mean((general - clean) ** 2)
            instance_mse = ((general - clean) ** 2).view(general.shape[0], -1).mean(1)
            # psnr = torch.mean(10 * torch.log10(1 / instance_mse))
            # print("MSE/PSNR of the general #%d: %f, %f" % (i, mse, psnr))
            psnr_obs = batch_PSNR(general, clean)
            ssim_obs = batch_SSIM(general, clean)
            snr_obs = batch_SNR(general, clean)
            print("MSE/PSNR/SNR/SSIM of the posterior sampling #%d: %f, %.2f, %.2f, %.4f" % (
            i, mse, psnr_obs, snr_obs, ssim_obs))

        mean = stochastic_variations[(2 + num_variations) * self.config.sampling.batch_size: (
                                                                                                         3 + num_variations) * self.config.sampling.batch_size,
               :, :, :]
        mse = torch.mean((mean - clean) ** 2)
        instance_mse = ((mean - clean) ** 2).view(mean.shape[0], -1).mean(1)
        psnr = torch.mean(10 * torch.log10(1 / instance_mse))
        print("PSNR of the observations %f" % (psnr))
        psnr_mean = batch_PSNR(mean, clean)
        ssim_mean = batch_SSIM(mean, clean)
        snr_mean = batch_SNR(mean, clean)
        print("MSE/PSNR/SNR/SSIM of the mean of posterior sampling:  %f, %.2f, %.2f, %.4f" % (
        mse, psnr_mean, snr_mean, ssim_mean))


    def sample_image(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x = efficient_generalized_steps(x, seq, model, self.betas, H_funcs, y_0, sigma_0, \
            etaB=self.args.etaB, etaA=self.args.eta, etaC=self.args.eta, cls_fn=cls_fn, classes=classes)
        if last:
            x = x[0][-1]
        return x

    def DPS_sample_image(self, x, model, H_funcs, y_0, sigma_0, scale, scale_guidance, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)

        x = dps_sampling(x, seq, model, self.betas, H_funcs, y_0, sigma_0, \
                                        scale=scale, scale_guidance=scale_guidance, cls_fn=cls_fn,
                                        classes=classes)
        if last:
            x = x[0][-1]
        return x

    def pseudoinverse_guidance_sample_image(self, x, model, H_funcs, y_0, sigma_0, scale, scale_guidance, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)

        # x = pseudoinverse_guidance_noisy_funcs(x, seq, model, self.betas, H_funcs, y_0, sigma_0, \
        #                                 scale=scale, scale_guidance=scale_guidance, cls_fn=cls_fn,
        #                                 classes=classes)
        # x = pseudoinverse_guidance_noiseless_funcs(x, seq, model, self.betas, H_funcs, y_0, sigma_0, \
        #                                        scale=scale, scale_guidance=scale_guidance, cls_fn=cls_fn,
        #                                        classes=classes)

        # define H_operator
        # H = torch.ones(x.shape, device=x.device)
        H = torch.zeros(x.shape, device=x.device)
        for i in range(128):
            H[:, :, i, i] = 1
        def linear_operator(x):
            return x
        x = pseudoinverse_guidance_noisy(x, seq, model, self.betas, H, y_0, sigma_0, scale, cls_fn=None, classes=None)


        # mask = torch.ones_like(x)
        # # prob = 0.75
        # # seed = 2025
        # # torch.manual_seed(seed)
        # # prob_tensor = torch.rand(mask.shape[1]) < prob
        # # mask[:, prob_tensor] = 0
        # x = pseudoinverse_guidance_inpainting(x, seq, model, self.betas, mask, H_funcs, y_0, sigma_0, scale, cls_fn=None, classes=None)

        if last:
            x = x[0][-1]
        return x


    def sample_image_(self, x, model, H_funcs, y_0, sigma_0, scale, scale_guidance, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)

        from omegaconf import OmegaConf

        # cfg_dict = {
        #     "algo": {
        #         "name": "dps",
        #         "deg": "SRI",
        #         "awd": True,
        #         "cond_awd": False,
        #         "grad_term_weight": 0.0,  # 0.1 for in2_20ff, and 1.0 for sr4 deon:1.5
        #         "sigma_y": 0.0,
        #         "eta": 0.0,
        #         "mcg": False,
        #         "original": True,
        #         "lr": 0.25,
        #     }
        # }
        #
        # cfg = OmegaConf.create(cfg_dict)
        # from algos.dps import DPS
        # sampler = DPS(model, cfg,  H_funcs, self.betas, scale)
        # x, _ = sampler.sample(x, y=None, ts=seq, y_0=y_0)

        # cfg_dict = {
        #     "algo": {
        #         "name": "ddrm",
        #         "sigma_y": float(sigma_0),
        #         "eta": 0.85,
        #         "eta_b": 1,
        #         "deg": "deno",
        #         "lr": 0.25,}
        # }
        # cfg = OmegaConf.create(cfg_dict)
        # from algos.ddrm import DDRM
        # sampler= DDRM(model, cfg,  H_funcs, self.betas)
        # x,_ = sampler.sample(x, y=None, ts=seq, y_0=y_0)

        cfg_dict = {
            "algo": {
                "name": "reddiff",
                "deg": "SRI", #deno
                "awd": True,
                "cond_awd": False,
                "obs_weight": 1, #1.0
                "grad_term_weight": 0.25, # 0.25
                "denoise_term_weight": "linear",
                # 可选: "linear", "sqrt", "log", "square", "trunc_linear", "const", "power2over3"
                "sigma_y": float(sigma_0),
                "eta": 0.8, #0.0
                "lr": 0.1,
                "sigma_x0": 0.00,#0.0
            },
            "exp": {
                "save_evolution": False
            }
        }
        cfg = OmegaConf.create(cfg_dict)
        from algos.reddiff import REDDIFF
        sampler = REDDIFF(model, cfg, H_funcs, self.betas)
        x, _ = sampler.sample(x, y=None, ts=seq, y_0=y_0)

        # cfg_dict = {
        #     "algo": {
        #         "name": "mcg",
        #         "deg": "deno",
        #         "grad_term_weight": 1,
        #         "sigma_y": 0.0,
        #         "eta": 0.0
        #     }
        # }
        # cfg = OmegaConf.create(cfg_dict)
        # from algos.mcg import MCG
        # sampler = MCG(model, cfg, H_funcs, self.betas)
        # x, _ = sampler.sample(x, y=None, ts=seq, y_0=y_0)

        if last:
            x = x[0][-1]
        return x



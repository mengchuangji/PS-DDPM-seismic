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

    def sample(self, obs):

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



        self.sample_sequence(model, obs,cls_fn)

    def sample_sequence(self, model, obs, cls_fn=None):

        args, config = self.args, self.config



        #get original images and corrupted y_0
        # dataset, test_dataset = get_dataset(args, config)



        print('processing single data')


        ## show stochastic variation mcj ##
        num_variations = self.args.num_variations
        stochastic_variations = torch.zeros((3 + num_variations) * self.config.sampling.batch_size,
                                            self.config.data.channels, self.config.data.image_shape[0], self.config.data.image_shape[1])
        stochastic_variations_R = torch.zeros((3 + num_variations) * self.config.sampling.batch_size,
                                              self.config.data.channels, self.config.data.image_shape[0],
                                              self.config.data.image_shape[1])
        stochastic_variations_LS = torch.zeros((3 + num_variations) * self.config.sampling.batch_size,
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
        args.obs=obs
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

        y_0 = obs
        y_0= data_transform(config, y_0)
        y_0 = H_funcs.H(y_0)



        # if deg[:3] == 'inp':
        #     y_0=H_funcs.H(y_0)
        #     pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], config.data.channels, self.config.data.image_size,
        #                                     self.config.data.image_size)
        # else:
        pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], config.data.channels, self.config.data.image_shape[0], self.config.data.image_shape[1])


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

        sample_y_0_ = inverse_data_transform(self.config, obs)

        for i in range(len(pinv_y_0)):
            tvu.save_image(
                inverse_data_transform(config, pinv_y_0[i]),
                os.path.join(self.args.image_folder, f"y0_{idx_so_far + i}.png")
            )
        # mcj

        stochastic_variations[0 * self.config.sampling.batch_size: 1 * self.config.sampling.batch_size, :, :,
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
            with torch.no_grad():
                x_, _ = self.sample_image(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn,
                                          classes=classes)
                print("--- {:.2f} seconds ---".format(time.time() - start_time))
            x_ = [inverse_data_transform(config, y) for y in x_]
            stochastic_variations[
            (self.config.sampling.batch_size) * (i + 1): (self.config.sampling.batch_size) * (i + 2), :, :, :] = x_[-1]
            stochastic_variations_R[
            (self.config.sampling.batch_size) * (i + 1): (self.config.sampling.batch_size) * (i + 2), :, :,
            :] = sample_y_0_.cpu() - x_[-1]

            # if deg == 'deno':
            #     from seis_utils.localsimi import localsimi
            #     LS=localsimi(x_[-1].squeeze().numpy(),
            #                  (sample_y_0_.cpu() - x_[-1]).squeeze().numpy(),
            #                  rect=[5, 5, 1], niter=20, eps=0.0, verb=1).squeeze()[np.newaxis,np.newaxis, :]
            #     energy_simi = np.sum(LS ** 1) / LS.size
            #     print("energy_simi=", energy_simi)
            #     LS=torch.from_numpy(LS).contiguous().type(torch.FloatTensor).to(sample_y_0_.device)
            #     stochastic_variations_LS[
            #     (self.config.sampling.batch_size) * (i + 1): (self.config.sampling.batch_size) * (i + 2), :, :,
            #     :] = LS

            for j, x_t in enumerate(x_[:]):
                x_t = x_t.view(x_[-1].shape[0], self.config.data.channels,
                               self.config.data.image_shape[0], self.config.data.image_shape[1]).to(self.config.device)
                stochastic_variations_x_t[
                (self.config.sampling.batch_size) * (j + 1 + (1+len(x_[:])) * i): (
                                                                                           self.config.sampling.batch_size) * (
                                                                                                   j + 2 + (1+len(
                                                                                               x_[:])) * i), :,
                :,
                :] = inverse_data_transform(self.config, x_t)
            stochastic_variations_x_t[i * (1 + (args.timesteps+1)): i * (1 + (args.timesteps+1)) + 1:, :, :] = inverse_data_transform(config, pinv_y_0)



        # calculate mean and std ##
        runs = stochastic_variations[
               (self.config.sampling.batch_size) * (1): (self.config.sampling.batch_size) * (
                       1 + num_variations), :, :, :]
        runs = runs.view(-1, self.config.sampling.batch_size, self.config.data.channels,
                         self.config.data.image_shape[0], self.config.data.image_shape[1])
        stochastic_variations[
        (self.config.sampling.batch_size) * (-2): (self.config.sampling.batch_size) * (-1), :, :,
        :] = torch.mean(runs, dim=0)
        stochastic_variations[(self.config.sampling.batch_size) * (-1):, :, :, :] = torch.std(runs, dim=0)
        stochastic_variations_R[(self.config.sampling.batch_size) * (-2): (self.config.sampling.batch_size) * (-1), :,
        :,
        :] = obs.cpu() - torch.mean(runs, dim=0)
        stochastic_variations_R[(self.config.sampling.batch_size) * (-1):, :, :, :] = 0

        sio.savemat(os.path.join(self.args.image_folder, "results.mat"),
                    {'data': stochastic_variations.cpu().squeeze().numpy()})
        from utils.make_grid_h import make_grid_h
        image_grid = make_grid_h(stochastic_variations, self.config.sampling.batch_size)

        # if deg == 'deno':
        #     from seis_utils.localsimi import localsimi
        #     LS = localsimi(torch.mean(runs, dim=0).cpu().squeeze().numpy(),
        #                    (sample_y_0_.cpu() - torch.mean(runs, dim=0)).cpu().squeeze().numpy(), rect=[5, 5, 1],
        #                    niter=20, eps=0.0, verb=1).squeeze()[np.newaxis, np.newaxis, :]
        #     energy_simi = np.sum(LS ** 2) / LS.size
        #     print("energy_simi=", energy_simi)
        #     LS = torch.from_numpy(LS).contiguous().type(torch.FloatTensor).to(sample_y_0_.device)
        #     stochastic_variations_LS[(self.config.sampling.batch_size) * (-2): (self.config.sampling.batch_size) * (-1), :,
        #     :,
        #     :] = LS
        #     stochastic_variations_LS[(self.config.sampling.batch_size) * (-1):, :, :, :] = 0
        plot_cmap(((stochastic_variations_LS.cpu()[1, :, :, :])).squeeze().numpy(), 300, (3.7, 3), data_range=[0, 1],
                  cmap=plt.cm.jet, cbar=True)


        ######### plot stochastic_variations ###############
        # import matplotlib.pyplot as plt
        plt.gcf().set_size_inches(10, 10)
        # plt.gcf().set_size_inches(3*(4 + num_variations), 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
        plt.imshow(image_grid.numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.seismic, vmin=-1, vmax=1)
        # plt.colorbar()  # 添加色标
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(os.path.join(self.args.image_folder, 'stochastic_variation.png'), dpi=300, bbox_inches='tight')
        sio.savemat(os.path.join(self.args.image_folder, "results.mat"),
                    {'data': stochastic_variations.cpu().squeeze().numpy()})
        plot_cmap(((stochastic_variations.cpu()[0, :, :, :])).squeeze().numpy(), 300, (3.0, 3), data_range=[-1, 1],
                  cmap=plt.cm.seismic, cbar=False)
        plot_cmap(((stochastic_variations.cpu()[1, :, :, :])).squeeze().numpy(), 300, (3.0, 3), data_range=[-1, 1], cmap=plt.cm.seismic, cbar=False)
        plot_std(((stochastic_variations.cpu()[-1, :, :, :])).squeeze().numpy(), dpi=300, figsize=(3.7, 3))
        ######### plot stochastic_variations ###############

        ######### plot stochastic_variations_R (residual) ###############
        image_grid = make_grid_h(stochastic_variations_R, self.config.sampling.batch_size, padding=8)
        plt.gcf().set_size_inches(10, 10)
        # plt.gcf().set_size_inches(3*(4 + num_variations), 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
        plt.imshow(image_grid.numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.seismic, vmin=-1, vmax=1)
        # plt.colorbar()  # 添加色标
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(os.path.join(self.args.image_folder, 'stochastic_variation_R.png'), dpi=300, bbox_inches='tight')
        sio.savemat(os.path.join(self.args.image_folder, "results_residual.mat"),
                    {'data': stochastic_variations_R.cpu().squeeze().numpy()})
        plot_cmap(((stochastic_variations_R.cpu()[1, :, :, :])).squeeze().numpy(), 300, (3.0, 3), data_range=[-1, 1],
                  cmap=plt.cm.seismic, cbar=False)

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
        plt.gcf().set_size_inches(10, 10)
        # plt.gcf().set_size_inches(3*(2 + len(x_t_list)*num_variations) , 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
        plt.imshow(image_grid.numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.seismic, vmin=-1, vmax=1)
        # plt.colorbar()  # 添加色标
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(os.path.join(self.args.image_folder, 'stochastic_variation_x_t.png'), dpi=300, bbox_inches='tight')
        sio.savemat(os.path.join(self.args.image_folder, "results_x_t.mat"),
                    {'data': stochastic_variations_x_t.cpu().squeeze().numpy()})
        ######### plot stochastic_variations_x_t ###############

        # for i in [-1]:  # range(len(x)):
        #     for j in range(x_[i].size(0)):
        #         tvu.save_image(
        #             x_[i][j], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{i}.png")
        #         )
        # idx_so_far += y_0.shape[0]




    def sample_image(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x = efficient_generalized_steps(x, seq, model, self.betas, H_funcs, y_0, sigma_0, \
            etaB=self.args.etaB, etaA=self.args.eta, etaC=self.args.eta, cls_fn=cls_fn, classes=classes)
        if last:
            x = x[0][-1]
        return x
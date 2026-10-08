import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb

from runners.diffusion_seis_GT import Diffusion

torch.set_printoptions(sci_mode=False)
import matplotlib.pyplot as plt

def plot(img,dpi,figsize):
    import matplotlib.pyplot as plt
    plt.figure(dpi=dpi, figsize=figsize)
    plt.imshow(img, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    # plt.title('fake_noisy')
    # plt.colorbar()
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

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, default='seismic.yml',  help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        default='church',
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--sample",
        # action="store_true",
        default=True,
        help="Whether to produce samples from the model",
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        # default="images",
        default= 'church_test_sigma_0.05',
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        # action="store_true",
        default=True,
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--timesteps", type=int, default=20, help="number of steps involved"
    )
    parser.add_argument(
        "--deg", type=str, default='deno', help="Degradation"
    )
    parser.add_argument(
        "--sigma_0", type=float, default=0.05, help="Sigma_0"
    )
    parser.add_argument(
        "--eta", type=float, default=0.85, help="Eta"
    )
    parser.add_argument(
        "--etaB", type=float, default=1, help="Eta_b (before)"
    )
    parser.add_argument(
        '--subset_start', type=int, default=-1
    )
    parser.add_argument(
        '--subset_end', type=int, default=-1
    )
    parser.add_argument(
        "--num_variations", type=int, default=3, help="number of sampling"
    )

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(
        args.exp, "image_samples", args.image_folder
    )
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(
                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
            )
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args, config = parse_args_and_config()

    ###########
    # deno：denoising,
    # int_r: Interpolation for regular missing data
    # int_ir: Interpolation for irregular missing data
    # int_c: Interpolation for consecutive missing data
    # int_u: Interpolation for unknown missing patterns
    # cs2, cs4: Compressed sensing

    args.deg='deno' #  deno int_r int_ir int_c int_u
    args.sigma_0=0.5#0.001
    args.timesteps=20
    args.eta=0.9#0.85
    args.etaB=1.0 #1
    args.num_variations=1 # Select acceleration 1000/args.timesteps times
    config.sampling.batch_size=1
    ###########

    # args.log_path_model ='/home/shendi_mcj/code/Reproducible/ddim-main/exp/logs/marmousi_lin'
    #MmsSegyopenf_pow4/320000  marmousi_R_pow4/400000
    args.log_path_model = '/home/shendi_mcj/code/Reproducible/ddim-main/exp/logs/MmsSegyopenf_pow4'


    #
    args.image_folder = 'exp/logs/marmousi/results/den'
    config.data.image_size = 128
    config.sampling.ckpt_id = 320000#250000  320000
    config.model.ch_mult=[1, 1, 2, 2, 4, 4] #(1, 2, 3, 4)   #[1, 2, 2, 3, 4] [1, 1, 2, 2, 4, 4]
    config.diffusion.beta_schedule='pow4' # 'cosine' "linear" "const" "jsd"  "sigmoid" "pow"
    args.subset_start=0
    config.data.seis_rescaled = False


    #### load sigma_map
    case = 2
    # Generate the sigma map
    from seis_utils.generateSigmaMap import peaks, gaussian_kernel, sincos_kernel, generate_gauss_kernel_mix, \
        Panke100_228_19_147Sigma, MonoPao
    if case == 1:
        # Test case 1
        sigma = peaks(256)
    elif case == 2:
        # Test case 2
        sigma = sincos_kernel()
    elif case == 3:
        # Test case 3
        sigma = generate_gauss_kernel_mix(256, 256)
    elif case == 4:
        sigma = Panke100_228_19_147Sigma()
    elif case == 5:
        sigma = MonoPao()
    elif case == 6:
        sigma = gaussian_kernel()
    sigma = 0.1 + (sigma - sigma.min()) / (sigma.max() - sigma.min()) * (0.5 - 0.1)
    # sigma_map = cv2.resize(sigma, (config.data.image_size, config.data.image_size))

    import scipy.io as sio
    original = sio.loadmat('/home/shendi_mcj/datasets/seismic/marmousi/marmousi35/marmousi35.mat')[
        'data']  # shape(2441, 13601)
    obs_GT1 = original[750:750+128,7200:7200+128]


    config.data.image_shape = obs_GT1.shape

    plot(obs_GT1, dpi=300, figsize=(3, 3))
    # plot_cmap(obs_GT1, 300, (3.7, 3), data_range=[-1, 1], cmap=plt.cm.seismic, cbar=True)
    obs_GT = torch.from_numpy(obs_GT1).contiguous().view(1, -1, obs_GT1.shape[-2], obs_GT1.shape[-1]).type(
        torch.FloatTensor).to(device)

    sigma_preset = args.sigma_0 * abs(obs_GT).max()
    print("sigma_preset:", sigma_preset)
    plot_cmap((sigma_preset * abs(obs_GT).max()).cpu().numpy() * np.ones_like(obs_GT1), 300, (3.7, 3),
              data_range=[0.0, 0.3], cmap=plt.cm.jet, cbar=True)



    obs = obs_GT + sigma_preset * torch.randn_like(obs_GT)


    from utils.estimate_sigma_using_VInonIID import estimate_sigma_using_VInonIID
    sigma_dict, sigma_map_prd = estimate_sigma_using_VInonIID(obs.view(1, -1, obs.shape[2], obs.shape[3]))
    plot_cmap(sigma_map_prd, 300, (3.7, 3), data_range=[0.0, 0.3], cmap=plt.cm.jet, cbar=True)
    args.sigma_0 = 1.0 * sigma_dict['median']
    # args.sigma_0 = sigma_dict['min']
    # args.sigma_0 = 1.0 * sigma_dict['max']  # 2*
    # args.sigma_0 = sigma_dict['median']+0.5*(sigma_dict['max']-sigma_dict['median'])
    # args.sigma_0 = sigma_dict['min']+0.3*(sigma_dict['median']-sigma_dict['min'])

    # args.sigma_0=1.0*sigma_preset



    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    try:
        runner = Diffusion(args, config)
        runner.sample(obs, obs_GT)
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())

import numpy as np
import tqdm
import torch


def get_degradation_matrix(args,config):
    ## get degradation matrix ##
    deg = args.deg
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    H_funcs = None
    if deg[:2] == 'cs':
        compress_by = int(deg[2:])
        from functions.svd_replacement import WalshHadamardCS,DCTCS,RandomProjectionCS
        # H_funcs = WalshHadamardCS(config.data.channels, config.data.image_shape, compress_by,
        #                           torch.randperm(config.data.image_shape[0] * config.data.image_shape[1],
        #                                          device=device), device)
        H_funcs = DCTCS(config.data.channels, config.data.image_shape, compress_by,
                                  torch.randperm(config.data.image_shape[0] * config.data.image_shape[1],
                                                 device=device), device)



    elif deg[:3] == 'inp':
        from functions.svd_replacement import Inpainting
        if deg == 'inp_lolcat':
            loaded = np.load("inp_masks/lolcat_extra.npy")
            mask = torch.from_numpy(loaded).to(device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 1
        elif deg == 'inp_lorem':
            loaded = np.load("inp_masks/lorem3.npy")
            mask = torch.from_numpy(loaded).to(device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 1
        else:
            missing_r = torch.randperm(config.data.image_size ** 2)[:config.data.image_size ** 2 // 2].to(
                device).long() * 1
        # missing_g = missing_r + 1
        # missing_b = missing_g + 1
        # missing = torch.cat([missing_r, missing_g, missing_b], dim=0)

        # missing_r = torch.randperm(config.data.image_size ** 2)[:config.data.image_size ** 2 // 2].to(
        #         device).long()
        H_funcs = Inpainting(config.data.channels, config.data.image_shape, missing_r, device)
    elif deg[:3] == 'int':
        from functions.svd_replacement import Inpainting
        if deg == 'int_u':
            samples=args.obs
            # M = (~torch.isnan(samples.squeeze()) & (samples.squeeze() != 0.5)).int()
            M = (~torch.isnan(samples.squeeze()) & (samples.squeeze() != 0)).int()
            print('mask.shape:', M.shape)
            mask = M
            mask = mask.reshape(-1)
            missing = torch.nonzero(mask == 0).long().reshape(-1)
        elif deg == 'int_r':
            def regular_mask(image_shape, a):
                n_col = image_shape[1]  # data.shape[-1]
                mask = torch.zeros((image_shape[0], image_shape[1])).to(device)
                for i in range(n_col):
                    if (i + 1) % a == 1:
                        mask[:, i] = 1
                    else:
                        mask[:, i] = 0
                return mask

            mask = regular_mask(image_shape=config.data.image_shape, a=3)
            mask = mask.reshape(-1)
            missing = torch.nonzero(mask == 0).long().reshape(-1)
        elif deg == 'int_ir':
            # 固定随机种子，确保每次运行相同
            seed = 2025
            torch.manual_seed(seed)

            M= torch.ones(config.data.image_shape[0],config.data.image_shape[1]).to(device)
            # 概率置零的概率（这里设置为 0.3）
            prob = 0.75
            # 生成概率矩阵（决定每列是否置零）
            prob_tensor = torch.rand(M.shape[1]) < prob
            # 将符合条件的列置零
            M[:, prob_tensor] = 0
            # M[:, 13 * config.data.image_shape[1] // 16: 14 * config.data.image_shape[1] // 16] = 0
            # M[:, 1 * config.data.image_shape[1] // 8: 2 * config.data.image_shape[1] // 8] = 0
            # M[:, 0 * config.data.image_shape[1] // 8: 3 * config.data.image_shape[1] // 8] = 1
            mask=M
            mask = mask.reshape(-1)
            missing = torch.nonzero(mask == 0).long().reshape(-1)
        elif deg == 'int_c':
            # # 创建一个全为1的矩阵
            M = torch.ones(config.data.image_shape[0], config.data.image_shape[1]).to(device)
            # 将后一半列的值置为0
            # M[:, config.data.image_shape[1] // 2:] = 0
            M[:, 2*config.data.image_shape[1] // 8: 3*config.data.image_shape[1] // 8] = 0
            M[:, 5 * config.data.image_shape[1] // 8: 6 * config.data.image_shape[1] // 8] = 0
            mask=M
            mask = mask.reshape(-1)
            missing = torch.nonzero(mask == 0).long().reshape(-1)
        elif deg == 'int_fix':
            # # 加载 .npy 文件为 NumPy 数组
            # M = np.load('/home/shendi_mcj/code/Reproducible/snips_torch-main/inp_masks/text_mask_2.npy')
            # M[M<255]=0
            # M = M / M.max()
            # # 将 NumPy 数组转换为 PyTorch 的 Tensor
            # M = torch.from_numpy(M).float().to(device)
            # M = torch.FloatTensor(M).to(device)

            # fiexed mask
            M_np=np.load('/home/shendi_mcj/code/Reproducible/snips_torch-main/inp_masks/mask_rnd_05.npy')
            #'/home/shendi_mcj/code/Reproducible/snips_torch-main/inp_masks'
            M = torch.from_numpy(M_np).float().to(device)

            mask=M
            mask = mask.reshape(-1)
            missing = torch.nonzero(mask == 0).long().reshape(-1)

        elif deg == 'int_sp':
            def generate_sparse_matrix(rows, cols, min_consecutive_zeros):
                # 生成初始的全一矩阵
                ones_matrix = torch.ones((rows, cols))

                # 计算需要置为0的列的总数（占一半）
                num_zeros_cols = cols // 2

                # 确保至少有 min_consecutive_zeros 列为0
                start_col = torch.randint(0, cols - min_consecutive_zeros + 1, (1,))
                ones_matrix[:, start_col:start_col + min_consecutive_zeros] = 0
                num_zeros_cols -= min_consecutive_zeros

                # 随机选择其他列置为0，直到满足总数要求
                while num_zeros_cols > 0:
                    col_index = torch.randint(0, cols, (1,))
                    if torch.all(ones_matrix[:, col_index:col_index + min_consecutive_zeros] == 1):
                        ones_matrix[:, col_index:col_index + min_consecutive_zeros] = 0
                        num_zeros_cols -= min_consecutive_zeros

                return ones_matrix
            M=generate_sparse_matrix(config.data.image_size,config.data.image_size,8).to(config.device)
            mask = M
            mask = mask.reshape(-1)
            missing = torch.nonzero(mask == 0).long().reshape(-1)


        H_funcs = Inpainting(config.data.channels, config.data.image_shape, missing, device)

    elif deg == 'deno':
        from functions.svd_replacement import Denoising
        H_funcs = Denoising(config.data.channels, config.data.image_shape, device)
    else:
        print("ERROR: degradation type not supported")
        quit()
    return H_funcs
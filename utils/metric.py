import numpy as np
def compare_SNR(real_img,recov_img):
    real_mean = np.mean(real_img)
    tmp1 = real_img - real_mean
    real_var = sum(sum(tmp1*tmp1))

    noise = real_img - recov_img
    noise_mean = np.mean(noise)
    tmp2 = noise - noise_mean
    noise_var = sum(sum(tmp2*tmp2))
    import math
    if noise_var ==0 or real_var==0:
      s = 999.99
    else:
      s = 10*math.log(real_var/noise_var,10)
    return s
def batch_PSNR(img, imclean):
        batch_size=img.shape[0]
        Img = img.data.cpu().numpy().squeeze()
        Iclean = imclean.data.cpu().numpy().squeeze()
        PSNR = 0
        from skimage.metrics import peak_signal_noise_ratio
        if len(Img.shape) == 2:
            PSNR = peak_signal_noise_ratio(Iclean, Img, data_range=2)
        else:
            for i in range(batch_size):
                PSNR += peak_signal_noise_ratio(Iclean[i, :, :], Img[i, :, :], data_range=2)
        return (PSNR / batch_size)


def batch_SNR(img, imclean):
    batch_size = img.shape[0]
    Img = img.data.cpu().numpy().squeeze()
    Iclean = imclean.data.cpu().numpy().squeeze()
    SNR = 0
    if len(Img.shape) == 2:
        SNR = compare_SNR(Iclean, Img)
    else:
        for i in range(batch_size):
            SNR += compare_SNR(Iclean[i, :, :], Img[i, :, :])
    return (SNR / batch_size)

def batch_SSIM(img, imclean):
        batch_size = img.shape[0]
        Img = img.data.cpu().numpy().squeeze()
        Iclean = imclean.data.cpu().numpy().squeeze()
        SSIM = 0
        from skimage.metrics import structural_similarity
        if len(Img.shape) == 2:
            SSIM = structural_similarity(Iclean[:, :], Img[:, :])
        else:
            for i in range(batch_size):
                SSIM += structural_similarity(Iclean[i, :, :], Img[i, :, :])
        return (SSIM / batch_size)

import torch

class H_functions:
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()
    
    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])
    
    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))
    
    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        return self.V(self.add_zeros(temp))

#a memory inefficient implementation for any general degradation H
class GeneralH(H_functions):
    def mat_by_vec(self, M, v):
        vshape = v.shape[1]
        if len(v.shape) > 2: vshape = vshape * v.shape[2]
        if len(v.shape) > 3: vshape = vshape * v.shape[3]
        return torch.matmul(M, v.view(v.shape[0], vshape,
                        1)).view(v.shape[0], M.shape[0])

    def __init__(self, H):
        self._U, self._singulars, self._V = torch.svd(H, some=False)
        self._Vt = self._V.transpose(0, 1)
        self._Ut = self._U.transpose(0, 1)

        ZERO = 1e-3
        self._singulars[self._singulars < ZERO] = 0
        print(len([x.item() for x in self._singulars if x == 0]))

    def V(self, vec):
        return self.mat_by_vec(self._V, vec.clone())

    def Vt(self, vec):
        return self.mat_by_vec(self._Vt, vec.clone())

    def U(self, vec):
        return self.mat_by_vec(self._U, vec.clone())

    def Ut(self, vec):
        return self.mat_by_vec(self._Ut, vec.clone())

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self._V.shape[0], device=vec.device)
        out[:, :self._U.shape[0]] = vec.clone().reshape(vec.shape[0], -1)
        return out

#Inpainting
class Inpainting(H_functions):
    def __init__(self, channels, image_shape, missing_indices, device):
        self.channels = channels
        # self.img_dim = img_dim
        self._singulars = torch.ones(channels * image_shape[0]*image_shape[1] - missing_indices.shape[0]).to(device)
        self.missing_indices = missing_indices
        self.kept_indices = torch.Tensor([i for i in range(channels * image_shape[0]*image_shape[1]) if i not in missing_indices]).to(device).long()
        self.device=device
        self.image_shape=image_shape

    def V(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, self.kept_indices] = temp[:, :self.kept_indices.shape[0]]
        out[:, self.missing_indices] = temp[:, self.kept_indices.shape[0]:]
        return out.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

    def Vt(self, vec):
        temp = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, :self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0]:] = temp[:, self.missing_indices]
        return out

    def _H(self):
        """
        Multiplies the input vector by H
        """
        vec=torch.eye(self.image_shape[0],self.image_shape[1]).reshape(1,-1).to(self.device)
        # vec=vec.clone().reshape(vec.shape[0], -1)
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        temp = torch.zeros((vec.shape[0], self.channels * self.image_shape[0]* self.image_shape[1]), device=vec.device)
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp


#Denoising
class Denoising(H_functions):
    def __init__(self, channels, image_shape, device):
        self._singulars = torch.ones(channels * image_shape[0]*image_shape[1], device=device)

    def V(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Vt(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

#Super Resolution
class SuperResolution(H_functions):
    def __init__(self, channels, img_dim, ratio, device): #ratio = 2 or 4
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        H = torch.Tensor([[1 / ratio**2] * ratio**2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        #reorder the vector back into patches (because singulars are ordered descendingly)
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(vec.shape[0], self.channels, self.y_dim**2, self.ratio**2, device=vec.device)
        patches[:, :, :, 0] = temp[:, :self.channels * self.y_dim**2].view(vec.shape[0], self.channels, -1)
        for idx in range(self.ratio**2-1):
            patches[:, :, :, idx+1] = temp[:, (self.channels*self.y_dim**2+idx)::self.ratio**2-1].view(vec.shape[0], self.channels, -1)
        #multiply each patch by the small V
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #repatch the patches into an image
        patches_orig = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        return recon

    def Vt(self, vec):
        #extract flattened patches
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        unfold_shape = patches.shape
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #multiply each by the small V transposed
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(vec.shape[0], self.channels * self.img_dim**2, device=vec.device)
        recon[:, :self.channels * self.y_dim**2] = patches[:, :, :, 0].view(vec.shape[0], self.channels * self.y_dim**2)
        for idx in range(self.ratio**2-1):
            recon[:, (self.channels*self.y_dim**2+idx)::self.ratio**2-1] = patches[:, :, :, idx+1].view(vec.shape[0], self.channels * self.y_dim**2)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec): #U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp




# class WalshHadamardCS(H_functions):
#     def __init__(self, channels, image_shape, ratio, perm, device):
#         self.channels = channels
#         self.image_shape = image_shape  # (height, width)
#         self.ratio = ratio
#         self.perm = perm
#         self.device = device
#         self._singulars = torch.ones(channels * image_shape[0] * image_shape[1] // ratio, device=device)
#
#     def fwht_1d(self, vec):
#         """In-place Fast Walsh–Hadamard Transform of a 1D vector."""
#         h = 1
#         while h < vec.shape[-1]:
#             vec = vec.reshape(-1, h * 2)
#             vec = torch.cat([(vec[:, :h] + vec[:, h:]), (vec[:, :h] - vec[:, h:])], dim=1)
#             h *= 2
#         return vec / (vec.shape[-1] ** 0.5)
#
#     def fwht_2d(self, mat):
#         """Apply FWHT along both dimensions of a 2D matrix."""
#         # Apply FWHT to each row
#         mat = torch.stack([self.fwht_1d(row) for row in mat])
#         # Apply FWHT to each column
#         mat = torch.stack([self.fwht_1d(col) for col in mat.T]).T
#         return mat
#
#     def V(self, vec):
#         batch_size = vec.shape[0]
#         temp = torch.zeros(batch_size, self.channels, self.image_shape[0] * self.image_shape[1], device=vec.device)
#         temp[:, :, self.perm] = vec.view(batch_size, -1, self.channels).permute(0, 2, 1)
#         temp = temp.view(batch_size * self.channels, self.image_shape[0], self.image_shape[1])
#         temp = torch.stack([self.fwht_2d(img) for img in temp])
#         return temp.view(batch_size, -1)
#
#     def Vt(self, vec):
#         batch_size = vec.shape[0]
#         temp = vec.view(batch_size * self.channels, self.image_shape[0], self.image_shape[1])
#         temp = torch.stack([self.fwht_2d(img) for img in temp])
#         temp = temp.view(batch_size, self.channels, self.image_shape[0] * self.image_shape[1])
#         temp = temp[:, :, self.perm].permute(0, 2, 1)
#         return temp.view(batch_size, -1)
#
#     def U(self, vec):
#         return vec.view(vec.shape[0], -1)
#
#     def Ut(self, vec):
#         return vec.view(vec.shape[0], -1)
#
#     def singulars(self):
#         return self._singulars
#
#     def add_zeros(self, vec):
#         out = torch.zeros(vec.shape[0], self.channels * self.image_shape[0] * self.image_shape[1], device=vec.device)
#         out[:, :self.channels * self.image_shape[0] * self.image_shape[1] // self.ratio] = vec.view(vec.shape[0], -1)
#         return out




import scipy.fftpack
class DCTCS(H_functions):
    def __init__(self, channels, image_shape, ratio, perm, device):
        self.channels = channels
        self.image_shape = image_shape  # (H, W)
        self.ratio = ratio
        self.perm = perm  # Subsampling permutation
        self.device = device

        total_dim = channels * image_shape[0] * image_shape[1]
        self._singulars = torch.ones(total_dim // ratio, device=device)

    # def dct_2d(self, x):
    #     # x: (B, C, H, W)
    #     x = x.cpu().numpy()
    #     x_dct = scipy.fftpack.dct(scipy.fftpack.dct(x, axis=2, norm='ortho'), axis=3, norm='ortho')
    #     return torch.tensor(x_dct, device=self.device, dtype=torch.float32)
    #
    # def idct_2d(self, x):
    #     x = x.cpu().numpy()
    #     x_idct = scipy.fftpack.idct(scipy.fftpack.idct(x, axis=2, norm='ortho'), axis=3, norm='ortho')
    #     return torch.tensor(x_idct, device=self.device, dtype=torch.float32)

    def dct_2d(self, x):
        if x.requires_grad:
            # 使用不破坏计算图的 PyTorch 实现
            return self.dct_2d_torch(x)
        else:
            # 非 autograd 情况下用 scipy 提高速度
            x_np = x.cpu().numpy()
            x_dct = scipy.fftpack.dct(scipy.fftpack.dct(x_np, axis=2, norm='ortho'), axis=3, norm='ortho')
            return torch.tensor(x_dct, device=self.device, dtype=torch.float32)

    def idct_2d(self, x):
        if x.requires_grad:
            return self.idct_2d_torch(x)
        else:
            x_np = x.cpu().numpy()
            x_idct = scipy.fftpack.idct(scipy.fftpack.idct(x_np, axis=2, norm='ortho'), axis=3, norm='ortho')
            return torch.tensor(x_idct, device=self.device, dtype=torch.float32)

    def dct_2d_torch(self, x):
        # 简单版本：用 DCT 矩阵乘法实现（慢，但有梯度）
        B, C, H, W = x.shape
        x = x.reshape(B * C, H, W)
        dct_mat = self.get_dct_matrix(H).to(x.device)  # 假设正方形图像
        out = torch.matmul(dct_mat, x)
        out = torch.matmul(out, dct_mat.t())
        return out.reshape(B, C, H, W)

    def idct_2d_torch(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B * C, H, W)
        dct_mat = self.get_dct_matrix(H).to(x.device)
        out = torch.matmul(dct_mat.t(), x)
        out = torch.matmul(out, dct_mat)
        return out.reshape(B, C, H, W)

    def get_dct_matrix(self, N):
        import math
        mat = torch.zeros((N, N))
        for k in range(N):
            for n in range(N):
                mat[k, n] = math.cos(math.pi * (n + 0.5) * k / N)
        mat[0, :] *= 1 / math.sqrt(N)
        mat[1:, :] *= math.sqrt(2 / N)
        return mat

    def V(self, vec):
        B = vec.shape[0]
        x = vec.reshape(B, self.channels, self.image_shape[0], self.image_shape[1])  # (B, C, H, W)
        x_dct = self.dct_2d(x).reshape(B, -1)  # (B, C*H*W)
        return x_dct[:, self.perm]  # (B, M)

    def Vt(self, vec):
        B = vec.shape[0]
        total_dim = self.channels * self.image_shape[0] * self.image_shape[1]
        x_full = torch.zeros(B, total_dim, device=vec.device)
        vec = vec.view(1, -1)
        x_full[:, self.perm] = vec  # 放回 perm 所代表的位置
        x_full = x_full.reshape(B, self.channels, self.image_shape[0], self.image_shape[1])
        x_idct = self.idct_2d(x_full)
        return x_idct.reshape(B, -1)

    def U(self, vec):
        return vec.reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        B = vec.shape[0]
        out = torch.zeros(B, self.channels * self.image_shape[0] * self.image_shape[1], device=vec.device)
        # out[:, self.perm] = vec.reshape(B, -1)
        out[:, :self.channels * self.image_shape[0] * self.image_shape[1] // self.ratio] = vec.view(vec.shape[0], -1)
        return out




import torch

import torch


class RandomProjectionCS(H_functions):
    def __init__(self, channels, image_shape, ratio, perm, device):
        self.channels = channels
        self.image_shape = image_shape
        self.ratio = ratio
        self.perm = perm
        self.device = device
        self.total_dim = channels * image_shape[0] * image_shape[1]
        # Define singular values (if needed)
        self._singulars = torch.ones(self.channels * self.image_shape[0] * self.image_shape[1] // self.ratio,
                                     device=device)
        self.proj_dim = self.total_dim // ratio
        self.proj_mat = torch.randn(self.proj_dim, self.total_dim, device=device) / (self.total_dim ** 0.5)
        self.inv_perm = torch.argsort(self.perm)


    def V(self, vec):
        # Permute the input vector and multiply by the projection matrix
        x_flat = vec.view(vec.shape[0], -1)
        x_perm = x_flat[:, self.perm]
        x_full = torch.matmul(x_perm, self.proj_mat)
        return x_full

    def Vt(self, vec):
        # Multiply by the transposed projection matrix and unpermute
        vec = vec.view(1, -1)
        x_full = torch.matmul(vec, self.proj_mat.t()).to(self.device)
        x_unperm = x_full[:, self.inv_perm]
        return x_unperm.view(vec.shape[0], -1)

    def U(self, vec):
        # Flatten the input vector
        return vec.view(vec.shape[0], -1)

    def Ut(self, vec):
        # Flatten the input vector
        return vec.view(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        # Add zeros to match the full dimensionality
        out = torch.zeros(vec.shape[0], self.total_dim, device=vec.device)
        out[:, :self.proj_dim] = vec
        return out


#Convolution-based super-resolution
class SRConv(H_functions):
    def mat_by_img(self, M, v, dim):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, dim,
                        dim)).reshape(v.shape[0], self.channels, M.shape[0], dim)

    def img_by_mat(self, v, M, dim):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, dim,
                        dim), M).reshape(v.shape[0], self.channels, dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, stride = 1):
        self.img_dim = img_dim
        self.channels = channels
        self.ratio = stride
        small_dim = img_dim // stride
        self.small_dim = small_dim
        #build 1D conv matrix
        H_small = torch.zeros(small_dim, img_dim, device=device)
        for i in range(stride//2, img_dim + stride//2, stride):
            for j in range(i - kernel.shape[0]//2, i + kernel.shape[0]//2):
                j_effective = j
                #reflective padding
                if j_effective < 0: j_effective = -j_effective-1
                if j_effective >= img_dim: j_effective = (img_dim - 1) - (j_effective - img_dim)
                #matrix building
                H_small[i // stride, j_effective] += kernel[j - i + kernel.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small.reshape(small_dim, 1), self.singulars_small.reshape(1, small_dim)).reshape(small_dim**2)
        #permutation for matching the singular values. See P_1 in Appendix D.5.
        self._perm = torch.Tensor([self.img_dim * i + j for i in range(self.small_dim) for j in range(self.small_dim)] + \
                                  [self.img_dim * i + j for i in range(self.small_dim) for j in range(self.small_dim, self.img_dim)]).to(device).long()

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)[:, :self._perm.shape[0], :]
        temp[:, self._perm.shape[0]:, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)[:, self._perm.shape[0]:, :]
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp, self.img_dim)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1), self.img_dim).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone(), self.img_dim)
        temp = self.img_by_mat(temp, self.V_small, self.img_dim).reshape(vec.shape[0], self.channels, -1)
        #permute the entries
        temp[:, :, :self._perm.shape[0]] = temp[:, :, self._perm]
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.small_dim**2, self.channels, device=vec.device)
        temp[:, :self.small_dim**2, :] = vec.clone().reshape(vec.shape[0], self.small_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp, self.small_dim)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1), self.small_dim).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone(), self.small_dim)
        temp = self.img_by_mat(temp, self.U_small, self.small_dim).reshape(vec.shape[0], self.channels, -1)
        #permute the entries
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat_interleave(1).reshape(-1)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp
#
#Deblurring
class Deblurring(H_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, ZERO = 3e-2):
        self.img_dim = img_dim
        self.channels = channels
        #build 1D conv matrix
        H_small = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel.shape[0]//2, i + kernel.shape[0]//2):
                if j < 0 or j >= img_dim: continue
                H_small[i, j] = kernel[j - i + kernel.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        #ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small.reshape(img_dim, 1), self.singulars_small.reshape(1, img_dim)).reshape(img_dim**2)
        #sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True) #, stable=True)

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, self.channels).reshape(-1) #mcj

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

#Anisotropic Deblurring
class Deblurring2D(H_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel1, kernel2, channels, img_dim, device):
        self.img_dim = img_dim
        self.channels = channels
        #build 1D conv matrix - kernel1
        H_small1 = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel1.shape[0]//2, i + kernel1.shape[0]//2):
                if j < 0 or j >= img_dim: continue
                H_small1[i, j] = kernel1[j - i + kernel1.shape[0]//2]
        #build 1D conv matrix - kernel2
        H_small2 = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel2.shape[0]//2, i + kernel2.shape[0]//2):
                if j < 0 or j >= img_dim: continue
                H_small2[i, j] = kernel2[j - i + kernel2.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small1, self.singulars_small1, self.V_small1 = torch.svd(H_small1, some=False)
        self.U_small2, self.singulars_small2, self.V_small2 = torch.svd(H_small2, some=False)
        ZERO = 3e-2
        self.singulars_small1[self.singulars_small1 < ZERO] = 0
        self.singulars_small2[self.singulars_small2 < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small1.reshape(img_dim, 1), self.singulars_small2.reshape(1, img_dim)).reshape(img_dim**2)
        #sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True) #, stable=True)

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small1, temp)
        out = self.img_by_mat(out, self.V_small2.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small1.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small2).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small1, temp)
        out = self.img_by_mat(out, self.U_small2.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small1.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small2).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, self.channels).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)
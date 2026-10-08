import torch
from tqdm import tqdm
import torchvision.utils as tvu
import os

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn=None, classes=None):
    with torch.no_grad():
        #setup vectors used in the algorithm
        singulars = H_funcs.singulars()
        Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
        Sigma[:singulars.shape[0]] = singulars
        U_t_y = H_funcs.Ut(y_0)
        Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

        #initialize x_T as given in the paper
        largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
        largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
        large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0)
        inv_singulars_and_zero = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3]).to(singulars.device)
        inv_singulars_and_zero[large_singulars_index] = sigma_0 / singulars[large_singulars_index]
        inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)     

        # implement p(x_T | x_0, y) as given in the paper
        # if eigenvalue is too small, we just treat it as zero (only for init) 
        init_y = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).to(x.device)
        init_y[:, large_singulars_index[0]] = U_t_y[:, large_singulars_index[0]] / singulars[large_singulars_index].view(1, -1)
        init_y = init_y.view(*x.size())
        remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2
        remaining_s = remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).clamp_min(0.0).sqrt()
        init_y = init_y + remaining_s * x
        init_y = init_y / largest_sigmas
        
        #setup iteration variables
        x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        #iterate over the timesteps
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            if cls_fn == None:
                et = model(xt, t)
            else:
                et = model(xt, t, classes)
                et = et[:, :x.shape[1]] #[:, :3]mcj
                et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
            
            if et.size(1) == 6:
                et = et[:, :3]  # [:, :3]mcj
            elif et.size(1) == 2:
                et = et[:, :1]  # [:, :3]mcj

            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            #variational inference conditioned on y
            sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
            sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            xt_mod = xt / at.sqrt()[0, 0, 0, 0]
            V_t_x = H_funcs.Vt(xt_mod)
            SVt_x = (V_t_x * Sigma)[:, :U_t_y.shape[1]]
            V_t_x0 = H_funcs.Vt(x0_t)
            SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

            falses = torch.zeros(V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device)
            cond_before_lite = singulars * sigma_next > sigma_0
            cond_after_lite = singulars * sigma_next < sigma_0
            cond_before = torch.hstack((cond_before_lite, falses))
            cond_after = torch.hstack((cond_after_lite, falses))

            std_nextC = sigma_next * etaC
            sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextC ** 2)

            std_nextA = sigma_next * etaA
            sigma_tilde_nextA = torch.sqrt(sigma_next**2 - std_nextA**2)
            
            diff_sigma_t_nextB = torch.sqrt(sigma_next ** 2 - sigma_0 ** 2 / singulars[cond_before_lite] ** 2 * (etaB ** 2))

            #missing pixels
            Vt_xt_mod_next = V_t_x0 + sigma_tilde_nextC * H_funcs.Vt(et) + std_nextC * torch.randn_like(V_t_x0)

            #less noisy than y (after)
            Vt_xt_mod_next[:, cond_after] = \
                V_t_x0[:, cond_after] + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0)[:, cond_after_lite] + std_nextA * torch.randn_like(V_t_x0[:, cond_after])
            
            #noisier than y (before)
            Vt_xt_mod_next[:, cond_before] = \
                (Sig_inv_U_t_y[:, cond_before_lite] * etaB + (1 - etaB) * V_t_x0[:, cond_before] + diff_sigma_t_nextB * torch.randn_like(U_t_y)[:, cond_before_lite])

            #aggregate all 3 cases and give next prediction
            xt_mod_next = H_funcs.V(Vt_xt_mod_next)
            xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))


    return xs, x0_preds




# DPS with DDPM and intrinsic scale
def dps_sampling(x, seq, model, b, H_funcs, y_0, sigma_0, scale, scale_guidance,cls_fn=None, classes=None):
    # with torch.no_grad():
    # Init random noise
    # x_T = torch.randn((1, 1, x.shape[2] , x.shape[3])).to(x.device)
    # x_t = x_T


    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []



    xs = [x]  # initial x is x_T

    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)

        # at_bar 表示 \bar{\alpha}_t，at_bar_next 表示 \bar{\alpha}_{t-1}
        at_bar = compute_alpha(b, t.long())  # shape: (B,1,1,1)
        at_bar_next = compute_alpha(b, next_t.long())  # shape: (B,1,1,1)
        # beta_t = 1 - at_bar / at_bar_next  # β_t = 1 - α_t / α_{t-1}
        beta_t=b[i]

        xt = xs[-1].to('cuda')
        # xt.requires_grad_()

        # # Predict noisy residual eps_theta(x_t)
        # x_t.requires_grad_()
        # epsilon_t = model(x_t, t).sample

        if cls_fn == None:
            et = model(xt, t)
        else:
            et = model(xt, t, classes)
            et = et[:, :x.shape[1]]  # [:, :3]mcj
            et = et - (1 - at_bar).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 2:
            et = et[:, :1]  # [:, :3]mcj

        # x0_hat = (xt - et * (1 - at_bar).sqrt()) / at_bar.sqrt()

        # Get x0_hat and unconditional
        # x_{t-1} = a_t * x_t + b_t * epsilon(x_t) + sigma_t z_t
        # with b_t = eta_t
        # predict = scheduler.step(epsilon_t, t, x_t)
        # x0_hat =predict.pred_original_sample
        # x_prev = predict.prev_sample  # unconditional DDPM sample x_{t-1}'

        # Step 1: predict x0 from epsilon
        # x0_hat = (xt - et * (1 - at_bar).sqrt()) / at_bar.sqrt()
        x0_hat = (1.0 / at_bar).sqrt() * xt - (1.0 / at_bar - 1).sqrt() * et


        # x0_preds.append(x0_hat.to('cpu'))

        # Step 2: compute mean of q(x_{t-1} | x_t, x0)
        mean = (
                       (at_bar_next.sqrt() * beta_t) * x0_hat +
                       ((1 - beta_t).sqrt() * (1 - at_bar_next)) * xt
               ) / (1.0 - at_bar)

        # Step 3: sample from posterior
        logvar = beta_t.log()
        noise = torch.randn_like(x)
        mask = 1 - (t == 0).float()
        mask = mask.view(-1, 1, 1, 1)
        x_prev = mean + mask * torch.exp(0.5 * logvar) * noise

        # beta_t = 1-at_bar / at_bar_next
        # sigma_t = (beta_t * (1 - at_bar_next) / (1 - at_bar)).sqrt()
        # x_prev = (
        #         (xt - (1 - at_bar).sqrt() * et) * (at_bar_next.sqrt() / at_bar.sqrt())
        #         + sigma_t * noise
        # )

        if type(H_funcs).__name__ == 'Inpainting':
            y = H_funcs.H_pinv(y_0).view(x.shape[0], x.shape[1], x.shape[2],
                         x.shape[3])
            # Guidance
            temp =H_funcs.H_pinv(H_funcs.H(x0_hat)).view(x.shape[0], x.shape[1], x.shape[2], x.shape[2])
            f = torch.norm(temp - y)
            # y_0=y_0.view(x.shape[0], x.shape[1], x.shape[2], x.shape[2])
            # f = torch.norm(x0_hat - y_0)
            g = torch.autograd.grad(f, x0_hat)[0]
        else:
            # Guidance
             y= y_0.view(x.shape[0], x.shape[1], x.shape[2],x.shape[3])
             temp = H_funcs.H(x0_hat).view(x.shape[0], x.shape[1], x.shape[2], x.shape[2])
             f = torch.norm(temp - y)
             g = torch.autograd.grad(f, x0_hat)[0]


        # compute variance schedule
        # 计算当前alpha_t 和 beta_t
        alpha_t = at_bar / at_bar_next  # α_t = \bar{\alpha}_t / \bar{\alpha}_{t-1}
        beta_t = 1 - alpha_t  # β_t = 1 - α_t

        # Guidance weight
        # eta_t = ...
        if (scale_guidance == 1):
            # residual = y_0 - x0_hat  # y - A(x0_hat)
            if type(H_funcs).__name__ == 'Inpainting':
                residual = H_funcs.H(y)-H_funcs.H(x0_hat)  # y - A(x0_hat)
                eta_t = 1.0 / (torch.norm(residual) + 1e-8)  # 防止除以0
            else:
                residual = y_0 - H_funcs.H(x0_hat)  # y - A(x0_hat)
                eta_t = 1.0 / (torch.norm(residual) + 1e-8)  # 防止除以0
        else:
            eta_t = 1.0

        # # DPS update rule = DDPM update rule + guidance
        # eta_t=
        x_t = x_prev - scale * eta_t * g
        # x_t = x_prev
        x_t = x_t.detach_()
        # x0_preds.append(x0_hat.to('cpu'))
        xs.append(x_t.to('cpu'))

    return xs #, x0_preds




from copy import deepcopy
def pseudoinverse_guidance_noisy_funcs(x, seq, model, b, H_funcs, y_0, sigma_noise, scale, scale_guidance,cls_fn=None, classes=None):
    """
    Pseudoinverse guidance sampling, matching DPS input signature.
    H is a linear operator matrix.
    """
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])

    # initialize
    x_0 = H_funcs.H_pinv(y_0).view(*x.size()).detach()
    ti = list(seq)[-1]
    t = torch.ones(n).to(x.device).long() * ti
    alpha_t = compute_alpha(b, t.long())
    x = alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * torch.randn_like(x_0)

    xs = [x]  # initial x is x_T



    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n, device=x.device) * i)
        next_t = (torch.ones(n, device=x.device) * j)

        # compute cumulative alphas and beta_t
        at_bar = compute_alpha(b, t.long())
        at_bar_next = compute_alpha(b, next_t.long())

        alpha_t = at_bar
        alpha_s = at_bar_next
        eta = 0
        c1 = ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt() * eta
        c2 = ((1 - alpha_s) - c1 ** 2).sqrt()

        # beta_t = 1 - at_bar / at_bar_next
        beta_t = b[i]

        xt = xs[-1].to(x.device)
        xt.requires_grad = True

        # predict noise
        et = model(xt, t)
        if et.size(1) == 2:
            et = et[:, :1]

        # estimate x0_hat
        x0_hat = (1.0 / at_bar).sqrt() * xt - torch.sqrt(1.0 / at_bar - 1.0) * et

        # # DDPM mean mu using torch operations
        # # Step 2: compute mean of q(x_{t-1} | x_t, x0)
        # mean = (
        #                (at_bar_next.sqrt() * beta_t) * x0_hat +
        #                ((1 - beta_t).sqrt() * (1 - at_bar_next)) * xt
        #        ) / (1.0 - at_bar)
        #
        # # Step 3: sample from posterior
        # logvar = beta_t.log()
        # noise = torch.randn_like(x)
        # mask = 1 - (t == 0).float()
        # mask = mask.view(-1, 1, 1, 1)
        # x_prev = mean + mask * torch.exp(0.5 * logvar) * noise

        # pseudoinverse guidance via H_funcs singular adjustment
        # create a copy of H_funcs to adjust singular values
        H_tilde = deepcopy(H_funcs)

        # r2_t = 1 - at_bar  # variance term
        r2_t = torch.clamp(1 - at_bar, min=1e-5)

        singulars = H_tilde.singulars().squeeze()  # original singular values
        singulars_inv = 1.0 / (singulars + (sigma_noise**2 / r2_t))
        H_tilde._singulars = singulars_inv.squeeze()  # override for pseudoinverse with noise regularization

        # # original singular values (D,)
        # singulars = H_tilde.singulars().squeeze()
        # # use sigma_noise as regularization parameter
        # lam =  (sigma_noise**2) / r2_t
        # # compute Tikhonov filter factors for all singulars
        # # for small s: s/(s^2 + lam^2); for large s: 1/s
        # filter_small = singulars.div(singulars.pow(2).add(lam.pow(2)))  # s/(s^2+lam^2)
        # singulars_inv = torch.where(singulars >= lam, 1.0 / singulars, filter_small)
        # # override singular values with broadcastable shape
        # H_tilde._singulars = singulars_inv.squeeze()


        # y_0 = H_funcs.H_pinv(y_0).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        # # Guidance
        # temp = H_funcs.H_pinv(H_funcs.H(x0_hat)).view(x.shape[0], x.shape[1], x.shape[2], x.shape[2])
        # # compute guided transform
        # residual_field = (y_0 - temp).view(x0_hat.size(0), -1)

        # compute guided transform
        residual_field = (y_0 - H_funcs.H(x0_hat)).view(x0_hat.size(0), -1)


        # mat = H_funcs.Ht(H_tilde.H(residual_field).view_as(x0_hat))  # H^T H_tilde (y - H x0_hat)

        mat = H_funcs.Ht(H_tilde.H(residual_field))  # H^T H_tilde (y - H x0_hat)

        mat_x = (mat.view_as(x0_hat).detach() * x0_hat).sum()
        guidance = torch.autograd.grad(mat_x, xt)[0]

        # zeta = scale * torch.sqrt(1.0 / torch.linalg.norm(mat))
        # diffusion update with guidance using torch.sqrt
        # x_t = x_prev + zeta * guidance



        # coeff = alpha_t.sqrt()
        coeff = alpha_s.sqrt()
        grad_term_weight=1
        coeff = coeff * alpha_t.sqrt() * grad_term_weight
        # coeff=0

        x_t = alpha_s.sqrt() * x0_hat + c1 * torch.randn_like(xt) + c2 * et + guidance * coeff

        x_t = x_t.detach()
        xs.append(x_t.to('cpu'))

    return xs


def pseudoinverse_guidance_noiseless_funcs(x, seq, model, b, H_funcs, y_0, sigma_noise, scale, scale_guidance,cls_fn=None, classes=None):
    """
    Pseudoinverse guidance sampling, matching DPS input signature.
    H is a linear operator matrix.
    """
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    xs = [x]  # initial x is x_T

    xt = x
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n, device=x.device) * i)
        next_t = (torch.ones(n, device=x.device) * j)

        # compute cumulative alphas and beta_t
        at_bar = compute_alpha(b, t.long())
        at_bar_next = compute_alpha(b, next_t.long())

        alpha_t=at_bar
        alpha_s=at_bar_next
        eta=0.0
        c1 = ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt() * eta
        c2 = ((1 - alpha_s) - c1 ** 2).sqrt()

        # xt = xs[-1].to(x.device)
        # xt.requires_grad = True
        xt = xt.clone().to('cuda').requires_grad_(True)

        # predict noise
        et = model(xt, t)
        if et.size(1) == 2:
            et = et[:, :1]

        # estimate x0_hat
        x0_hat = (1.0 / at_bar).sqrt() * xt - torch.sqrt(1.0 / at_bar - 1.0) * et

        # DDPM mean mu using torch operations
        # Step 2: compute mean of q(x_{t-1} | x_t, x0)
        # mean = (
        #                (at_bar_next.sqrt() * beta_t) * x0_hat +
        #                ((1 - beta_t).sqrt() * (1 - at_bar_next)) * xt
        #        ) / (1.0 - at_bar)
        #
        # # Step 3: sample from posterior
        # logvar = beta_t.log()
        # noise = torch.randn_like(x)
        # mask = 1 - (t == 0).float()
        # mask = mask.view(-1, 1, 1, 1)
        # x_prev = mean + mask * torch.exp(0.5 * logvar) * noise


        # y_0 = H_funcs.H_pinv(y_0).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        # Guidance
        # temp = H_funcs.H_pinv(H_funcs.H(x0_hat)).view(x.shape[0], x.shape[1], x.shape[2], x.shape[2])
        # compute guided transform
        # residual_field = (y_0 - temp).view(x0_hat.size(0), -1)

        # compute guided transform
        mat = (H_funcs.H_pinv(y_0) - H_funcs.H_pinv(H_funcs.H(x0_hat))).reshape(n, -1)
        mat_x = (mat.detach() * x0_hat.reshape(n, -1)).sum()
        guidance = torch.autograd.grad(mat_x, xt)[0]

        # x0_hat = x0_hat * (1 - H_funcs.singulars().view(x0_hat.size())) + (y_0 * H_funcs.singulars()).view(x0_hat.size())
        # guidance = guidance * (1 - H_funcs.singulars().view(guidance.size()))

        # coeff = alpha_s.sqrt()
        # grad_term_weight=1
        # coeff = coeff * alpha_t.sqrt() * grad_term_weight
        coeff = alpha_t.sqrt()

        x_t = alpha_s.sqrt() * x0_hat+ c1 * torch.randn_like(xt) + c2 * et + guidance * coeff

        x_t = x_t.detach()
        xs.append(x_t.to('cpu'))

    return xs



def pseudoinverse_guidance_noisy(x, seq, model, b, H, y_0, sigma_y, scale, cls_fn=None, classes=None):
    """ Pseudoinverse guidance reconstruction for linear inverse problems. """

    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]  # initial x is x_T

    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)

        # at_bar 表示 \bar{\alpha}_t，at_bar_next 表示 \bar{\alpha}_{t-1}
        at_bar = compute_alpha(b, t.long())  # shape: (B,1,1,1)
        at_bar_next = compute_alpha(b, next_t.long())  # shape: (B,1,1,1)
        # beta_t = 1 - at_bar / at_bar_next  # β_t = 1 - α_t / α_{t-1}
        beta_t=b[i]

        xt = xs[-1].to('cuda')
        xt.requires_grad_()

        if cls_fn == None:
            et = model(xt, t)
        else:
            et = model(xt, t, classes)
            et = et[:, :x.shape[1]]  # [:, :3]mcj
            et = et - (1 - at_bar).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 2:
            et = et[:, :1]  # [:, :3]mcj

        x0_hat = (1.0 / at_bar).sqrt() * xt - (1.0 / at_bar - 1).sqrt() * et
        # x0_hat.clamp_(min=-1, max=1.0)

        # x0_preds.append(x0_hat.to('cpu'))

        # Step 2: compute mean of q(x_{t-1} | x_t, x0)
        mean = (
                       (at_bar_next.sqrt() * beta_t) * x0_hat +
                       ((1 - beta_t).sqrt() * (1 - at_bar_next)) * xt
               ) / (1.0 - at_bar)
        # mean = (
        #         (1.0 / at_bar).sqrt() * xt
        #         - (1.0 / at_bar - 1).sqrt() * et
        # )

        # Step 3: sample from posterior
        logvar = beta_t.log()
        noise = torch.randn_like(x)
        mask = 1 - (t == 0).float()
        mask = mask.view(-1, 1, 1, 1)
        x_prev = mean + mask * torch.exp(0.5 * logvar) * noise

        g1 = y_0.view(x.shape) - torch.matmul(H, x0_hat)

        # r2_t = 1 - at_bar
        r2_t = torch.clamp(1 - at_bar, min=1e-4)
        I = torch.zeros(H.shape, device=x.device)
        for c in range(H.shape[1]):
            I[0, c, :, :] = torch.eye(H.shape[2], device=x.device)

        g2 = (torch.matmul(H, H.transpose(2, 3)) + (sigma_y ** 2 / r2_t) * I).inverse()
        g = torch.matmul(torch.matmul(g1.transpose(2, 3), g2), H).transpose(2, 3)


        mat = (g.detach() * x0_hat).sum()
        guidance = torch.autograd.grad(mat, xt)[0]

        zeta = scale * torch.sqrt(1.0 / torch.norm(mat))
        # zeta = scale*1.0 / (torch.norm(mat)+ 1e-8)
        # zeta=torch.sqrt(at_bar)

        x_t = x_prev + zeta * guidance
        # x_t.clamp_(min=-1,max=1)


        # current_beta_t= 1 - at_bar / at_bar_next
        # variance = (1 - at_bar_next) / (1 - at_bar) * current_beta_t
        # r = torch.sqrt(variance / (variance + 1))
        # current_alpha_t = 1 / (1 + variance)
        # x_t = x_prev + scale * guidance * (r ** 2) * torch.sqrt(current_alpha_t)


        x_t = x_t.detach_()
        # x0_preds.append(x0_hat.to('cpu'))
        xs.append(x_t.to('cpu'))


    return xs  # , x0_preds


def pseudoinverse_guidance_inpainting(x, seq, model, b, mask, H_funcs, y, sigma_noise, scale, cls_fn=None, classes=None):
    """ Pseudoinverse guidance reconstruction for inpainting + denoising """

    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]  # initial x is x_T

    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)

        # at_bar 表示 \bar{\alpha}_t，at_bar_next 表示 \bar{\alpha}_{t-1}
        at_bar = compute_alpha(b, t.long())  # shape: (B,1,1,1)
        at_bar_next = compute_alpha(b, next_t.long())  # shape: (B,1,1,1)

        alpha_t = at_bar
        alpha_s = at_bar_next
        eta = 0.0
        c1 = ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt() * eta
        c2 = ((1 - alpha_s) - c1 ** 2).sqrt()

        beta_t = b[i]

        xt = xs[-1].to(x.device)
        xt.requires_grad = True

        if cls_fn == None:
            et = model(xt, t)
        else:
            et = model(xt, t, classes)
            et = et[:, :x.shape[1]]  # [:, :3]mcj
            et = et - (1 - at_bar).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 2:
            et = et[:, :1]  # [:, :3]mcj

        x0_hat = (1.0 / at_bar).sqrt() * xt - (1.0 / at_bar - 1).sqrt() * et
        # x0_hat.clamp_(min=-1, max=1.0)

        # x0_preds.append(x0_hat.to('cpu'))

        # # Step 2: compute mean of q(x_{t-1} | x_t, x0)
        # mean = (
        #                (at_bar_next.sqrt() * beta_t) * x0_hat +
        #                ((1 - beta_t).sqrt() * (1 - at_bar_next)) * xt
        #        ) / (1.0 - at_bar)
        #
        #
        # # Step 3: sample from posterior
        # logvar = beta_t.log()
        # noise = torch.randn_like(x)
        # mask = 1 - (t == 0).float()
        # mask = mask.view(-1, 1, 1, 1)
        # x_prev = mean + mask * torch.exp(0.5 * logvar) * noise

        # r2_t = 1 - at_bar
        r2_t = torch.clamp(1 - at_bar, min=1e-5) #

        # y = H_funcs.H_pinv(y).view(x.shape)
        g = y.view(x.shape) - mask * x0_hat
        g = g * mask * (1 / (1 + sigma_noise ** 2 / r2_t))

        mat = (g.detach() * x0_hat).sum()
        guidance = torch.autograd.grad(mat, xt)[0]

        # zeta = scale * torch.sqrt(1.0 / torch.linalg.norm(mat))
        # x_t = x_prev + zeta * guidance

        coeff = alpha_t.sqrt()
        # coeff = alpha_s.sqrt()
        # grad_term_weight=1
        # coeff = coeff * alpha_t.sqrt() * grad_term_weight
        x_t = alpha_s.sqrt() * x0_hat + c1 * torch.randn_like(xt) + c2 * et + guidance * coeff


        x_t = x_t.detach_()
        # x0_preds.append(x0_hat.to('cpu'))
        xs.append(x_t.to('cpu'))


    return xs  # , x0_preds
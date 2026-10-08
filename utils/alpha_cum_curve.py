import numpy as np
import matplotlib.pyplot as plt
import math

# Number of timesteps
num_diffusion_timesteps = 1000

# Define the betas for alpha_bar function
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
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 1, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


beta_schedule='sigmoid' #'cosine' "linear" "const" "jsd"  "sigmoid"
beta_start=0.0001
beta_end=0.02
num_diffusion_timesteps=1000

betas = get_beta_schedule(
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )

alphas = 1.0 - betas




# Calculate cumulative sum of betas
cumprod_alphas = np.cumprod(alphas)

# Plot the beta values
plt.figure(figsize=(10, 6))
plt.plot((1-cumprod_alphas), label='1-cumprod_alphas', color='b')
plt.title('Beta Values from'+beta_schedule+'Schedule')
plt.xlabel('Timesteps')
plt.ylabel('Beta Value')
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# # 生成从 0.01 到 1 的等比数列
# beta_values = np.geomspace(0.01, 1, 1000)
#
# # 绘制等比数列曲线
# plt.plot(beta_values)
# plt.title("Geometric Series from 0.01 to 1")
# plt.xlabel("Steps")
# plt.ylabel("Beta Value")
# plt.grid(True)
# plt.show()

# 定义 alpha_bar 函数，使用等比数列递减函数
# def alpha_bar(t, ratio=0.99):
#     return ratio**t  # ratio 为等比数列的公比 lambda t: alpha_bar(t, ratio=0.99)

# alpha_bar = lambda t: np.exp(-5 * t)



betas = np.linspace(0, 1, num_diffusion_timesteps)
betas= pow(betas, 5)
betas = betas * (beta_end - beta_start) + beta_start

alphas = 1.0 - betas




# Calculate cumulative sum of betas
cumprod_alphas = np.cumprod(alphas)

# Plot the beta values
plt.figure(figsize=(10, 6))
plt.plot((1-cumprod_alphas), label='1-cumprod_alphas', color='b')
plt.title('Beta Values from'+beta_schedule+'Schedule')
plt.xlabel('Timesteps')
plt.ylabel('Beta Value')
plt.legend()
plt.grid(True)
plt.show()

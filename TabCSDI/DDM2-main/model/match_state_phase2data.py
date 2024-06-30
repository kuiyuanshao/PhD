import torch
import argparse
import logging
import os
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from functools import partial
from scipy.stats import norm
import math
from tqdm import tqdm
from numpy import genfromtxt    

class Object:
    def __init__(self, config):
        self.config = config
        self.phase = 'train'
        self.gpu_ids = None
        self.debug = False

def _rev_warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_start * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[n_timestep - warmup_time:] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


#######
to_torch = partial(torch.tensor, dtype=torch.float32, device='cuda:0')

betas = _rev_warmup_beta(0, 1, 1000, 0.7)

alphas = 1. - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
sqrt_alphas_cumprod_prev_np = np.sqrt(
            np.append(1., alphas_cumprod))
sqrt_alphas_cumprod_prev = to_torch(np.sqrt(
            np.append(1., alphas_cumprod)))


data = genfromtxt('data.csv', delimiter=',') # original data
denoised = genfromtxt('denoised.csv', delimiter=',') # denoised data after the noise model

data = torch.from_numpy(data).float().to("cuda")
denoised = torch.from_numpy(denoised).float().to("cuda")


min_lh = [999, 999, 999, 999]
min_t = [-1, -1, -1, -1]
prev_diff = [999., 999., 999., 999.]


for t in range(sqrt_alphas_cumprod_prev.shape[0]): # linear search with early stopping
    for col in range(4):
        noise = data[:, col] - sqrt_alphas_cumprod_prev[t] * denoised[:, col] # difference between original data and denoised data, scaled by a parameter alpha
        noise_mean = torch.mean(noise) # mean of the noise
        noise = noise - noise_mean # distances.

        mu, std = norm.fit(noise.cpu().numpy()) # get the mean and std info from the current noise

        diff = np.abs((1 - sqrt_alphas_cumprod_prev[t]**2).sqrt().cpu().numpy() - std)
        #print(mu, std, (1 - sqrt_alphas_cumprod_prev[t]**2).sqrt(), diff)

        if diff < min_lh[col]:
            min_lh[col] = diff
            min_t[col] = t

        if diff > prev_diff[col]:
            break # find a match!
        else:
            prev_diff[col] = diff
min_t = max(min_t)
print(min_t)
print('done!')
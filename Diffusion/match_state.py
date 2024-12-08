import pickle
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import math
from tqdm import tqdm


def match_state(filename, config=""):
    start = config["diffusion"]["beta_start"]
    end = config["diffusion"]["beta_end"]
    num_steps = config["diffusion"]["num_steps"]
    schedule = config["diffusion"]["schedule"]
    if schedule == "linear":
        betas = np.linspace(start, end, num_steps, dtype=np.float64)
    elif schedule == "quad":
        betas = np.linspace(start**0.5, end**0.5, num_steps, dtype=np.float64) ** 2
        
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)

    with open(filename, "rb") as f:
                observed_values, observed_masks, gt_masks, phase1_cols, phase2_cols, strata_info, binary_cols = pickle.load(f)

    index = np.where(observed_masks[:, phase2_cols[0]] != 0)
    phase1 = observed_values[:, phase1_cols]
    phase2 = observed_values[:, phase2_cols]

    min_lh = [999] * len(phase1_cols)
    min_t = [-1] * len(phase1_cols)
    prev_diff = [999.] * len(phase1_cols)
    
    mu_t = [999] * len(phase1_cols)
    std_t = [999] * len(phase1_cols)
    for col in range(len(phase1_cols)):
        curr_phase1 = phase1[index, col]
        curr_phase2 = phase2[index, col]
        for t in range(sqrt_alphas_cumprod.shape[0]):
            noise = curr_phase1 - sqrt_alphas_cumprod[t] * curr_phase2
            noise_mean = np.mean(noise)
            noise = noise - noise_mean

            mu, std = norm.fit(noise)

            diff = np.abs(np.sqrt(1 - sqrt_alphas_cumprod[t]**2) - std)
        
            if diff < min_lh[col]:
                min_lh[col] = diff
                min_t[col] = t
                mu_t[col] = mu
                std_t[col] = std

            if diff > prev_diff[col]:
                print("find a match!", t, "mean:", mu, "std:", std)
                break # find a match!
            else:
                prev_diff[col] = diff
    
    return min_t
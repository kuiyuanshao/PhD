import pickle
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import math
from tqdm import tqdm
import torch
from tokenizer import Tokenizer

def match_state(dataset, tokenized_values, config="", device="cpu"):
    start = config["diffusion"]["beta_start"]
    end = config["diffusion"]["beta_end"]
    num_steps = config["diffusion"]["num_steps"]
    schedule = config["diffusion"]["schedule"]    
    token_dim = config["model"]["token_emb_dim"]
    
    if schedule == "linear":
        betas = np.linspace(start, end, num_steps, dtype=np.float64)
    elif schedule == "quad":
        betas = np.linspace(start**0.5, end**0.5, num_steps, dtype=np.float64) ** 2
        
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas)
    sqrt_alphas_cumprod = torch.tensor(np.sqrt(alphas_cumprod)).float().to(device)

    index = np.where(dataset.observed_masks[:, dataset.phase2_cols[0]] != 0)
    tokenized_values = tokenized_values[index[0], :]
    
    min_lh = [999] * (len(dataset.phase1_cols) * token_dim)
    min_t = [-1] * (len(dataset.phase1_cols) * token_dim)
    prev_diff = [999.] * (len(dataset.phase1_cols) * token_dim)
    mu_t = [999] * (len(dataset.phase1_cols) * token_dim)
    std_t = [999] * (len(dataset.phase1_cols) * token_dim)
    
    p1 = [1] * (len(dataset.phase1_cols) * token_dim)
    p2 = [1] * (len(dataset.phase1_cols) * token_dim)
    for col in range(len(dataset.phase1_cols)):
        for dim in range(token_dim):
            idx1 = dataset.phase1_cols[col] * token_dim + dim
            p1[dim + col * token_dim] = idx1
            idx2 = dataset.phase2_cols[col] * token_dim + dim
            p2[dim + col * token_dim] = idx2
            for t in range(sqrt_alphas_cumprod.shape[0]):
                noise = tokenized_values[:, idx1] - sqrt_alphas_cumprod[t] * tokenized_values[:, idx2]
                noise_mean = torch.mean(noise)
                noise = noise - noise_mean

                #mu = torch.mean(noise, dim=0)
                #std = torch.std(noise, dim=0)

                mu, std = norm.fit(noise.cpu().numpy())
                #curr_alpha = torch.sqrt(1 - sqrt_alphas_cumprod[t]**2)
                #curr_alpha_rep = curr_alpha.repeat(token_dim)
                diff = np.abs(np.sqrt(1 - sqrt_alphas_cumprod[t].cpu().numpy()**2) - std)

                #diff = np.abs(curr_alpha_rep.cpu().numpy() - std.cpu().numpy())
                index = dim + col * token_dim
                if diff < min_lh[index]:
                    min_lh[index] = diff
                    min_t[index] = t
                    mu_t[index] = mu
                    std_t[index] = std

                if diff > prev_diff[index]:
                    print("find a match!", t, "mean:", mu, "std:", std)
                    break # find a match!
                else:
                    prev_diff[index] = diff
    return min_t, p1, p2
    
    

def calculate_flattened_difference(output_flat, curr_alphas_cumprod, var_idx1, var_idx2, embedding_dim=8):
    start_idx1 = var_idx1 * embedding_dim
    end_idx1 = start_idx1 + embedding_dim
    start_idx2 = var_idx2 * embedding_dim
    end_idx2 = start_idx2 + embedding_dim
    
    # Extract the embeddings
    embedding1 = output_flat[:, start_idx1:end_idx1]
    embedding2 = output_flat[:, start_idx2:end_idx2]

    # Compute the difference
    difference = embedding1 - curr_alphas_cumprod * embedding2
    return difference